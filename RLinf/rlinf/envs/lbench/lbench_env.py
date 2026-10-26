# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vector RLinf environment backed by isolated LBench/Isaac processes."""

from __future__ import annotations

import copy
import json
import operator
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .action import executed_action_to_policy_action, map_action_batch
from .client import cfg_get, create_vector_client, import_lbench_modules
from .norm_range_diagnostics import NormRangeDiagnostics
from .observation import map_observation_batch
from .reward import RewardAdapter, resolve_lbench_success
from .scheduler import BalancedTaskScheduler


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "task"


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    """Return an exact configuration integer without lossy coercion."""

    if isinstance(value, bool):
        raise TypeError(f"LBench {name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"LBench {name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"LBench {name} must be >= {minimum}")
    # ``operator.index`` may preserve a NumPy integer scalar.  Convert only
    # after exactness validation so later counter multiplication uses Python's
    # unbounded integer arithmetic before the explicit int64 range check.
    return int(result)


def _metric(raw_obs: Any) -> dict[str, Any]:
    robot = raw_obs.get("robot", {}) if isinstance(raw_obs, Mapping) else {}
    value = robot.get("metric", {}) if isinstance(robot, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _context(raw_obs: Any) -> dict[str, Any]:
    robot = raw_obs.get("robot", {}) if isinstance(raw_obs, Mapping) else {}
    value = robot.get("context", {}) if isinstance(robot, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _raw_metadata(raw_obs: Any, info: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "context": _context(raw_obs),
        "metric": _metric(raw_obs),
    }
    if isinstance(info, Mapping):
        metadata["step_info"] = dict(info)
    if isinstance(raw_obs, Mapping):
        for key in ("failure_seed_variation", "filter_scene", "domain_randomization"):
            if key in raw_obs:
                metadata[key] = raw_obs[key]
    return metadata


def _scalar_bool(value: Any, *, name: str) -> bool:
    """Return one protocol boolean without accepting ambiguous arrays."""

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"LBench {name} must be a scalar boolean")
    return bool(array.reshape(-1)[0])


class LBenchEnv:
    """Expose the LBench process pool through RLinf's vector-env contract.

    This class runs in an RLinf EnvWorker process.  It launches one independent
    Isaac Python subprocess per local environment over private Unix sockets;
    the RLinf interpreter never imports Isaac Sim.

    ``lazy_start=true`` defers the process pool and task discovery until reset.
    ``offload()`` releases a pool between non-autoreset episodes; the next full
    reset recreates it without rewinding task/seed counters. This permits train
    and evaluation pools to share GPUs without being resident simultaneously.
    Defaults retain eager startup and the caller controls when offload occurs.
    """

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        *,
        vector_client: Any = None,
    ) -> None:
        self.cfg = cfg
        self.num_envs = _require_int("num_envs", num_envs, minimum=1)
        self.seed_offset = _require_int("seed_offset", seed_offset)
        self.total_num_processes = _require_int(
            "total_num_processes", total_num_processes, minimum=1
        )
        self.worker_info = worker_info

        configured_total_value = cfg_get(cfg, "total_num_envs", self.num_envs)
        if configured_total_value is None:
            raise ValueError("LBench total_num_envs must be configured")
        configured_total = _require_int(
            "total_num_envs", configured_total_value, minimum=1
        )
        if configured_total % self.total_num_processes != 0:
            raise ValueError(
                "LBench total_num_envs must be divisible by env ranks × pipeline "
                f"stages: {configured_total} % {self.total_num_processes} != 0"
            )
        expected_local = configured_total // self.total_num_processes
        if expected_local != self.num_envs:
            raise ValueError(
                "LBench local environment allocation mismatch: "
                f"expected {expected_local}, received {self.num_envs}"
            )

        self.group_size = _require_int(
            "group_size", cfg_get(cfg, "group_size", 1), minimum=1
        )
        if self.group_size < 1 or self.num_envs % self.group_size != 0:
            raise ValueError("LBench group_size must divide the local num_envs")
        self.num_group = self.num_envs // self.group_size
        self.auto_reset = bool(cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(cfg_get(cfg, "ignore_terminations", False))
        self.is_eval = bool(cfg_get(cfg, "is_eval", False))
        self._episode_log_cfg = cfg_get(cfg, "episode_log", {})
        self._episode_log_path: Path | None = None
        if bool(cfg_get(self._episode_log_cfg, "enabled", False)):
            if not self.is_eval:
                raise ValueError("LBench episode_log is supported only for evaluation")
            log_dir = cfg_get(self._episode_log_cfg, "dir", None)
            if not isinstance(log_dir, str) or not log_dir.strip():
                raise ValueError("LBench episode_log.dir must be a non-empty path")
            directory = Path(log_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            # seed_offset identifies the environment rank and pipeline stage.
            self._episode_log_path = directory / f"rank_{self.seed_offset:03d}.jsonl"
        self._norm_range_diagnostics = None
        norm_paths = cfg_get(self._episode_log_cfg, "norm_stats_paths", None)
        if norm_paths:
            if self._episode_log_path is None:
                raise ValueError(
                    "LBench norm range diagnostics require episode_log.enabled"
                )
            self._norm_range_diagnostics = NormRangeDiagnostics(norm_paths)
        self.task_sampling = str(cfg_get(cfg, "task_sampling", "balanced_round_robin"))
        if self.task_sampling != "balanced_round_robin":
            raise ValueError(
                "LBench currently supports only task_sampling='balanced_round_robin'"
            )
        self.max_episode_steps = _require_int(
            "max_episode_steps",
            cfg_get(cfg, "max_episode_steps", 1200),
            minimum=1,
        )
        gripper_threshold = float(cfg_get(cfg, "gripper_threshold", 0.5))
        if not np.isfinite(gripper_threshold) or not 0.0 <= gripper_threshold <= 1.0:
            raise ValueError("LBench gripper_threshold must be finite and in [0, 1]")
        self._observation_gripper_joint_threshold = float(
            cfg_get(cfg, "observation_gripper_joint_threshold", 0.4)
        )
        if (
            not np.isfinite(self._observation_gripper_joint_threshold)
            or self._observation_gripper_joint_threshold < 0.0
        ):
            raise ValueError(
                "LBench observation_gripper_joint_threshold must be finite and non-negative"
            )

        self._closed = False
        self._worker_id_base = self.seed_offset * self.num_envs
        self._client = vector_client
        self._client_initialized = False
        self._all_task_names: list[str] | None = None
        self._task_indices: list[int] = []
        self._task_scheduler: BalancedTaskScheduler | None = None
        try:
            lazy_start = cfg_get(cfg, "lazy_start", False)
            if not isinstance(lazy_start, bool):
                raise TypeError("LBench lazy_start must be boolean")
            self._needs_full_reset = lazy_start
            self._fixed_seeds = [
                _require_int("fixed seed", value)
                for value in cfg_get(cfg, "fixed_seeds", [])
            ]
            self._base_seed = _require_int("seed", cfg_get(cfg, "seed", 0))
            self._run_id = _safe_name(str(cfg_get(cfg, "run_id", "rlinf-lbench")))

            self._episode_counts = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self._episode_success = np.zeros(self.num_envs, dtype=bool)
            self._failure_stage_ids = np.zeros(self.num_envs, dtype=np.int64)
            self._task_ids = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_seeds = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_ids = [""] * self.num_envs
            self._episode_override_counts = np.zeros(self.num_envs, dtype=np.int64)
            self._episode_active = np.zeros(self.num_envs, dtype=bool)
            self._current_raw_obs: list[Any] = [None] * self.num_envs
            self._reward_adapter = RewardAdapter.from_config(
                self.num_envs, cfg_get(cfg, "reward_adapter", {})
            )
            self._is_start = True
            if not lazy_start:
                self._ensure_client()
        except Exception:
            self.close()
            raise

    def _ensure_client(self) -> None:
        """Start the pool and validate its task catalog before using it."""
        if self._closed:
            raise RuntimeError("LBench environment is closed")
        if self._client_initialized:
            return
        previous_names = self._all_task_names
        try:
            if self._client is None:
                self._client = create_vector_client(
                    self.cfg, self.num_envs, self._worker_id_base
                )
            names = list(self._client.list_tasks())
            if not names:
                raise RuntimeError("LBench worker pool returned no tasks")
            if previous_names is not None and names != previous_names:
                raise RuntimeError("LBench task catalog changed after offload")
            self._all_task_names = names
            indices = self._resolve_task_indices()
            if previous_names is not None and indices != self._task_indices:
                raise RuntimeError("LBench task selection changed after offload")
            self._task_indices = indices
            self._task_scheduler = BalancedTaskScheduler(indices)
            self._client_initialized = True
        except BaseException:
            self._all_task_names = previous_names
            self._release_client()
            raise

    def _release_client(self) -> None:
        """Drop transport and image ownership; retain scheduling counters."""
        client = self._client
        try:
            if client is not None:
                client.close()
        finally:
            self._client = None
            self._client_initialized = False
            self._needs_full_reset = True
            self._is_start = True
            self._current_raw_obs = [None] * self.num_envs
            active = getattr(self, "_episode_active", None)
            if active is not None:
                active.fill(False)

    def _require_started_client(self) -> None:
        if self._closed:
            raise RuntimeError("LBench environment is closed")
        if not self._client_initialized:
            raise RuntimeError(
                "Reset LBench before stepping an unstarted/offloaded pool"
            )

    @property
    def seed(self) -> int:
        """Return a stable worker identifier for wrapper video directories.

        Episode randomization continues to use ``_episode_spec`` so task and
        seed coverage are independent of how environments are distributed.
        """
        return self._base_seed + self.seed_offset

    def get_wrapper_attr(self, name: str) -> Any:
        """Support Gymnasium wrapper traversal without importing its Env base."""
        return getattr(self, name)

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._episode_lengths.copy()

    @property
    def task_descriptions(self) -> list[str]:
        descriptions: list[str] = []
        for raw_obs in self._current_raw_obs:
            robot = raw_obs.get("robot", {}) if isinstance(raw_obs, Mapping) else {}
            descriptions.append(str(robot.get("instruction", "")))
        return descriptions

    def restore_rollout_progress(self, global_step: int) -> int:
        """Rebuild the deterministic episode cursor after a PPO checkpoint load.

        LBench PPO uses ``auto_reset=false``: every rollout epoch starts exactly
        one episode in every environment slot.  A checkpoint named
        ``global_step_N`` was written after ``N * rollout_epoch`` such starts,
        so the cursor can be reconstructed without serializing Isaac state.

        This method is deliberately valid only before the first reset.  Restoring
        an in-flight simulator episode would require a full simulator snapshot and
        must never be approximated by changing only the scheduler counters.
        """

        if self._closed:
            raise RuntimeError("LBench environment is closed")
        restored_step = _require_int("restore global_step", global_step)
        if self.auto_reset and restored_step > 0:
            raise ValueError(
                "LBench rollout progress can be reconstructed only with "
                "auto_reset=false; auto-reset may complete a variable number of "
                "episodes per PPO step"
            )
        if self._episode_active.any() or any(
            raw_obs is not None for raw_obs in self._current_raw_obs
        ):
            raise RuntimeError(
                "LBench rollout progress must be restored before the first reset"
            )

        rollout_epoch = _require_int(
            "rollout_epoch", cfg_get(self.cfg, "rollout_epoch", 1), minimum=1
        )
        completed_episodes_per_slot = restored_step * rollout_epoch
        if completed_episodes_per_slot > np.iinfo(np.int64).max:
            raise OverflowError("LBench restored episode counter exceeds int64")

        current = self._episode_counts
        if np.any(current != 0) and np.any(current != completed_episodes_per_slot):
            raise RuntimeError(
                "LBench rollout progress conflicts with existing episode counters"
            )
        current.fill(completed_episodes_per_slot)
        return completed_episodes_per_slot

    def _resolve_task_indices(self) -> list[int]:
        requested_indices = cfg_get(self.cfg, "task_indices", None)
        requested_names = cfg_get(self.cfg, "task_names", None)
        if requested_indices is not None and requested_names is not None:
            raise ValueError("set only one of LBench task_indices or task_names")
        if requested_names is not None:
            name_to_index = {
                name: index for index, name in enumerate(self._all_task_names)
            }
            missing = [
                str(name) for name in requested_names if str(name) not in name_to_index
            ]
            if missing:
                raise ValueError(f"unknown LBench task_names: {missing}")
            indices = [name_to_index[str(name)] for name in requested_names]
        elif requested_indices is not None:
            indices = [_require_int("task index", value) for value in requested_indices]
        else:
            indices = list(range(len(self._all_task_names)))
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("LBench task selection must be non-empty and unique")
        invalid = [
            value
            for value in indices
            if value < 0 or value >= len(self._all_task_names)
        ]
        if invalid:
            raise ValueError(f"LBench task_indices out of range: {invalid}")
        metric_names = [_safe_name(self._all_task_names[index]) for index in indices]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError(
                "selected LBench task names collide after metric-name sanitization"
            )
        return indices

    def _global_ordinal(self, env_id: int) -> int:
        configured_total = int(cfg_get(self.cfg, "total_num_envs", self.num_envs))
        return (
            int(self._episode_counts[env_id]) * configured_total
            + self._worker_id_base
            + env_id
        )

    def _episode_spec(self, env_id: int, explicit_seed: int | None = None):
        _, _, _, episode_spec_cls, _ = import_lbench_modules(self.cfg)
        ordinal = self._global_ordinal(env_id)
        task_index = self._task_scheduler.task_for_ordinal(ordinal)
        task_name = self._all_task_names[task_index]
        if explicit_seed is not None:
            episode_seed = int(explicit_seed)
        elif self._fixed_seeds:
            seed_slot = (ordinal // len(self._task_indices)) % len(self._fixed_seeds)
            episode_seed = self._fixed_seeds[seed_slot]
        else:
            episode_seed = self._base_seed + ordinal // len(self._task_indices)
        if episode_seed < 0:
            raise ValueError("LBench episode seed must be non-negative")
        local_index = int(self._episode_counts[env_id])
        episode_id = (
            f"{self._run_id}/{_safe_name(task_name)}/{episode_seed}/"
            f"{self._worker_id_base + env_id:03d}/{local_index:06d}"
        )
        return episode_spec_cls(
            env_id=env_id,
            worker_id=self._worker_id_base + env_id,
            task_index=task_index,
            task_name=task_name,
            seed=episode_seed,
            episode_id=episode_id,
            mode="rlinf_eval" if self.is_eval else "rlinf_rollout",
            randomization_profile=cfg_get(self.cfg, "randomization_profile", None),
            metadata={
                "rlinf_rank": self.seed_offset,
                "global_env_slot": self._worker_id_base + env_id,
            },
        )

    def _map_obs(self, raw_observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return map_observation_batch(
            raw_observations,
            camera_mapping=cfg_get(self.cfg, "camera_mapping", None),
            observation_gripper_joint_threshold=self._observation_gripper_joint_threshold,
        )

    @staticmethod
    def _normalize_env_ids(
        num_envs: int, env_idx: Any = None, env_ids: Any = None
    ) -> list[int]:
        if env_idx is not None and env_ids is not None:
            raise ValueError("pass only one of env_idx or env_ids")
        value = env_ids if env_ids is not None else env_idx
        if value is None:
            result = list(range(num_envs))
        elif isinstance(value, (int, np.integer)):
            result = [_require_int("reset env id", value)]
        else:
            result = [_require_int("reset env id", item) for item in value]
        if len(set(result)) != len(result):
            raise ValueError("LBench reset env ids contain duplicates")
        invalid = [item for item in result if item < 0 or item >= num_envs]
        if invalid:
            raise IndexError(f"LBench reset env ids out of range: {invalid}")
        return result

    def reset(
        self,
        seed: Any = None,
        env_idx: Any = None,
        env_ids: Any = None,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("LBench environment is closed")
        selected = self._normalize_env_ids(self.num_envs, env_idx, env_ids)
        if self._needs_full_reset and len(selected) != self.num_envs:
            raise ValueError(
                "LBench first reset after lazy startup/offload must reset all envs"
            )
        if self._is_start and len(selected) == self.num_envs:
            # EnvWorker marks every explicit rollout bootstrap as ``is_start``.
            # Do not reset the per-slot counters here: doing so would replay the
            # same task, seed and episode id in every rollout_epoch. Counters
            # start at zero in __init__ and intentionally advance for the life
            # of this environment worker.
            self._is_start = False
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = [_require_int("reset seed", value) for value in seed]
            if len(seeds) != len(selected):
                raise ValueError("LBench seed sequence length must match reset env ids")
        else:
            seeds = [None if seed is None else _require_int("reset seed", seed)] * len(
                selected
            )
        starting_client = not self._client_initialized
        self._ensure_client()
        try:
            specs = {
                env_id: self._episode_spec(env_id, explicit_seed=explicit_seed)
                for env_id, explicit_seed in zip(selected, seeds)
            }
            _, raw = self._client.reset(env_ids=selected, episode_specs=specs)
        except BaseException:
            if starting_client:
                # A new pool can be partially reset when one worker fails.
                # Do not leave its Isaac processes resident or advance cursors.
                self._release_client()
            raise
        self._needs_full_reset = False
        self._current_raw_obs = list(raw)
        for env_id in selected:
            self._task_ids[env_id] = specs[env_id].task_index
            self._episode_seeds[env_id] = specs[env_id].seed
            self._episode_ids[env_id] = specs[env_id].episode_id
            self._episode_counts[env_id] += 1
            self._episode_lengths[env_id] = 0
            self._episode_returns[env_id] = 0.0
            self._episode_success[env_id] = False
            self._failure_stage_ids[env_id] = 0
            self._episode_override_counts[env_id] = 0
            self._episode_active[env_id] = True
        # Prime shaping from the reset state.  Some LBench tasks start with a
        # non-zero stage/progress; treating that as progress made by the first
        # policy action would create a synthetic reward.
        self._reward_adapter.reset(selected, self._current_raw_obs)
        return self._map_obs(self._current_raw_obs), {}

    def _episode_info(self) -> dict[str, torch.Tensor]:
        episode_info = {
            "success_once": torch.from_numpy(self._episode_success.copy()),
            "return": torch.from_numpy(self._episode_returns.copy()),
            "episode_len": torch.from_numpy(self._episode_lengths.copy()),
            "task_index": torch.from_numpy(self._task_ids.copy()),
            "seed": torch.from_numpy(self._episode_seeds.copy()),
            "failure_stage_id": torch.from_numpy(self._failure_stage_ids.copy()),
        }
        for task_index in self._task_indices:
            task_name = _safe_name(self._all_task_names[task_index])
            task_mask = self._task_ids == task_index
            episode_info[f"task/{task_name}/success"] = torch.from_numpy(
                np.where(task_mask, self._episode_success.astype(np.float32), np.nan)
            )
            episode_info[f"task/{task_name}/return"] = torch.from_numpy(
                np.where(task_mask, self._episode_returns, np.nan).astype(np.float32)
            )
            episode_info[f"task/{task_name}/episode_len"] = torch.from_numpy(
                np.where(task_mask, self._episode_lengths.astype(np.float32), np.nan)
            )
            episode_info[f"__sum__/task/{task_name}/count"] = torch.from_numpy(
                task_mask.astype(np.float32)
            )
        return episode_info

    def _log_completed_episodes(
        self,
        newly_done: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        raw: Sequence[Mapping[str, Any]],
        info_list: Sequence[Mapping[str, Any]],
    ) -> None:
        """Persist terminal scalars before reset or padded chunk steps lose them."""
        if self._episode_log_path is None:
            return
        for env_id in np.flatnonzero(newly_done):
            info = info_list[env_id]
            info = info if isinstance(info, Mapping) else {}
            metric = _metric(raw[env_id])
            components = info.get("reward_components", {})
            components = components if isinstance(components, Mapping) else {}
            failure_reason = str(
                metric.get("failure_reason") or components.get("failure_reason") or ""
            )
            success = bool(self._episode_success[env_id])
            done_reason = str(info.get("done_reason") or "")
            reason_source = "backend" if done_reason else "derived"
            if not done_reason:
                if success:
                    done_reason = "success"
                elif failure_reason:
                    done_reason = failure_reason
                elif truncated[env_id]:
                    done_reason = "max_step"
                else:
                    done_reason = "terminated"
            task_index = int(self._task_ids[env_id])
            record = {
                "schema_version": 1,
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "writer_pid": os.getpid(),
                "env_rank": self.seed_offset,
                "global_env_slot": self._worker_id_base + int(env_id),
                "run_id": self._run_id,
                "model_label": cfg_get(
                    self._episode_log_cfg, "model_label", self._run_id
                ),
                "checkpoint": cfg_get(self._episode_log_cfg, "checkpoint", None),
                "episode_id": self._episode_ids[env_id],
                "task_index": task_index,
                "task_name": self._all_task_names[task_index],
                "scene_seed": int(self._episode_seeds[env_id]),
                "success": success,
                "episode_length": int(self._episode_lengths[env_id]),
                "done_reason": done_reason,
                "done_reason_source": reason_source,
                "failure_reason": failure_reason,
                "stage_id": int(self._failure_stage_ids[env_id]),
                "terminated": bool(terminated[env_id]),
                "truncated": bool(truncated[env_id]),
                "policy_action_override_count": int(
                    self._episode_override_counts[env_id]
                ),
            }
            if self._norm_range_diagnostics is not None:
                record["norm_range"] = self._norm_range_diagnostics.finish(
                    self._episode_ids[env_id]
                )
            # Each worker owns a separate file. Open in append mode and flush
            # each complete row so a bounded evaluation can retain partial work.
            with self._episode_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
                stream.write("\n")
                stream.flush()

    def _step_no_reset(
        self, actions: Any, inactive_mask: torch.Tensor | None = None
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        inactive = (
            np.zeros(self.num_envs, dtype=bool)
            if inactive_mask is None
            else inactive_mask.detach().cpu().numpy().astype(bool)
        )
        if inactive.all():
            infos = {
                "episode": self._episode_info(),
                "lbench_metadata": [
                    _raw_metadata(raw_obs, {"inactive": True})
                    for raw_obs in self._current_raw_obs
                ],
                "executed_action": torch.zeros((self.num_envs, 8), dtype=torch.float32),
                "policy_action_overridden": torch.zeros(
                    self.num_envs, dtype=torch.bool
                ),
                "transition_executed": torch.zeros(self.num_envs, dtype=torch.bool),
            }
            zeros = torch.zeros(self.num_envs, dtype=torch.float32)
            false = torch.zeros(self.num_envs, dtype=torch.bool)
            return self._map_obs(self._current_raw_obs), zeros, false, false, infos

        # Inactive policy slots are not transitions.  Zero them before strict
        # action validation so a stale terminal-policy output can never abort an
        # otherwise valid partial batch, then mark them transport-inactive.
        safe_actions = np.asarray(actions, dtype=np.float32).copy()
        safe_actions[inactive] = 0.0
        commands = map_action_batch(safe_actions, self.cfg)
        for env_id in np.flatnonzero(inactive):
            commands[int(env_id)] = {}
        previous_raw = list(self._current_raw_obs)
        _, rewards, terminated, truncated, info_list, raw = self._client.step(commands)
        terminated_array = np.asarray(terminated, dtype=bool)
        truncated_array = np.asarray(truncated, dtype=bool)
        transition_executed = np.zeros(self.num_envs, dtype=bool)
        policy_action_overridden = np.zeros(self.num_envs, dtype=bool)
        executed_actions = safe_actions.copy()
        executed_actions[:, -1] = (
            executed_actions[:, -1]
            >= float(cfg_get(self.cfg, "gripper_threshold", 0.5))
        ).astype(np.float32)
        for env_id, info in enumerate(info_list):
            if inactive[env_id]:
                continue
            if not isinstance(info, Mapping):
                info = {}
            transition_executed[env_id] = _scalar_bool(
                info.get("transition_executed", True),
                name="transition_executed",
            )
            reported_override_value = info.get("action_overridden")
            reported_override = (
                _scalar_bool(
                    reported_override_value,
                    name="action_overridden",
                )
                if reported_override_value is not None
                else False
            )
            reported_action = info.get("executed_action")
            if reported_action is None:
                if transition_executed[env_id] and reported_override:
                    raise ValueError(
                        "LBench reported action_overridden=True without an "
                        "executed_action; refusing to label the controller "
                        "transition with the policy action"
                    )
                continue
            if not transition_executed[env_id]:
                # Boundary/control-only calls deliberately report a no-op. The
                # value is masked and must not be turned into a fake action.
                continue
            policy_action = executed_actions[env_id].copy()
            executed_actions[env_id] = executed_action_to_policy_action(
                reported_action,
                self.cfg,
                # An explicit override must be a complete action. Falling back
                # to the policy gripper would silently create a hybrid label.
                fallback=None if reported_override else policy_action,
            )
            actions_differ = not np.allclose(
                executed_actions[env_id],
                policy_action,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            # Treat the decoded action as authoritative even when an older or
            # buggy backend incorrectly reports action_overridden=False.
            policy_action_overridden[env_id] = reported_override or actions_differ
        # Empty commands are sent for transport alignment, but inactive slots
        # retain their terminal observation and contribute no transition.
        for env_id in np.flatnonzero(inactive):
            raw[env_id] = previous_raw[env_id]
        if self._norm_range_diagnostics is not None:
            diagnostics = self._norm_range_diagnostics
            for env_id in np.flatnonzero(transition_executed):
                episode_id = self._episode_ids[env_id]
                diagnostics.observe(
                    episode_id,
                    "transition_state",
                    diagnostics.state(
                        previous_raw[env_id], self._observation_gripper_joint_threshold
                    ),
                )
                diagnostics.observe(episode_id, "policy_action", safe_actions[env_id])
                diagnostics.observe(
                    episode_id, "executed_action", executed_actions[env_id]
                )
        self._episode_lengths[transition_executed] += 1
        if self._episode_log_path is not None:
            self._episode_override_counts += (
                transition_executed & policy_action_overridden
            )
        truncated_array |= (~inactive) & (
            self._episode_lengths >= self.max_episode_steps
        )
        rewards_array, success_events, failure_events = self._reward_adapter.adapt(
            raw,
            info_list,
            rewards,
            transition_executed,
            timeout_mask=truncated_array,
        )
        terminated_array |= success_events | failure_events
        rewards_array[inactive] = 0.0
        terminated_array[inactive] = False
        truncated_array[inactive] = False
        self._episode_returns += rewards_array
        for env_id, raw_obs in enumerate(raw):
            if inactive[env_id]:
                continue
            metric = _metric(raw_obs)
            self._failure_stage_ids[env_id] = int(
                metric.get("current_stage_id", self._failure_stage_ids[env_id]) or 0
            )
            success = resolve_lbench_success(
                raw_obs,
                info_list[env_id],
                env_id=env_id,
            )
            self._episode_success[env_id] |= success
        if self.ignore_terminations:
            terminated_array[:] = False
        newly_done = (~inactive) & (terminated_array | truncated_array)
        if newly_done.any():
            # SimBoxSubprocVectorEnv decides which subprocesses receive STEP
            # from its own active set.  Merely replacing an action with {} is
            # not a no-op: without deactivate(), a terminal Isaac worker would
            # still be stepped again on the next sub-action.
            self._client.deactivate(np.flatnonzero(newly_done).tolist())
            self._log_completed_episodes(
                newly_done, terminated_array, truncated_array, raw, info_list
            )
        self._current_raw_obs = list(raw)
        infos: dict[str, Any] = {
            "episode": self._episode_info(),
            "lbench_metadata": [
                _raw_metadata(raw_obs, info) for raw_obs, info in zip(raw, info_list)
            ],
            "executed_action": torch.from_numpy(executed_actions),
            "policy_action_overridden": torch.from_numpy(policy_action_overridden),
            "transition_executed": torch.from_numpy(transition_executed),
        }
        return (
            self._map_obs(raw),
            torch.from_numpy(rewards_array),
            torch.from_numpy(terminated_array),
            torch.from_numpy(truncated_array),
            infos,
        )

    def _auto_reset_done(
        self,
        dones: torch.Tensor,
        final_obs: dict[str, Any],
        final_info: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        done_ids = torch.nonzero(dones, as_tuple=False).reshape(-1).tolist()
        obs, reset_infos = self.reset(env_ids=done_ids)
        reset_infos.update(
            {
                "final_observation": final_obs,
                "final_info": final_info,
                "_final_observation": dones.clone(),
                "_final_info": dones.clone(),
            }
        )
        return obs, reset_infos

    def step(
        self, actions: Any, auto_reset: bool = True
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        self._require_started_client()
        inactive = torch.from_numpy(~self._episode_active.copy())
        obs, rewards, terminations, truncations, infos = self._step_no_reset(
            actions, inactive_mask=inactive
        )
        dones = torch.logical_or(terminations, truncations)
        if dones.any():
            self._episode_active[dones.detach().cpu().numpy()] = False
        if dones.any() and auto_reset and self.auto_reset:
            obs, reset_infos = self._auto_reset_done(
                dones, copy.deepcopy(obs), copy.deepcopy(infos)
            )
            reset_infos.setdefault("episode", infos["episode"])
            reset_infos.setdefault("lbench_metadata", infos["lbench_metadata"])
            for key in (
                "executed_action",
                "policy_action_overridden",
                "transition_executed",
            ):
                if key in infos:
                    reset_infos.setdefault(key, infos[key])
            infos = reset_infos
        return obs, rewards, terminations, truncations, infos

    def chunk_step(self, chunk_actions: Any):
        """Execute a chunk with an exact first-terminal action mask."""

        self._require_started_client()
        actions = np.asarray(chunk_actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape != (
            self.num_envs,
            actions.shape[1],
            8,
        ):
            raise ValueError(
                f"LBench chunk actions must be [{self.num_envs}, C, 8], "
                f"got {actions.shape}"
            )
        chunk_size = int(actions.shape[1])
        if chunk_size < 1:
            raise ValueError("LBench action chunk must not be empty")

        if self._norm_range_diagnostics is not None:
            diagnostics = self._norm_range_diagnostics
            for env_id in np.flatnonzero(self._episode_active):
                diagnostics.observe(
                    self._episode_ids[env_id],
                    "policy_state",
                    diagnostics.state(
                        self._current_raw_obs[env_id],
                        self._observation_gripper_joint_threshold,
                    ),
                )

        obs_list: list[dict[str, Any]] = []
        infos_list: list[dict[str, Any]] = []
        reward_steps: list[torch.Tensor] = []
        finished = torch.from_numpy(~self._episode_active.copy())
        completed_this_chunk = torch.zeros(self.num_envs, dtype=torch.bool)
        chunk_terminations = torch.zeros((self.num_envs, chunk_size), dtype=torch.bool)
        chunk_truncations = torch.zeros_like(chunk_terminations)
        action_mask = torch.zeros_like(chunk_terminations)
        transition_mask = torch.zeros_like(chunk_terminations)
        executed_action_chunk = torch.zeros(
            (self.num_envs, chunk_size, 8), dtype=torch.float32
        )
        policy_action_overridden = torch.zeros_like(chunk_terminations)
        terminal_obs: dict[str, Any] | None = None
        terminal_info: dict[str, Any] | None = None

        for chunk_index in range(chunk_size):
            active_before_step = ~finished
            if finished.all():
                obs = (
                    copy.deepcopy(obs_list[-1])
                    if obs_list
                    else self._map_obs(self._current_raw_obs)
                )
                infos = (
                    copy.deepcopy(infos_list[-1])
                    if infos_list
                    else {
                        "episode": self._episode_info(),
                        "lbench_metadata": [
                            _raw_metadata(raw_obs, {})
                            for raw_obs in self._current_raw_obs
                        ],
                    }
                )
                obs_list.append(obs)
                infos_list.append(infos)
                reward_steps.append(torch.zeros(self.num_envs, dtype=torch.float32))
                continue
            step_actions = actions[:, chunk_index].copy()
            obs, rewards, terminations, truncations, infos = self._step_no_reset(
                step_actions, inactive_mask=finished
            )
            step_transition_executed = infos["transition_executed"].bool()
            step_overridden = infos["policy_action_overridden"].bool()
            transition_mask[:, chunk_index] = (
                active_before_step & step_transition_executed
            )
            action_mask[:, chunk_index] = (
                transition_mask[:, chunk_index] & ~step_overridden
            )
            executed_action_chunk[:, chunk_index] = infos["executed_action"]
            policy_action_overridden[:, chunk_index] = step_overridden
            rewards = rewards.masked_fill(finished, 0.0)
            newly_terminated = terminations & ~finished
            newly_truncated = truncations & ~finished
            newly_done = newly_terminated | newly_truncated
            chunk_terminations[:, chunk_index] = newly_terminated
            chunk_truncations[:, chunk_index] = newly_truncated
            if newly_done.any():
                if terminal_obs is None:
                    terminal_obs = copy.deepcopy(obs)
                    terminal_info = copy.deepcopy(infos)
                else:
                    for key, value in obs.items():
                        if isinstance(value, torch.Tensor):
                            terminal_obs[key][newly_done] = value[newly_done]
                    assert terminal_info is not None
                    for key, value in infos["episode"].items():
                        terminal_info["episode"][key][newly_done] = value[newly_done]
                    done_ids = (
                        torch.nonzero(newly_done, as_tuple=False).reshape(-1).tolist()
                    )
                    for env_id in done_ids:
                        terminal_info["lbench_metadata"][env_id] = copy.deepcopy(
                            infos["lbench_metadata"][env_id]
                        )
                    # Preserve each environment's own terminal-step protocol
                    # fields.  Different slots may finish on different chunk
                    # indices, so retaining the first terminal snapshot would
                    # relabel later slots with stale executed actions.
                    for key, value in infos.items():
                        terminal_value = terminal_info.get(key)
                        if (
                            isinstance(value, torch.Tensor)
                            and isinstance(terminal_value, torch.Tensor)
                            and value.ndim >= 1
                            and terminal_value.shape == value.shape
                            and value.shape[0] == self.num_envs
                        ):
                            terminal_value[newly_done] = value[newly_done]
            finished |= newly_done
            completed_this_chunk |= newly_done
            if newly_done.any():
                self._episode_active[newly_done.detach().cpu().numpy()] = False
            obs_list.append(obs)
            infos_list.append(infos)
            reward_steps.append(rewards)

        if completed_this_chunk.any() and self.auto_reset:
            assert terminal_obs is not None and terminal_info is not None
            # Keep the last physical-step protocol for environments that did
            # not terminate. ``_auto_reset_done`` stores terminal env values
            # under final_info, but its reset info otherwise replaces the
            # whole batch and would make surviving envs lose executed_action.
            last_step_info = infos_list[-1]
            obs, final_infos = self._auto_reset_done(
                completed_this_chunk, terminal_obs, terminal_info
            )
            for key, value in last_step_info.items():
                if key not in {
                    "final_observation",
                    "final_info",
                    "_final_observation",
                    "_final_info",
                }:
                    # final_info is the per-env terminal aggregate. The
                    # ordinary top-level batch must remain the actual last
                    # chunk step, especially for environments that survived
                    # while peers auto-reset earlier in the same chunk.
                    final_infos.setdefault(key, value)
            obs_list[-1] = obs
            infos_list[-1] = final_infos
        for infos in infos_list:
            infos["action_mask"] = action_mask
            infos["transition_mask"] = transition_mask
            infos["executed_action_chunk"] = executed_action_chunk
            infos["policy_action_overridden_chunk"] = policy_action_overridden
        return (
            obs_list,
            torch.stack(reward_steps, dim=1),
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def update_reset_state_ids(self) -> None:
        """Compatibility hook; scheduling advances atomically on each reset."""

    def offload(self) -> None:
        """Release Isaac processes at an episode boundary, allowing restart.

        Only non-autoreset pools with no active episodes can be offloaded.
        This method does not save simulator state, flush wrapper-owned videos,
        or reset episode/task/seed counters. Callers must finish their rollout
        and video/export work first, then fully reset before using this pool.
        An unstarted, already offloaded, or permanently closed pool is a no-op.
        """
        if self._closed or self._client is None:
            return
        if self.auto_reset:
            raise RuntimeError("LBench offload requires auto_reset=false")
        if self._episode_active.any():
            raise RuntimeError(
                "LBench cannot offload active episodes; finish the rollout first"
            )
        self._release_client()

    def close(self) -> None:
        """Permanently release the process pool, including an active episode."""
        if self._closed:
            return
        self._closed = True
        self._release_client()


__all__ = ["LBenchEnv"]
