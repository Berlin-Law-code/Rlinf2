# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import atexit
import copy
import json
import os
import pickle
import re
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from rlinf.utils.logging import get_logger

_VALID_FORMATS = ("pickle", "lerobot", "diagnostic")
# ``lbench_v3`` is retained as a backwards-compatible alias. Both LBench
# values select feature names only; the installed LeRobot writer decides
# whether the physical dataset layout is v2.1 or v3.0.
_LBENCH_JOINT_SCHEMAS = ("lbench_joint", "lbench_v3")
_VALID_LEROBOT_SCHEMAS = ("default", *_LBENCH_JOINT_SCHEMAS)


_ID_DIR_RE = re.compile(r"^id_(\d+)$")


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    """Return an exact integer, rejecting bools and lossy float coercion."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"collect_episode {name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"collect_episode {name} must be >= {minimum}")
    return result


def _scan_existing_lerobot_shards(save_dir: str, rank: int) -> tuple[int, int]:
    """Return ``(total_episodes, next_shard_id)`` for resume.

    ``next_shard_id`` is ``max(existing_id_numbers) + 1`` over every
    ``id_<int>/`` directory (regardless of whether ``meta/info.json`` is
    finalized) so a crashed session's partial shard is never overwritten.
    Shards with unparseable ``info.json`` contribute 0 to ``total_episodes``.
    """
    rank_dir = os.path.join(save_dir, f"rank_{rank}")
    if not os.path.isdir(rank_dir):
        return 0, 0

    total = 0
    max_id = -1
    for entry in os.listdir(rank_dir):
        m = _ID_DIR_RE.match(entry)
        if m is None:
            continue
        if not os.path.isdir(os.path.join(rank_dir, entry)):
            continue
        max_id = max(max_id, int(m.group(1)))

        info_path = os.path.join(rank_dir, entry, "meta", "info.json")
        try:
            with open(info_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        count = meta.get("total_episodes")
        if isinstance(count, int) and count > 0:
            total += count
    next_shard_id = max_id + 1 if max_id >= 0 else 0
    return total, next_shard_id


class CollectEpisode(gym.Wrapper):
    """Wrapper for collecting rollout data episode by episode.

    Records observations, actions, rewards, termination flags, and info dicts
    at each step. Completed episodes are asynchronously saved to disk in either
    pickle or LeRobot format.

    Supports both single and vectorized environments. When used with
    auto-resetting environments (those that embed ``final_observation`` in
    ``info``), the pre-reset observation is correctly attributed to the finished
    episode, and the post-reset observation is carried over to the next episode.

    Args:
        env: The gymnasium environment to wrap.
        save_dir: Directory for saving collected episode data.
        rank: Worker rank for file naming in distributed settings. Defaults to 0.
        num_envs: Number of parallel environments. Defaults to 1.
        show_goal_site: Whether to show goal visualization in renders.
            Defaults to True.
        export_format: Episode export format, ``"pickle"`` or ``"lerobot"``.
            Defaults to ``"pickle"``.
        robot_type: Robot type for LeRobot metadata. Defaults to ``"panda"``.
        fps: FPS for LeRobot metadata. Defaults to 10.
        only_success: Whether to save only successful episodes. Defaults to False.
        finalize_interval: Call ``writer.finalize()`` every this many completed
            episodes to flush ``info.json`` and ``stats.json`` as a checkpoint.
            ``0`` disables periodic flushing (lerobot only). Defaults to 100.
        max_pending_exports: Maximum number of completed episodes allowed to
            remain in the asynchronous export queue. Once the limit is
            exceeded, rollout waits for the oldest write, bounding camera-data
            memory. ``0`` preserves the historical unbounded behavior.
            Defaults to 0.
        resume: If True and ``export_format == "lerobot"``, reuse ``save_dir``
            across sessions — new episodes land in a fresh ``id_{N}`` shard
            (N = max existing shard id + 1, including partial shards) so the
            in-progress write never touches previously-finalized data. Ignored
            for pickle.
            Defaults to False.
        lerobot_schema: Feature naming contract for LeRobot export.
            ``"lbench_joint"`` emits the exact LBench SFT keys but does not
            select the physical LeRobot storage version. ``"lbench_v3"`` is a
            backwards-compatible alias. Ignored for pickle export. Defaults
            to ``"default"``.
        lbench_gripper_threshold: Threshold used to canonicalize the saved
            LBench action gripper channel to ``0=closed`` / ``1=open``. It must
            match the threshold used by the SimBox action mapper. Defaults to
            0.5.
    """

    def __init__(
        self,
        env: gym.Env,
        save_dir: str,
        rank: int = 0,
        num_envs: int = 1,
        show_goal_site: bool = True,
        export_format: str = "pickle",
        robot_type: str = "panda",
        fps: int = 10,
        only_success: bool = False,
        finalize_interval: int = 100,
        max_pending_exports: int = 0,
        resume: bool = False,
        lerobot_schema: str = "default",
        lbench_gripper_threshold: float = 0.5,
        diagnostic_image_interval: int = 20,
    ):
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if export_format not in _VALID_FORMATS:
            raise ValueError(
                f"Unsupported export_format={export_format!r}, "
                f"expected one of {_VALID_FORMATS}"
            )
        if lerobot_schema not in _VALID_LEROBOT_SCHEMAS:
            raise ValueError(
                f"Unsupported lerobot_schema={lerobot_schema!r}, expected one "
                f"of {_VALID_LEROBOT_SCHEMAS}"
            )

        self.save_dir = save_dir
        self.rank = _require_int("rank", rank, minimum=0)
        self.num_envs = _require_int("num_envs", num_envs, minimum=1)
        self.show_goal_site = show_goal_site
        self.export_format = export_format
        self.robot_type = robot_type
        self.fps = _require_int("fps", fps, minimum=1)
        self.only_success = only_success
        self.finalize_interval = _require_int(
            "finalize_interval", finalize_interval, minimum=0
        )
        self.max_pending_exports = _require_int(
            "max_pending_exports", max_pending_exports, minimum=0
        )
        self.lerobot_schema = lerobot_schema
        self.lbench_gripper_threshold = float(lbench_gripper_threshold)
        self.diagnostic_image_interval = _require_int(
            "diagnostic_image_interval", diagnostic_image_interval, minimum=1
        )
        if not 0.0 <= self.lbench_gripper_threshold <= 1.0:
            raise ValueError("lbench_gripper_threshold must be finite and in [0, 1]")

        self._preexisting_episode_count = 0
        self._next_shard_id = 0
        if export_format == "lerobot":
            self._lerobot_writer: Optional[Any] = None
            self._lerobot_lock = Lock()
            if resume:
                (
                    self._preexisting_episode_count,
                    self._next_shard_id,
                ) = _scan_existing_lerobot_shards(save_dir, rank)
            self._episodes_written = (
                self._preexisting_episode_count
            )  # guarded by _lerobot_lock

        # Single-worker executor keeps write ordering deterministic.
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"collect_episode_export_rank_{self.rank}",
        )
        self._futures: list[Future] = []

        # Per-environment episode state.
        self._episode_ids = [0] * self.num_envs
        self._episode_success = [False] * self.num_envs
        self._segment_ids: list[int] = [0] * self.num_envs
        self._global_step = 0
        # Holds the post-reset obs for auto-reset envs to prepend to next episode.
        self._pending_obs: list[Any] = [None] * self.num_envs
        self._pending_info: list[Any] = [None] * self.num_envs
        self._buffers: list[dict[str, list]] = [
            self._new_buffer() for _ in range(self.num_envs)
        ]

        self._closed = False
        self.logger = get_logger()

        os.makedirs(self.save_dir, exist_ok=True)
        atexit.register(self._finalize_on_exit)

    @property
    def preexisting_episode_count(self) -> int:
        """Number of episodes on disk at construction time (resume mode only)."""
        return self._preexisting_episode_count

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def restore_rollout_progress(self, global_step: int) -> int | None:
        """Restore collection cursors together with a resumable base env.

        The base environment owns the task/seed cursor.  The collector mirrors
        its completed-episode count for collision-free pickle names and, for
        LeRobot output, always advances past every existing (including partial)
        rank-local shard.  It is called only for an actual runner checkpoint
        resume, independent of the collection config's ``resume`` convenience
        flag.
        """

        restore = getattr(self.env, "restore_rollout_progress", None)
        if not callable(restore):
            return None
        if self._closed:
            raise RuntimeError("collect_episode is closed")
        if self._futures:
            raise RuntimeError(
                "collect_episode rollout progress must be restored before writes start"
            )
        if self.export_format == "lerobot" and self._lerobot_writer is not None:
            raise RuntimeError(
                "collect_episode rollout progress must be restored before "
                "writer creation"
            )
        if any(any(values for values in buffer.values()) for buffer in self._buffers):
            raise RuntimeError(
                "collect_episode rollout progress must be restored before reset"
            )
        if any(value is not None for value in self._pending_obs + self._pending_info):
            raise RuntimeError(
                "collect_episode rollout progress conflicts with pending observations"
            )

        completed = restore(global_step)
        completed = _require_int(
            "restored episodes per environment", completed, minimum=0
        )
        if any(value not in (0, completed) for value in self._episode_ids):
            raise RuntimeError(
                "collect_episode rollout progress conflicts with episode ids"
            )
        self._episode_ids = [completed] * self.num_envs

        if self.export_format == "lerobot":
            existing_count, next_shard_id = _scan_existing_lerobot_shards(
                self.save_dir, self.rank
            )
            self._preexisting_episode_count = existing_count
            self._episodes_written = existing_count
            self._next_shard_id = next_shard_id
        return completed

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """Reset the environment and initialise episode buffers.

        Args:
            seed: Optional random seed for environment reset.
            options: Optional dictionary of reset options.

        Returns:
            Tuple of (observation, info) from the underlying environment.
        """
        self._buffers = [self._new_buffer() for _ in range(self.num_envs)]
        self._episode_success = [False] * self.num_envs
        self._pending_obs = [None] * self.num_envs
        self._pending_info = [None] * self.num_envs

        try:
            obs, info = self.env.reset(seed=seed, options=options)
        except TypeError:
            obs, info = self.env.reset()

        self._show_goal_site_visual()
        self._record_reset_obs(obs)
        return obs, info

    def step(self, action, **kwargs):
        """Execute a step and record the transition.

        Args:
            action: Action to execute in the environment.
            **kwargs: Additional arguments forwarded to the underlying step.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        transition_executed = (
            info.get("transition_executed") if isinstance(info, dict) else None
        )
        self._record_step(
            action,
            obs,
            reward,
            terminated,
            truncated,
            info,
            record_mask=transition_executed,
        )
        self._maybe_flush(terminated, truncated, info)
        if (
            transition_executed is not None
            and isinstance(info, dict)
            and "final_observation" in info
        ):
            for env_idx in range(self.num_envs):
                executed = self._scalar_flag(transition_executed, env_idx)
                done = self._scalar_flag(terminated, env_idx) or self._scalar_flag(
                    truncated, env_idx
                )
                if not executed and done and not self._buffers[env_idx]["observations"]:
                    self._seed_reset_frame(env_idx, self._slice_copy(obs, env_idx))
        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        """Execute a chunk of actions, recording each sub-step individually.

        Both pickle and lerobot formats receive step-level records for maximum
        data fidelity.

        Args:
            chunk_actions: Action chunk, typically a tensor of shape
                ``[num_envs, chunk_size, action_dim]``.

        Returns:
            Tuple of (obs_list, rewards, terminations, truncations, infos_list).
        """
        obs_list, rewards, terminations, truncations, infos_list = self.env.chunk_step(
            chunk_actions
        )

        chunk_size = len(obs_list) if isinstance(obs_list, (list, tuple)) else 1
        action_mask = self._extract_chunk_action_mask(infos_list, chunk_size)
        for step_idx in range(chunk_size):
            step_action = (
                chunk_actions[:, step_idx]
                if isinstance(chunk_actions, (torch.Tensor, np.ndarray))
                and chunk_actions.ndim > 1
                else chunk_actions
            )
            step_obs = (
                obs_list[step_idx] if isinstance(obs_list, (list, tuple)) else obs_list
            )
            step_reward = (
                rewards[:, step_idx] if getattr(rewards, "ndim", 1) > 1 else rewards
            )
            step_term = (
                terminations[:, step_idx]
                if getattr(terminations, "ndim", 1) > 1
                else terminations
            )
            step_trunc = (
                truncations[:, step_idx]
                if getattr(truncations, "ndim", 1) > 1
                else truncations
            )
            step_info = (
                copy.deepcopy(infos_list[step_idx])
                if isinstance(infos_list, (list, tuple))
                else infos_list
            )
            self._record_step(
                step_action,
                step_obs,
                step_reward,
                step_term,
                step_trunc,
                step_info,
                record_mask=(
                    action_mask[:, step_idx] if action_mask is not None else None
                ),
            )
            self._maybe_flush(step_term, step_trunc, step_info)

        # Some chunked environments (notably LBench) stop an environment at
        # its first terminal sub-action, pad the remainder of the chunk, and
        # auto-reset only after the whole chunk has finished.  The terminal
        # step therefore cannot carry the post-reset observation.  Seed the
        # already-cleared episode buffer here so the next real action is again
        # state/action aligned.  Environments that terminate on the final slot
        # are already handled by _record_step's normal final_observation path.
        self._seed_deferred_chunk_resets(
            obs_list, terminations, truncations, infos_list, action_mask
        )

        return obs_list, rewards, terminations, truncations, infos_list

    def _extract_chunk_action_mask(self, infos_list, chunk_size: int):
        """Return the physical-transition ``[num_envs, chunk_size]`` mask.

        ``action_mask`` may additionally exclude controller-overridden actions
        from policy-gradient losses. A data collector must retain those real
        transitions, so LBench exposes ``transition_mask`` separately.
        """

        if not isinstance(infos_list, (list, tuple)) or not infos_list:
            return None
        last_info = infos_list[-1]
        if not isinstance(last_info, dict):
            return None
        value = last_info.get("transition_mask", last_info.get("action_mask"))
        if value is None:
            return None
        mask = self._to_numpy(value)
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=bool)
        expected = (self.num_envs, int(chunk_size))
        if mask.shape != expected:
            raise ValueError(
                "collect_episode transition mask must have shape "
                f"{expected}, got {mask.shape}"
            )
        return mask

    def _seed_deferred_chunk_resets(
        self, obs_list, terminations, truncations, infos_list, action_mask
    ) -> None:
        """Seed post-reset observations after a masked, chunk-level reset."""

        if action_mask is None or not isinstance(infos_list, (list, tuple)):
            return
        if not infos_list or not isinstance(infos_list[-1], dict):
            return
        final_step_info = infos_list[-1]
        if "final_observation" not in final_step_info:
            return

        term = np.asarray(self._to_numpy(terminations), dtype=bool)
        trunc = np.asarray(self._to_numpy(truncations), dtype=bool)
        if term.ndim == 1:
            term = term[:, None]
        if trunc.ndim == 1:
            trunc = trunc[:, None]
        completed = np.logical_or(term, trunc).any(axis=1)
        final_obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) else obs_list
        for env_idx in np.flatnonzero(completed):
            # The ordinary final_observation path has already seeded this
            # buffer when termination occurred on the final chunk slot.
            if self._buffers[int(env_idx)]["observations"]:
                continue
            self._seed_reset_frame(
                int(env_idx), self._slice_copy(final_obs, int(env_idx))
            )

    def close(self):
        if self._closed:
            return None
        self._closed = True
        self._finalize_lerobot()
        self._wait_futures()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def flush_collection(self):
        """Finish pending exports before workers offload or the runner exits."""
        self._wait_futures()
        self._finalize_lerobot()

    def _new_buffer(self) -> dict[str, list]:
        return {
            "observations": [],
            "actions": [],
            "policy_actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
            "segment_ids": [],
        }

    def _seed_reset_frame(self, env_idx: int, env_obs: Any) -> None:
        """Seed a fresh buffer with the post-reset state-aligned entry.

        State-aligned fields (observations / rewards / terminated / truncated /
        infos) get a leading reset entry; action-aligned fields (actions,
        segment_ids) stay empty and fill on the first regular step.
        """
        buf = self._buffers[env_idx]
        buf["observations"].append(env_obs)
        buf["rewards"].append(0.0)
        buf["terminated"].append(False)
        buf["truncated"].append(False)
        buf["infos"].append({})

    def _record_reset_obs(self, obs) -> None:
        """Record the initial observation from reset into every env's buffer."""
        for env_idx in range(self.num_envs):
            self._seed_reset_frame(env_idx, self._slice_copy(obs, env_idx))

    @staticmethod
    def _bool_from_env_info(env_info: Any, key: str) -> bool:
        """Read a per-env bool flag (scalar / 0-d / size-1 array all OK)."""
        if not isinstance(env_info, dict) or env_info.get(key) is None:
            return False
        return bool(np.asarray(env_info[key]).any())

    def _record_step(
        self,
        action,
        obs,
        reward,
        terminated,
        truncated,
        info,
        record_mask=None,
    ) -> None:
        """Record one transition into every env's buffer."""
        if record_mask is not None and not any(
            self._scalar_flag(record_mask, env_idx) for env_idx in range(self.num_envs)
        ):
            return
        self._global_step += 1

        has_final_obs = isinstance(info, dict) and "final_observation" in info
        has_final_info = isinstance(info, dict) and "final_info" in info
        if has_final_obs != has_final_info:
            raise ValueError(
                "collect_episode auto-reset protocol requires final_observation "
                "and final_info to be provided together"
            )
        if has_final_obs:
            final_observation = info["final_observation"]
            final_info_batch = info["final_info"]
            info_no_reset = copy.deepcopy(info)
            info_no_reset.pop("final_observation", None)
            info_no_reset.pop("final_info", None)

        for env_idx in range(self.num_envs):
            if record_mask is not None and not self._scalar_flag(record_mask, env_idx):
                continue
            # Auto-reset envs store the pre-reset obs in info["final_observation"];
            # the current `obs` is the post-reset obs for the *next* episode.
            # Only use final_observation for envs that are actually done this step.
            env_done = self._scalar_flag(terminated, env_idx) or self._scalar_flag(
                truncated, env_idx
            )
            if has_final_obs and env_done:
                env_obs = self._slice_copy(final_observation, env_idx)
                env_info = self._slice_copy(final_info_batch, env_idx)
                if not isinstance(env_info, dict):
                    raise TypeError(
                        "collect_episode final_info must provide one mapping per env"
                    )
                self._pending_obs[env_idx] = self._slice_copy(obs, env_idx)
                self._pending_info[env_idx] = self._slice_copy(info_no_reset, env_idx)
                if "intervene_action" in env_info:
                    if "intervene_flag" not in env_info:
                        raise ValueError(
                            "collect_episode final_info must pair intervene_action "
                            "with intervene_flag"
                        )
                    env_info["intervene_action"] = env_info["intervene_action"][-1]
                    env_info["intervene_flag"] = env_info["intervene_flag"][-1]
            else:
                env_obs = self._slice_copy(obs, env_idx)
                env_info = self._slice_copy(info, env_idx)
                if isinstance(env_info, dict):
                    env_info.pop("final_observation", None)
                    env_info.pop("final_info", None)

            record_reset = self._bool_from_env_info(env_info, "record_reset")
            pre_record = self._bool_from_env_info(env_info, "pre_record")

            if record_reset:
                self._buffers[env_idx] = self._new_buffer()
                self._episode_success[env_idx] = False
                self._segment_ids[env_idx] = 0
                self._seed_reset_frame(env_idx, env_obs)
                continue

            if pre_record:
                continue

            if self._bool_from_env_info(env_info, "segment_advance"):
                self._segment_ids[env_idx] += 1

            recorded_action = self._slice_copy(action, env_idx)
            if (
                isinstance(env_info, dict)
                and env_info.get("intervene_action") is not None
                and self._intervene_flag_from_info(env_info)
            ):
                recorded_action = self._copy(env_info["intervene_action"])
            if (
                isinstance(env_info, dict)
                and env_info.get("executed_action") is not None
            ):
                # Recovery/controller wrappers can replace the policy command.
                # Label this transition with the command that produced env_obs.
                recorded_action = self._copy(env_info["executed_action"])

            buf = self._buffers[env_idx]
            buf["observations"].append(env_obs)
            buf["actions"].append(recorded_action)
            if self.export_format == "diagnostic":
                buf["policy_actions"].append(self._slice_copy(action, env_idx))
            buf["rewards"].append(self._slice_copy(reward, env_idx))
            buf["terminated"].append(self._slice_copy(terminated, env_idx))
            buf["truncated"].append(self._slice_copy(truncated, env_idx))
            buf["infos"].append(env_info)
            buf["segment_ids"].append(int(self._segment_ids[env_idx]))

            self._update_success(env_idx, env_info)

    def _reset_env_buffer(self, env_idx: int) -> None:
        """Advance episode counter, clear the buffer, and carry over pending obs."""
        self._episode_ids[env_idx] += 1
        self._buffers[env_idx] = self._new_buffer()
        self._episode_success[env_idx] = False
        self._segment_ids[env_idx] = 0

        if self._pending_obs[env_idx] is not None:
            self._buffers[env_idx]["observations"].append(self._pending_obs[env_idx])
            self._pending_obs[env_idx] = None

            if self._pending_info[env_idx] is not None:
                self._buffers[env_idx]["infos"].append(self._pending_info[env_idx])
                self._pending_info[env_idx] = None
            else:
                self._buffers[env_idx]["infos"].append({})

            self._buffers[env_idx]["rewards"].append(0.0)
            self._buffers[env_idx]["terminated"].append(False)
            self._buffers[env_idx]["truncated"].append(False)

    def _maybe_flush(self, terminated, truncated, info=None) -> None:
        """Save finished episodes and reset their buffers."""
        for env_idx in range(self.num_envs):
            done_by_term = self._scalar_flag(terminated, env_idx)
            done_by_trunc = self._scalar_flag(truncated, env_idx)
            if (done_by_term or done_by_trunc) and self._buffers[env_idx]["actions"]:
                # A terminal control-only notification can carry the decisive
                # success/reason metadata without a physical transition. Fold
                # that metadata into the last real post-state before flushing.
                terminal_info = None
                if isinstance(info, dict) and "final_info" in info:
                    terminal_info = self._slice_copy(info["final_info"], env_idx)
                elif isinstance(info, dict):
                    terminal_info = self._slice_copy(info, env_idx)
                    if isinstance(terminal_info, dict):
                        terminal_info.pop("final_observation", None)
                        terminal_info.pop("final_info", None)
                if isinstance(terminal_info, dict):
                    previous_info = self._buffers[env_idx]["infos"][-1]
                    if not isinstance(previous_info, dict):
                        previous_info = {}
                    merged_info = copy.deepcopy(previous_info)
                    merged_info.update(terminal_info)
                    self._buffers[env_idx]["infos"][-1] = merged_info
                    self._update_success(env_idx, terminal_info)
                # A control-only terminal notification has no new transition.
                # Mark the last physical state as the episode boundary.
                self._buffers[env_idx]["terminated"][-1] = bool(done_by_term)
                self._buffers[env_idx]["truncated"][-1] = bool(done_by_trunc)
            is_success = self._get_episode_success(self._buffers[env_idx], env_idx)
            if self.only_success:
                if done_by_term or done_by_trunc:
                    if is_success and done_by_term:
                        self._flush_episode(env_idx, is_success)
                    # Failed terminal episodes must also be discarded before
                    # an auto-reset environment starts recording the next one.
                    self._reset_env_buffer(env_idx)
            else:
                if done_by_term or done_by_trunc:
                    self._flush_episode(env_idx, is_success)
                    self._reset_env_buffer(env_idx)

    def _flush_episode(self, env_idx: int, is_success: bool) -> None:
        """Dispatch a completed episode to the appropriate format writer."""
        self.logger.info(f"Flush env {env_idx}")
        buf = self._buffers[env_idx]
        if not buf["actions"]:
            return

        if self.export_format == "diagnostic":
            from rlinf.envs.wrappers.episode_diagnostics import write_episode

            destination = os.path.join(
                self.save_dir,
                f"rank_{self.rank:03d}_env_{env_idx:03d}_"
                f"episode_{self._episode_ids[env_idx]:06d}",
            )
            # Completed buffers are detached by _reset_env_buffer. The writer
            # reads these immutable CPU snapshots without another image copy.
            self._submit(
                write_episode,
                destination,
                buf,
                is_success,
                self.fps,
                self.diagnostic_image_interval,
            )
        elif self.export_format == "lerobot":
            ep_data = self._buffer_to_lerobot_ep(buf, env_idx, is_success)
            if ep_data is not None:
                self._submit(self._write_lerobot_episode, ep_data)
        else:
            episode_data = self._copy(
                {
                    "rank": self.rank,
                    "env_idx": env_idx,
                    "episode_id": self._episode_ids[env_idx],
                    "step": self._global_step,
                    "success": is_success,
                    "observations": buf["observations"],
                    "actions": buf["actions"],
                    "rewards": buf["rewards"],
                    "terminated": buf["terminated"],
                    "truncated": buf["truncated"],
                    "infos": buf["infos"],
                }
            )
            label = "success" if is_success else "fail"
            filename = (
                f"rank_{self.rank}_env_{env_idx}_"
                f"episode_{self._episode_ids[env_idx]}_"
                f"step_{self._global_step}_"
                f"{label}.pkl"
            )
            self._submit(
                self._write_pickle, os.path.join(self.save_dir, filename), episode_data
            )

    def _buffer_to_lerobot_ep(
        self, buf: dict, env_idx: int, is_success: bool
    ) -> Optional[list[dict[str, Any]]]:
        """Convert a raw episode buffer into a list of per-step frame dicts.
        Produces the format expected by ``LeRobotDatasetWriter.add_episode``:
        a ``list[dict]`` where every dict represents one step and carries the
        fields ``image``, ``state``, ``actions``, ``task``, ``is_success``,
        ``done``, ``intervene_flag``, and optionally ``wrist_image`` /
        ``extra_view_image``.
        The observations list contains one extra entry prepended at reset time,
        so it is aligned to the actions list by taking the leading N entries.
        Generic-schema steps missing state/action are skipped. The LBench joint
        schema instead rejects missing state, action, camera, or task fields so
        corrupt shards cannot enter the next SFT merge unnoticed.
        Args:
            buf: Raw episode buffer produced by ``_new_buffer``.
            env_idx: Index of the parallel environment this buffer belongs to.
            is_success: Whether the episode was successful.
        Returns:
            A list of per-step frame dicts, or ``None`` if no valid frames
            could be extracted.
        """
        actions = buf["actions"]
        terminated = buf["terminated"]
        truncated = buf["truncated"]
        obs_steps = buf["observations"]
        seg_ids = buf.get("segment_ids", [])
        if not actions:
            return None
        expected_state_entries = len(actions) + 1
        state_aligned = {
            "observations": obs_steps,
            "rewards": buf["rewards"],
            "terminated": terminated,
            "truncated": truncated,
            "infos": buf["infos"],
        }
        misaligned = {
            key: len(values)
            for key, values in state_aligned.items()
            if len(values) != expected_state_entries
        }
        if misaligned or len(seg_ids) != len(actions):
            raise ValueError(
                "collect_episode state/action buffer alignment is invalid: "
                f"actions={len(actions)}, state_aligned={misaligned}, "
                f"segment_ids={len(seg_ids)}"
            )
        # Observation i is the state immediately before action i. The final
        # post-action observation is retained only for termination metadata.
        obs_steps = obs_steps[: len(actions)]
        task_desc = self._extract_task_description(buf, env_idx)
        steps: list[dict[str, Any]] = []
        first_term_step: Optional[int] = None
        for i, action in enumerate(actions):
            obs = obs_steps[i] if i < len(obs_steps) else None
            image, wrist_image, extra_view_image, state = self._extract_obs_image_state(
                obs
            )
            # Overwrite action with intervene action if present.
            np_action = self._to_numpy(action)
            raw_info = buf["infos"][i + 1]
            if not isinstance(raw_info, dict):
                if self.lerobot_schema in _LBENCH_JOINT_SCHEMAS:
                    raise TypeError(
                        "LBench joint-schema export requires per-step info mappings"
                    )
                raw_info = {}
            if isinstance(raw_info, dict) and "final_info" in raw_info:
                # _record_step normally pops final_info before storing; if it
                # leaked through (e.g. nested wrapper that also auto-resets)
                # drop the frame instead of crashing the writer.
                self.logger.warning(
                    "collect_episode: dropping frame %d because info still "
                    "carries final_info; check upstream auto-reset wrappers",
                    i,
                )
                continue
            info_with_intervene = copy.deepcopy(raw_info)

            if (
                "intervene_flag" in info_with_intervene
                and "intervene_action" in info_with_intervene
            ):
                if bool(
                    np.asarray(
                        self._to_numpy(info_with_intervene["intervene_flag"]),
                        dtype=bool,
                    ).all()
                ):
                    np_action = self._to_numpy(info_with_intervene["intervene_action"])
            if state is None or np_action is None:
                if self.lerobot_schema in _LBENCH_JOINT_SCHEMAS:
                    raise ValueError(
                        "LBench joint-schema export cannot omit state or action"
                    )
                continue
            intervene_flag = self._intervene_flag_from_info(info_with_intervene)
            seg_id = int(seg_ids[i]) if i < len(seg_ids) else 0
            frame: dict[str, Any] = {
                "state": np.asarray(state).astype(np.float32),
                "actions": np.asarray(np_action).astype(np.float32).flatten(),
                "task": task_desc,
                "is_success": np.array([is_success], dtype=bool),
                "done": np.array([False], dtype=bool),
                "intervene_flag": np.array([intervene_flag], dtype=bool),
                "segment_id": np.array([seg_id], dtype=np.uint8),
            }
            if self.lerobot_schema in _LBENCH_JOINT_SCHEMAS:
                cameras = {
                    "image": image,
                    "wrist_image": wrist_image,
                    "extra_view_image": extra_view_image,
                }
                for key, camera in cameras.items():
                    frame[key] = self._lbench_camera_frame(camera, key)
                lbench_state = np.asarray(frame["state"], dtype=np.float32)
                lbench_action = np.asarray(frame["actions"], dtype=np.float32).copy()
                if lbench_state.shape != (8,) or lbench_action.shape != (8,):
                    raise ValueError(
                        "LBench joint-schema export requires state/action [8], "
                        f"got {lbench_state.shape}/{lbench_action.shape}"
                    )
                if (
                    not np.isfinite(lbench_state).all()
                    or not np.isfinite(lbench_action).all()
                ):
                    raise ValueError("LBench state/action contains NaN or Inf")
                if lbench_state[-1] < -1.0e-4 or lbench_state[-1] > 1.0 + 1.0e-4:
                    raise ValueError("LBench state gripper must use 0=closed / 1=open")
                if (
                    not isinstance(frame["task"], str)
                    or not frame["task"].strip()
                    or frame["task"].strip() == "unknown task"
                ):
                    raise ValueError(
                        "LBench joint-schema export requires a language instruction"
                    )
                # Store the action that was actually executed by SimBox.  The
                # transport maps the policy scalar through this same threshold
                # to the binary 1=open / 0=closed convention.
                lbench_action[-1] = float(
                    lbench_action[-1] >= self.lbench_gripper_threshold
                )
                frame = {
                    "observation.state": lbench_state,
                    "action": lbench_action,
                    "observation.images.obs_camera": frame["image"],
                    "observation.images.obs_camera_2": frame["wrist_image"],
                    "observation.images.realsense": frame["extra_view_image"],
                    "task": frame["task"].strip(),
                    "is_success": frame["is_success"],
                    "done": frame["done"],
                    "intervene_flag": frame["intervene_flag"],
                    "segment_id": frame["segment_id"],
                }
            else:
                if image is not None:
                    frame["image"] = self._to_uint8(np.asarray(image))
                for key, img in self._expand_multi_view_images(
                    "wrist_image", wrist_image
                ).items():
                    frame[key] = self._to_uint8(np.asarray(img))
                for key, img in self._expand_multi_view_images(
                    "extra_view_image", extra_view_image
                ).items():
                    frame[key] = self._to_uint8(np.asarray(img))
            steps.append(frame)
            # State-aligned flag arrays contain the leading reset entry, just
            # like observations/infos.  The action at i therefore ends at
            # flag i+1.  Check both termination and truncation so a defensive
            # conversion also cuts any accidentally retained post-done tail.
            flag_index = i + 1
            if (
                flag_index < len(terminated)
                and (bool(terminated[flag_index]) or bool(truncated[flag_index]))
                and first_term_step is None
            ):
                first_term_step = len(steps)
        if not steps:
            return None
        end = first_term_step if first_term_step is not None else len(steps)
        steps = steps[:end]
        steps[-1]["done"] = np.array([True], dtype=bool)
        return steps

    def _ensure_lerobot_writer(self, ep_data: dict):
        """Get-or-create the LeRobot writer. Must be called under ``_lerobot_lock``."""
        from rlinf.data.storage.lerobot import LeRobotDatasetWriter

        if self._lerobot_writer is None:
            self._lerobot_writer = LeRobotDatasetWriter()
        shard_id = self._next_shard_id

        if self._lerobot_writer.dataset is None:
            first = ep_data[0]
            if self.lerobot_schema in _LBENCH_JOINT_SCHEMAS:
                features = {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": tuple(first["observation.state"].shape),
                        "names": ["state"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": tuple(first["action"].shape),
                        "names": ["action"],
                    },
                    "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
                    "is_success": {
                        "dtype": "bool",
                        "shape": (1,),
                        "names": ["is_success"],
                    },
                    "intervene_flag": {
                        "dtype": "bool",
                        "shape": (1,),
                        "names": ["intervene_flag"],
                    },
                    "segment_id": {
                        "dtype": "uint8",
                        "shape": (1,),
                        "names": ["segment_id"],
                    },
                }
                for key in (
                    "observation.images.obs_camera",
                    "observation.images.obs_camera_2",
                    "observation.images.realsense",
                ):
                    features[key] = {
                        "dtype": "image",
                        "shape": tuple(first[key].shape),
                        "names": ["height", "width", "channel"],
                    }
                self._lerobot_writer.create(
                    repo_id=os.path.join(
                        self.save_dir, f"rank_{self.rank}", f"id_{shard_id}"
                    ),
                    robot_type=self.robot_type,
                    fps=self.fps,
                    features=features,
                )
                self._next_shard_id = shard_id + 1
                return self._lerobot_writer
            wrist_image_keys = self._collect_image_keys(first, "wrist_image")
            extra_view_image_keys = self._collect_image_keys(first, "extra_view_image")
            self._lerobot_writer.create(
                repo_id=os.path.join(
                    self.save_dir, f"rank_{self.rank}", f"id_{shard_id}"
                ),
                robot_type=self.robot_type,
                fps=self.fps,
                image_shape=first["image"].shape if "image" in first else None,
                state_dim=int(first["state"].shape[-1]),
                action_dim=int(first["actions"].shape[-1]),
                has_image="image" in first,
                wrist_image_keys=wrist_image_keys,
                extra_view_image_keys=extra_view_image_keys,
                has_intervene_flag="intervene_flag" in first,
                has_segment_id="segment_id" in first,
            )
            self._next_shard_id = shard_id + 1
        return self._lerobot_writer

    @staticmethod
    def _collect_image_keys(
        frame: dict[str, Any],
        prefix: str,
    ) -> dict[str, tuple[int, ...]]:
        """Return ``{key: (H, W, C)}`` for all frame keys matching *prefix*.

        Matches both the bare ``prefix`` (e.g. ``wrist_image``) and indexed
        variants (``wrist_image-0``, ``wrist_image-1``, …). The separator is a
        hyphen, not ``/``: lerobot >= 0.3 rejects feature names containing
        ``/`` in ``LeRobotDatasetMetadata.create``.
        """
        return {
            k: tuple(frame[k].shape)
            for k in frame
            if (k == prefix or k.startswith(f"{prefix}-"))
            and isinstance(frame[k], np.ndarray)
            and frame[k].ndim == 3
        }

    def _write_lerobot_episode(self, ep_data: dict) -> None:
        with self._lerobot_lock:
            writer = self._ensure_lerobot_writer(ep_data)
            writer.add_episode(ep_data)
            self._episodes_written += 1
            count = self._episodes_written
            if self.finalize_interval > 0 and count % self.finalize_interval == 0:
                writer.finalize()
                # ``finalize`` releases the underlying LeRobotDataset and sets
                # writer.dataset to None. Drop the wrapper as well so close()
                # cannot finalize it a second time when the total episode count
                # lands exactly on the interval boundary. The next episode, if
                # any, creates a fresh rank-isolated shard in
                # ``_ensure_lerobot_writer``.
                self._lerobot_writer = None

    def _finalize_lerobot(self) -> None:
        """Drain pending futures then write the LeRobot dataset metadata."""
        if self.export_format != "lerobot":
            return
        self._wait_futures()
        with self._lerobot_lock:
            if self._lerobot_writer is not None:
                self._lerobot_writer.finalize()
                self._lerobot_writer = None

    def _write_pickle(self, save_path: str, episode_data: dict) -> None:
        with open(save_path, "wb") as f:
            pickle.dump(episode_data, f)

    def _submit(self, fn, *args) -> None:
        if self._executor is None:
            return
        self._futures.append(self._executor.submit(fn, *args))
        self.logger.info(f"Futures queue length: {len(self._futures)}")
        self._drain_futures()
        while (
            self.max_pending_exports > 0
            and len(self._futures) > self.max_pending_exports
        ):
            # Long LBench episodes can retain several GiB of three-camera
            # frames. Apply backpressure instead of letting a slower encoder
            # create an unbounded queue across rollout epochs.
            oldest = self._futures[0]
            oldest.result()
            self._futures.pop(0)
            self._drain_futures()

    def _drain_futures(self) -> None:
        remaining = []
        for f in self._futures:
            if f.done():
                f.result()
            else:
                remaining.append(f)
        self._futures = remaining

    def _wait_futures(self) -> None:
        for f in self._futures:
            f.result()
        self._futures = []

    def _finalize_on_exit(self) -> None:
        self.close()

    def _update_success(self, env_idx: int, env_info) -> None:
        """Update the per-env success flag from a single-env info dict."""
        if not isinstance(env_info, dict):
            return

        success = self._extract_success_from_info(env_info)
        if success is not None:
            # Keep success sticky during an episode.
            self._episode_success[env_idx] = self._episode_success[env_idx] or success

    def _get_episode_success(self, buf: dict, env_idx: int) -> bool:
        """Determine final episode success by scanning recorded info dicts.

        Checks (in priority order): ``final_info``, ``episode``, and the root
        info dict, looking for ``success_once``, ``success_at_end``, and
        ``success`` keys. Falls back to the incrementally-updated
        ``_episode_success`` flag.
        """
        if self._episode_success[env_idx]:
            return True

        found_any = False
        is_success = False

        for info in buf["infos"]:
            if not isinstance(info, dict):
                continue
            success = self._extract_success_from_info(info)
            if success is not None:
                found_any = True
                is_success = is_success or success

        if found_any:
            return is_success
        return self._episode_success[env_idx]

    @staticmethod
    def _intervene_flag_from_info(info: Any) -> bool:
        """Whether this timestep used human / expert intervention (per-env info)."""
        if not isinstance(info, dict):
            return False
        overridden = info.get("policy_action_overridden")
        if overridden is not None:
            arr = CollectEpisode._to_numpy(overridden)
            if arr is not None and bool(np.asarray(arr, dtype=bool).reshape(-1).any()):
                return True
        val = info.get("intervene_flag")
        if val is None:
            return False
        arr = CollectEpisode._to_numpy(val)
        if arr is None:
            return False
        return bool(np.asarray(arr, dtype=bool).reshape(-1).any())

    @staticmethod
    def _to_bool_scalar(val) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            if val.numel() != 1:
                return None
            return bool(val.item())
        if isinstance(val, np.ndarray):
            if val.size != 1:
                return None
            return bool(val.reshape(-1)[0])
        return bool(val)

    def _extract_success_from_source(self, src) -> Optional[bool]:
        if not isinstance(src, dict):
            return None
        for key in ("success_once", "success_at_end", "success"):
            val = self._to_bool_scalar(src.get(key))
            if val is not None:
                return val
        return None

    def _extract_success_from_info(self, info: dict) -> Optional[bool]:
        """Extract success with episode-level fields taking priority."""
        episode_values: list[bool] = []

        final_info = info.get("final_info", None)
        if isinstance(final_info, dict):
            final_info_success = self._extract_success_from_source(final_info)
            if final_info_success is not None:
                episode_values.append(final_info_success)
            final_episode_success = self._extract_success_from_source(
                final_info.get("episode")
            )
            if final_episode_success is not None:
                episode_values.append(final_episode_success)

        current_episode_success = self._extract_success_from_source(info.get("episode"))
        if current_episode_success is not None:
            episode_values.append(current_episode_success)

        if episode_values:
            return any(episode_values)

        return self._extract_success_from_source(info)

    def _extract_task_description(self, buf: dict, env_idx: int) -> str:
        for obs in reversed(buf["observations"]):
            if not isinstance(obs, dict) or "task_descriptions" not in obs:
                continue
            desc = obs["task_descriptions"]
            if isinstance(desc, (list, tuple)):
                return str(desc[env_idx] if len(desc) == self.num_envs else desc[0])
            return str(desc)
        return "unknown task"

    def _extract_obs_image_state(self, obs):
        """Return ``(image, wrist_image, extra_view_image, state)`` from an obs dict.

        ``wrist_image`` and ``extra_view_image`` are returned as raw numpy
        arrays and may have shape ``[H, W, C]`` *or* ``[N, H, W, C]``.
        Use :meth:`_expand_multi_view_images` to fan them out into
        individually-keyed views before writing.
        """
        if not isinstance(obs, dict):
            return None, None, None, None
        image = obs.get("main_images", obs.get("image", obs.get("full_image")))
        wrist_image = obs.get("wrist_images", obs.get("wrist_image"))
        extra_view_image = obs.get("extra_view_images", obs.get("extra_view_image"))
        state = obs.get("states", obs.get("state"))
        return (
            self._to_numpy(image),
            self._to_numpy(wrist_image),
            self._to_numpy(extra_view_image),
            self._to_numpy(state),
        )

    @staticmethod
    def _expand_multi_view_images(
        base_key: str,
        arr: Optional[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Expand a potentially batched image array into per-view entries.

        * ``[H, W, C]``           → ``{base_key: img}``
        * ``[1, H, W, C]``        → ``{base_key: img[0]}``
        * ``[N, H, W, C]`` (N>1)  → ``{base_key-0: img[0], …, base_key-N-1: img[N-1]}``
        """
        if arr is None:
            return {}
        if arr.ndim == 3:
            return {base_key: arr}
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                return {base_key: arr[0]}
            return {f"{base_key}-{i}": arr[i] for i in range(arr.shape[0])}
        return {base_key: arr}

    def _slice_data(self, data, env_idx: int):
        """Slice batched data for a single env without copying."""
        if isinstance(data, torch.Tensor):
            return (
                data[env_idx]
                if data.dim() > 0 and data.shape[0] == self.num_envs
                else data
            )
        if isinstance(data, np.ndarray):
            return (
                data[env_idx]
                if data.ndim > 0 and data.shape[0] == self.num_envs
                else data
            )
        if isinstance(data, dict):
            return {k: self._slice_data(v, env_idx) for k, v in data.items()}
        if isinstance(data, list):
            return data[env_idx] if len(data) == self.num_envs else data
        return data

    def _slice_copy(self, data, env_idx: int):
        """Slice batched data for a single env and deep-copy the result."""
        return self._copy(self._slice_data(data, env_idx))

    def _scalar_flag(self, flags, env_idx: int) -> bool:
        """Extract a boolean flag for ``env_idx`` from a batched flag."""
        if isinstance(flags, torch.Tensor):
            if flags.dim() > 1:
                return bool(flags[env_idx, -1].item())
            if flags.dim() == 1 and flags.shape[0] == self.num_envs:
                return bool(flags[env_idx].item())
            return bool(flags.item())
        if isinstance(flags, np.ndarray):
            if flags.ndim > 0 and flags.shape[0] == self.num_envs:
                return bool(flags[env_idx])
            return bool(flags)
        return bool(flags)

    @staticmethod
    def _to_numpy(data) -> Optional[np.ndarray]:
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        return (
            (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        )

    @staticmethod
    def _lbench_camera_frame(value: Any, key: str) -> np.ndarray:
        """Validate the exact uint8 HWC image contract emitted by LBench."""

        if value is None:
            raise ValueError(f"LBench joint-schema export is missing camera {key!r}")
        image = np.asarray(value)
        if (
            image.ndim != 3
            or image.shape[-1] != 3
            or any(size < 1 for size in image.shape)
        ):
            raise ValueError(
                "LBench joint-schema cameras must be non-empty HWC RGB; "
                f"{key} has shape {image.shape}"
            )
        if image.dtype != np.uint8:
            raise TypeError(
                "LBench observation mapping must emit uint8 camera frames before "
                f"collection; {key} has dtype {image.dtype}"
            )
        return np.ascontiguousarray(image)

    def _copy(self, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().clone()
        if isinstance(data, np.ndarray):
            return data.copy()
        if isinstance(data, dict):
            return {k: self._copy(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._copy(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self._copy(item) for item in data)
        return data

    def _show_goal_site_visual(self) -> None:
        """Unhide the goal site in environments that support it."""
        if not self.show_goal_site:
            return

        unwrapped = self.env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "unwrapped"):
            unwrapped = unwrapped.unwrapped

        if not hasattr(unwrapped, "goal_site"):
            return

        goal_site = unwrapped.goal_site
        if hasattr(unwrapped, "_hidden_objects"):
            while goal_site in unwrapped._hidden_objects:
                unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
