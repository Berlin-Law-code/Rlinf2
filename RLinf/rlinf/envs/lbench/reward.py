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

"""Stateful rewards for LBench reinforcement fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _robot_mapping(raw: Any, key: str) -> Mapping[str, Any]:
    robot = raw.get("robot", {}) if isinstance(raw, Mapping) else {}
    value = robot.get(key, {}) if isinstance(robot, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _protocol_bool(value: Any, *, name: str, env_id: int | None) -> bool:
    """Decode one strict boolean carried by the LBench step protocol."""

    array = np.asarray(value)
    location = "" if env_id is None else f" for env {env_id}"
    if array.size != 1:
        raise ValueError(f"LBench {name} must be a scalar boolean{location}")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, float, np.integer, np.floating)):
        numeric = float(scalar)
        if np.isfinite(numeric) and numeric in (0.0, 1.0):
            return bool(numeric)
    raise ValueError(f"LBench {name} must be boolean or 0/1{location}")


def resolve_lbench_success(
    raw: Any,
    info: Any,
    *,
    env_id: int | None = None,
) -> bool:
    """Resolve success without bypassing LBench's success-hold contract.

    Updated workers expose ``success_confirmed`` only after the configured
    hold has completed.  Its presence is authoritative even when the raw
    evaluator's ``episode_sr`` (a candidate signal) is already one.  Metrics
    remain a compatibility fallback for older workers that omit the field.
    """

    if isinstance(info, Mapping) and "success_confirmed" in info:
        return _protocol_bool(
            info["success_confirmed"],
            name="success_confirmed",
            env_id=env_id,
        )

    metric = _robot_mapping(raw, "metric")
    components = (
        info.get("reward_components", {}) if isinstance(info, Mapping) else {}
    )
    if not isinstance(components, Mapping):
        components = {}
    episode_success = float(metric.get("episode_sr", 0.0) or 0.0)
    component_success = float(components.get("success", 0.0) or 0.0)
    if not np.isfinite(episode_success) or not np.isfinite(component_success):
        location = "" if env_id is None else f" for env {env_id}"
        raise ValueError(f"LBench success metric is non-finite{location}")
    return bool(episode_success >= 1.0 or component_success >= 1.0)


@dataclass(frozen=True)
class RewardConfig:
    kind: str = "backend"
    progress_key: str = "progress"
    progress_scale: float = 1.0
    stage_bonus: float = 0.1
    success_bonus: float = 1.0
    failure_penalty: float = 1.0


class RewardAdapter:
    """Convert raw task metrics into sparse or shaped, event-once rewards."""

    def __init__(self, num_envs: int, config: RewardConfig):
        self.num_envs = int(num_envs)
        self.config = config
        if self.num_envs < 1:
            raise ValueError("LBench reward num_envs must be positive")
        if config.kind not in {
            "backend",
            "libero_sparse_success",
            "sparse_success",
            "progress_delta",
        }:
            raise ValueError(f"unsupported LBench reward type: {config.kind}")
        coefficients = {
            "progress_scale": config.progress_scale,
            "stage_bonus": config.stage_bonus,
            "success_bonus": config.success_bonus,
            "failure_penalty": config.failure_penalty,
        }
        non_finite = [
            name for name, value in coefficients.items() if not np.isfinite(value)
        ]
        if non_finite:
            raise ValueError(
                f"LBench reward coefficients must be finite: {non_finite}"
            )
        if config.kind == "progress_delta" and not config.progress_key.strip():
            raise ValueError("LBench progress_key must not be empty")
        self._progress = np.zeros(self.num_envs, dtype=np.float32)
        self._stage = np.full(self.num_envs, -1, dtype=np.int64)
        self._success = np.zeros(self.num_envs, dtype=bool)
        self._failure = np.zeros(self.num_envs, dtype=bool)

    @classmethod
    def from_config(cls, num_envs: int, value: Any) -> "RewardAdapter":
        get = value.get if hasattr(value, "get") else lambda _key, default=None: default
        return cls(
            num_envs,
            RewardConfig(
                kind=str(get("type", "backend")),
                progress_key=str(get("progress_key", "progress")),
                progress_scale=float(get("progress_scale", 1.0)),
                stage_bonus=float(get("stage_bonus", 0.1)),
                success_bonus=float(get("success_bonus", 1.0)),
                failure_penalty=float(get("failure_penalty", 1.0)),
            ),
        )

    def reset(
        self,
        env_ids: Sequence[int],
        raw_observations: Sequence[Any] | None = None,
    ) -> None:
        ids = np.asarray(list(env_ids), dtype=np.int64)
        self._progress[ids] = 0.0
        self._stage[ids] = -1
        self._success[ids] = False
        self._failure[ids] = False
        if raw_observations is None:
            return
        if len(raw_observations) != self.num_envs:
            raise ValueError(
                "LBench reward reset observations must be a full environment batch"
            )
        for env_id in ids.tolist():
            metric = _robot_mapping(raw_observations[env_id], "metric")
            progress = float(metric.get(self.config.progress_key, 0.0) or 0.0)
            if not np.isfinite(progress):
                raise ValueError(
                    f"LBench reset progress is non-finite for env {env_id}: "
                    f"{progress}"
                )
            self._progress[env_id] = progress
            self._stage[env_id] = int(metric.get("current_stage_id", 0) or 0)

    def adapt(
        self,
        raw_observations: Sequence[Any],
        infos: Sequence[Any],
        backend_rewards: Sequence[float],
        active_mask: Sequence[bool],
        timeout_mask: Sequence[bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = {
            "raw_observations": raw_observations,
            "infos": infos,
            "backend_rewards": backend_rewards,
            "active_mask": active_mask,
        }
        for name, value in values.items():
            if len(value) != self.num_envs:
                raise ValueError(
                    f"LBench reward {name} must have length {self.num_envs}, "
                    f"got {len(value)}"
                )
        if timeout_mask is None:
            timeout_mask = [False] * self.num_envs
        elif len(timeout_mask) != self.num_envs:
            raise ValueError(
                f"LBench reward timeout_mask must have length {self.num_envs}, "
                f"got {len(timeout_mask)}"
            )

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        success_events = np.zeros(self.num_envs, dtype=bool)
        # Shaped/sparse adapters can promote an explicit task failure to a
        # termination. The LIBERO-compatible adapter leaves all failure/done
        # semantics to SimBox, and a timeout always remains a truncation.
        failure_events = np.zeros(self.num_envs, dtype=bool)
        for env_id, (raw, info, active, timed_out) in enumerate(
            zip(raw_observations, infos, active_mask, timeout_mask)
        ):
            if not active:
                continue
            metric = _robot_mapping(raw, "metric")
            components = (
                info.get("reward_components", {})
                if isinstance(info, Mapping)
                else {}
            )
            if not isinstance(components, Mapping):
                components = {}
            success = resolve_lbench_success(raw, info, env_id=env_id)
            explicit_failure = bool(
                metric.get("failure", False)
                or metric.get("episode_failure", False)
                or components.get("failure", False)
            ) and not success
            # Reaching the aligned simulator/RL horizon without success is an
            # unsuccessful episode for reward purposes, not a process error.
            timeout_failure = bool(timed_out) and not success
            failure = explicit_failure or timeout_failure
            success_event = success and not self._success[env_id]
            failure_event = failure and not self._failure[env_id]
            success_events[env_id] = success_event
            # The LIBERO-compatible baseline never creates a termination from
            # adapter-side failure detection. SimBox may still terminate an
            # irrecoverably failed episode through its native ``terminated``
            # flag; the adapter neither suppresses nor rewards that event.
            failure_events[env_id] = (
                failure_event
                and explicit_failure
                and self.config.kind != "libero_sparse_success"
            )

            if self.config.kind == "backend":
                reward = float(backend_rewards[env_id])
            elif self.config.kind == "libero_sparse_success":
                # Match RLinf's LIBERO PPO reward contract: the first success
                # transition receives reward_coef (success_bonus here), while
                # progress, stages, backend reward, explicit failures and
                # horizon truncations all receive zero reward.
                reward = self.config.success_bonus if success_event else 0.0
            elif self.config.kind == "sparse_success":
                reward = self.config.success_bonus if success_event else 0.0
                if failure_event:
                    reward -= self.config.failure_penalty
            else:
                progress = float(metric.get(self.config.progress_key, 0.0) or 0.0)
                if not np.isfinite(progress):
                    raise ValueError(
                        f"LBench progress is non-finite for env {env_id}: {progress}"
                    )
                reward = self.config.progress_scale * (
                    progress - float(self._progress[env_id])
                )
                stage = int(metric.get("current_stage_id", 0) or 0)
                if self._stage[env_id] >= 0 and stage > self._stage[env_id]:
                    reward += self.config.stage_bonus
                if success_event:
                    reward += self.config.success_bonus
                if failure_event:
                    reward -= self.config.failure_penalty
                self._progress[env_id] = progress
                self._stage[env_id] = max(self._stage[env_id], stage)
            if not np.isfinite(reward):
                raise ValueError(
                    f"LBench adapted reward is non-finite for env {env_id}: {reward}"
                )
            rewards[env_id] = reward
            self._success[env_id] |= success
            self._failure[env_id] |= failure
        return rewards, success_events, failure_events


__all__ = ["RewardAdapter", "RewardConfig", "resolve_lbench_success"]
