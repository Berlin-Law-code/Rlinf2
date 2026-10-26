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

"""RLinf-facing access to the shared LBench action mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .client import cfg_get, import_lbench_modules


def map_action(action: Any, cfg: Any) -> dict[str, Any]:
    """Validate and map one 8-D policy action to a SimBox command."""

    _, mapper, _, _, _ = import_lbench_modules(cfg)
    return mapper(
        np.asarray(action, dtype=np.float32),
        gripper_open=cfg_get(cfg, "gripper_open", (0.0, 0.0)),
        gripper_close=cfg_get(cfg, "gripper_close", (5.0, 5.0)),
        gripper_threshold=float(cfg_get(cfg, "gripper_threshold", 0.5)),
    )


def map_action_batch(actions: Any, cfg: Any) -> list[dict[str, Any]]:
    """Map a ``[B, 8]`` action batch in stable environment order."""

    batch = np.asarray(actions, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[1] != 8:
        raise ValueError(f"expected LBench action batch [B, 8], got {batch.shape}")
    _, mapper, _, _, _ = import_lbench_modules(cfg)
    mapping_kwargs = {
        "gripper_open": cfg_get(cfg, "gripper_open", (0.0, 0.0)),
        "gripper_close": cfg_get(cfg, "gripper_close", (5.0, 5.0)),
        "gripper_threshold": float(cfg_get(cfg, "gripper_threshold", 0.5)),
    }
    return [mapper(row, **mapping_kwargs) for row in batch]


def _configured_gripper_scalar(values: Any, cfg: Any) -> float:
    """Map a SimBox gripper target back to ``1=open`` / ``0=closed``."""

    target = np.asarray(values, dtype=np.float32).reshape(-1)
    open_target = np.asarray(
        cfg_get(cfg, "gripper_open", (0.0, 0.0)), dtype=np.float32
    ).reshape(-1)
    close_target = np.asarray(
        cfg_get(cfg, "gripper_close", (5.0, 5.0)), dtype=np.float32
    ).reshape(-1)
    if target.size < 1 or not np.isfinite(target).all():
        raise ValueError("LBench executed gripper target must be finite and non-empty")
    if (
        open_target.size < 1
        or open_target.shape != close_target.shape
        or not np.isfinite(open_target).all()
        or not np.isfinite(close_target).all()
    ):
        raise ValueError("LBench configured gripper targets are invalid")
    count = min(target.size, open_target.size)
    distance_open = float(np.mean(np.abs(target[:count] - open_target[:count])))
    distance_close = float(np.mean(np.abs(target[:count] - close_target[:count])))
    return float(distance_open <= distance_close)


def _controller_gripper_scalar(values: Any) -> float:
    """Decode the raw CuRobo controller convention used by rescue actions.

    LBench's CuRobo controller emits physical gripper targets with zero for a
    closed gripper and a positive target (normally ``0.04``) for open. They can
    have a different scale from the policy-facing SimBox configuration, so
    comparing them with ``gripper_close`` (often ``5``) is not reliable.
    """

    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.size < 1 or not np.isfinite(target).all():
        raise ValueError("LBench controller gripper target must be finite")
    return float(np.any(np.abs(target) > 1.0e-6))


def _nested_controller_action(action: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Find the robot-local articulation command in a SimBox action mapping."""

    for value in action.values():
        if isinstance(value, Mapping) and (
            "joint_positions" in value or "joint_position" in value
        ):
            return value
    return None


def executed_action_to_policy_action(
    executed_action: Any,
    cfg: Any,
    *,
    fallback: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Convert an executed SimBox joint command to LBench's policy-space 8D action.

    The external LBench backend reports the command that actually advanced the
    simulator in ``info['executed_action']``.  Normal policy commands use the
    transport-neutral ``joint_position`` / ``gripper_joint_position`` schema,
    while CuRobo rescue commands may retain the robot-local
    ``joint_positions`` representation.  This function accepts both and fails
    closed when an explicitly reported physical command cannot be represented
    by RLinf's joint-only LBench action space.
    """

    if not isinstance(executed_action, Mapping):
        raise TypeError("LBench executed_action must be a mapping")
    if executed_action.get("type") == "no_op":
        if fallback is None:
            raise ValueError("LBench no-op action requires a fallback action")
        result = np.asarray(fallback, dtype=np.float32).reshape(-1).copy()
        if result.shape != (8,) or not np.isfinite(result).all():
            raise ValueError("LBench fallback action must be finite [8]")
        return result

    command: Mapping[str, Any] = executed_action
    controller_schema = "joint_positions" in command and (
        "joint_indices" in command or "joint_position" in command
    )
    if "joint_position" not in command and "joint_positions" not in command:
        nested = _nested_controller_action(command)
        if nested is None:
            raise ValueError(
                "LBench executed_action has no supported joint command"
            )
        command = nested
        controller_schema = "joint_positions" in command

    arm_value = command.get("joint_position", command.get("joint_positions"))
    joints = np.asarray(arm_value, dtype=np.float32).reshape(-1)
    if joints.size < 7 or not np.isfinite(joints).all():
        raise ValueError("LBench executed arm target must contain 7 finite joints")
    arm = joints[:7]

    explicit_openness = command.get("gripper_openness")
    if explicit_openness is not None:
        openness_array = np.asarray(explicit_openness, dtype=np.float32).reshape(-1)
        if (
            openness_array.size != 1
            or not np.isfinite(openness_array).all()
            or not 0.0 <= float(openness_array[0]) <= 1.0
        ):
            raise ValueError(
                "LBench executed gripper_openness must be one finite value in [0, 1]"
            )
        gripper = float(
            float(openness_array[0])
            >= float(cfg_get(cfg, "gripper_threshold", 0.5))
        )
        return np.concatenate(
            [
                arm.astype(np.float32, copy=False),
                np.asarray([gripper], dtype=np.float32),
            ]
        )

    gripper_value = command.get(
        "gripper_joint_position", command.get("gripper_action")
    )
    if gripper_value is None and joints.size > 7:
        gripper_value = joints[7:]
        controller_schema = True
    if gripper_value is None:
        if fallback is None:
            raise ValueError("LBench executed action is missing a gripper target")
        fallback_array = np.asarray(fallback, dtype=np.float32).reshape(-1)
        if fallback_array.shape != (8,) or not np.isfinite(fallback_array).all():
            raise ValueError("LBench fallback action must be finite [8]")
        gripper = float(fallback_array[7])
    elif controller_schema:
        gripper = _controller_gripper_scalar(gripper_value)
    else:
        gripper = _configured_gripper_scalar(gripper_value, cfg)

    return np.concatenate(
        [arm.astype(np.float32, copy=False), np.asarray([gripper], dtype=np.float32)]
    )


__all__ = [
    "executed_action_to_policy_action",
    "map_action",
    "map_action_batch",
]
