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

"""Strict LBench raw-observation to OpenPI input mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

DEFAULT_CAMERA_MAPPING = {
    "main": "obs_camera",
    "wrist": "obs_camera_2",
    "additional": "realsense",
}


def _camera_name(camera_mapping: Any, role: str) -> str | None:
    if camera_mapping is None:
        return DEFAULT_CAMERA_MAPPING[role]
    value = camera_mapping.get(role) if hasattr(camera_mapping, "get") else None
    if value is None and role != "additional":
        raise ValueError(f"camera_mapping.{role} is required")
    return None if value is None else str(value)


def _rgb(raw_obs: Mapping[str, Any], camera_name: str, role: str) -> np.ndarray:
    robot = raw_obs.get("robot", {})
    sensors = robot.get("sensors", {}) if isinstance(robot, Mapping) else {}
    camera = sensors.get(camera_name, {}) if isinstance(sensors, Mapping) else {}
    value = camera.get("rgb") if isinstance(camera, Mapping) else None
    if value is None:
        raise KeyError(f"LBench {role} camera {camera_name!r} is missing")
    image = np.asarray(value)
    if (
        image.ndim != 3
        or image.shape[-1] not in (3, 4)
        or any(size < 1 for size in image.shape)
    ):
        raise ValueError(
            f"LBench {role} camera must be HWC RGB(A), got {image.shape}"
        )
    image = image[..., :3]
    if not (
        np.issubdtype(image.dtype, np.number)
        or np.issubdtype(image.dtype, np.bool_)
    ):
        raise TypeError(f"LBench {role} camera must be numeric, got {image.dtype}")
    if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
        raise ValueError(f"LBench {role} camera contains NaN or Inf")
    if image.dtype != np.uint8:
        if (
            np.issubdtype(image.dtype, np.floating)
            and image.size
            and image.max() <= 1.0
        ):
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _gripper_scalar(
    robot: Mapping[str, Any],
    positions: np.ndarray,
    *,
    joint_threshold: float,
) -> float:
    """Match FrankaRobotiq85's observed ``1=open`` / ``0=closed`` state."""
    if "states.gripper.position" in robot:
        value = np.asarray(robot["states.gripper.position"], dtype=np.float64)
        if value.size != 1 or not np.isin(value, (0.0, 1.0)).all():
            raise ValueError(
                "LBench states.gripper.position must be one finite binary value"
            )
        return float(value.reshape(-1)[0])
    # These are measured joints, not the commanded [0, 0] / [5, 5] targets.
    # Compare in the same precision as positions so the 0.4 boundary stays open.
    threshold = np.asarray(joint_threshold, dtype=positions.dtype)
    return float(positions[7] <= threshold)


def map_observation_batch(
    raw_observations: Sequence[Mapping[str, Any]],
    *,
    camera_mapping: Any = None,
    gripper_open: Sequence[float] = (0.0, 0.0),
    gripper_close: Sequence[float] = (5.0, 5.0),
    gripper_threshold: float = 0.5,
    observation_gripper_joint_threshold: float = 0.4,
) -> dict[str, Any]:
    """Map raw observations to the exact RLinf/OpenPI LBench schema.

    The legacy ``gripper_open``, ``gripper_close`` and ``gripper_threshold``
    keywords remain accepted, but action targets do not define observed state.
    """

    if not raw_observations:
        raise ValueError("raw_observations must not be empty")
    joint_threshold = float(observation_gripper_joint_threshold)
    if not np.isfinite(joint_threshold) or joint_threshold < 0.0:
        raise ValueError(
            "LBench observation_gripper_joint_threshold must be finite and non-negative"
        )
    main_name = _camera_name(camera_mapping, "main")
    wrist_name = _camera_name(camera_mapping, "wrist")
    additional_name = _camera_name(camera_mapping, "additional")
    assert main_name is not None and wrist_name is not None

    main_images = []
    wrist_images = []
    additional_images = []
    states = []
    task_descriptions = []
    for env_id, raw_obs in enumerate(raw_observations):
        if not isinstance(raw_obs, Mapping):
            raise TypeError(f"raw observation {env_id} must be a mapping")
        main = _rgb(raw_obs, main_name, "main")
        wrist = _rgb(raw_obs, wrist_name, "wrist")
        additional = (
            np.zeros_like(main)
            if additional_name is None
            else _rgb(raw_obs, additional_name, "additional")
        )
        robot = raw_obs.get("robot", {})
        joints = robot.get("joints_state", {}) if isinstance(robot, Mapping) else {}
        positions = np.asarray(
            joints.get("positions", []), dtype=np.float32
        ).reshape(-1)
        if positions.size < 8 or not np.isfinite(positions).all():
            raise ValueError(
                f"LBench env {env_id} requires at least 8 finite joint positions"
            )
        state = np.concatenate(
            [
                positions[:7],
                np.asarray(
                    [
                        _gripper_scalar(
                            robot,
                            positions,
                            joint_threshold=joint_threshold,
                        )
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        main_images.append(main)
        wrist_images.append(wrist)
        additional_images.append(additional)
        states.append(state)
        instruction = robot.get("instruction", "")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"LBench env {env_id} requires a non-empty language instruction"
            )
        task_descriptions.append(instruction.strip())

    try:
        return {
            "main_images": torch.from_numpy(np.stack(main_images)),
            "wrist_images": torch.from_numpy(np.stack(wrist_images)),
            # RLinf's shared embodied observation name. OpenPI maps this to
            # observation/extra_view_image and then to the third image slot.
            "extra_view_images": torch.from_numpy(np.stack(additional_images)),
            "states": torch.from_numpy(np.stack(states).astype(np.float32)),
            "task_descriptions": task_descriptions,
        }
    except ValueError as exc:
        raise ValueError("LBench camera shapes differ across workers") from exc


__all__ = ["DEFAULT_CAMERA_MAPPING", "map_observation_batch"]
