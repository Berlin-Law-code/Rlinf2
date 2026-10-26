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

"""OpenPI transforms for the LBench joint-space contract."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _image(value: object, name: str, *, preserve_float: bool = False) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must be a 3-D RGB image, got {image.shape}")
    if any(size <= 0 for size in image.shape):
        raise ValueError(f"{name} must have non-empty dimensions, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must have three RGB channels, got {image.shape}")
    if not (
        np.issubdtype(image.dtype, np.number) or np.issubdtype(image.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must have a numeric dtype, got {image.dtype}")
    if preserve_float:
        is_integer = not np.issubdtype(image.dtype, np.floating)
        image = image.astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if is_integer or float(np.max(image)) > 1.0:
            image = image / 255.0
        return np.ascontiguousarray(np.clip(image, 0.0, 1.0))
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"{name} contains NaN or Inf")
        scale = 255.0 if float(np.max(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class LBenchInputs(transforms.DataTransformFn):
    """Map the public LBench schema to Pi0/Pi0.5's three image slots."""

    model_type: _model.ModelType
    state_dim: int = 8
    action_horizon: int | None = None
    default_prompt: str | None = None
    # LeRobot resizes floating-point images before mapping them to [-1, 1].
    preserve_float_images: bool = False

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"LBench observation.state must be finite [{self.state_dim}], "
                f"got {state.shape}"
            )
        if state[-1] < -1.0e-4 or state[-1] > 1.0 + 1.0e-4:
            raise ValueError("LBench state gripper channel must use 0=closed / 1=open")
        base = _image(
            data["observation/image"],
            "obs_camera",
            preserve_float=self.preserve_float_images,
        )
        wrist = _image(
            data["observation/wrist_image"],
            "obs_camera_2",
            preserve_float=self.preserve_float_images,
        )
        additional = _image(
            data["observation/extra_view_image"],
            "realsense",
            preserve_float=self.preserve_float_images,
        )
        result = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": additional,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 8:
                raise ValueError(
                    f"LBench actions must be [horizon, 8], got {actions.shape}"
                )
            if (
                self.action_horizon is not None
                and actions.shape[0] != self.action_horizon
            ):
                raise ValueError(
                    "LBench action horizon must match the OpenPI model: "
                    f"expected {self.action_horizon}, got {actions.shape[0]}"
                )
            if not np.isfinite(actions).all():
                raise ValueError("LBench actions contain NaN or Inf")
            if (actions[:, -1] < -1.0e-4).any() or (
                actions[:, -1] > 1.0 + 1.0e-4
            ).any():
                raise ValueError(
                    "LBench action gripper channel must use 0=closed / 1=open"
                )
            result["actions"] = actions
        prompt = data.get("prompt", data.get("task", self.default_prompt))
        if isinstance(prompt, np.ndarray) and prompt.ndim == 0:
            prompt = prompt.item()
        if isinstance(prompt, (bytes, np.bytes_)):
            prompt = prompt.decode("utf-8")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("LBench language instruction must be a non-empty string")
        result["prompt"] = prompt.strip()
        return result


@dataclasses.dataclass(frozen=True)
class LBenchOutputs(transforms.DataTransformFn):
    """Slice padded OpenPI actions back to LBench's absolute 8-D target."""

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < self.action_dim:
            raise ValueError(
                f"OpenPI actions must be [horizon, >= {self.action_dim}], "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("OpenPI actions contain NaN or Inf")
        return {"actions": actions[:, : self.action_dim]}


__all__ = ["LBenchInputs", "LBenchOutputs"]
