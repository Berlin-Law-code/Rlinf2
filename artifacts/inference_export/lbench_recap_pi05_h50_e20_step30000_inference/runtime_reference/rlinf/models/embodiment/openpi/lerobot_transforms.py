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

"""Opt-in OpenPI transforms matching LeRobot's PI0.5 preprocessing."""

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F
from openpi import transforms
from openpi.shared.normalize import NormStats


def _quantile_parameters(stats: NormStats) -> tuple[np.ndarray, np.ndarray]:
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("LeRobot quantile normalization requires q01 and q99")
    q01 = np.asarray(stats.q01, dtype=np.float32)
    q99 = np.asarray(stats.q99, dtype=np.float32)
    denominator = q99 - q01
    denominator = np.where(denominator == 0, np.float32(1e-8), denominator)
    return q01, denominator


@dataclasses.dataclass(frozen=True)
class LeRobotNormalize(transforms.Normalize):
    """Match LeRobot float32 quantiles while retaining OpenPI's other modes."""

    def _normalize_quantile(self, x, stats: NormStats):
        x = np.asarray(x, dtype=np.float32)
        q01, denominator = _quantile_parameters(stats)
        q01 = q01[..., : x.shape[-1]]
        denominator = denominator[..., : x.shape[-1]]
        return np.float32(2.0) * (x - q01) / denominator - np.float32(1.0)


@dataclasses.dataclass(frozen=True)
class LeRobotUnnormalize(transforms.Unnormalize):
    """Match LeRobot inverse quantiles and preserve padded action channels."""

    def _unnormalize_quantile(self, x, stats: NormStats):
        x = np.asarray(x, dtype=np.float32)
        q01, denominator = _quantile_parameters(stats)
        dimension = q01.shape[-1]
        result = (x[..., :dimension] + np.float32(1.0)) * denominator / np.float32(
            2.0
        ) + q01
        if dimension < x.shape[-1]:
            result = np.concatenate((result, x[..., dimension:]), axis=-1)
        return result


@dataclasses.dataclass(frozen=True)
class LeRobotResizeImages(transforms.ResizeImages):
    """Resize HWC [0, 1] images into float32 CHW [-1, 1] for OpenPI PyTorch.

    uint8 input is also accepted. Bilinear interpolation runs in float32
    before normalization, with black padding and no antialiasing.
    OpenPI's Observation.from_dict only transposes uint8 PyTorch images;
    pre-normalized float32 images must already have channels first.
    """

    def __call__(self, data: dict) -> dict:
        data["image"] = {
            key: self._resize(image) for key, image in data["image"].items()
        }
        return data

    def _resize(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected an HWC RGB image, got {image.shape}")
        if self.height <= 0 or self.width <= 0 or min(image.shape[:2]) <= 0:
            raise ValueError("Image and target dimensions must be positive")
        is_uint8 = image.dtype == np.uint8
        image = np.asarray(image, dtype=np.float32)
        if is_uint8:
            image = image / np.float32(255.0)
        if not np.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError("LeRobot images must be finite float32 [0, 1] or uint8")

        tensor = (
            torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        )
        current_height, current_width = image.shape[:2]
        if (current_height, current_width) != (self.height, self.width):
            ratio = max(current_width / self.width, current_height / self.height)
            resized_height = int(current_height / ratio)
            resized_width = int(current_width / ratio)
            tensor = F.interpolate(
                tensor,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            ).clamp(0.0, 1.0)
            pad_top, remainder_height = divmod(self.height - resized_height, 2)
            pad_left, remainder_width = divmod(self.width - resized_width, 2)
            tensor = F.pad(
                tensor,
                (
                    pad_left,
                    pad_left + remainder_width,
                    pad_top,
                    pad_top + remainder_height,
                ),
                value=0.0,
            )
        tensor = tensor * 2.0 - 1.0
        return tensor.squeeze(0).contiguous().numpy()


__all__ = ["LeRobotNormalize", "LeRobotUnnormalize", "LeRobotResizeImages"]
