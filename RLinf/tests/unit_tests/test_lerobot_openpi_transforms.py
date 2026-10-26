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

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from openpi import transforms
from openpi.shared.normalize import NormStats

from rlinf.models.embodiment.openpi.lerobot_transforms import (
    LeRobotNormalize,
    LeRobotResizeImages,
    LeRobotUnnormalize,
)


def _stats():
    return NormStats(
        mean=np.array([0.2, -0.7, 0.5]),
        std=np.array([0.4, 0.6, 0.1]),
        q01=np.array([-0.648656964302063, -2.648320436477661, 0.5]),
        q99=np.array([0.44183725118637085, -1.6074007749557495, 0.5]),
    )


def _torch_quantile_parameters(stats):
    low = torch.tensor(stats.q01, dtype=torch.float32)
    high = torch.tensor(stats.q99, dtype=torch.float32)
    denominator = high - low
    denominator = torch.where(denominator == 0, torch.tensor(1e-8), denominator)
    return low, denominator


def test_quantile_normalization_is_float32_exact_and_does_not_clip():
    stats = _stats()
    low, denominator = _torch_quantile_parameters(stats)
    values = np.asarray(
        [stats.q01, stats.q99, [-1.1, -0.8, 0.6], [0.027, -2.21, 0.5]],
        dtype=np.float32,
    )
    expected = 2.0 * (torch.from_numpy(values) - low) / denominator - 1.0
    actual = LeRobotNormalize({"state": stats}, use_quantiles=True)(
        {"state": values, "prompt": "unchanged"}
    )
    assert actual["state"].dtype == np.float32
    np.testing.assert_array_equal(actual["state"], expected.numpy())
    assert actual["prompt"] == "unchanged"
    assert actual["state"][2, 0] < -1
    assert actual["state"][2, 1] > 1


def test_quantile_inverse_is_float32_exact_and_leaves_padding_unchanged():
    stats = _stats()
    low, denominator = _torch_quantile_parameters(stats)
    values = np.array(
        [[[-1.0, 1.0, 0.0, 1.27, -2.34], [2.13, -0.49, 3.0, 0.003, 9.0]]],
        dtype=np.float32,
    )
    expected = (torch.from_numpy(values[..., :3]) + 1.0) * denominator / 2.0 + low
    actual = LeRobotUnnormalize({"actions": stats}, use_quantiles=True)(
        {"actions": values}
    )["actions"]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual[..., :3], expected.numpy())
    np.testing.assert_array_equal(actual[..., 3:], values[..., 3:])


def test_normalization_can_use_a_prefix_of_stats():
    stats = _stats()
    values = np.array([[0.15, -1.8]], dtype=np.float32)
    low, denominator = _torch_quantile_parameters(stats)
    expected = 2.0 * (torch.from_numpy(values) - low[:2]) / denominator[:2] - 1.0
    actual = LeRobotNormalize({"state": stats}, use_quantiles=True)({"state": values})[
        "state"
    ]
    np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize(
    ("compatible", "original"),
    [
        (LeRobotNormalize, transforms.Normalize),
        (LeRobotUnnormalize, transforms.Unnormalize),
    ],
)
def test_nonquantile_behavior_is_unchanged(compatible, original):
    values = np.array([[0.15, -1.8, 0.73]], dtype=np.float32)
    stats = {"state": _stats()}
    actual = compatible(stats, use_quantiles=False)({"state": values})["state"]
    expected = original(stats, use_quantiles=False)({"state": values})["state"]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("shape", "resized_shape", "padding"),
    [
        ((480, 640), (168, 224), (0, 0, 28, 28)),
        ((640, 480), (224, 168), (28, 28, 0, 0)),
        ((223, 400), (124, 224), (0, 0, 50, 50)),
        ((221, 400), (123, 224), (0, 0, 50, 51)),
        ((224, 224), None, None),
    ],
)
def test_resize_matches_lerobot_float32_operation_order(shape, resized_shape, padding):
    image = np.random.default_rng(17).random((*shape, 3), dtype=np.float32)
    reference = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    if resized_shape is not None:
        reference = F.interpolate(
            reference, size=resized_shape, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        reference = F.pad(reference, padding, value=0.0)
    reference = (reference * 2.0 - 1.0).squeeze(0).numpy()
    actual = LeRobotResizeImages(224, 224)(
        {"image": {"base_0_rgb": image}, "state": np.array([0.1])}
    )
    np.testing.assert_array_equal(actual["image"]["base_0_rgb"], reference)
    np.testing.assert_array_equal(actual["state"], np.array([0.1]))
    assert actual["image"]["base_0_rgb"].dtype == np.float32
    assert actual["image"]["base_0_rgb"].shape == (3, 224, 224)
    assert actual["image"]["base_0_rgb"].flags.c_contiguous


def test_resize_uint8_and_float_inputs_match():
    image = np.random.default_rng(18).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    transform = LeRobotResizeImages(224, 224)
    uint8_output = transform({"image": {"camera": image}})["image"]["camera"]
    float_output = transform(
        {"image": {"camera": image.astype(np.float32) / np.float32(255.0)}}
    )["image"]["camera"]
    np.testing.assert_array_equal(uint8_output, float_output)
    np.testing.assert_array_equal(uint8_output[:, :28], -np.ones((3, 28, 224)))


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_resize_rejects_invalid_float_range(value):
    with pytest.raises(ValueError, match="finite float32"):
        LeRobotResizeImages(224, 224)(
            {"image": {"camera": np.full((8, 8, 3), value, dtype=np.float32)}}
        )
