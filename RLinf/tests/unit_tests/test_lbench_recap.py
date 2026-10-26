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

"""Regression tests for LBench RECAP camera and normalization contracts."""

import copy
import json

import numpy as np
import pytest
from omegaconf import OmegaConf
from openpi import transforms

from rlinf.data.datasets.recap.value_dataset import ValueDataset
from rlinf.models.embodiment.openpi_cfg.contract import resolve_cfg_contract
from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
    build_input_transforms,
)


def test_value_training_and_inference_keep_all_three_cameras():
    cameras = [np.full((3, 16, 20), value, np.float32) for value in (0.1, 0.5, 0.9)]
    sample = {
        "observation.images.obs_camera": cameras[0],
        "observation.images.obs_camera_2": cameras[1],
        "observation.images.realsense": cameras[2],
        "observation.state": np.zeros(8, np.float32),
        "action": np.zeros((20, 8), np.float32),
        "prompt": "Place the object in the drawer.",
    }
    train = ValueDataset._build_transform("lbench", "pi05", 32, None)(
        copy.deepcopy(sample)
    )
    infer = transforms.compose(
        build_input_transforms("lbench", "pi05", 32, None, None, True)
    )(
        {
            "observation/image": cameras[0],
            "observation/wrist_image": cameras[1],
            "observation/extra_view_image": cameras[2],
            "observation/state": sample["observation.state"],
            "prompt": sample["prompt"],
        }
    )
    assert len(train["image"]) == 3
    for key, expected in zip(
        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), cameras
    ):
        np.testing.assert_array_equal(train["image"][key], expected.transpose(1, 2, 0))
        np.testing.assert_array_equal(train["image"][key], infer["image"][key])
    np.testing.assert_array_equal(train["state"], infer["state"])
    assert train["actions"].shape == (20, 32)


def test_cfg_requires_explicit_opt_in_to_change_checkpoint_norm(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    bundled = checkpoint / "lbench/norm_stats.json"
    bundled.parent.mkdir()
    statistics = {
        k: {"mean": [0.0] * 8, "std": [1.0] * 8, "q01": [-1.0] * 8, "q99": [1.0] * 8}
        for k in ("state", "actions")
    }
    bundled.write_text(json.dumps({"norm_stats": statistics}))
    (checkpoint / "openpi_assets.json").write_text(
        json.dumps({"norm_stats": ["lbench/norm_stats.json"]})
    )
    new = tmp_path / "new/norm_stats.json"
    new.parent.mkdir()
    statistics["actions"]["q99"][0] = 2.0
    new.write_text(json.dumps({"norm_stats": statistics}))
    cfg = OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "openpi": {"config_name": "pi05_lbench_joint", "action_horizon": 50},
            "openpi_data": {"norm_stats_path": str(new), "lerobot_compatible": True},
        }
    )
    with pytest.raises(ValueError, match="normalization"):
        resolve_cfg_contract(cfg)
    cfg.allow_norm_stats_override = True
    config, _ = resolve_cfg_contract(cfg)
    assert config.model.action_horizon == 50
    data = config.data.create(config.assets_dirs, config.model)
    assert data.norm_stats["actions"].q99[0] == 2.0
    assert json.loads(bundled.read_text())["norm_stats"]["actions"]["q99"][0] == 1.0
