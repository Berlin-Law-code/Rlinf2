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
"""Read-only CPU preflight for the local LBench STEAM/CFGRL recipe."""

import copy
import json
import os
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from openpi import transforms

from rlinf.data.datasets.recap.cfg_model import TokenizePromptWithGuidance
from rlinf.models.embodiment.openpi_cfg.contract import resolve_cfg_contract


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("REPO_PATH", str(root))
    with initialize_config_dir(
        str(root / "examples/offline_rl/config"), version_base="1.1"
    ):
        cfg = compose(config_name="lbench_cfgrl")
        value_cfg = compose(config_name="lbench_steam_value")
    config, precision = resolve_cfg_contract(cfg.actor.model)
    data = config.data.create(config.assets_dirs, config.model)
    model_transforms = [
        TokenizePromptWithGuidance(
            t.tokenizer, getattr(t, "discrete_state_input", False)
        )
        if type(t).__name__ == "TokenizePrompt"
        else t
        for t in data.model_transforms.inputs
    ]
    sft_transform = transforms.compose(
        [
            *data.repack_transforms.inputs,
            *data.data_transforms.inputs,
            *data.model_transforms.inputs,
        ]
    )
    cfg_transform = transforms.compose(
        [
            *data.repack_transforms.inputs,
            *data.data_transforms.inputs,
            *model_transforms,
        ]
    )
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    report = {
        "precision": precision,
        "horizon": config.model.action_horizon,
        "datasets": [],
    }
    for entry in cfg.data.train_data_paths:
        path = Path(entry.dataset_path)
        info = json.loads((path / "meta/info.json").read_text())
        episodes = [
            json.loads(line)
            for line in (path / "meta/episodes.jsonl").read_text().splitlines()
        ]
        dataset = LeRobotDataset(
            path.name,
            root=path,
            download_videos=False,
            delta_timestamps={"action": [t / info["fps"] for t in range(50)]},
        )
        starts = []
        offset = 0
        seen = set()
        for ep in episodes:
            task = tuple(ep["tasks"])
            if task not in seen:
                starts.append(offset)
                seen.add(task)
            offset += ep["length"]
        for index in starts:
            sample = dict(dataset[index])
            sample["prompt"] = sample["task"]
            expected = sft_transform(copy.deepcopy(sample))
            actual = cfg_transform(copy.deepcopy(sample))
            for key in [
                "state",
                "actions",
                "tokenized_prompt",
                "tokenized_prompt_mask",
            ]:
                np.testing.assert_array_equal(actual[key], expected[key])
            for key in expected["image"]:
                np.testing.assert_array_equal(
                    actual["image"][key], expected["image"][key]
                )
                assert actual["image"][key].shape == (3, 224, 224)
            assert actual["actions"].shape == (50, 32)
            assert np.isfinite(actual["actions"]).all()
            assert not np.array_equal(
                actual["tokenized_prompt"], actual["tokenized_positive_guidance_prompt"]
            )
        report["datasets"].append(
            {
                "path": str(path),
                "task_samples_checked": len(starts),
                "advantage_file_exists": (
                    path / "meta" / f"advantages_{cfg.data.advantage_tag}.parquet"
                ).is_file(),
            }
        )
        del dataset
    report["steam_backbones"] = {
        key: {
            "path": str(value_cfg.actor.model[key]),
            "config_exists": (
                Path(value_cfg.actor.model[key]) / "config.json"
            ).is_file(),
            "weights_exist": bool(
                list(Path(value_cfg.actor.model[key]).glob("*.safetensors"))
            ),
        }
        for key in ["vision_repo_id", "language_repo_id"]
    }
    report["status"] = "CPU transform parity passed; GPU training is not tested"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
