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

"""Validate the local LBench RECAP recipe and optionally smoke-test the policy."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from openpi import transforms

from rlinf.data.datasets.recap.cfg_model import TokenizePromptWithGuidance
from rlinf.data.datasets.recap.value_dataset import ValueDataset
from rlinf.models.embodiment.openpi_cfg.contract import resolve_cfg_contract
from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
    build_input_transforms,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-backbone-check", action="store_true")
    parser.add_argument("--policy-smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("REPO_PATH", str(root))
    with initialize_config_dir(
        str(root / "examples/offline_rl/config"), version_base="1.1"
    ):
        cfg = compose(config_name="lbench_recap_policy")
        value_cfg = compose(config_name="lbench_recap_value")
    config, precision = resolve_cfg_contract(cfg.actor.model)
    data = config.data.create(config.assets_dirs, config.model)
    mt = [
        TokenizePromptWithGuidance(
            t.tokenizer, getattr(t, "discrete_state_input", False)
        )
        if type(t).__name__ == "TokenizePrompt"
        else t
        for t in data.model_transforms.inputs
    ]
    policy_transform = transforms.compose(
        [*data.repack_transforms.inputs, *data.data_transforms.inputs, *mt]
    )
    value_transform = ValueDataset._build_transform("lbench", "pi05", 32, None)
    value_inference = transforms.compose(
        build_input_transforms("lbench", "pi05", 32, None, None, True)
    )
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    prepared = Path("/root/autodl-tmp/datasets/lbench_recap_round1")
    manifest = json.loads((prepared / "preparation_manifest.json").read_text())
    norm_path = Path(cfg.actor.model.openpi_data.norm_stats_path)
    assert (
        hashlib.sha256(norm_path.read_bytes()).hexdigest()
        == manifest["norm_stats_sha256"]
    )
    assert manifest["sft_reproduces_reference_exactly_float32"]
    assert config.model.action_horizon == 50
    assert cfg.actor.model.openpi.action_chunk == 20
    assert cfg.runner.val_check_interval == -1
    assert value_cfg.runner.val_check_interval == -1
    assert cfg.data.prefetch_factor == 2 and not cfg.data.persistent_workers
    report = {
        "episodes": manifest["episodes"],
        "frames": manifest["frames"],
        "precision": precision,
        "datasets": [],
    }
    smoke_samples = []
    for entry in cfg.data.train_data_paths:
        path = Path(entry.dataset_path)
        info = json.loads((path / "meta/info.json").read_text())
        dataset = LeRobotDataset(
            path.name,
            root=path,
            download_videos=False,
            delta_timestamps={"action": [t / info["fps"] for t in range(50)]},
        )
        for index in (0, len(dataset) - 1):
            sample = dict(dataset[index])
            sample["prompt"] = sample["task"]
            actual = policy_transform(copy.deepcopy(sample))
            assert actual["actions"].shape == (50, 32)
            assert np.isfinite(actual["actions"]).all()
            assert len(actual["image"]) == 3
            for image in actual["image"].values():
                assert image.shape == (3, 224, 224)
            # Training and advantage inference must preserve identical camera
            # ordering and pixels before the shared value processor.
            train = value_transform(copy.deepcopy(sample))
            infer = value_inference(
                {
                    "observation/image": sample["observation.images.obs_camera"],
                    "observation/wrist_image": sample[
                        "observation.images.obs_camera_2"
                    ],
                    "observation/extra_view_image": sample[
                        "observation.images.realsense"
                    ],
                    "observation/state": sample["observation.state"],
                    "prompt": sample["prompt"],
                }
            )
            for key in train["image"]:
                np.testing.assert_array_equal(train["image"][key], infer["image"][key])
            if len(smoke_samples) < 2:
                smoke_samples.append(actual)
        report["datasets"].append(
            {"path": str(path), "frames": len(dataset), "checked_first_and_last": True}
        )
        print(f"Checked {path.name}", flush=True)
        del dataset
    if not args.skip_backbone_check:
        for key in ("siglip_path", "gemma3_path", "tokenizer_path"):
            path = Path(value_cfg.actor.model[key])
            assert (path / "config.json").is_file(), path
            assert (path / "model.safetensors").is_file(), path
    if args.policy_smoke:
        from rlinf.models.embodiment.openpi_cfg import get_model
        from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
            Observation,
        )

        print(
            "Loading requested policy checkpoint for GPU forward/backward", flush=True
        )
        model = get_model(cfg.actor.model).cuda().train()
        model.gradient_checkpointing_enable()
        batch = torch.utils._pytree.tree_map(
            lambda *values: torch.as_tensor(np.stack(values)).cuda(), *smoke_samples
        )
        # Synthetic routing labels are only for this in-memory smoke test.
        # Production labels are generated from the trained value model.
        import psutil

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, foreach=False)
        memory_samples = []
        for step in range(args.smoke_steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(
                    data={
                        "observation": Observation.from_dict(batch),
                        "actions": batch["actions"],
                        "advantage": torch.tensor([True, False], device="cuda"),
                    }
                )
                loss = loss.mean()
            assert torch.isfinite(loss)
            loss.backward()
            grad = model.action_out_proj.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
            loss_value, grad_value = loss.item(), grad.norm().item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del loss, grad
            torch.cuda.synchronize()
            sample = {
                "step": step + 1,
                "loss": loss_value,
                "action_head_grad_norm": grad_value,
                "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "gpu_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "cpu_rss_gib": psutil.Process().memory_info().rss / 2**30,
            }
            memory_samples.append(sample)
            print(json.dumps(sample), flush=True)
        report["policy_gpu_smoke"] = memory_samples
    output = Path(
        os.environ.get(
            "LBENCH_RECAP_OUTPUT", "/root/autodl-tmp/outputs/lbench_recap_round1"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
