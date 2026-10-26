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

"""CPU checks for SFT adaptation assets and action-horizon propagation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from openpi.training import data_loader

from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
    build_official_openpi_sft_dataloader,
)
from rlinf.models.embodiment.openpi import dataconfig
from rlinf.models.embodiment.openpi.checkpoint_assets import (
    resolve_checkpoint_norm_stats,
)
from toolkits.lerobot.prepare_lbench_sft_adaptation import prepare_initialization


@pytest.mark.parametrize("horizon", [None, 20])
def test_horizon_reaches_actual_dataset_timestamp_queries(monkeypatch, horizon):
    reference = dataconfig.get_openpi_config("pi05_lbench_joint")
    captured = {}

    def dataset(repo_id, *, delta_timestamps):
        captured.update(delta_timestamps)
        return []

    monkeypatch.setattr(
        data_loader,
        "_import_lerobot_dataset",
        lambda: SimpleNamespace(
            LeRobotDatasetMetadata=lambda _: SimpleNamespace(fps=30, tasks={}),
            LeRobotDataset=dataset,
        ),
    )

    def loader(config, **kwargs):
        data_loader.create_torch_dataset(
            SimpleNamespace(
                repo_id="test/data",
                action_sequence_keys=("action",),
                prompt_from_task=False,
            ),
            config.model.action_horizon,
            config.model,
        )
        return SimpleNamespace(
            _data_loader=SimpleNamespace(_data_loader=SimpleNamespace(batch_size=16)),
            data_config=lambda: config.data,
        )

    monkeypatch.setattr(data_loader, "create_data_loader", loader)
    cfg = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": 16,
                "model": {
                    "model_type": "openpi",
                    "model_path": "/unused",
                    "openpi": {"config_name": "pi05_lbench_joint"},
                },
            },
            "data": {"num_workers": 0},
        }
    )
    if horizon is not None:
        cfg.actor.model.openpi.action_horizon = horizon
    build_official_openpi_sft_dataloader(cfg, 1, 0, "test/data")
    assert captured["action"] == [t / 30 for t in range(horizon or 50)]
    assert reference.model.action_horizon == 50


def test_adaptation_snapshots_new_stats_without_changing_original(tmp_path):
    source = tmp_path / "source"
    weights = source / "actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"original weights")
    original_manifest = {"compatibility": {"parameter_precision": "float32"}}
    (source / "openpi_assets.json").write_text(json.dumps(original_manifest))
    norm = {
        key: {"mean": [0.0] * 8, "std": [1.0] * 8, "q01": [-1.0] * 8, "q99": [1.0] * 8}
        for key in ("state", "actions")
    }
    stats = tmp_path / "norm_stats.json"
    stats.write_text(json.dumps({"norm_stats": norm}))
    output = tmp_path / "initialization"
    prepare_initialization(source, stats, output)
    assert (output / "actor/model_state_dict/full_weights.pt").resolve() == weights
    assert weights.read_bytes() == b"original weights"
    assert json.loads((source / "openpi_assets.json").read_text()) == original_manifest
    resolved = resolve_checkpoint_norm_stats(output, stats)
    assert resolved != stats
    with pytest.raises(FileExistsError):
        prepare_initialization(source, stats, output)
    norm["state"]["mean"][0] = 5.0
    stats.write_text(json.dumps({"norm_stats": norm}))
    with pytest.raises(ValueError, match="differs"):
        resolve_checkpoint_norm_stats(output, stats)


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize(
    "launcher,env_prefix,gpu_suffix,gpus",
    [
        ("15_sft_adapt_1gpu.sh", "LBENCH_SFT_ADAPT", "GPU", "0"),
        ("17_sft_base_2gpu.sh", "LBENCH_SFT_BASE", "GPUS", "0,1"),
        ("18_sft_base_4gpu.sh", "LBENCH_SFT_BASE4", "GPUS", "0,1,2,3"),
    ],
)
def test_launcher_dry_run_and_failure_exit_status(
    tmp_path, dry_run, launcher, env_prefix, gpu_suffix, gpus
):
    """Use a fake interpreter: no model, Ray worker, or GPU process is launched."""
    root = Path(__file__).resolve().parents[2]
    calls = tmp_path / "calls.jsonl"
    interpreter = tmp_path / "fake_python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"with open({str(calls)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(0 if sys.argv[1] == '-' or '--cfg' in sys.argv else 7)\n"
    )
    interpreter.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(root / "scripts/lbench/remote" / launcher)]
        + (["--dry-run"] if dry_run else []),
        env={
            **os.environ,
            "RLINF_ROOT": str(root),
            "RLINF_PYTHON": str(interpreter),
            f"{env_prefix}_OUTPUT": str(output),
            f"{env_prefix}_{gpu_suffix}": gpus,
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(recorded) == 2
    if dry_run:
        assert result.returncode == 0
        assert recorded[-1][-3:] == ["--cfg", "job", "--resolve"]
        assert not output.exists()
    else:
        assert result.returncode == 7
        assert "--cfg" not in recorded[-1]


@pytest.mark.parametrize("gpus", ["0", "0,0", "0,1,2", "0,", ""])
def test_two_gpu_launcher_rejects_invalid_placement_before_python(tmp_path, gpus):
    root = Path(__file__).resolve().parents[2]
    # Empty uses the documented default; explicitly blank identifiers do not.
    if not gpus:
        gpus = " "
    result = subprocess.run(
        ["bash", str(root / "scripts/lbench/remote/17_sft_base_2gpu.sh"), "--dry-run"],
        env={
            **os.environ,
            "RLINF_PYTHON": str(tmp_path / "must_not_run"),
            "LBENCH_SFT_BASE_GPUS": gpus,
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "GPU IDs" in result.stderr


@pytest.mark.parametrize("gpus", ["0,1", "0,1,2", "0,1,2,2", "0,1,2,00", "0,1,2,3,4"])
def test_four_gpu_launcher_rejects_invalid_placement_before_python(tmp_path, gpus):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts/lbench/remote/18_sft_base_4gpu.sh"), "--dry-run"],
        env={
            **os.environ,
            "RLINF_PYTHON": str(tmp_path / "must_not_run"),
            "LBENCH_SFT_BASE4_GPUS": gpus,
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "GPU IDs" in result.stderr
