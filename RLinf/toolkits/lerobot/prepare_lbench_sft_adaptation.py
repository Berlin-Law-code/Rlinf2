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

"""Prepare an explicitly untrained adaptation initialization from SFT weights."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from rlinf.models.embodiment.openpi.checkpoint_assets import (
    build_checkpoint_assets,
    export_checkpoint_assets,
    read_checkpoint_manifest,
    resolve_checkpoint_norm_stats,
)


def prepare_initialization(source: Path, stats: Path, output: Path) -> Path:
    """Reuse original weights and snapshot new stats without modifying the source."""
    source, stats, output = source.resolve(), stats.resolve(), output.absolute()
    weights = source / "actor/model_state_dict/full_weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    manifest = read_checkpoint_manifest(source)
    if manifest.get("compatibility", {}).get("parameter_precision") != "float32":
        raise ValueError("Expected an original FP32 SFT checkpoint")
    payload = json.loads(stats.read_text())
    norm = payload.get("norm_stats", payload)
    if set(norm) != {"state", "actions"}:
        raise ValueError("Expected state and actions normalization statistics")
    for feature in norm.values():
        if any(len(feature[key]) != 8 for key in ("mean", "std", "q01", "q99")):
            raise ValueError("Expected eight physical state/action dimensions")
    assets = build_checkpoint_assets(
        config_name="pi05_lbench_joint",
        norm_stats=norm,
        asset_ids=["lbench_sft_adapt_h20", "lbench/lbench_multitask_joint"],
        parameter_precision="float32",
        lerobot_compatible=True,
        action_horizon=20,
        execute_horizon=20,
        action_dim=8,
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace initialization: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".sft-init-") as temp:
        staged = Path(temp) / "initialization"
        export_checkpoint_assets(assets, staged)
        target = staged / "actor/model_state_dict/full_weights.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(weights)
        (staged / "initialization.json").write_text(
            json.dumps(
                {
                    "status": "untrained_adaptation_initialization",
                    "source_checkpoint": str(source),
                    "source_weights": str(weights),
                    "source_weights_bytes": weights.stat().st_size,
                    "source_norm_stats": str(stats),
                    "optimizer_restored": False,
                    "action_horizon": 20,
                    "execute_horizon": 20,
                },
                indent=2,
            )
            + "\n"
        )
        resolve_checkpoint_norm_stats(staged, stats)
        staged.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_initialization(args.source, args.stats, args.output))


if __name__ == "__main__":
    main()
