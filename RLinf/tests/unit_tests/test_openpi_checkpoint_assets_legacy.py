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

"""Regression coverage for non-LBench asset ids and offloaded save failures."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi import checkpoint_assets as assets


def test_absolute_dataset_repo_id_preserves_external_legacy_assets(tmp_path):
    stats = {"state": {"mean": [0.0], "std": [1.0]}}
    external = tmp_path / "external/norm_stats.json"
    external.parent.mkdir()
    external.write_text(json.dumps(stats))
    selected = assets.resolve_checkpoint_norm_stats(
        tmp_path / "legacy",
        external,
        canonical_asset_id=str(tmp_path / "dataset"),
    )
    assert selected == external.resolve()
    contract = assets.build_checkpoint_assets(
        config_name="pi0_custom",
        norm_stats=stats,
        asset_ids=[str(tmp_path / "dataset"), None],
        parameter_precision="openpi",
        lerobot_compatible=False,
        action_horizon=8,
        execute_horizon=8,
        action_dim=7,
    )
    assert contract["manifest"]["asset_ids"] == ["normalization"]
    assets.export_checkpoint_assets(contract, tmp_path / "next")
    restored = assets.resolve_checkpoint_norm_stats(tmp_path / "next/actor", None)
    assert restored == (tmp_path / "next/actor/normalization/norm_stats.json").resolve()


def test_offloaded_state_is_restored_after_asset_export_failure(tmp_path, monkeypatch):
    source = (
        Path(__file__).resolve().parents[2]
        / "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FSDPModelManager"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    calls = []

    def export(*_):
        calls.append("assets")
        raise RuntimeError("rank zero disk failure")

    monkeypatch.setattr(assets, "save_checkpoint_assets_distributed", export)
    holder = SimpleNamespace(
        model=None,
        optimizer=None,
        lr_scheduler=None,
        device="cpu",
        is_weight_offloaded=True,
        is_optimizer_offloaded=True,
        load_param_and_grad=lambda *_: calls.append("load weights"),
        load_optimizer=lambda *_: calls.append("load optimizer"),
        offload_param_and_grad=lambda *_: calls.append("offload weights"),
        offload_optimizer=lambda *_: calls.append("offload optimizer"),
        _openpi_checkpoint_assets={},
        _cfg=OmegaConf.create({"fsdp_config": {"save_full_model_weights": True}}),
        _strategy=SimpleNamespace(
            save_checkpoint=lambda *_args, **_kwargs: calls.append("weights")
        ),
    )
    holder.load_param_and_grad = lambda *_: (
        calls.append("load weights"),
        setattr(holder, "is_weight_offloaded", False),
    )
    holder.load_optimizer = lambda *_: (
        calls.append("load optimizer"),
        setattr(holder, "is_optimizer_offloaded", False),
    )
    with pytest.raises(RuntimeError, match="rank zero disk failure"):
        namespace["save_checkpoint"](holder, str(tmp_path / "actor"))
    assert calls == [
        "load weights",
        "load optimizer",
        "weights",
        "assets",
        "offload weights",
        "offload optimizer",
    ]
