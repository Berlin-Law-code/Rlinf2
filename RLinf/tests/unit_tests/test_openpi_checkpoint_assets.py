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

"""CPU tests for assets across the actual loader and FSDP checkpoint hook."""

import ast
import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment import openpi as model_loader
from rlinf.models.embodiment.openpi import checkpoint_assets as assets
from rlinf.models.embodiment.openpi.lerobot_transforms import (
    LeRobotNormalize,
    LeRobotUnnormalize,
)

ROOT = Path(__file__).resolve().parents[2]


def _stats():
    return {
        feature: {
            "mean": [0.5] * 8,
            "std": [0.25] * 8,
            "q01": [0.0] * 8,
            "q99": [1.0] * 8,
        }
        for feature in ("state", "actions")
    }


def _assets():
    return assets.build_checkpoint_assets(
        config_name="pi05_lbench_joint",
        norm_stats=_stats(),
        asset_ids=["exact", "lbench/lbench_multitask_joint"],
        parameter_precision="float32",
        lerobot_compatible=True,
        action_horizon=50,
        execute_horizon=50,
        action_dim=8,
    )


def _write_stats(directory, payload=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "norm_stats.json"
    path.write_text(json.dumps(payload if payload is not None else _stats()))
    return path


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.bfloat16))

    def to_bfloat16_for_selected_params(self, precision):
        assert precision == "bfloat16"
        self.to(dtype=torch.bfloat16)


class _TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.paligemma_with_expert = _Backbone()

    def freeze_vlm(self):
        self.paligemma_with_expert.requires_grad_(False)

    def setup_wrappers(self, transforms, output_transforms):
        from openpi.transforms import compose

        self._input_transform = compose(transforms)
        self._output_transform = compose(output_transforms)


@pytest.fixture
def tiny_loader(monkeypatch):
    """Replace only model allocation/config; exercise real loading and assets."""
    from openpi.shared.normalize import NormStats
    from openpi.transforms import Group

    def get_config(config_name, model_path=None, data_kwargs=None):
        data_kwargs = data_kwargs or {}
        stats_path = data_kwargs.get("norm_stats_path")
        compatible = bool(data_kwargs.get("lerobot_compatible", False))

        def create(*_):
            payload = json.loads(Path(stats_path).read_text())
            payload = payload.get("norm_stats", payload)
            stats = {
                key: NormStats(
                    **{name: np.asarray(value) for name, value in values.items()}
                )
                for key, values in payload.items()
            }
            inputs = [LeRobotNormalize(stats, use_quantiles=True)] if compatible else []
            outputs = (
                [LeRobotUnnormalize(stats, use_quantiles=True)] if compatible else []
            )
            return SimpleNamespace(
                norm_stats=stats,
                asset_id=Path(stats_path).parent.name,
                use_quantile_norm=True,
                data_transforms=Group(inputs=inputs, outputs=outputs),
                model_transforms=Group(),
            )

        data = SimpleNamespace(
            repo_id="lbench/lbench_multitask_joint",
            lerobot_compatible=compatible,
            create=create,
        )
        return SimpleNamespace(
            data=data,
            assets_dirs=model_path,
            model=SimpleNamespace(
                action_horizon=50,
                action_chunk=50,
                action_env_dim=8,
                train_expert_only=False,
                add_value_head=False,
            ),
        )

    configuration = ModuleType("rlinf.models.embodiment.openpi.dataconfig")
    configuration.get_openpi_config = get_config
    modeling = ModuleType("rlinf.models.embodiment.openpi.openpi_action_model")
    modeling.OpenPi0Config = SimpleNamespace
    modeling.OpenPi0ForRLActionPrediction = _TinyModel
    monkeypatch.setitem(sys.modules, configuration.__name__, configuration)
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    return model_loader.get_model


def _config(checkpoint, explicit_stats=None):
    data = {} if explicit_stats is None else {"norm_stats_path": str(explicit_stats)}
    return OmegaConf.create(
        {
            "model_path": str(checkpoint),
            "openpi_data": data,
            "openpi": {
                "config_name": "pi05_lbench_joint",
                "action_horizon": 50,
                "action_chunk": 20,
                "action_env_dim": 8,
                "add_value_head": True,
            },
        }
    )


def _save_weights(checkpoint, state=None):
    directory = checkpoint / "actor/model_state_dict"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        state
        if state is not None
        else {"paligemma_with_expert.weight": torch.tensor([1.0001])},
        directory / "full_weights.pt",
    )


def _manager_save_method():
    # Avoid constructing Ray/CUDA workers while exercising the real save hook.
    source = ROOT / "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py"
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
    return namespace["save_checkpoint"]


@pytest.mark.parametrize("entry", ["", "actor"])
def test_sft_to_ppo_to_second_load_keeps_effective_contract(
    tmp_path, tiny_loader, entry
):
    sft = tmp_path / "sft"
    assets.export_checkpoint_assets(_assets(), sft / "actor")
    _save_weights(sft)
    raw_config = _config(sft / entry)
    model = tiny_loader(raw_config)
    assert "parameter_precision" not in raw_config.openpi
    assert "lerobot_compatible" not in raw_config.openpi_data
    assert model.paligemma_with_expert.weight.dtype == torch.float32
    torch.testing.assert_close(
        model.paligemma_with_expert.weight, torch.tensor([1.0001]), rtol=0, atol=0
    )

    ppo = tmp_path / "ppo"
    holder = SimpleNamespace(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        _openpi_checkpoint_assets=model._openpi_checkpoint_assets,
        _cfg=OmegaConf.create({"fsdp_config": {"save_full_model_weights": True}}),
        _strategy=SimpleNamespace(
            save_checkpoint=lambda *_args, **_kwargs: _save_weights(
                ppo, model.state_dict()
            )
        ),
    )
    _manager_save_method()(holder, str(ppo / "actor"), 40)
    restored = tiny_loader(_config(ppo / entry))
    manifest = assets.read_checkpoint_manifest(ppo / entry)
    assert manifest["compatibility"] == {
        "lerobot_compatible": True,
        "parameter_precision": "float32",
    }
    assert manifest["action_horizon"] == 50
    assert manifest["execute_horizon"] == 20
    assert manifest["action_dim"] == 8
    assert restored.paligemma_with_expert.weight.dtype == torch.float32
    torch.testing.assert_close(
        restored.paligemma_with_expert.weight,
        model.paligemma_with_expert.weight,
        rtol=0,
        atol=0,
    )
    state = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    expected = state * 2 - 1
    np.testing.assert_array_equal(
        restored._input_transform({"state": state})["state"], expected
    )
    action = np.tile(expected, (50, 1))
    np.testing.assert_allclose(
        restored._output_transform({"actions": action, "state": expected})["actions"],
        np.tile(state, (50, 1)),
        rtol=0,
        atol=3e-8,
    )


def test_wrong_external_norm_fails_before_model_allocation(
    tmp_path, tiny_loader, monkeypatch
):
    checkpoint = tmp_path / "sft"
    assets.export_checkpoint_assets(_assets(), checkpoint)
    payload = _stats()
    payload["actions"]["q99"][0] = 2.0
    wrong = _write_stats(tmp_path / "wrong", payload)
    monkeypatch.setattr(
        _TinyModel,
        "__init__",
        lambda *_: pytest.fail("model allocated before norm validation"),
    )
    with pytest.raises(ValueError, match="differs from checkpoint.*actions.q99"):
        tiny_loader(_config(checkpoint, wrong))


def test_equal_external_stats_selects_bundled_and_ignores_json_layout(tmp_path):
    checkpoint = tmp_path / "sft"
    assets.export_checkpoint_assets(_assets(), checkpoint)
    explicit = _write_stats(tmp_path / "external")
    selected = assets.resolve_checkpoint_norm_stats(checkpoint, explicit)
    assert selected == (checkpoint / "exact/norm_stats.json").resolve()


def test_explicit_finetuning_norm_loads_weights_and_saves_new_stats(
    tmp_path, tiny_loader
):
    checkpoint = tmp_path / "source"
    assets.export_checkpoint_assets(_assets(), checkpoint)
    _save_weights(checkpoint)
    before = {str(p): p.read_bytes() for p in checkpoint.rglob("*.json")}
    payload = _stats()
    payload["state"]["q99"] = [2.0] * 8
    payload["actions"]["q99"] = [4.0] * 8
    external = _write_stats(tmp_path / "new_dataset", payload)
    config = _config(checkpoint, external)
    config.allow_norm_stats_override = True
    model = tiny_loader(config)
    torch.testing.assert_close(
        model.paligemma_with_expert.weight, torch.tensor([1.0001]), rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        model._input_transform({"state": np.ones(8, np.float32)})["state"],
        np.zeros(8, np.float32),
    )
    np.testing.assert_array_equal(
        model._output_transform(
            {
                "actions": np.zeros((50, 8), np.float32),
                "state": np.zeros(8, np.float32),
            }
        )["actions"],
        np.full((50, 8), 2.0, np.float32),
    )
    assert before == {str(p): p.read_bytes() for p in checkpoint.rglob("*.json")}
    destination = tmp_path / "finetuned"
    assets.export_checkpoint_assets(model._openpi_checkpoint_assets, destination)
    _save_weights(destination, model.state_dict())
    restored = tiny_loader(_config(destination, external))
    assert restored._openpi_checkpoint_assets["norm_stats"]["norm_stats"] == payload
    with pytest.raises(ValueError, match="differs from checkpoint"):
        tiny_loader(_config(destination, checkpoint / "exact/norm_stats.json"))


def test_finetuning_override_requires_explicit_stats(tmp_path):
    with pytest.raises(ValueError, match="requires an explicit norm_stats_path"):
        assets.resolve_checkpoint_norm_stats(
            tmp_path, None, allow_explicit_override=True
        )


def test_finetuning_override_rejects_non_finite_external_stats(tmp_path):
    checkpoint = tmp_path / "source"
    assets.export_checkpoint_assets(_assets(), checkpoint)
    payload = _stats()
    payload["actions"]["q99"][0] = float("nan")
    external = _write_stats(tmp_path / "external", payload)
    with pytest.raises(ValueError, match="differs from checkpoint"):
        assets.resolve_checkpoint_norm_stats(
            checkpoint, external, allow_explicit_override=True
        )


def test_finetuning_override_does_not_hide_corrupt_checkpoint_assets(tmp_path):
    checkpoint = tmp_path / "source"
    assets.export_checkpoint_assets(_assets(), checkpoint)
    external = _write_stats(tmp_path / "external")
    bundled_alias = checkpoint / "lbench/lbench_multitask_joint/norm_stats.json"
    payload = _stats()
    payload["actions"]["q99"][0] = 5.0
    bundled_alias.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs from checkpoint"):
        assets.resolve_checkpoint_norm_stats(
            checkpoint, external, allow_explicit_override=True
        )
    bundled_alias.unlink()
    (checkpoint / "actor/lbench/lbench_multitask_joint/norm_stats.json").unlink()
    with pytest.raises(FileNotFoundError, match="asset is missing"):
        assets.resolve_checkpoint_norm_stats(
            checkpoint, external, allow_explicit_override=True
        )


def test_legacy_without_manifest_accepts_external_and_keeps_legacy_precision(
    tmp_path, tiny_loader
):
    checkpoint = tmp_path / "legacy"
    _save_weights(checkpoint)
    stats = _write_stats(tmp_path / "external")
    model = tiny_loader(_config(checkpoint, stats))
    assert model.paligemma_with_expert.weight.dtype == torch.bfloat16
    assert model._openpi_checkpoint_assets["manifest"]["compatibility"] == {
        "lerobot_compatible": False,
        "parameter_precision": "openpi",
    }


def test_legacy_bundled_canonical_asset_is_preferred(tmp_path):
    stats = _write_stats(tmp_path / "legacy/lbench/lbench_multitask_joint")
    assert (
        assets.resolve_checkpoint_norm_stats(
            tmp_path / "legacy",
            None,
            canonical_asset_id="lbench/lbench_multitask_joint",
        )
        == stats.resolve()
    )


def test_missing_declared_stats_cannot_be_replaced_by_external_file(tmp_path):
    checkpoint = tmp_path / "sft"
    checkpoint.mkdir()
    (checkpoint / "openpi_assets.json").write_text(json.dumps(_assets()["manifest"]))
    external = _write_stats(tmp_path / "external")
    with pytest.raises(FileNotFoundError, match="asset is missing"):
        assets.resolve_checkpoint_norm_stats(checkpoint, external)


def test_stats_snapshot_is_independent_of_later_mutation(tmp_path):
    payload = _stats()
    contract = assets.build_checkpoint_assets(
        config_name="pi05_lbench_joint",
        norm_stats=payload,
        asset_ids=["exact"],
        parameter_precision="float32",
        lerobot_compatible=True,
        action_horizon=50,
        execute_horizon=20,
        action_dim=8,
    )
    payload["state"]["q01"][0] = 99
    assets.export_checkpoint_assets(contract, tmp_path)
    actual = json.loads((tmp_path / "exact/norm_stats.json").read_text())
    assert actual["norm_stats"]["state"]["q01"][0] == 0


@pytest.mark.parametrize("rank", [0, 1])
def test_asset_write_failure_is_raised_on_every_rank(tmp_path, monkeypatch, rank):
    monkeypatch.setattr(assets.dist, "is_available", lambda: True)
    monkeypatch.setattr(assets.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(assets.dist, "get_rank", lambda: rank)
    calls = []

    def export(*_):
        calls.append("export")
        raise OSError("disk full")

    def broadcast(status, src):
        assert src == 0
        calls.append("broadcast")
        if rank == 0:
            assert status == ["OSError: disk full"]
        else:
            assert status == [None]
            status[0] = "OSError: disk full"

    monkeypatch.setattr(assets, "export_checkpoint_assets", export)
    monkeypatch.setattr(assets.dist, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="Failed exporting.*disk full"):
        assets.save_checkpoint_assets_distributed(_assets(), tmp_path)
    assert calls == (["export", "broadcast"] if rank == 0 else ["broadcast"])


def test_non_openpi_manager_save_does_not_add_collectives(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        assets,
        "save_checkpoint_assets_distributed",
        lambda *_: pytest.fail("non-OpenPI asset export"),
    )
    holder = SimpleNamespace(
        model=None,
        optimizer=None,
        lr_scheduler=None,
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        _cfg=OmegaConf.create({"fsdp_config": {"save_full_model_weights": True}}),
        _strategy=SimpleNamespace(
            save_checkpoint=lambda *_args, **_kwargs: calls.append("weights")
        ),
    )
    _manager_save_method()(holder, str(tmp_path))
    assert calls == ["weights"]


@pytest.mark.parametrize("name", ["../outside/norm_stats.json", "/tmp/norm_stats.json"])
def test_manifest_rejects_paths_outside_checkpoint(tmp_path, name):
    manifest = copy.deepcopy(_assets()["manifest"])
    manifest["norm_stats"] = [name]
    (tmp_path / "openpi_assets.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint-relative"):
        assets.resolve_checkpoint_norm_stats(tmp_path, None)


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("fail_save", [False, True])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_cpu_checkpoint_restores_residency_even_when_save_fails(
    tmp_path, offloaded, fail_save, rank
):
    holder = SimpleNamespace(
        model=SimpleNamespace(rank=rank),
        optimizer=None,
        lr_scheduler=None,
        device=0,
        is_weight_offloaded=offloaded,
        is_optimizer_offloaded=offloaded,
        _cfg=OmegaConf.create(
            {
                "fsdp_config": {
                    "strategy": "fsdp",
                    "sharding_strategy": "no_shard",
                    "use_orig_params": True,
                    "checkpoint_cpu_staging": True,
                }
            }
        ),
    )
    holder.offload_param_and_grad = lambda: setattr(holder, "is_weight_offloaded", True)
    holder.offload_optimizer = lambda: setattr(holder, "is_optimizer_offloaded", True)
    holder.load_param_and_grad = lambda _: setattr(holder, "is_weight_offloaded", False)
    holder.load_optimizer = lambda _: setattr(holder, "is_optimizer_offloaded", False)

    def save(*args, **kwargs):
        expected_offload = offloaded or rank == 0
        assert holder.is_weight_offloaded == expected_offload
        assert holder.is_optimizer_offloaded == expected_offload
        assert kwargs["rank0_only"] is True
        if fail_save:
            raise OSError("disk full")

    holder._strategy = SimpleNamespace(save_checkpoint=save)
    if fail_save:
        with pytest.raises(OSError, match="disk full"):
            _manager_save_method()(holder, str(tmp_path))
    else:
        _manager_save_method()(holder, str(tmp_path))
    assert holder.is_weight_offloaded == offloaded
    assert holder.is_optimizer_offloaded == offloaded


@pytest.mark.parametrize(
    "key,value",
    [
        ("strategy", "fsdp2"),
        ("sharding_strategy", "full_shard"),
        ("use_orig_params", False),
    ],
)
def test_cpu_checkpoint_rejects_unsupported_fsdp_modes(tmp_path, key, value):
    config = {
        "strategy": "fsdp",
        "sharding_strategy": "no_shard",
        "use_orig_params": True,
        "checkpoint_cpu_staging": True,
    }
    config[key] = value
    holder = SimpleNamespace(
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        _cfg=OmegaConf.create({"fsdp_config": config}),
    )
    with pytest.raises(ValueError, match="checkpoint_cpu_staging requires"):
        _manager_save_method()(holder, str(tmp_path))
