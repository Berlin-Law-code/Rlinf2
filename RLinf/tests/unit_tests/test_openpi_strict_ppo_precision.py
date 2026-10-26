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

"""CPU contracts for explicit FP32 PPO computation in actor and rollout."""

import ast
import json
import logging
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment import openpi as model_loader
from rlinf.models.embodiment.openpi import precision


@pytest.fixture
def backend(monkeypatch):
    """Exercise backend policy without initializing CUDA or leaking switches."""
    state = SimpleNamespace(precision="high", events=[])

    def set_precision(value):
        state.events.append(value)
        state.precision = value

    monkeypatch.setattr(torch, "set_float32_matmul_precision", set_precision)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: state.precision)
    monkeypatch.setattr(torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=True))
    monkeypatch.setattr(torch.backends, "cudnn", SimpleNamespace(allow_tf32=True))
    return state


def _config(**openpi_overrides):
    return OmegaConf.create(
        {
            "precision": None,
            "openpi": {
                "strict_ppo_precision": True,
                "parameter_precision": "float32",
                **openpi_overrides,
            },
        }
    )


def test_disabled_flag_does_not_validate_or_touch_existing_precision(backend):
    cfg = _config(strict_ppo_precision=False, parameter_precision="openpi")
    cfg.precision = "bf16"
    assert not precision.validate_strict_ppo_precision_config(
        cfg, fsdp_config={"amp_autocast": {"enabled": True}}
    )
    # No parameter inspection, backend changes, or model attributes when disabled.
    assert precision.apply_strict_ppo_precision(object()) is None
    assert backend.precision == "high"
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32
    assert backend.events == []


@pytest.mark.parametrize("value", ["true", "false", 1, None])
def test_non_boolean_opt_in_is_rejected(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        precision.validate_strict_ppo_precision_config(
            _config(strict_ppo_precision=value)
        )


@pytest.mark.parametrize("model_precision", [None, "null", 32, "32", "fp32", "32-true"])
def test_fp32_model_precision_and_policy_only_evaluation_are_accepted(model_precision):
    cfg = _config(add_value_head=False)
    cfg.precision = model_precision
    assert precision.validate_strict_ppo_precision_config(
        cfg, torch_dtype=torch.float32
    )


def test_checkpoint_resolved_fp32_can_satisfy_explicit_opt_in():
    cfg = _config()
    del cfg.openpi.parameter_precision
    assert precision.validate_strict_ppo_precision_config(
        cfg, parameter_precision="float32"
    )
    with pytest.raises(ValueError, match="parameter_precision"):
        precision.validate_strict_ppo_precision_config(cfg)


@pytest.mark.parametrize("value", ["bf16", "bf16-mixed", "fp16", 16])
def test_model_storage_precision_conflicts_fail(value):
    cfg = _config()
    cfg.precision = value
    with pytest.raises(ValueError, match="model.precision"):
        precision.validate_strict_ppo_precision_config(cfg)


def test_loader_torch_dtype_conflict_fails():
    with pytest.raises(ValueError, match="torch_dtype"):
        precision.validate_strict_ppo_precision_config(
            _config(), torch_dtype=torch.bfloat16
        )


@pytest.mark.parametrize("field", ["param_dtype", "reduce_dtype", "buffer_dtype"])
def test_fsdp_non_fp32_cast_is_rejected(field):
    fsdp = {"mixed_precision": {field: "bf16"}}
    with pytest.raises(ValueError, match=field):
        precision.validate_strict_ppo_precision_config(_config(), fsdp_config=fsdp)


def test_fsdp_amp_conflict_is_rejected():
    with pytest.raises(ValueError, match="AMP disabled"):
        precision.validate_strict_ppo_precision_config(
            _config(), fsdp_config={"amp_autocast": {"enabled": True}}
        )


def test_fp32_fsdp_without_amp_is_accepted():
    fsdp = {
        "amp_autocast": {"enabled": False},
        "mixed_precision": {
            "param_dtype": "fp32",
            "reduce_dtype": "fp32",
            "buffer_dtype": "fp32",
        },
    }
    assert precision.validate_strict_ppo_precision_config(_config(), fsdp_config=fsdp)


def test_apply_logs_actual_precision_and_validates_parameters(backend, caplog):
    model = torch.nn.Linear(2, 1)
    with caplog.at_level(logging.INFO):
        state = precision.apply_strict_ppo_precision(model, enabled=True)
    assert state == {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cuda_autocast_enabled": False,
        "cpu_autocast_enabled": False,
    }
    assert model._openpi_strict_ppo_precision == state
    assert "highest" in caplog.text
    assert "cuda_matmul_allow_tf32" in caplog.text
    assert backend.events == ["highest"]


def test_non_fp32_model_is_rejected_without_mutating_backend(backend):
    with pytest.raises(ValueError, match="all parameters to be FP32"):
        precision.apply_strict_ppo_precision(
            torch.nn.Linear(2, 1).bfloat16(), enabled=True
        )
    assert backend.events == []


def test_active_cpu_amp_is_rejected_during_apply_and_runtime(backend):
    model = torch.nn.Linear(2, 1)
    precision.apply_strict_ppo_precision(model, enabled=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="active AMP"):
            precision.apply_strict_ppo_precision(model, enabled=True)
        with pytest.raises(RuntimeError, match="active AMP"):
            precision.assert_strict_ppo_precision_runtime(model)


@pytest.mark.parametrize(
    "changed", ["matmul_precision", "cuda_tf32", "cudnn_tf32", "parameters"]
)
def test_runtime_rejects_contract_changed_after_loading(backend, changed):
    model = torch.nn.Linear(2, 1)
    precision.apply_strict_ppo_precision(model, enabled=True)
    if changed == "matmul_precision":
        torch.set_float32_matmul_precision("high")
    elif changed == "cuda_tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif changed == "cudnn_tf32":
        torch.backends.cudnn.allow_tf32 = True
    else:
        model.bfloat16()
    with pytest.raises(RuntimeError, match="after model"):
        precision.assert_strict_ppo_precision_runtime(model)


@pytest.fixture
def tiny_loader(monkeypatch, tmp_path, backend):
    """Keep the real loader; replace architecture allocation and data factories."""
    from openpi.transforms import Group

    stats = {
        feature: {"q01": [0.0] * 8, "q99": [1.0] * 8}
        for feature in ("state", "actions")
    }
    loaded_stats = {
        feature: SimpleNamespace(mean=None, std=None, **values)
        for feature, values in stats.items()
    }
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    (tmp_path / "openpi_assets.json").write_text(
        json.dumps(
            {
                "norm_stats": ["norm_stats.json"],
                "compatibility": {"parameter_precision": "float32"},
            }
        )
    )
    weight_dir = tmp_path / "actor/model_state_dict"
    weight_dir.mkdir(parents=True)
    torch.save(
        {"paligemma_with_expert.weight": torch.tensor([1.0001])},
        weight_dir / "full_weights.pt",
    )
    # Exercise the actor-directory entry too, without making duplicate weights.
    (tmp_path / "actor/openpi_assets.json").write_text(
        (tmp_path / "openpi_assets.json").read_text()
    )
    (tmp_path / "actor/norm_stats.json").write_text(json.dumps(stats))

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

    class Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            torch.set_float32_matmul_precision(
                "high"
            )  # Real upstream constructor behavior.
            self.paligemma_with_expert = Backbone()

        def setup_wrappers(self, **kwargs):
            assert torch.equal(
                self.paligemma_with_expert.weight, torch.tensor([1.0001])
            )
            backend.events.append("loaded")

    def get_config(*args, **kwargs):
        data = SimpleNamespace(
            repo_id="lbench/lbench_multitask_joint",
            lerobot_compatible=False,
            create=lambda *_: SimpleNamespace(
                norm_stats=loaded_stats,
                asset_id="normalization",
                use_quantile_norm=True,
                data_transforms=Group(),
                model_transforms=Group(),
            ),
        )
        return SimpleNamespace(
            data=data,
            assets_dirs=tmp_path,
            model=SimpleNamespace(
                action_horizon=50,
                action_chunk=20,
                action_env_dim=8,
                train_expert_only=False,
                add_value_head=False,
            ),
        )

    config_module = ModuleType("rlinf.models.embodiment.openpi.dataconfig")
    config_module.get_openpi_config = get_config
    model_module = ModuleType("rlinf.models.embodiment.openpi.openpi_action_model")
    model_module.OpenPi0Config = SimpleNamespace
    model_module.OpenPi0ForRLActionPrediction = Model
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    return tmp_path


@pytest.mark.parametrize("entry", ["", "actor"])
@pytest.mark.parametrize("value_head", [False, True])
def test_actual_loader_applies_opt_in_after_constructor_and_load(
    tiny_loader, backend, entry, value_head
):
    cfg = _config(config_name="pi05_lbench_joint", add_value_head=value_head)
    del cfg.openpi.parameter_precision  # The checkpoint is authoritative for storage.
    cfg.model_path = str(tiny_loader / entry)
    model = model_loader.get_model(cfg)
    assert model.config.strict_ppo_precision
    assert model.config.parameter_precision == "float32"
    assert backend.events == ["high", "loaded", "highest"]
    precision.assert_strict_ppo_precision_runtime(model)


def test_loader_without_opt_in_preserves_existing_backend_behavior(
    tiny_loader, backend
):
    cfg = _config(config_name="pi05_lbench_joint")
    del cfg.openpi.strict_ppo_precision
    cfg.model_path = str(tiny_loader)
    model = model_loader.get_model(cfg)
    assert not model.config.strict_ppo_precision
    assert not hasattr(model, "_openpi_strict_ppo_precision")
    assert backend.events == ["high", "loaded"]


def _forward_method(name):
    source = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/openpi/openpi_action_model.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "OpenPi0ForRLActionPrediction"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {
        "assert_strict_ppo_precision_runtime": precision.assert_strict_ppo_precision_runtime
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("method_name", ["default_forward", "predict_action_batch"])
@pytest.mark.parametrize("enabled", [False, True])
def test_actor_and_rollout_guard_before_any_inference(backend, method_name, enabled):
    class InferenceReached(Exception):
        pass

    def reached(*args, **kwargs):
        raise InferenceReached

    holder = SimpleNamespace(
        config=SimpleNamespace(strict_ppo_precision=enabled),
        input_transform=reached,
        obs_processor=reached,
        named_parameters=lambda: iter(()),
    )
    method = MethodType(_forward_method(method_name), holder)
    data = {"chains": torch.zeros(1), "denoise_inds": torch.zeros(1)}
    with pytest.raises(RuntimeError if enabled else InferenceReached):
        method(data)
