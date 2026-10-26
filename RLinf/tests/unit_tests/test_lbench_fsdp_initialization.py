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

"""Exercise the real PPO FSDP initialization configuration without a GPU."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import (
    GemmaDecoderLayer,
    GemmaRotaryEmbedding,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

import rlinf.hybrid_engines.fsdp.strategy.fsdp as fsdp_module
from rlinf.hybrid_engines.fsdp import ShardingStrategy
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(
    params=[
        "lbench_ppo_openpi_pi05_drawer_2gpu",
        "lbench_ppo_openpi_pi05_two_tasks_3gpu",
    ]
)
def profile(request, monkeypatch):
    embodiment = _REPO_ROOT / "examples" / "embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(embodiment))
    monkeypatch.setenv("LOCAL_RANK", "0")
    with initialize_config_dir(
        config_dir=str(embodiment / "config"), version_base="1.3"
    ):
        return compose(config_name=request.param)


@pytest.fixture
def tiny_openpi_layers():
    # Only a small hierarchy of real installed layer types. The large OpenPI
    # constructor and pretrained checkpoint are never instantiated or loaded.
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        train_expert_only=True, noise_method="flow_sde", use_rlt=False
    )
    model._no_split_modules = OpenPi0ForRLActionPrediction._no_split_modules.fget(model)
    model._no_split_names = OpenPi0ForRLActionPrediction._no_split_names.fget(model)
    gemma_config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        vocab_size=32,
        use_adarms=False,
    )
    model.paligemma = torch.nn.Module()
    model.paligemma.layer = GemmaDecoderLayer(gemma_config, 0)
    model.paligemma.rotary_emb = GemmaRotaryEmbedding(gemma_config)
    model.paligemma.vision = SiglipVisionEmbeddings(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=14,
            patch_size=14,
        )
    )
    model.paligemma.requires_grad_(False)
    model.expert = GemmaDecoderLayer(gemma_config, 0)
    model.action_in_proj = torch.nn.Linear(8, 16)
    model.action_out_proj = torch.nn.Linear(16, 8)
    # OpenPI constructs its critic with ReLU, including for pi05_lbench_joint.
    model.value_head = ValueHead(
        input_dim=16, hidden_sizes=(8,), output_dim=1, activation="relu"
    )
    for name, module in model.named_modules():
        module._fsdp_wrap_name = name.split(".")[-1]
    assert sum(parameter.numel() for parameter in model.parameters()) < 20000
    return model


def test_profiles_reach_fsdp_constructor_with_supported_options(
    profile, tiny_openpi_layers, monkeypatch
):
    captured = {}

    def capture_fsdp(**kwargs):
        captured.update(kwargs)
        return kwargs["module"]

    # Everything before the actual distributed/GPU wrapper is production code.
    monkeypatch.setattr(fsdp_module, "FSDP", capture_fsdp)
    strategy = FSDPStrategyBase.create(profile.actor, world_size=1)
    assert (
        strategy.wrap_model(tiny_openpi_layers, device_mesh=None) is tiny_openpi_layers
    )
    assert captured["sharding_strategy"] == ShardingStrategy.NO_SHARD
    assert captured["use_orig_params"] is True
    assert captured["cpu_offload"].offload_params is False
    assert captured["backward_prefetch"] is None
    assert captured["forward_prefetch"] is False
    assert captured["limit_all_gathers"] is False
    assert captured["sync_module_states"] is True
    assert captured["device_id"] == 0
    assert profile.actor.fsdp_config.gradient_checkpointing is False
    assert profile.actor.fsdp_config.amp_autocast.enabled is False
    for name in ("param_dtype", "reduce_dtype", "buffer_dtype"):
        assert getattr(captured["mixed_precision"], name) is torch.float32
    assert profile.actor.model.openpi.parameter_precision == "float32"

    policy = captured["auto_wrap_policy"]
    # Verify real class-name resolution and the frozen VLM boundary, rather
    # than merely checking that a callable wrapping policy was returned.
    for name in (
        "paligemma",
        "paligemma.layer",
        "paligemma.vision",
        "paligemma.rotary_emb",
        "expert",
        "action_in_proj",
        "action_out_proj",
        "value_head",
    ):
        module = tiny_openpi_layers.get_submodule(name)
        assert policy(
            module,
            recurse=False,
            nonwrapped_numel=sum(
                parameter.numel() for parameter in module.parameters()
            ),
        ), name
    assert not any(
        parameter.requires_grad
        for parameter in tiny_openpi_layers.paligemma.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in tiny_openpi_layers.expert.parameters()
    )


@pytest.mark.parametrize("field", ["param_dtype", "reduce_dtype", "buffer_dtype"])
def test_float32_spelling_reproduces_failure_before_fsdp_allocation(
    profile, tiny_openpi_layers, monkeypatch, field
):
    profile.actor.fsdp_config.mixed_precision[field] = "float32"

    def forbidden_fsdp(**kwargs):
        pytest.fail("Invalid precision reached the distributed FSDP constructor")

    monkeypatch.setattr(fsdp_module, "FSDP", forbidden_fsdp)
    strategy = FSDPStrategyBase.create(profile.actor, world_size=1)
    with pytest.raises(ValueError, match="Could not parse the precision of `float32`"):
        strategy.wrap_model(tiny_openpi_layers, device_mesh=None)
