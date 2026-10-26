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

"""Exercise OpenPI's real FSDP wrapping and repeated exports on tiny CPU layers."""

import contextlib
import warnings
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
)
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import (
    GemmaDecoderLayer,
    GemmaRotaryEmbedding,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)

HEAD_KEY = "paligemma_with_expert.gemma_expert.lm_head.weight"
EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
VLM_PREFIX = "paligemma_with_expert.paligemma."


class _TinyPolicy(nn.Module):
    """Keep OpenPI's module ownership and bypassed-container call pattern."""

    def __init__(self, *, old_policy=False):
        super().__init__()
        self.config = SimpleNamespace(
            train_expert_only=True, noise_method="flow_sde", use_rlt=False
        )
        self._no_split_modules = OpenPi0ForRLActionPrediction._no_split_modules.fget(
            self
        )
        self._no_split_names = OpenPi0ForRLActionPrediction._no_split_names.fget(self)
        assert "gemma_expert" in self._no_split_names
        if old_policy:
            self._no_split_names.remove("gemma_expert")

        cfg = GemmaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_hidden_layers=1,
            vocab_size=16,
            use_adarms=False,
        )
        cfg._attn_implementation = "eager"
        self.paligemma_with_expert = nn.Module()
        vlm = nn.Module()
        vlm.language_model = nn.Module()
        vlm.language_model.embed_tokens = nn.Embedding(16, 8)
        vlm.language_model.layer = GemmaDecoderLayer(cfg, 0)
        vlm.language_model.rotary_emb = GemmaRotaryEmbedding(cfg)
        vlm.lm_head = nn.Linear(8, 16, bias=False)
        vlm.lm_head.weight = vlm.language_model.embed_tokens.weight
        vlm.vision = SiglipVisionEmbeddings(
            SiglipVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=14,
                patch_size=14,
            )
        )
        vlm.requires_grad_(False)
        self.paligemma_with_expert.paligemma = vlm

        expert = nn.Module()
        expert.model = nn.Module()
        expert.model.embed_tokens = nn.Embedding(16, 8)
        expert.lm_head = nn.Linear(8, 16, bias=False)
        expert.lm_head.weight = expert.model.embed_tokens.weight
        # OpenPI supplies embeddings itself, leaving the tied output head unused.
        expert.model.embed_tokens = None
        expert.model.layer = GemmaDecoderLayer(cfg, 0)
        expert.model.rotary_emb = GemmaRotaryEmbedding(cfg)
        self.paligemma_with_expert.gemma_expert = expert
        self.action_in_proj = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 4)
        self.value_head = ValueHead(
            input_dim=8, hidden_sizes=(4,), output_dim=1, activation="relu"
        )
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.split(".")[-1]
        assert sum(parameter.numel() for parameter in self.parameters()) < 10000

    def forward(self, inputs):
        positions = torch.arange(inputs.shape[1]).expand(inputs.shape[0], -1)
        # OpenPI accesses the inner language/decoder models directly, so the
        # frozen PaliGemma and expert containers' FSDP.forward are bypassed.
        with torch.no_grad():
            vlm = self.paligemma_with_expert.paligemma.language_model
            embeddings = vlm.rotary_emb(inputs, positions)
            prefix = vlm.layer(inputs, position_embeddings=embeddings)[0]
        suffix = self.action_in_proj(inputs) + prefix
        expert = self.paligemma_with_expert.gemma_expert.model
        embeddings = expert.rotary_emb(suffix, positions)
        suffix = expert.layer(suffix, position_embeddings=embeddings)[0]
        return self.action_out_proj(suffix), self.value_head(prefix.mean(1).detach())


@pytest.fixture(scope="module")
def cpu_process_group(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("This regression requires the CPU Gloo backend")
    if dist.is_initialized():
        pytest.skip("This test owns an isolated one-rank CPU process group")
    rendezvous = tmp_path_factory.mktemp("unused-head-gloo") / "rendezvous"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous.as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        yield
    finally:
        dist.destroy_process_group()
        torch.set_num_threads(previous_threads)


def _wrapped_policy(*, old_policy=False):
    torch.manual_seed(431)
    model = _TinyPolicy(old_policy=old_policy)
    policy = get_fsdp_wrap_policy(
        model, config={"use_orig_params": True}, model_type="openpi"
    )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        device_id=torch.device("cpu"),
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        # CPU FSDP cannot perform GPU-only initialization broadcasts.
        sync_module_states=False,
    )


def _export(model, *, full):
    return get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=full, cpu_offload=full),
    )


def _update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    for micro in range(4):
        sync = model.no_sync() if micro < 3 else contextlib.nullcontext()
        with sync:
            inputs = torch.linspace(-1, 1, 48).reshape(2, 3, 8) + micro * 0.01
            actions, value = model(inputs)
            loss = ((actions - 0.4).square().mean() + (value + 0.2).square().mean()) / 4
            assert torch.isfinite(loss)
            loss.backward()
    for group in ("gemma_expert.model", "value_head"):
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if group in name.replace("_fsdp_wrapped_module.", "")
            and parameter.is_leaf
            and parameter.grad is not None
        ]
        assert gradients, group
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.square().sum()) for gradient in gradients) > 0
    optimizer.step()


def test_old_policy_reproduces_nonleaf_warning_and_rollout_export_failure(
    cpu_process_group,
):
    model = _wrapped_policy(old_policy=True)
    initial = _export(model, full=False)
    assert HEAD_KEY in initial
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
        weight_decay=0,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _update(model, optimizer)
        head = model.paligemma_with_expert.gemma_expert.lm_head
        assert not head.weight.is_leaf
        assert "weight" not in head._parameters
        with pytest.raises(AssertionError, match=r"gemma_expert\.lm_head\.weight"):
            _export(model, full=False)
    assert any(
        "not a leaf Tensor" in str(warning.message)
        and warning.filename.endswith("_flat_param.py")
        for warning in captured
    )


def test_expert_container_preserves_gradients_sync_and_checkpoint_exports(
    cpu_process_group, tmp_path
):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = _wrapped_policy()
        assert isinstance(model.paligemma_with_expert.gemma_expert, FSDP)
        initial = {
            key: value.clone() for key, value in _export(model, full=False).items()
        }
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.01,
            weight_decay=0,
        )
        for iteration in range(2):
            _update(model, optimizer)
            head = model.paligemma_with_expert.gemma_expert.lm_head
            assert head.weight.is_leaf
            assert "weight" in head._parameters
            for full in (False, True):
                state = _export(model, full=full)
                assert set(state) == set(initial)
                for key in state:
                    if key == HEAD_KEY or key.startswith(VLM_PREFIX):
                        torch.testing.assert_close(
                            state[key], initial[key], rtol=0, atol=0
                        )
                for prefix in (EXPERT_PREFIX, "value_head."):
                    assert any(
                        not torch.equal(state[key], initial[key])
                        for key in state
                        if key.startswith(prefix)
                    ), prefix
            optim_state = get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            assert optim_state["state"]
            for prefix in (EXPERT_PREFIX, "value_head."):
                assert any(key.startswith(prefix) for key in optim_state["state"])
            checkpoint = tmp_path / f"iteration_{iteration}.pt"
            torch.save({"model": state, "optimizer": optim_state}, checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert set(saved["model"]) == set(initial)
            assert saved["optimizer"]["state"]
    assert not any("not a leaf Tensor" in str(warning.message) for warning in captured)
