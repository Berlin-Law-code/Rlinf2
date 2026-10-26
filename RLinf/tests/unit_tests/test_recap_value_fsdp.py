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

"""Exercise RECAP's tied language embedding through real two-rank FSDP."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding

from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.embodiment.value_model.recap.modeling_critic import ValueCriticModel
from rlinf.models.embodiment.value_model.recap.value_expert import ValueExpert


class _TinyValueModel(nn.Module):
    # Keep the production name policy and embedding call. The unrelated large
    # vision/decoder modules are omitted so the distributed regression runs on CPU.
    _no_split_modules = []
    _no_split_names = ValueCriticModel._no_split_names

    def __init__(self):
        super().__init__()
        self.value_expert = ValueExpert.__new__(ValueExpert)
        nn.Module.__init__(self.value_expert)
        self.value_expert.gemma3 = nn.Module()
        gemma3 = self.value_expert.gemma3
        gemma3.model = nn.Module()
        gemma3.model.embed_tokens = Gemma3TextScaledWordEmbedding(
            32, 8, padding_idx=0, embed_scale=8**0.5
        )
        gemma3.lm_head = nn.Linear(8, 32, bias=False)
        gemma3.lm_head.weight = gemma3.model.embed_tokens.weight
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.split(".")[-1]

    def forward(self, tokens):
        return self.value_expert.embed_language_tokens(tokens).square().mean()


def _run_tied_embedding_update(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(42)
        model = _TinyValueModel()
        reference = copy.deepcopy(model)
        policy = get_fsdp_wrap_policy(model, model_type="recap_value_model")
        wrapped = FSDP(
            model,
            auto_wrap_policy=policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=torch.device("cpu"),
        )
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for step in range(3):
            global_tokens = torch.arange(12).reshape(4, 3) + 1 + step
            local_tokens = global_tokens.chunk(2)[rank]
            optimizer.zero_grad(set_to_none=True)
            loss = wrapped(local_tokens)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()

            reference_optimizer.zero_grad(set_to_none=True)
            reference(global_tokens).backward()
            reference_optimizer.step()
            with FSDP.summon_full_params(wrapped, writeback=False):
                gemma3 = model.value_expert.gemma3
                assert gemma3.lm_head.weight is gemma3.model.embed_tokens.weight
                torch.testing.assert_close(
                    gemma3.model.embed_tokens.weight,
                    reference.value_expert.gemma3.model.embed_tokens.weight,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires CPU Gloo")
def test_tied_gemma3_embedding_updates_match_unsharded_model(tmp_path):
    """Sharded forward/backward must preserve the shared embedding's updates."""
    mp.spawn(
        _run_tied_embedding_update,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
