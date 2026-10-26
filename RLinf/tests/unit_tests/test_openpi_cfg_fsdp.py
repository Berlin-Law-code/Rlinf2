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

"""Exercise OpenPI CFG's shared embedding through actual two-rank FSDP."""

import copy
from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    OpenPi0ForCFGActionPrediction,
)


class _TinyCFGModel(nn.Module):
    # Use the production name policy and embedding method, with small tensors
    # and no unrelated transformer blocks so the regression runs on CPU.
    _no_split_modules = []
    _no_split_names = OpenPi0ForCFGActionPrediction._no_split_names

    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = PaliGemmaWithExpertModel.__new__(
            PaliGemmaWithExpertModel
        )
        nn.Module.__init__(self.paligemma_with_expert)
        self.paligemma_with_expert.paligemma = nn.Module()
        paligemma = self.paligemma_with_expert.paligemma
        paligemma.language_model = nn.Module()
        paligemma.language_model.embed_tokens = nn.Embedding(32, 8)
        paligemma.lm_head = nn.Linear(8, 32, bias=False)
        paligemma.lm_head.weight = paligemma.language_model.embed_tokens.weight
        self.action_in_proj = nn.Linear(8, 4)
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.split(".")[-1]

    def forward(self, tokens):
        embeddings = self.paligemma_with_expert.embed_language_tokens(tokens)
        return self.action_in_proj(embeddings).square().mean()


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
        model = _TinyCFGModel()
        reference = copy.deepcopy(model)
        wrapped = FSDP(
            model,
            auto_wrap_policy=get_fsdp_wrap_policy(model, model_type="cfg_model"),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            device_id=torch.device("cpu"),
        )
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for step in range(3):
            global_tokens = torch.arange(24).reshape(8, 3) + 1 + step
            local_tokens = global_tokens.chunk(2)[rank]
            optimizer.zero_grad(set_to_none=True)
            for micro_index, micro_tokens in enumerate(local_tokens.chunk(2)):
                with wrapped.no_sync() if micro_index == 0 else nullcontext():
                    loss = wrapped(micro_tokens) / 2
                    assert torch.isfinite(loss)
                    loss.backward()
            optimizer.step()

            reference_optimizer.zero_grad(set_to_none=True)
            reference(global_tokens).backward()
            reference_optimizer.step()
            with FSDP.summon_full_params(wrapped, writeback=False):
                paligemma = model.paligemma_with_expert.paligemma
                assert (
                    paligemma.lm_head.weight
                    is paligemma.language_model.embed_tokens.weight
                )
                for name, parameter in model.named_parameters():
                    reference_name = name.replace("_fsdp_wrapped_module.", "")
                    torch.testing.assert_close(
                        parameter, reference.get_parameter(reference_name)
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires CPU Gloo")
def test_cfg_tied_embedding_updates_match_unsharded_model(tmp_path):
    """Sharded accumulation preserves shared embeddings and optimizer updates."""
    mp.spawn(
        _run_tied_embedding_update,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
