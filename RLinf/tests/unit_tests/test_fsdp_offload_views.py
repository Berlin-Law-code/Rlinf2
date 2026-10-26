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

"""FSDP offload must release saved views without changing the next update."""

import copy
import weakref
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.checkpoint import checkpoint

from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy


class _Policy(nn.Module):
    def __init__(self, activation_checkpointing):
        super().__init__()
        self.frozen = nn.Linear(4, 4).requires_grad_(False)
        self.expert = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, inputs):
        with torch.no_grad():
            prefix = self.frozen(inputs)
        hidden = (
            checkpoint(self.expert, prefix, use_reentrant=False)
            if self.activation_checkpointing
            else self.expert(prefix)
        )
        return self.head(hidden.tanh())


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo is required")
    if dist.is_initialized():
        pytest.skip("This test owns its CPU process group")
    rendezvous = tmp_path_factory.mktemp("offload-views") / "rendezvous"
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


@pytest.mark.parametrize("activation_checkpointing", [False, True])
@pytest.mark.parametrize("offload_grad", [False, True])
def test_offload_releases_forward_views_and_preserves_repeated_updates(
    process_group, activation_checkpointing, offload_grad
):
    torch.manual_seed(37)
    raw = _Policy(activation_checkpointing)
    reference = copy.deepcopy(raw)
    model = FSDP(
        raw,
        device_id=torch.device("cpu"),
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003)
    strategy = FSDPStrategy(SimpleNamespace(), 1)
    original_ids = [id(p) for p in model.parameters()]

    for cycle in range(8):
        inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10 + cycle / 100
        for current, optim in ((model, optimizer), (reference, reference_optimizer)):
            optim.zero_grad(set_to_none=True)
            current(inputs).square().mean().backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        # PPO's final probability probe is a no-grad FSDP forward.
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
        handles = strategy._iter_fsdp_handles(model)
        refs = [
            weakref.ref(t)
            for h in handles
            for t in h.flat_param._tensors
            if t is not None
        ]
        assert refs, "Exercise the cached forward views that retained CUDA storage"
        strategy.offload_optimizer(optimizer)
        strategy.offload_param_and_grad(model, offload_grad)
        assert all(ref() is None for ref in refs)
        assert [id(p) for p in model.parameters()] == original_ids
        strategy.onload_param_and_grad(model, torch.device("cpu"), offload_grad)
        strategy.onload_optimizer(optimizer, torch.device("cpu"))
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
