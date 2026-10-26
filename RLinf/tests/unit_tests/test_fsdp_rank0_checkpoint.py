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

"""Replicated checkpoints must skip nonwriter exports and preserve updates."""

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(6, 6).requires_grad_(False)
        self.expert = nn.Linear(6, 6)
        self.head = nn.Linear(6, 2)
        self.register_buffer("offset", torch.tensor(0.01))

    def forward(self, inputs):
        return self.head(self.expert(self.frozen(inputs)).tanh()) + self.offset


def _checkpoint_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(43)
        model = FSDP(
            _Policy(),
            device_id=torch.device("cpu"),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
        cfg = OmegaConf.create(
            {
                "fsdp_config": {
                    "strategy": "fsdp",
                    "sharding_strategy": "no_shard",
                    "use_orig_params": True,
                    "checkpoint_cpu_staging": True,
                    "save_full_model_weights": True,
                }
            }
        )
        strategy = FSDPStrategy(cfg, 3)
        manager = object.__new__(FSDPModelManager)
        manager._cfg = cfg
        manager._strategy = strategy
        manager.model = model
        manager.optimizer = optimizer
        manager.lr_scheduler = scheduler
        manager.device = torch.device("cpu")
        manager.is_weight_offloaded = False
        manager.is_optimizer_offloaded = False
        inputs = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10 + rank / 10

        def update():
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                model(inputs)
            return loss.item()

        update()
        for initially_offloaded in (False, True):
            if initially_offloaded:
                manager.offload_param_and_grad()
                manager.offload_optimizer()
            export_count = 0

            def check_export(*_):
                nonlocal export_count
                assert rank == 0, "A nonwriter tried to clone the complete model"
                export_count += 1

            hook = model.register_state_dict_pre_hook(check_export)
            path = str(Path(directory) / str(initially_offloaded))
            manager.save_checkpoint(path)
            hook.remove()
            assert export_count == (1 if rank == 0 else 0)
            assert manager.is_weight_offloaded == initially_offloaded
            assert manager.is_optimizer_offloaded == initially_offloaded
            if initially_offloaded:
                manager.load_param_and_grad(torch.device("cpu"))
                manager.load_optimizer(torch.device("cpu"))

            expected_loss = update()
            expected = [p.detach().clone() for p in model.parameters()]
            expected_optim = copy.deepcopy(optimizer.state_dict())
            expected_schedule = copy.deepcopy(scheduler.state_dict())
            strategy.load_checkpoint(model, optimizer, scheduler, path)
            assert update() == expected_loss
            for actual, reference in zip(model.parameters(), expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            actual_optim = optimizer.state_dict()
            assert actual_optim["param_groups"] == expected_optim["param_groups"]
            for key, reference in expected_optim["state"].items():
                for name, value in reference.items():
                    torch.testing.assert_close(
                        actual_optim["state"][key][name], value, rtol=0, atol=0
                    )
            assert scheduler.state_dict() == expected_schedule
    finally:
        dist.destroy_process_group()


def test_three_rank_checkpoint_skips_replica_exports_and_restores_next_update(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo is required")
    mp.spawn(_checkpoint_worker, args=(str(tmp_path),), nprocs=3, join=True)
