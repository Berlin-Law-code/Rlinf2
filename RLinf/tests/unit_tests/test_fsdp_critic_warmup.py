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

"""Real CPU FSDP updates across critic warmup and actor-only optimizer skips."""

import copy
import logging
import warnings
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

import rlinf.hybrid_engines.fsdp.fsdp_model_manager as manager_module
import rlinf.hybrid_engines.fsdp.strategy.fsdp as strategy_module
from rlinf.algorithms.ppo_update_guard import (
    ActorOptimizerSnapshot,
    chunk_probability_stats,
)
from rlinf.algorithms.registry import policy_loss
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.utils.utils import collect_param_names_need_sync


class _TinyPolicy(nn.Module):
    def __init__(self, *, add_value_head=True):
        super().__init__()
        self.paligemma = nn.Linear(4, 4)
        self.paligemma.requires_grad_(False)
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = nn.Linear(4, 4)
        # Like OpenPI, the outer expert is bypassed and holds an unused head.
        self.gemma_expert.lm_head = nn.Linear(4, 5, bias=False)
        self.action_out_proj = nn.Linear(4, 2)
        if add_value_head:
            self.value_head = nn.Linear(4, 1)
        self._no_split_names = [
            "paligemma",
            "gemma_expert",
            "model",
            "action_out_proj",
            "value_head",
        ]
        for name, module in self.named_modules():
            module._fsdp_wrap_name = name.split(".")[-1]

    def forward(self, inputs):
        with torch.no_grad():
            prefix = self.paligemma(inputs)
        actions = self.action_out_proj(self.gemma_expert.model(prefix + inputs))
        value = (
            self.value_head(prefix.detach()) if hasattr(self, "value_head") else None
        )
        return actions, value


@pytest.fixture(scope="module")
def cpu_process_group(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo is required")
    if dist.is_initialized():
        pytest.skip("This test owns an isolated CPU process group")
    rendezvous = tmp_path_factory.mktemp("critic-warmup-gloo") / "rendezvous"
    threads = torch.get_num_threads()
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
        torch.set_num_threads(threads)


@pytest.fixture
def make_manager(cpu_process_group, monkeypatch):
    cpu_worker = SimpleNamespace(torch_device_type="cpu")
    monkeypatch.setattr(manager_module, "Worker", cpu_worker)
    monkeypatch.setattr(strategy_module, "Worker", cpu_worker)
    monkeypatch.setenv("LOCAL_RANK", "0")

    def make(*, warmup=2, scaler_enabled=False, add_value_head=True):
        torch.manual_seed(43)
        module = _TinyPolicy(add_value_head=add_value_head)
        names = collect_param_names_need_sync(module)
        model = FSDP(
            module,
            auto_wrap_policy=get_fsdp_wrap_policy(
                module, config={"use_orig_params": True}, model_type="openpi"
            ),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=torch.device("cpu"),
            sync_module_states=False,
        )
        manager = object.__new__(FSDPModelManager)
        manager._cfg = OmegaConf.create(
            {
                "optim": {
                    "lr": 0.01,
                    "value_lr": 0.02,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95,
                    "adam_eps": 1e-8,
                    "weight_decay": 0.1,
                    "clip_grad": 0.5,
                    "total_training_steps": 10,
                    "lr_scheduler": "cosine",
                    "min_lr_rate": 0.1,
                },
                "fsdp_config": {"sharding_strategy": "no_shard"},
            }
        )
        manager._logger = logging.getLogger("test-critic-warmup")
        manager.device = torch.device("cpu")
        manager.model = model
        manager.param_names_need_sync = names
        manager.optimizer_steps = 0
        manager.critic_warmup_steps = warmup
        manager.last_optimizer_step_applied = False
        manager.last_actor_optimizer_step_applied = False
        manager.is_weight_offloaded = manager.is_optimizer_offloaded = False
        manager._strategy = object.__new__(FSDPStrategy)
        manager._strategy.cfg = manager._cfg
        manager.optimizer = manager.build_optimizer(
            model, enable_critic_warmup=warmup > 0
        )
        manager.lr_scheduler = manager.build_lr_scheduler(
            manager.optimizer, manager._cfg.optim
        )
        manager.grad_scaler = ShardedGradScaler(
            device="cpu", enabled=scaler_enabled, init_scale=8.0
        )
        return manager

    return make


def _backward(manager, *, critic_only=False):
    manager.optimizer.zero_grad(set_to_none=True)
    inputs = torch.linspace(-1, 1, 16).reshape(4, 4)
    for micro in range(2):
        actions, values = manager.model(inputs + micro * 0.1)
        if values is None:
            loss = actions.square().mean()
        else:
            loss, _ = policy_loss(
                task_type="embodied",
                loss_type="actor_critic",
                logprob_type="chunk_level",
                reward_type="chunk_level",
                single_action_dim=2,
                logprobs=actions.unsqueeze(1),
                old_logprobs=actions.detach().unsqueeze(1),
                advantages=torch.ones(4, 1),
                values=values,
                prev_values=values.detach(),
                returns=torch.full((4, 1), 0.7),
                loss_mask=torch.ones(4, 1, dtype=torch.bool),
                clip_ratio_low=0.2,
                clip_ratio_high=0.2,
                value_clip=0.2,
                huber_delta=10.0,
                critic_warmup=critic_only,
            )
        manager.grad_scaler.scale(loss / 2).backward()


def _state(params, optimizer):
    return [
        (param.detach().clone(), copy.deepcopy(optimizer.state[param]))
        for param in params
    ]


def _assert_unchanged(params, optimizer, before):
    for param, (value, state) in zip(params, before):
        torch.testing.assert_close(param, value, rtol=0, atol=0)
        actual = optimizer.state[param]
        assert set(actual) == set(state)
        for key, expected in state.items():
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
            else:
                assert actual[key] == expected


def _export(manager, *, full):
    return get_model_state_dict(
        manager.model, options=StateDictOptions(full_state_dict=full, cpu_offload=full)
    )


def test_warmup_keeps_state_schedule_grad_flags_and_fsdp_exports(
    make_manager, tmp_path
):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        manager = make_manager(warmup=2)
        initial = {
            key: value.clone() for key, value in _export(manager, full=False).items()
        }
        initial_flags = {
            name: param.requires_grad
            for name, param in manager.model.named_parameters()
        }
        sync_names = list(manager.param_names_need_sync)
        optimizer, scheduler = manager.optimizer, manager.lr_scheduler
        actors, critics = (
            manager._actor_optimizer_params,
            manager._critic_optimizer_params,
        )
        actor_initial = _state(actors, optimizer)
        critic_state = optimizer.state[critics[0]]
        critic_momentum = critic_state["exp_avg"]
        assert [group["rlinf_role"] for group in optimizer.param_groups] == [
            "actor",
            "critic",
        ]
        assert all(param.requires_grad for param in actors)

        for update in range(3):
            scheduled_lrs = [group["lr"] for group in optimizer.param_groups]
            _backward(manager, critic_only=update < 2)
            norm, applied_lrs = manager.optimizer_step()
            assert torch.isfinite(torch.as_tensor(norm))
            assert manager.last_optimizer_step_applied
            assert manager.last_actor_optimizer_step_applied == (update == 2)
            assert applied_lrs == (
                [0.0, scheduled_lrs[1]] if update < 2 else scheduled_lrs
            )
            assert manager.optimizer_steps == update + 1
            assert manager.critic_warmup_active == (update == 0)
            assert manager.optimizer is optimizer and manager.lr_scheduler is scheduler
            assert optimizer.state[critics[0]] is critic_state
            assert critic_state["exp_avg"] is critic_momentum
            assert int(critic_state["step"]) == update + 1
            assert critic_momentum.abs().sum() > 0
            if update < 2:
                _assert_unchanged(actors, optimizer, actor_initial)
            scheduler.step()
            assert scheduler.last_epoch == update + 1
            assert manager.param_names_need_sync == sync_names
            assert {
                name: param.requires_grad
                for name, param in manager.model.named_parameters()
            } == initial_flags
            for full in (False, True):
                exported = _export(manager, full=full)
                assert set(exported) == set(initial)
                for key in initial:
                    if (
                        key.startswith("paligemma.")
                        or key == "gemma_expert.lm_head.weight"
                    ):
                        torch.testing.assert_close(
                            exported[key], initial[key], rtol=0, atol=0
                        )
                assert any(
                    not torch.equal(exported[key], initial[key])
                    for key in initial
                    if key.startswith("value_head.")
                )
            optim_export = get_optimizer_state_dict(
                manager.model,
                optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            assert [group["rlinf_role"] for group in optim_export["param_groups"]] == [
                "actor",
                "critic",
            ]
            assert all(
                group["rlinf_optimizer_steps"] == update + 1
                for group in optim_export["param_groups"]
            )
            torch.save(
                {
                    "model": exported,
                    "optimizer": optim_export,
                    "scheduler": scheduler.state_dict(),
                },
                tmp_path / f"update_{update}.pt",
            )
        assert any(
            not torch.equal(param, saved[0])
            for param, saved in zip(actors, actor_initial)
        )
        assert not any("not a leaf Tensor" in str(item.message) for item in captured)


def test_skip_actor_excludes_its_nan_before_scaling_clip_and_preserves_momentum(
    make_manager,
):
    manager = make_manager(warmup=0, scaler_enabled=True)
    _backward(manager)
    manager.optimizer_step()
    actors, critics = manager._actor_optimizer_params, manager._critic_optimizer_params
    before_actor = _state(actors, manager.optimizer)
    before_critic = _state(critics, manager.optimizer)
    _backward(manager)
    expected_norm = (
        torch.linalg.vector_norm(torch.cat([param.grad.flatten() for param in critics]))
        / manager.grad_scaler.get_scale()
    )
    for param in actors:
        if param.grad is not None:
            param.grad.fill_(float("nan"))
    norm, lrs = manager.optimizer_step(skip_actor_update=True)
    assert norm == pytest.approx(float(expected_norm), rel=1e-5)
    assert lrs == [0.0, manager.optimizer.param_groups[1]["lr"]]
    assert (
        manager.last_optimizer_step_applied
        and not manager.last_actor_optimizer_step_applied
    )
    assert manager.optimizer_steps == 2
    _assert_unchanged(actors, manager.optimizer, before_actor)
    assert all(param.grad is None for param in actors)
    assert any(
        not torch.equal(param, old[0]) for param, old in zip(critics, before_critic)
    )


def test_nonfinite_and_empty_updates_do_not_consume_warmup(make_manager):
    manager = make_manager(warmup=2, scaler_enabled=True)
    initial = _state(manager._critic_optimizer_params, manager.optimizer)
    _backward(manager, critic_only=True)
    for param in manager._critic_optimizer_params:
        param.grad.fill_(float("inf"))
    old_scale = manager.grad_scaler.get_scale()
    norm, lrs = manager.optimizer_step()
    assert not torch.isfinite(torch.as_tensor(norm))
    assert lrs == [0.0, 0.0]
    assert manager.optimizer_steps == 0 and manager.critic_warmup_active
    assert not manager.last_optimizer_step_applied
    assert manager.grad_scaler.get_scale() < old_scale
    _assert_unchanged(manager._critic_optimizer_params, manager.optimizer, initial)
    manager.optimizer.zero_grad(set_to_none=True)
    assert manager.optimizer_step() == (0.0, [0.0, 0.0])
    assert manager.optimizer_steps == 0 and not manager.last_optimizer_step_applied
    for _ in range(2):
        _backward(manager, critic_only=True)
        manager.optimizer_step()
    assert manager.optimizer_steps == 2 and not manager.critic_warmup_active


def test_no_warmup_actor_only_behavior_and_explicit_skip(make_manager):
    manager = make_manager(warmup=0, add_value_head=False)
    _backward(manager)
    _, lrs = manager.optimizer_step()
    assert lrs == [manager._cfg.optim.lr]
    assert (
        manager.last_optimizer_step_applied
        and manager.last_actor_optimizer_step_applied
    )
    assert manager.optimizer_steps == 1
    before = _state(manager._actor_optimizer_params, manager.optimizer)
    _backward(manager)
    assert manager.optimizer_step(skip_actor_update=True) == (0.0, [0.0])
    assert manager.optimizer_steps == 1 and not manager.last_optimizer_step_applied
    _assert_unchanged(manager._actor_optimizer_params, manager.optimizer, before)


def test_load_restores_successful_update_counter_and_optimizer_metadata(
    make_manager, monkeypatch
):
    manager = make_manager(warmup=2)
    for _ in range(2):
        _backward(manager, critic_only=True)
        manager.optimizer_step()
        manager.lr_scheduler.step()
    saved_optimizer = copy.deepcopy(manager.optimizer.state_dict())
    saved_scheduler = copy.deepcopy(manager.lr_scheduler.state_dict())
    resumed = make_manager(warmup=2)

    def load(model, optimizer, scheduler, path):
        optimizer.load_state_dict(saved_optimizer)
        scheduler.load_state_dict(saved_scheduler)

    monkeypatch.setattr(resumed._strategy, "load_checkpoint", load)
    resumed.load_checkpoint("unused-test-path")
    assert resumed.optimizer_steps == 2 and not resumed.critic_warmup_active
    assert resumed.lr_scheduler.last_epoch == 2
    assert [group["rlinf_role"] for group in resumed.optimizer.param_groups] == [
        "actor",
        "critic",
    ]
    assert not resumed.last_optimizer_step_applied


def test_warmup_requires_separable_critic_parameters(make_manager):
    with pytest.raises(ValueError, match="identifiable trainable value_head"):
        make_manager(warmup=2, add_value_head=False)


def test_legacy_nonwarmup_checkpoint_restores_group_roles(make_manager, monkeypatch):
    manager = make_manager(warmup=0)
    _backward(manager)
    manager.optimizer_step()
    saved = copy.deepcopy(manager.optimizer.state_dict())
    for group in saved["param_groups"]:
        group.pop("rlinf_role")
        group.pop("rlinf_optimizer_steps")
    resumed = make_manager(warmup=0)

    def load(model, optimizer, scheduler, path):
        optimizer.load_state_dict(saved)

    monkeypatch.setattr(resumed._strategy, "load_checkpoint", load)
    resumed.load_checkpoint("legacy-test-path")
    assert [group["rlinf_role"] for group in resumed.optimizer.param_groups] == [
        "actor",
        "critic",
    ]
    assert not resumed.critic_warmup_active
    _backward(resumed)
    resumed.optimizer_step()
    assert resumed.last_actor_optimizer_step_applied


def test_guard_snapshot_probe_restore_then_critic_update_keeps_fsdp_views(make_manager):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        manager = make_manager(warmup=2)
        initial_keys = set(_export(manager, full=False))
        sync_names = list(manager.param_names_need_sync)
        for _ in range(2):
            _backward(manager, critic_only=True)
            manager.optimizer_step()
        actors, critics = (
            manager._actor_optimizer_params,
            manager._critic_optimizer_params,
        )
        optimizer_ids = [
            id(param)
            for group in manager.optimizer.param_groups
            for param in group["params"]
        ]
        _backward(manager)
        saved_actor = _state(actors, manager.optimizer)
        saved_critic = _state(critics, manager.optimizer)
        snapshot = ActorOptimizerSnapshot(manager.optimizer)
        assert snapshot.entries
        manager.optimizer_step()
        assert manager.last_actor_optimizer_step_applied
        manager.optimizer.zero_grad(set_to_none=True)
        # Mirror post-update probability probes: multiple no-grad forwards after
        # optimizer.step, before restoring the original optimizer parameter refs.
        with torch.no_grad():
            for offset in range(4):
                inputs = torch.full((1, 4), offset * 0.1)
                actions, _ = manager.model(inputs)
                stats = chunk_probability_stats(
                    actions,
                    actions.clone(),
                    torch.ones(1, 1, dtype=torch.bool),
                    0.2,
                    0.2,
                )
                assert stats["count"] == 1 and stats["kl_sum"] == 0
        snapshot.restore()
        _assert_unchanged(actors, manager.optimizer, saved_actor)
        assert any(
            not torch.equal(param, saved[0])
            for param, saved in zip(critics, saved_critic)
        )
        assert manager.optimizer_steps == 3
        assert manager.optimizer.param_groups[0]["rlinf_optimizer_steps"] == 2
        assert manager.optimizer.param_groups[1]["rlinf_optimizer_steps"] == 3

        _backward(manager, critic_only=True)
        manager.optimizer_step(skip_actor_update=True)
        assert manager.optimizer_steps == 4
        _assert_unchanged(actors, manager.optimizer, saved_actor)
        assert optimizer_ids == [
            id(param)
            for group in manager.optimizer.param_groups
            for param in group["params"]
        ]
        assert all(
            param.is_leaf
            for group in manager.optimizer.param_groups
            for param in group["params"]
        )
        assert manager.param_names_need_sync == sync_names
        for full in (False, True):
            assert set(_export(manager, full=full)) == initial_keys
        state = get_optimizer_state_dict(
            manager.model,
            manager.optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        assert all(
            group["rlinf_optimizer_steps"] == 4 for group in state["param_groups"]
        )
        assert not any("not a leaf Tensor" in str(item.message) for item in captured)
