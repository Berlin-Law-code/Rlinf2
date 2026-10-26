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

"""Probability, rollback, and actual worker regressions for guarded OpenPI PPO."""

import asyncio
import copy
import inspect
import json
import math
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
from rlinf.algorithms.ppo_update_guard import (
    ActorOptimizerSnapshot,
    add_probability_stats,
    chunk_probability_stats,
    empty_probability_stats,
    make_probability_probe,
    probability_limit_exceeded,
    probability_stats_metrics,
    validate_ppo_update_guard_config,
)
from rlinf.algorithms.utils import safe_normalize
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logprobs = torch.nn.Parameter(torch.zeros(20, 8))
        self.value_head = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.value_head.weight)

    def forward(self, forward_inputs, **kwargs):
        batch = len(forward_inputs["obs"])
        return {
            "logprobs": self.logprobs.expand(batch, -1, -1),
            "values": self.value_head(torch.ones(batch, 1)).reshape(batch),
        }


class TestWorker(SimpleNamespace):
    __test__ = False

    @property
    def critic_warmup_active(self):
        return self.optimizer_steps < self.critic_warmup_steps


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_rollout_state_dict_staging_uses_configured_device(cpu_offload):
    calls = []
    state = {"weight": torch.tensor([1.0, 2.0])}

    def get_state(**kwargs):
        calls.append(kwargs)
        return state

    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "actor": {"rollout_state_dict_cpu_offload": cpu_offload},
                "weight_syncer": {"type": "bucket" if cpu_offload else "patch"},
            }
        ),
        get_model_state_dict=get_state,
    )
    result = inspect.unwrap(EmbodiedFSDPActor.get_rollout_state_dict)(worker)
    assert result is state
    assert calls == [{"cpu_offload": cpu_offload, "full_state_dict": False}]


def test_cpu_rollout_state_dict_rejects_gpu_patch_builder():
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "actor": {"rollout_state_dict_cpu_offload": True},
                "weight_syncer": {"type": "patch"},
            }
        )
    )
    with pytest.raises(ValueError, match="requires bucket sync"):
        inspect.unwrap(EmbodiedFSDPActor.get_rollout_state_dict)(worker)


@pytest.mark.parametrize("is_sender", [False, True])
def test_cpu_rollout_sync_releases_actor_before_export_and_repeat_transfer(is_sender):
    from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer

    source = torch.nn.Linear(2, 1, bias=False)
    target = torch.nn.Linear(2, 1, bias=False)
    events = []
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "actor": {
                    "rollout_state_dict_cpu_offload": True,
                    "fsdp_config": {
                        "strategy": "fsdp",
                        "sharding_strategy": "no_shard",
                        "use_orig_params": True,
                    },
                },
                "weight_syncer": {"type": "bucket"},
            }
        ),
        enable_offload=True,
        is_weight_offloaded=True,
        is_optimizer_offloaded=True,
        _is_weight_sender=is_sender,
        _group_name="actor",
        _rollout_group_name="rollout",
        _rollout_all_ranks=[0],
        _sync_weight_comm_options=None,
        param_names_need_sync=["weight"],
        version=0,
        weight_syncer=BucketWeightSyncer(16, None, "cpu"),
    )

    def offload_optimizer():
        events.append("offload optimizer")
        worker.is_optimizer_offloaded = True

    def offload_weights(*args):
        events.append("offload weights")
        worker.is_weight_offloaded = True

    def export(**kwargs):
        assert is_sender, "Non-senders must not clone the complete CPU state dict"
        assert worker.is_weight_offloaded and worker.is_optimizer_offloaded
        assert kwargs == {"cpu_offload": True, "full_state_dict": False}
        events.append("export")
        return {name: tensor.clone() for name, tensor in source.state_dict().items()}

    worker.offload_optimizer = offload_optimizer
    worker.offload_param_and_grad = offload_weights
    worker.load_param_and_grad = lambda *args: pytest.fail("onloaded CPU staging actor")
    worker.get_model_state_dict = export
    worker.get_rollout_state_dict = MethodType(
        EmbodiedFSDPActor.get_rollout_state_dict, worker
    )

    async def run():
        queue = asyncio.Queue()

        def broadcast(data, **kwargs):
            assert worker.is_weight_offloaded and worker.is_optimizer_offloaded
            events.append("send")

            async def wait():
                await queue.put({key: tensor.clone() for key, tensor in data.items()})

            return SimpleNamespace(async_wait=wait)

        worker.broadcast = broadcast
        receiver = BucketWeightSyncer(16, None, "cpu")
        for version in (0, 1):
            if version:
                # Model/optimizer reside on GPU after training, unlike startup.
                worker.is_weight_offloaded = worker.is_optimizer_offloaded = False
                with torch.no_grad():
                    source.weight.add_(1)
            worker.version = version
            if is_sender:
                _, applied = await asyncio.gather(
                    inspect.unwrap(EmbodiedFSDPActor.sync_model_to_rollout)(worker),
                    receiver.apply(target, queue.get),
                )
                assert applied == version
                torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)
            else:
                await inspect.unwrap(EmbodiedFSDPActor.sync_model_to_rollout)(worker)
                assert not worker.weight_syncer.sender_initialized()
                assert queue.empty()
            assert worker.is_weight_offloaded and worker.is_optimizer_offloaded

    asyncio.run(run())
    assert events == (
        ["export", "send", "offload optimizer", "offload weights", "export", "send"]
        if is_sender
        else ["offload optimizer", "offload weights"]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("strategy", "fsdp2"),
        ("sharding_strategy", "full_shard"),
        ("use_orig_params", False),
    ],
)
def test_cpu_rollout_sync_rejects_unsupported_fsdp(key, value):
    fsdp = {
        "strategy": "fsdp",
        "sharding_strategy": "no_shard",
        "use_orig_params": True,
    }
    fsdp[key] = value
    worker = SimpleNamespace(
        enable_offload=True,
        cfg=OmegaConf.create(
            {
                "actor": {"rollout_state_dict_cpu_offload": True, "fsdp_config": fsdp},
                "weight_syncer": {"type": "bucket"},
            }
        ),
    )
    with pytest.raises(ValueError, match="CPU rollout staging requires"):
        asyncio.run(inspect.unwrap(EmbodiedFSDPActor.sync_model_to_rollout)(worker))


@pytest.mark.parametrize("cpu_staging", [False, True])
def test_actor_initializes_fsdp_before_first_cpu_state_export(monkeypatch, cpu_staging):
    events = []
    model = object()

    @contextmanager
    def summon(actual_model, **kwargs):
        assert actual_model is model
        assert kwargs == {"writeback": False}
        events.append("initialize FSDP")
        yield

    monkeypatch.setattr(actor_module.FSDP, "summon_full_params", summon)
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {"actor": {"rollout_state_dict_cpu_offload": cpu_staging}}
        ),
        model=model,
        enable_offload=True,
        setup_model_and_optimizer=lambda: events.append("setup"),
        offload_param_and_grad=lambda: events.append("offload weights"),
        offload_optimizer=lambda: events.append("offload optimizer"),
    )
    inspect.unwrap(EmbodiedFSDPActor.init_worker)(worker)
    expected = ["setup"]
    if cpu_staging:
        expected.append("initialize FSDP")
    assert events == expected + ["offload weights", "offload optimizer"]


def make_worker(monkeypatch, *, zero_reward=False, warmup=0, lr=0.001):
    platform = SimpleNamespace(
        current_device=lambda: torch.device("cpu"), empty_cache=lambda: None
    )
    monkeypatch.setattr(actor_module.Worker, "torch_platform", platform)
    monkeypatch.setattr(actor_module, "all_reduce_dict", lambda x, **kw: x)
    monkeypatch.setattr(actor_module, "clear_memory", lambda **kw: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "actor": {
                "seed": 7,
                "global_batch_size": 4,
                "micro_batch_size": 1,
                "fsdp_config": {
                    "sharding_strategy": "no_shard",
                    "use_orig_params": True,
                },
                "model": {
                    "model_type": "openpi",
                    "action_dim": 8,
                    "num_action_chunks": 20,
                    "openpi": {
                        "train_expert_only": True,
                        "value_after_vlm": True,
                        "detach_critic_input": True,
                        "strict_ppo_precision": True,
                    },
                },
            },
            "algorithm": {
                "adv_type": "gae",
                "loss_type": "actor_critic",
                "logprob_type": "chunk_level",
                "reward_type": "chunk_level",
                "entropy_bonus": 0,
                "clip_ratio_high": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_c": 3.0,
                "value_clip": 0.2,
                "huber_delta": 10.0,
                "normalize_advantages": True,
                "update_epoch": 3,
                "ppo_update_guard": {
                    "enabled": True,
                    "target_kl": 0.02,
                    "max_initial_logratio": 0.01,
                    "probe_batch_size": 4,
                    "skip_actor_on_zero_reward": True,
                },
            },
            "env": {
                "train": {
                    "max_episode_steps": 40,
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
        }
    )
    model = TinyPolicy()
    actor = [model.logprobs]
    critic = list(model.value_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": actor, "lr": lr, "rlinf_role": "actor"},
            {"params": critic, "lr": 0.01, "rlinf_role": "critic"},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    worker = TestWorker(
        cfg=cfg,
        device=torch.device("cpu"),
        torch_platform=platform,
        model=model,
        amp_context=nullcontext(),
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        optimizer_steps=0,
        critic_warmup_steps=warmup,
        _actor_optimizer_params=tuple(actor),
        _critic_optimizer_params=tuple(critic),
        _logger=SimpleNamespace(info=lambda *x: None, warning=lambda *x: None),
        log_info=lambda *args: None,
        _strategy=SimpleNamespace(
            clip_grad_norm_=lambda model: torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1
            )
        ),
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        before_micro_batch=lambda *a, **kw: nullcontext(),
        enable_sft_co_train=False,
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        _rollout_batch_in_use=False,
        _world_size=1,
        _rank=0,
        version=0,
        gradient_accumulation=4,
    )
    for name in (
        "run_training",
        "_run_training_on_rollout_batch",
        "_run_guarded_updates",
        "_measure_policy_probe",
        "_reduce_probability_stats",
        "train_micro_batch",
    ):
        method = getattr(EmbodiedFSDPActor, name)
        if name != "_measure_policy_probe":
            method = inspect.unwrap(method)
        setattr(worker, name, MethodType(method, worker))
    worker.optimizer_step = MethodType(FSDPModelManager.optimizer_step, worker)
    worker.rollout_batch = {
        "rewards": torch.zeros(2, 2, 20),
        "advantages": torch.ones(2, 2, 1),
        "prev_logprobs": torch.zeros(2, 2, 20, 8),
        "returns": torch.ones(2, 2, 1),
        "prev_values": torch.zeros(3, 2, 1),
        "loss_mask": torch.ones(2, 2, 1, dtype=torch.bool),
        "loss_mask_sum": torch.full((2, 2, 1), 40),
        "action_mask": torch.ones(2, 2, 20, dtype=torch.bool),
        "forward_inputs": {"obs": torch.zeros(2, 2, 1)},
    }
    if not zero_reward:
        worker.rollout_batch["rewards"][-1, 0, -1] = 1
    return worker


def test_probability_statistics_ignore_padding_and_use_positive_kl():
    new = torch.tensor([math.log(2), 999.0, 0.0]).reshape(3, 1, 1)
    old = torch.zeros_like(new)
    stats = chunk_probability_stats(new, old, torch.tensor([[1], [0], [1]]), 0.2, 0.2)
    result = probability_stats_metrics(stats)
    assert result["valid_count"] == 2
    assert result["ratio"] == pytest.approx(1.5)
    assert result["kl"] == pytest.approx((1 - math.log(2)) / 2)
    assert result["outside_clip_fraction"] == 0.5
    assert stats["invalid_count"] == 0
    assert probability_limit_exceeded(stats, 0.02)


@pytest.mark.parametrize("valid_count", [0, 1, 4])
def test_fixed_size_probe_masks_padding_without_mutating_rollout(valid_count):
    mask = torch.arange(4).reshape(4, 1) < valid_count
    batch = {"loss_mask": mask, "prev_logprobs": torch.zeros(4, 20, 8)}
    probe = make_probability_probe(batch, 4)
    assert probe["prev_logprobs"].shape[0] == 4
    assert probe["loss_mask"].sum().item() == valid_count
    assert torch.equal(batch["loss_mask"], mask)


def _distributed_guard_rank(rank, rendezvous, output, reject, warmup, enforce_kl_limit):
    """Exercise actual worker updates through three CPU DDP/Gloo ranks."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=40),
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            original_world_size = torch.distributed.get_world_size
            worker = make_worker(patch, lr=0.001 if reject else 0.0001, warmup=warmup)
            worker.cfg.algorithm.ppo_update_guard.enforce_kl_limit = enforce_kl_limit
            patch.setattr(torch.distributed, "get_world_size", original_world_size)

            def reduce_stats(
                values, dtype=torch.float32, op=torch.distributed.ReduceOp.SUM
            ):
                keys = sorted(values)
                tensor = torch.tensor([values[key] for key in keys], dtype=dtype)
                average = op == torch.distributed.ReduceOp.AVG
                torch.distributed.all_reduce(
                    tensor, op=torch.distributed.ReduceOp.SUM if average else op
                )
                if average:
                    tensor /= 3
                return dict(zip(keys, tensor.tolist()))

            patch.setattr(actor_module, "all_reduce_dict", reduce_stats)
            worker._world_size = 3
            worker._rank = rank
            worker.cfg.actor.global_batch_size = 12
            worker.cfg.algorithm.update_epoch = 4
            # Local valid counts are 4, 1 and 0. Rank 2 must still participate
            # in probe forwards, backward and any global rollback decision.
            valid_count = [4, 1, 0][rank]
            worker.rollout_batch["loss_mask"].reshape(-1)[valid_count:] = False
            worker.rollout_batch["action_mask"] &= worker.rollout_batch["loss_mask"]
            if rank:
                worker.rollout_batch["rewards"].zero_()
            raw_model = worker.model
            # A buffer makes probe forwards exercise DDP's broadcast ordering.
            raw_model.register_buffer("probe_buffer", torch.zeros(1))
            worker.model = torch.nn.parallel.DistributedDataParallel(
                raw_model, find_unused_parameters=True, broadcast_buffers=True
            )
            result = worker.run_training()
            torch.save(
                {
                    "result": result,
                    "model": raw_model.state_dict(),
                    "optimizer": worker.optimizer.state_dict(),
                    "scheduler": worker.lr_scheduler.state_dict(),
                },
                Path(output) / f"rank{rank}.pt",
            )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    ("reject", "warmup", "enforce_kl_limit"),
    [(False, 0, True), (True, 0, True), (True, 1, True), (True, 0, False)],
)
def test_three_actor_ranks_agree_with_uneven_and_empty_shards(
    tmp_path, monkeypatch, reject, warmup, enforce_kl_limit
):
    context = torch.multiprocessing.start_processes(
        _distributed_guard_rank,
        args=(str(tmp_path / "gloo"), str(tmp_path), reject, warmup, enforce_kl_limit),
        nprocs=3,
        join=False,
        start_method="spawn",
    )
    try:
        # A bounded wait ensures a collective ordering regression fails rather
        # than leaving the test suite hung indefinitely.
        import time

        deadline = time.monotonic() + 90
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Three-rank PPO collective test timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    states = [
        torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(3)
    ]
    reference = states[0]
    result = reference["result"]
    assert result["guard/abort"] == 0
    assert result["guard/valid_chunks"] == result["guard/initial_valid_count"] == 5
    assert result["guard/critic_updates"] == 4
    assert result["guard/actor_updates"] > 0
    assert result["guard/rejected_actor_updates"] == int(reject and enforce_kl_limit)
    if enforce_kl_limit:
        assert result["guard/final_kl"] <= 0.02
    else:
        assert result["guard/final_kl"] > 0.02
        assert result["guard/actor_updates"] == 4
        assert result["guard/pre_update_kl_stop"] == 0
    for state in states[1:]:
        # Constant synthetic returns have undefined explained variance.
        assert state["result"] == pytest.approx(result, nan_ok=True)
        assert state["scheduler"] == reference["scheduler"]
        for key, value in reference["model"].items():
            torch.testing.assert_close(state["model"][key], value, rtol=0, atol=0)
        for key, value in reference["optimizer"]["state"].items():
            for name, tensor in value.items():
                torch.testing.assert_close(
                    state["optimizer"]["state"][key][name], tensor, rtol=0, atol=0
                )
    actor_state = reference["optimizer"]["state"][0]
    assert actor_state["step"].item() == result["guard/actor_updates"]

    # Compare against a single actor receiving the same global batch. This
    # checks valid-chunk gradient normalization, not just rank agreement.
    baseline = make_worker(monkeypatch, lr=0.001 if reject else 0.0001, warmup=warmup)
    baseline.cfg.algorithm.ppo_update_guard.enforce_kl_limit = enforce_kl_limit
    baseline.cfg.actor.global_batch_size = 12
    baseline.cfg.algorithm.update_epoch = 4
    baseline.gradient_accumulation = 12
    shards = []
    for rank, valid_count in enumerate([4, 1, 0]):
        shard = copy.deepcopy(baseline.rollout_batch)
        shard["loss_mask"].reshape(-1)[valid_count:] = False
        shard["action_mask"] &= shard["loss_mask"]
        if rank:
            shard["rewards"].zero_()
        shards.append(shard)

    def concatenate(values):
        if isinstance(values[0], dict):
            return {
                key: concatenate([value[key] for value in values]) for key in values[0]
            }
        return torch.cat(values, dim=1)

    baseline.rollout_batch = concatenate(shards)
    baseline_result = baseline.run_training()
    assert baseline_result["guard/actor_updates"] == result["guard/actor_updates"]
    assert baseline_result["actor/grad_norm"] == pytest.approx(
        result["actor/grad_norm"], rel=1e-5
    )
    for key, value in baseline.model.state_dict().items():
        torch.testing.assert_close(reference["model"][key], value, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), 21.0, -21.0]
)
def test_invalid_probabilities_are_not_hidden_by_clamping(value):
    stats = chunk_probability_stats(
        torch.tensor([[[value]]]), torch.zeros(1, 1, 1), None, 0.2, 0.2
    )
    assert probability_limit_exceeded(stats, 0.02)
    assert stats["invalid_count"] == 1
    assert all(math.isfinite(v) for v in probability_stats_metrics(stats).values())


def test_probability_diagnostics_distinguish_overlapping_causes_and_ignore_padding():
    deltas = torch.tensor(
        [
            21.0,
            -23.0,
            float("nan"),
            float("inf"),
            -float("inf"),
            20.0,
            -20.0,
            0.0,
            999.0,
        ]
    ).reshape(-1, 1, 1)
    weights = torch.tensor(
        [1.0, 0.0, 1.0, -1.0, 1.0, float("nan"), 1.0, 1.0, float("nan")]
    )
    mask = torch.tensor([True] * 8 + [False]).reshape(-1, 1)
    stats = chunk_probability_stats(
        deltas, torch.zeros_like(deltas), mask, 0.2, 0.2, sample_weights=weights
    )
    assert stats["count"] == 8
    assert stats["invalid_count"] == 6  # Union, not the sum of overlapping causes.
    assert stats["nonfinite_logratio_count"] == 3
    assert stats["finite_out_of_range_logratio_count"] == 2
    assert stats["extreme_positive_logratio_count"] == 1
    assert stats["extreme_negative_logratio_count"] == 1
    assert stats["invalid_weight_count"] == 3
    assert stats["max_abs_logratio"] == 20
    assert stats["max_abs_logratio_unfiltered"] == 23
    assert all(math.isfinite(value) for value in stats.values())
    assert probability_limit_exceeded(stats, 0.02, enforce_kl_limit=False)


@pytest.mark.parametrize("delta", [-20.0, 20.0])
def test_finite_probability_boundary_preserves_existing_guard_decision(delta):
    tensor = torch.tensor([[[delta]]])
    stats = chunk_probability_stats(tensor, torch.zeros_like(tensor), None, 0.2, 0.2)
    assert stats["invalid_count"] == 0
    assert stats["finite_out_of_range_logratio_count"] == 0
    assert stats["max_abs_logratio"] == stats["max_abs_logratio_unfiltered"] == 20
    assert not probability_limit_exceeded(stats, 0.02, enforce_kl_limit=False)


def test_probability_diagnostics_merge_counts_and_maxima_across_microbatches():
    delta = torch.tensor([21.0, -24.0, float("nan"), -2.0, 1.0]).reshape(-1, 1, 1)
    old = torch.zeros_like(delta)
    expected = chunk_probability_stats(delta, old, None, 0.2, 0.2)
    total = empty_probability_stats()
    for chunk in delta.split(2):
        add_probability_stats(
            total,
            chunk_probability_stats(chunk, torch.zeros_like(chunk), None, 0.2, 0.2),
        )
    assert total == pytest.approx(expected)
    assert total["invalid_count"] == 3
    assert total["max_abs_logratio"] == 2
    assert total["max_abs_logratio_unfiltered"] == 24


def test_probability_rank_reduction_uses_sum_for_counts_and_max_for_extrema(
    monkeypatch,
):
    worker = make_worker(monkeypatch)
    deltas = [
        torch.tensor(values).reshape(-1, 1, 1)
        for values in ([21.0, 2.0], [-24.0, float("nan")], [0.0])
    ]
    shards = [
        chunk_probability_stats(x, torch.zeros_like(x), None, 0.2, 0.2) for x in deltas
    ]
    calls = []

    def reduce(values, dtype, op):
        assert dtype == torch.float64
        calls.append(op)
        if op == torch.distributed.ReduceOp.MAX:
            assert set(values) == {"max_abs_logratio", "max_abs_logratio_unfiltered"}
            return {key: max(shard[key] for shard in shards) for key in values}
        assert op == torch.distributed.ReduceOp.SUM
        assert not {"max_abs_logratio", "max_abs_logratio_unfiltered"}.intersection(
            values
        )
        return {key: sum(shard[key] for shard in shards) for key in values}

    monkeypatch.setattr(actor_module, "all_reduce_dict", reduce)
    result = worker._reduce_probability_stats(shards[0])
    complete = torch.cat(deltas)
    assert result == pytest.approx(
        chunk_probability_stats(complete, torch.zeros_like(complete), None, 0.2, 0.2)
    )
    assert calls == [torch.distributed.ReduceOp.MAX, torch.distributed.ReduceOp.SUM]


@pytest.mark.parametrize("fault", [21.0, -21.0, float("nan"), float("inf")])
@pytest.mark.parametrize("rank", [0, 1])
def test_worker_emits_exact_pre_update_probability_abort_without_progress_logging(
    monkeypatch, fault, rank
):
    worker = make_worker(monkeypatch, lr=0.0001)
    worker._rank = rank
    worker.version = 41
    worker.cfg.algorithm.ppo_update_guard.enforce_kl_limit = False
    messages = []
    worker.log_info = messages.append
    original = worker.model.forward
    injected = False

    def forward(**kwargs):
        nonlocal injected
        result = original(**kwargs)
        if torch.is_grad_enabled() and worker.optimizer_steps == 1 and not injected:
            result["logprobs"] = result["logprobs"].clone()
            result["logprobs"][0, 0, 0] += fault
            injected = True
        return result

    worker.model.forward = forward
    result = worker.run_training()
    assert injected
    assert result["guard/abort"] == result["guard/invalid_probability"] == 1
    assert result["guard/abort_round"] == 42
    assert result["guard/abort_pass"] == 2
    assert result["guard/abort_minibatch"] == 1
    assert worker.optimizer_steps == 1
    assert result["guard/actor_updates"] == result["guard/critic_updates"] == 1
    if rank:
        assert not messages
        return
    assert len(messages) == 1
    assert messages[0].startswith("PPO_GUARD_ABORT ")
    event = json.loads(messages[0].removeprefix("PPO_GUARD_ABORT "))
    assert event["reason"] == "invalid_probability"
    assert event["stage"] == "pre_update"
    assert event["global_step"] == 41 and event["round"] == 42
    assert event["pass"] == 2 and event["minibatch"] == 1
    assert event["logging_rank"] == 0
    assert event["statistics_scope"] == "all_actor_ranks"
    assert (
        event["optimizer_steps"]
        == event["actor_updates"]
        == event["critic_updates"]
        == 1
    )
    probability = event["probability"]
    assert probability["valid_count"] == 4
    assert probability["invalid_count"] == 1
    assert probability["nonfinite_logratio_count"] == int(not math.isfinite(fault))
    assert probability["finite_out_of_range_logratio_count"] == int(
        math.isfinite(fault)
    )
    assert probability["extreme_positive_logratio_count"] == int(fault == 21)
    assert probability["extreme_negative_logratio_count"] == int(fault == -21)
    assert probability["invalid_weight_count"] == 0
    if math.isfinite(fault):
        assert probability["max_abs_logratio_unfiltered"] > 20
        assert probability["max_abs_logratio"] < 20
    assert event["nonfinite_statistics"] == []


def test_initial_probability_abort_sanitizes_reduction_overflow_in_json(monkeypatch):
    worker = make_worker(monkeypatch)
    messages = []
    worker.log_info = messages.append
    stats = empty_probability_stats()
    stats.update(count=1.0, weight_sum=1.0, weighted_kl_sum=float("inf"))
    worker._measure_policy_probe = lambda probe: stats
    result = worker.run_training()
    assert result["guard/abort"] == result["guard/initial_probability_mismatch"] == 1
    assert worker.optimizer_steps == 0
    assert len(messages) == 1
    event = json.loads(messages[0].removeprefix("PPO_GUARD_ABORT "))
    assert event["stage"] == "initial_probe"
    assert event["pass"] is event["minibatch"] is None
    assert event["probability"]["kl"] is None
    assert event["probability"]["invalid_count"] == 0
    assert event["nonfinite_statistics"] == ["weighted_kl_sum"]


def test_final_probability_probe_abort_records_unfiltered_finite_outlier(monkeypatch):
    worker = make_worker(monkeypatch, lr=0.0001)
    worker.cfg.algorithm.ppo_update_guard.enforce_kl_limit = False
    messages = []
    worker.log_info = messages.append
    original = worker.model.forward

    def forward(**kwargs):
        result = original(**kwargs)
        if not torch.is_grad_enabled() and worker.optimizer_steps == 3:
            result["logprobs"] = result["logprobs"].clone()
            result["logprobs"][0, 0, 0] -= 21.0
        return result

    worker.model.forward = forward
    result = worker.run_training()
    assert result["guard/abort"] == result["guard/final_probability_mismatch"] == 1
    assert worker.optimizer_steps == 3
    assert len(messages) == 1
    event = json.loads(messages[0].removeprefix("PPO_GUARD_ABORT "))
    assert event["stage"] == "final_probe"
    assert event["pass"] is event["minibatch"] is None
    assert event["probability"]["finite_out_of_range_logratio_count"] == 4
    assert event["probability"]["nonfinite_logratio_count"] == 0
    assert event["probability"]["max_abs_logratio_unfiltered"] > 20


def test_worker_rolls_back_excessive_actor_step_and_adam_but_keeps_critic(monkeypatch):
    worker = make_worker(monkeypatch)
    result = worker.run_training()
    assert result["guard/actor_updates"] == 1
    assert result["guard/rejected_actor_updates"] == 1
    assert result["guard/critic_updates"] == 3
    assert worker.optimizer.state[worker.model.logprobs]["step"].item() == 1
    assert worker.optimizer.state[worker.model.value_head.weight]["step"].item() == 3
    assert result["guard/final_kl"] <= 0.02
    assert 1.16 < result["guard/final_ratio"] < 1.19
    assert torch.allclose(worker.model.logprobs, torch.full((20, 8), 0.001), atol=1e-7)
    assert worker.model.value_head.weight.item() > 0
    assert worker.rollout_batch is None


def test_worker_rejects_initial_probability_mismatch_before_updates(monkeypatch):
    worker = make_worker(monkeypatch)
    worker.model.logprobs.data.fill_(0.1)
    before = copy.deepcopy(worker.model.state_dict())
    result = worker.run_training()
    assert result["guard/abort"] == 1
    assert result["guard/initial_probability_mismatch"] == 1
    assert worker.optimizer_steps == 0
    assert all(torch.equal(before[k], v) for k, v in worker.model.state_dict().items())
    assert worker.rollout_batch is None


@pytest.mark.parametrize("warmup", [0, 2])
def test_zero_reward_batch_cannot_change_actor_even_with_nonzero_advantages(
    monkeypatch, warmup
):
    worker = make_worker(monkeypatch, zero_reward=True, warmup=warmup)
    result = worker.run_training()
    assert result["guard/actor_skipped_zero_reward"] == 1
    assert result["guard/actor_updates"] == 0
    assert torch.count_nonzero(worker.model.logprobs) == 0
    assert not worker.optimizer.state.get(worker.model.logprobs)
    assert result["guard/critic_updates"] == 3
    assert worker.model.value_head.weight.item() > 0


def test_warmup_keeps_critic_moments_then_allows_policy_updates(monkeypatch):
    worker = make_worker(monkeypatch, warmup=2)
    result = worker.run_training()
    assert result["guard/critic_updates"] == 3
    assert result["guard/actor_updates"] == 1
    assert worker.optimizer.state[worker.model.value_head.weight]["step"].item() == 3
    assert worker.optimizer.state[worker.model.logprobs]["step"].item() == 1


def test_entire_invalid_rollout_does_not_step_optimizer(monkeypatch):
    worker = make_worker(monkeypatch)
    worker.rollout_batch["loss_mask"].fill_(False)
    result = worker.run_training()
    assert result["guard/abort"] == 1
    assert worker.optimizer_steps == 0


def test_snapshot_restores_existing_actor_momentum_and_leaves_critic_state():
    a = torch.nn.Parameter(torch.tensor(1.0))
    c = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.AdamW(
        [
            {"params": [a], "rlinf_role": "actor"},
            {"params": [c], "rlinf_role": "critic"},
        ],
        lr=0.1,
    )
    (a.square() + c.square()).backward()
    opt.step()
    opt.zero_grad()
    a_before = a.detach().clone()
    old_state = copy.deepcopy(opt.state[a])
    (a.square() + c.square()).backward()
    snap = ActorOptimizerSnapshot(opt)
    opt.step()
    c_after = c.detach().clone()
    snap.restore()
    assert torch.equal(a, a_before) and torch.equal(c, c_after)
    assert all(torch.equal(opt.state[a][k], v) for k, v in old_state.items())
    assert opt.state[c]["step"].item() == 2


def test_single_valid_advantage_normalizes_without_nan():
    result = safe_normalize(torch.tensor([1.0, 2.0]), torch.tensor([True, False]))
    assert result[0] == 0 and torch.isfinite(result).all()


@pytest.mark.parametrize(
    "field,value",
    [("target_kl", 0), ("max_initial_logratio", float("nan")), ("probe_batch_size", 0)],
)
def test_invalid_guard_configuration_is_rejected(monkeypatch, field, value):
    worker = make_worker(monkeypatch)
    worker.cfg.algorithm.ppo_update_guard[field] = value
    with pytest.raises(ValueError):
        validate_ppo_update_guard_config(worker.cfg)


@pytest.mark.parametrize("normalize", [False, True])
def test_actual_actor_forwards_normalization_option(monkeypatch, normalize):
    worker = make_worker(monkeypatch)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    worker.cfg.algorithm.normalize_advantages = normalize
    worker.rollout_batch["dones"] = torch.zeros(3, 2, 20, dtype=torch.bool)
    worker.rollout_batch["dones"][-1] = True
    method = inspect.unwrap(EmbodiedFSDPActor.compute_advantages_and_returns)
    method(worker)
    actual = worker.rollout_batch["advantages"]
    raw = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
    if normalize:
        expected = (raw - raw.mean()) / (raw.std() + 1e-5)
    else:
        expected = raw
    assert torch.allclose(actual, expected)


def test_guard_supports_larger_microbatch_with_gradient_accumulation(monkeypatch):
    worker = make_worker(monkeypatch)
    worker.cfg.actor.micro_batch_size = 2
    worker.gradient_accumulation = 2
    result = worker.run_training()
    assert result["guard/abort"] == 0
    assert worker.optimizer_steps == worker.cfg.algorithm.update_epoch


def test_all_padding_global_batches_do_not_advance_adam(monkeypatch):
    from rlinf.utils.nested_dict_process import process_nested_dict_for_train

    worker = make_worker(monkeypatch, zero_reward=True)
    worker.cfg.actor.global_batch_size = 2
    worker.gradient_accumulation = 2
    worker.rollout_batch["action_mask"][1] = False
    worker.rollout_batch["loss_mask"][1] = False
    worker.rollout_batch = process_nested_dict_for_train(
        worker.rollout_batch, torch.arange(4)
    )
    result = worker._run_guarded_updates(
        dict(worker.cfg.algorithm.ppo_update_guard), 4, 2
    )
    assert result["guard/skipped_empty_batches"] == 3
    assert result["guard/critic_updates"] == 3
    assert worker.optimizer.state[worker.model.value_head.weight]["step"].item() == 3
    assert worker.optimizer_steps == 3


def test_nonfinite_targets_abort_before_any_model_update(monkeypatch):
    worker = make_worker(monkeypatch)
    worker.rollout_batch["returns"][0, 0] = float("nan")
    result = worker.run_training()
    assert result["guard/abort"] == 1 and result["guard/invalid_rollout"] == 1
    assert worker.optimizer_steps == 0


def test_gradient_enabled_initial_mismatch_is_caught(monkeypatch):
    worker = make_worker(monkeypatch)
    original = worker.model.forward

    def forward(**kwargs):
        output = original(**kwargs)
        if torch.is_grad_enabled():
            output["logprobs"] = output["logprobs"] + 0.1
        return output

    worker.model.forward = forward
    result = worker.run_training()
    assert result["guard/initial_probability_mismatch"] == 1
    assert result["guard/abort"] == 1
    assert worker.optimizer_steps == 0


def test_rejected_update_does_not_keep_actor_momentum_changes(monkeypatch):
    worker = make_worker(monkeypatch, lr=0.1)
    result = worker.run_training()
    assert result["guard/rejected_actor_updates"] == 1
    assert result["guard/actor_updates"] == 0
    assert torch.count_nonzero(worker.model.logprobs) == 0
    assert worker.optimizer.state[worker.model.logprobs] == {}
    assert result["guard/critic_updates"] == 3
    assert result["guard/final_ratio"] == 1
