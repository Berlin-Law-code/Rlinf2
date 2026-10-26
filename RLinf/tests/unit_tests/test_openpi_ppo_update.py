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

"""CPU regressions for the real embodied PPO update and batch lifecycle."""

import asyncio
import inspect
import math
import weakref
from collections import Counter, deque
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
import rlinf.workers.actor.fsdp_actor_worker_pipeline as pipeline_module
from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


def _method(name):
    # Bypass only Worker RPC/timing wrappers; execute the production method.
    return inspect.unwrap(getattr(EmbodiedFSDPActor, name))


class _TinyPolicy(torch.nn.Module):
    def __init__(self, log_ratio):
        super().__init__()
        self.logprobs = torch.nn.Parameter(torch.full((20, 8), log_ratio / 160))
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, forward_inputs, **kwargs):
        batch_size = forward_inputs["obs"].shape[0]
        logprobs = self.logprobs.expand(batch_size, -1, -1)
        return {
            "logprobs": logprobs,
            "prev_logprobs": kwargs.get("prev_logprobs", torch.zeros_like(logprobs)),
            "values": self.value.expand(batch_size),
        }


@pytest.fixture
def cpu_platform(monkeypatch):
    platform = SimpleNamespace(
        current_device=lambda: torch.device("cpu"), empty_cache=lambda: None
    )
    monkeypatch.setattr(actor_module.Worker, "torch_platform", platform)
    monkeypatch.setattr(actor_module, "clear_memory", lambda **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        actor_module, "all_reduce_dict", lambda values, **kwargs: values
    )
    return platform


def _worker(cpu_platform, log_ratio=0.0, model_type="openpi"):
    config = OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "actor": {
                "seed": 123,
                "global_batch_size": 4,
                "micro_batch_size": 1,
                "model": {"model_type": model_type, "action_dim": 8},
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
                "clip_log_ratio_min": -20.0,
                "clip_log_ratio_max": 20.0,
                "value_clip": 0.2,
                "huber_delta": 10.0,
                "update_epoch": 3,
            },
            "env": {"train": {"max_episode_steps": 1200}},
            "rollout": {"sampling_params": {"temperature_train": 1.0, "top_k": 0}},
        }
    )
    worker = SimpleNamespace(
        cfg=config,
        device=torch.device("cpu"),
        torch_platform=cpu_platform,
        amp_context=nullcontext(),
        model=_TinyPolicy(log_ratio),
        before_micro_batch=lambda *args, **kwargs: nullcontext(),
        optimizer_steps=0,
        critic_warmup_steps=0,
        enable_sft_co_train=False,
        gradient_accumulation=1,
        grad_scaler=SimpleNamespace(scale=lambda loss: loss),
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        rollout_batch=None,
        _rollout_batch_in_use=False,
        _world_size=1,
        _rank=0,
    )
    worker.train_micro_batch = MethodType(_method("train_micro_batch"), worker)
    worker._run_training_on_rollout_batch = MethodType(
        _method("_run_training_on_rollout_batch"), worker
    )
    return worker


def _micro_batch(advantage=-1.0):
    return {
        "advantages": torch.tensor([[advantage]]),
        "prev_logprobs": torch.zeros(1, 20, 8),
        "returns": torch.zeros(1, 1),
        "prev_values": torch.zeros(1, 1),
        "loss_mask": torch.ones(1, 1, dtype=torch.bool),
        "loss_mask_sum": torch.tensor([[1200]]),
        "action_mask": torch.ones(1, 20, dtype=torch.bool),
        "forward_inputs": {"obs": torch.zeros(1, 4)},
    }


@pytest.mark.parametrize("model_type", ["openpi", "gr00t_n1d6", "gr00t_n1d7"])
def test_worker_passes_configured_dual_clip(cpu_platform, model_type):
    worker = _worker(cpu_platform, math.log(10), model_type)
    metrics = {}
    worker.train_micro_batch(_micro_batch(), metrics, is_last=True)
    assert metrics["actor/policy_loss"] == pytest.approx([3.0])
    assert torch.count_nonzero(worker.model.logprobs.grad) == 0


@pytest.mark.parametrize("log_ratio", [-100.0, 100.0])
@pytest.mark.parametrize("advantage", [-1.0, 1.0])
def test_worker_bounds_large_finite_ratios_before_exponentiation(
    cpu_platform, log_ratio, advantage
):
    worker = _worker(cpu_platform, log_ratio)
    metrics = {}
    worker.train_micro_batch(_micro_batch(advantage), metrics, is_last=True)
    assert metrics["actor/ratio"] == pytest.approx([math.exp(log_ratio / 5)])
    assert torch.isfinite(worker.model.logprobs.grad).all()
    assert torch.count_nonzero(worker.model.logprobs.grad) == 0
    assert all(
        torch.isfinite(torch.as_tensor(values)).all() for values in metrics.values()
    )


@pytest.mark.parametrize(
    ("model_type", "configured", "expected"),
    [("openpi", False, 10.0), ("gr00t_n1d6", False, 3.0), ("openvla", True, 10.0)],
)
def test_other_model_and_unconfigured_defaults_are_preserved(
    cpu_platform, model_type, configured, expected
):
    worker = _worker(cpu_platform, math.log(10), model_type)
    if not configured:
        for key in ("clip_ratio_c", "clip_log_ratio_min", "clip_log_ratio_max"):
            del worker.cfg.algorithm[key]
    metrics = {}
    worker.train_micro_batch(_micro_batch(), metrics, is_last=True)
    assert metrics["actor/policy_loss"] == pytest.approx([expected])


def test_unexecuted_tail_has_no_ratio_or_gradient_contribution(cpu_platform):
    worker = _worker(cpu_platform)
    batch = _micro_batch()
    batch["action_mask"][:, 1:] = False
    with torch.no_grad():
        worker.model.logprobs[0].fill_(math.log(1.1) / 8)
        worker.model.logprobs[1:].fill_(100.0)
    batch["prev_logprobs"][:, 1:] = -100.0
    metrics = {}
    worker.train_micro_batch(batch, metrics, is_last=True)
    assert metrics["actor/ratio"] == pytest.approx([1.1])
    assert torch.isfinite(worker.model.logprobs.grad).all()
    assert torch.count_nonzero(worker.model.logprobs.grad[0]) == 8
    assert torch.count_nonzero(worker.model.logprobs.grad[1:]) == 0


@pytest.mark.parametrize(
    ("mask", "expected"),
    [(None, 0.5), ([True, True, True, False], 1 / 3), ([False] * 4, 0.0)],
)
def test_critic_clip_metric_uses_unclipped_valid_deltas(mask, expected):
    values = torch.tensor([2.0, 0.2, 0.1, -2.0], requires_grad=True)
    mask = None if mask is None else torch.tensor(mask)
    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=torch.ones(4),
        prev_values=torch.zeros(4),
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=mask,
    )
    assert metrics["critic/value_clip_ratio"].item() == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(values.grad).all()


def _prepare_training(worker):
    worker.gradient_accumulation = 4
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.001)
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1
    )

    def optimizer_step():
        worker.optimizer.step()
        worker.optimizer_steps += 1
        return 0.0, [0.001]

    worker.optimizer_step = optimizer_step
    worker.rollout_batch = {
        "advantages": torch.ones(2, 2, 1),
        "prev_logprobs": torch.zeros(2, 2, 20, 8),
        "returns": torch.arange(4, dtype=torch.float32).reshape(2, 2, 1),
        "prev_values": torch.zeros(3, 2, 1),
        "loss_mask": torch.ones(2, 2, 1, dtype=torch.bool),
        "loss_mask_sum": torch.full((2, 2, 1), 1200),
        "action_mask": torch.ones(2, 2, 20, dtype=torch.bool),
        "forward_inputs": {"obs": torch.zeros(2, 2, 4)},
    }


def test_sync_batch_lives_through_all_epochs_and_metrics_then_is_released(
    cpu_platform, monkeypatch
):
    worker = _worker(cpu_platform)
    _prepare_training(worker)
    batch_refs = []
    train_micro_batch = worker.train_micro_batch

    def update(**kwargs):
        assert worker.rollout_batch is not None and worker._rollout_batch_in_use
        batch_refs.append(weakref.ref(worker.rollout_batch["forward_inputs"]["obs"]))
        train_micro_batch(**kwargs)

    def reduce_metrics(metrics, **kwargs):
        assert worker.rollout_batch is not None and worker._rollout_batch_in_use
        assert len(batch_refs) == 12
        return metrics

    worker.train_micro_batch = update
    monkeypatch.setattr(actor_module, "all_reduce_dict", reduce_metrics)
    metrics = _method("run_training")(worker)
    assert worker.optimizer_steps == 3
    assert worker.rollout_batch is None and not worker._rollout_batch_in_use
    assert all(reference() is None for reference in batch_refs)
    assert all(math.isfinite(float(value)) for value in metrics.values())


@pytest.mark.parametrize("failure_stage", ["update", "metrics"])
def test_sync_batch_is_released_on_failure(cpu_platform, monkeypatch, failure_stage):
    worker = _worker(cpu_platform)
    _prepare_training(worker)

    def fail(*args, **kwargs):
        raise ValueError("injected failure")

    if failure_stage == "update":
        worker.train_micro_batch = fail
    else:
        monkeypatch.setattr(actor_module, "all_reduce_dict", fail)
    with pytest.raises(ValueError, match="injected failure"):
        _method("run_training")(worker)
    assert worker.rollout_batch is None and not worker._rollout_batch_in_use


@pytest.mark.parametrize("processing_fails", [False, True])
def test_recv_releases_old_storage_before_waiting_and_publishes_only_complete_batch(
    cpu_platform, monkeypatch, processing_fails
):
    worker = _worker(cpu_platform)
    worker.rollout_batch = {"old": torch.zeros(16)}
    old_storage = weakref.ref(worker.rollout_batch["old"])
    worker.stage_num = 1
    worker._component_placement = SimpleNamespace(get_world_size=lambda name: 1)

    async def receive():
        assert worker.rollout_batch is None and old_storage() is None
        return "trajectory"

    channel = SimpleNamespace(get=lambda **kwargs: SimpleNamespace(async_wait=receive))
    new_batch = {"new": torch.ones(1)}
    monkeypatch.setattr(
        actor_module,
        "convert_trajectories_to_batch",
        lambda _, **kwargs: new_batch,
    )

    def process(batch):
        assert worker.rollout_batch is None
        if processing_fails:
            raise ValueError("invalid rollout")
        return batch

    worker._process_received_rollout_batch = process
    if processing_fails:
        with pytest.raises(ValueError, match="invalid rollout"):
            asyncio.run(_method("recv_rollout_trajectories")(worker, channel))
        assert worker.rollout_batch is None
    else:
        asyncio.run(_method("recv_rollout_trajectories")(worker, channel))
        assert worker.rollout_batch is new_batch


def test_receive_cannot_replace_a_batch_during_training(cpu_platform):
    worker = _worker(cpu_platform)
    worker.rollout_batch = {"in_use": torch.ones(1)}
    worker._rollout_batch_in_use = True
    with pytest.raises(RuntimeError, match="during actor training"):
        asyncio.run(_method("recv_rollout_trajectories")(worker, None))
    assert "in_use" in worker.rollout_batch and worker._rollout_batch_in_use


def test_pipeline_reuses_micro_batches_for_all_epochs_and_rollout_metrics(
    cpu_platform, monkeypatch
):
    worker = _worker(cpu_platform)
    _prepare_training(worker)
    worker.rollout_batch = None
    worker.gradient_accumulation = 2
    worker.update_epoch = 3
    worker.micro_batches_per_step = 2
    worker.global_batches_per_step = 1
    worker.finish_global_batch = MethodType(_method("finish_global_batch"), worker)
    pipeline_actor = pipeline_module.PipelineEmbodiedFSDPActor
    worker.select_global_batch = MethodType(
        inspect.unwrap(pipeline_actor.select_global_batch), worker
    )
    batches = [_micro_batch(1.0), _micro_batch(1.0)]
    queue = deque(batches)
    worker.try_recv_micro_batch = lambda _: queue.popleft() if queue else None
    train_micro_batch = worker.train_micro_batch
    calls = Counter()

    def update(**kwargs):
        calls[id(kwargs["micro_batch"])] += 1
        train_micro_batch(**kwargs)

    worker.train_micro_batch = update
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline_module, "all_reduce_dict", lambda metrics, **kwargs: metrics
    )
    metrics = inspect.unwrap(pipeline_actor.run_training)(worker, None)
    assert calls == {id(batch): 3 for batch in batches}
    assert worker.optimizer_steps == 3
    assert metrics["rollout_metrics"]["advantages_mean"] == pytest.approx(1.0)
    assert worker.rollout_batch is None
