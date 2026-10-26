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

"""CPU gradient contracts for guarded chunk and LIBERO length aggregation."""

import copy
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from test_ppo_update_guard import make_worker

import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
from rlinf.algorithms.ppo_update_guard import validate_ppo_update_guard_config
from rlinf.algorithms.registry import policy_loss
from rlinf.utils.nested_dict_process import process_nested_dict_for_train


def _shard(rank: int) -> dict:
    # Each column is one trajectory. Real lengths range from 5 to 80 actions;
    # chunks after termination are padding, and the final chunk may be partial.
    counts = torch.tensor(
        [
            [[20, 7], [20, 0], [20, 0], [10, 0]],
            [[20, 20], [10, 20], [0, 20], [0, 20]],
            [[5, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ][rank]
    )
    action_mask = torch.arange(20) < counts.unsqueeze(-1)
    mask = counts.unsqueeze(-1) > 0
    lengths = counts.sum(0).float().reshape(1, 2, 1).expand(4, 2, 1).clone()
    # Include both zero and NaN padding sentinels. Neither may enter H/L.
    lengths[~mask] = float("nan") if rank % 2 else 0
    rewards = torch.zeros(4, 2, 20)
    if rank == 0:
        rewards[-1, 0, 9] = 1
    return {
        "rewards": rewards,
        "advantages": torch.tensor([1.2, -0.7, 0.5, -1.0, 0.3, 0.9, -0.4, 0.8])
        .reshape(4, 2, 1)
        .mul(1 + rank * 0.2),
        "prev_logprobs": torch.zeros(4, 2, 20, 8),
        "returns": torch.linspace(0.2, 1.3, 8).reshape(4, 2, 1) + rank * 0.1,
        "prev_values": torch.zeros(5, 2, 1),
        "loss_mask": mask,
        "loss_mask_sum": lengths,
        "action_mask": action_mask,
        "forward_inputs": {"obs": torch.zeros(4, 2, 1)},
    }


def _global_batch() -> dict:
    def concatenate(values):
        if isinstance(values[0], dict):
            return {
                key: concatenate([value[key] for value in values]) for key in values[0]
            }
        return torch.cat(values, dim=1)

    return concatenate([_shard(rank) for rank in range(4)])


def _configure_worker(monkeypatch, mode: str, micro: int, *, rank=None):
    worker = make_worker(monkeypatch, lr=1e-6)
    world = 1 if rank is None else 4
    worker._world_size = world
    worker._rank = 0 if rank is None else rank
    worker.cfg.actor.global_batch_size = 32
    worker.cfg.actor.micro_batch_size = micro
    worker.gradient_accumulation = 32 // world // micro
    worker.cfg.algorithm.update_epoch = 1
    worker.cfg.algorithm.clip_ratio_c = None
    worker.cfg.algorithm.clip_log_ratio_min = None
    worker.cfg.algorithm.clip_log_ratio_max = None
    worker.cfg.algorithm.ppo_update_guard.loss_weighting = mode
    worker.cfg.env.train.max_episode_steps = 80
    worker.rollout_batch = _global_batch() if rank is None else _shard(rank)
    worker.recorded_gradients = []

    def record_and_clip(model):
        worker.recorded_gradients.append(
            torch.cat(
                [parameter.grad.flatten().clone() for parameter in model.parameters()]
            )
        )
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    worker._strategy.clip_grad_norm_ = record_and_clip
    return worker


def _analytic_gradient(mode: str) -> torch.Tensor:
    """Independent closed form at ratio=1 and value=0, before gradient clipping."""
    batch = _global_batch()
    mask = batch["loss_mask"].float()
    if mode == "valid_chunk":
        weights = mask / mask.sum()
    else:
        lengths = torch.where(mask.bool(), batch["loss_mask_sum"], 1)
        weights = mask * (80 / lengths) / 32
    # At ratio=1 the clipped PPO surrogate has derivative -advantage.
    # The final action mask must apply before the joint chunk probability sum.
    actor = (
        -(batch["advantages"] * weights * batch["action_mask"])
        .sum(dim=(0, 1))
        .unsqueeze(-1)
        .expand(20, 8)
    )
    # All residuals are inside the quadratic part of the actual Huber loss.
    critic = -(batch["returns"] * weights).sum().reshape(1)
    return torch.cat((actor.flatten(), critic))


@pytest.mark.parametrize("mode", ["valid_chunk", "libero_trajectory"])
@pytest.mark.parametrize("micro", [1, 2, 4, 8, 32])
def test_actual_worker_gradients_match_global_objective(monkeypatch, mode, micro):
    worker = _configure_worker(monkeypatch, mode, micro)
    result = worker.run_training()
    assert result["guard/abort"] == 0
    assert result["guard/valid_chunks"] == 12
    assert result["guard/actor_updates"] == result["guard/critic_updates"] == 1
    assert len(worker.recorded_gradients) == 1
    torch.testing.assert_close(
        worker.recorded_gradients[0], _analytic_gradient(mode), rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize("mode", ["valid_chunk", "libero_trajectory"])
def test_extra_padding_retains_each_modes_explicit_denominator(monkeypatch, mode):
    worker = _configure_worker(monkeypatch, mode, 4)
    padding = copy.deepcopy(worker.rollout_batch)
    for key in ("loss_mask", "action_mask", "loss_mask_sum", "rewards"):
        padding[key].zero_()

    def append_padding(batch, tail):
        if isinstance(batch, dict):
            return {key: append_padding(batch[key], tail[key]) for key in batch}
        return torch.cat((batch, tail), dim=1)

    worker.rollout_batch = append_padding(worker.rollout_batch, padding)
    worker.cfg.actor.global_batch_size = 64
    worker.gradient_accumulation = 16
    result = worker.run_training()
    assert result["guard/abort"] == 0
    assert result["guard/valid_chunks"] == 12
    # LIBERO's fixed-slot mean includes added padding in B; the default mean
    # only includes valid chunks. Never advertise the two as interchangeable.
    expected = _analytic_gradient(mode) * (0.5 if mode == "libero_trajectory" else 1)
    torch.testing.assert_close(
        worker.recorded_gradients[0], expected, rtol=2e-6, atol=2e-7
    )


def test_libero_objective_matches_direct_official_actor_critic_loss(monkeypatch):
    worker = _configure_worker(monkeypatch, "libero_trajectory", 32)
    batch = process_nested_dict_for_train(
        copy.deepcopy(worker.rollout_batch), torch.arange(32)
    )
    output = worker.model(forward_inputs=batch["forward_inputs"])
    lengths = torch.where(batch["loss_mask"], batch["loss_mask_sum"], 1)
    loss, _ = policy_loss(
        loss_type="actor_critic",
        task_type="embodied",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=8,
        logprobs=output["logprobs"] * batch["action_mask"].unsqueeze(-1),
        old_logprobs=batch["prev_logprobs"],
        values=output["values"],
        advantages=batch["advantages"],
        returns=batch["returns"],
        prev_values=batch["prev_values"],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=batch["loss_mask"],
        loss_mask_sum=lengths,
        max_episode_steps=80,
    )
    loss.backward()
    gradient = torch.cat(
        [parameter.grad.flatten() for parameter in worker.model.parameters()]
    )
    torch.testing.assert_close(gradient, _analytic_gradient("libero_trajectory"))


def _distributed_rank(rank: int, rendezvous: str, output: str, mode: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            original_world_size = torch.distributed.get_world_size
            worker = _configure_worker(patch, mode, 2, rank=rank)
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
                    tensor /= 4
                return dict(zip(keys, tensor.tolist()))

            patch.setattr(actor_module, "all_reduce_dict", reduce_stats)
            raw_model = worker.model
            raw_model.register_buffer("probe_buffer", torch.zeros(1))
            worker.model = torch.nn.parallel.DistributedDataParallel(
                raw_model, find_unused_parameters=True, broadcast_buffers=True
            )
            result = worker.run_training()
            torch.save(
                {
                    "result": result,
                    "gradients": worker.recorded_gradients,
                    "model": raw_model.state_dict(),
                },
                Path(output) / f"rank{rank}.pt",
            )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("mode", ["valid_chunk", "libero_trajectory"])
def test_four_ranks_with_all_padding_rank_match_unsharded_gradient(tmp_path, mode):
    context = torch.multiprocessing.start_processes(
        _distributed_rank,
        args=(str(tmp_path / "gloo"), str(tmp_path), mode),
        nprocs=4,
        join=False,
        start_method="spawn",
    )
    try:
        deadline = time.monotonic() + 100
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Four-rank gradient weighting collective test timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
    states = [
        torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(4)
    ]
    for state in states:
        assert state["result"]["guard/abort"] == 0
        assert state["result"]["guard/valid_chunks"] == 12
        assert state["result"]["guard/actor_updates"] == 1
        assert len(state["gradients"]) == 1
        torch.testing.assert_close(
            state["gradients"][0], _analytic_gradient(mode), rtol=2e-6, atol=2e-7
        )
        for name, value in states[0]["model"].items():
            torch.testing.assert_close(state["model"][name], value, rtol=0, atol=0)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), None])
def test_bad_valid_episode_lengths_abort_before_optimizer(monkeypatch, value):
    worker = _configure_worker(monkeypatch, "libero_trajectory", 2)
    if value is None:
        del worker.rollout_batch["loss_mask_sum"]
    else:
        worker.rollout_batch["loss_mask_sum"][0, 0, 0] = value
    result = worker.run_training()
    assert result["guard/abort"] == 1
    assert result["guard/invalid_episode_length_count"] > 0
    assert worker.optimizer_steps == 0
    assert not worker.recorded_gradients


def test_default_weighting_retains_valid_chunk_mode(monkeypatch):
    worker = make_worker(monkeypatch)
    assert (
        validate_ppo_update_guard_config(worker.cfg)["loss_weighting"] == "valid_chunk"
    )


@pytest.mark.parametrize("micro", [0, -1, True, 1.5])
def test_invalid_microbatch_sizes_fail_validation(monkeypatch, micro):
    worker = make_worker(monkeypatch)
    worker.cfg.actor.micro_batch_size = micro
    with pytest.raises(ValueError, match="positive integer micro_batch_size"):
        validate_ppo_update_guard_config(worker.cfg)


@pytest.mark.parametrize("mode", ["episode_equal", "typo", None])
def test_unknown_weighting_fails_validation(monkeypatch, mode):
    worker = make_worker(monkeypatch)
    worker.cfg.algorithm.ppo_update_guard.loss_weighting = mode
    with pytest.raises(ValueError, match="loss_weighting"):
        validate_ppo_update_guard_config(worker.cfg)


def test_libero_mode_rejects_microbatch_dependent_entropy(monkeypatch):
    worker = _configure_worker(monkeypatch, "libero_trajectory", 2)
    worker.cfg.algorithm.entropy_bonus = 0.01
    with pytest.raises(ValueError, match="entropy_bonus=0"):
        validate_ppo_update_guard_config(worker.cfg)


@pytest.mark.parametrize("horizon", [0, -1, True, 90.5])
def test_libero_mode_requires_positive_integer_episode_cap(monkeypatch, horizon):
    worker = _configure_worker(monkeypatch, "libero_trajectory", 2)
    worker.cfg.env.train.max_episode_steps = horizon
    with pytest.raises(ValueError, match="max_episode_steps"):
        validate_ppo_update_guard_config(worker.cfg)
