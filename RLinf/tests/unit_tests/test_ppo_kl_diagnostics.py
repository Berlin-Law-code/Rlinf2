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

"""Actual worker contracts for KL diagnostics and optional threshold enforcement."""

import copy

import pytest
import torch
from test_ppo_guard_batch_weighting import _configure_worker
from test_ppo_optimizer_progress import _capture
from test_ppo_update_guard import make_worker

import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
from rlinf.algorithms.ppo_update_guard import (
    chunk_probability_stats,
    probability_stats_metrics,
)
from rlinf.utils.nested_dict_process import process_nested_dict_for_train


def test_diagnostic_mode_accepts_finite_kl_without_snapshots_or_candidate_probes(
    monkeypatch,
):
    worker = make_worker(monkeypatch)
    worker.cfg.algorithm.ppo_update_guard.enforce_kl_limit = False
    worker.cfg.actor.log_optimizer_progress = True
    events = _capture(worker)
    probe_calls = []
    measure = worker._measure_policy_probe

    def counted_probe(probe):
        probe_calls.append(worker.optimizer_steps)
        return measure(probe)

    def forbidden_snapshot(*args, **kwargs):
        pytest.fail("Diagnostic-only KL must not copy actor/Adam snapshots")

    worker._measure_policy_probe = counted_probe
    monkeypatch.setattr(actor_module, "ActorOptimizerSnapshot", forbidden_snapshot)
    result = worker.run_training()
    assert result["guard/abort"] == result["guard/rejected_actor_updates"] == 0
    assert result["guard/actor_updates"] == result["guard/critic_updates"] == 3
    assert result["guard/final_kl"] > 0.02
    assert result["guard/pre_update_kl_exceeded_batches"] > 0
    assert result["guard/pre_update_kl_stop"] == 0
    assert result["guard/kl_limit_enforced"] == 0
    assert probe_calls == [0, 3]
    assert all(event["outcome"] == "accepted" for event in events)
    assert all(event["actor_lr"] == 0.001 for event in events)
    assert all(not event["pre_update_actor_stopped"] for event in events)
    assert all(event["candidate_post_kl"] is None for event in events)
    assert all(event["last_verified_policy_kl"] is None for event in events)


def test_pre_update_stop_records_triggering_batch_even_when_fixed_probe_is_small(
    monkeypatch,
):
    worker = make_worker(monkeypatch)
    worker.cfg.actor.log_optimizer_progress = True
    events = _capture(worker)
    # Model a heterogeneous rollout: the separate fixed probe can be quiet
    # while the actual optimizer minibatch exhibits a much larger change.
    quiet = torch.zeros(4, 1)
    worker._measure_policy_probe = lambda probe: chunk_probability_stats(
        quiet, quiet, None, 0.2, 0.2
    )
    result = worker.run_training()
    assert result["guard/abort"] == 0
    assert result["guard/pre_update_kl_stop"] == 1
    stops = [event for event in events if event["pre_update_actor_stopped"]]
    assert len(stops) == 1
    stop = stops[0]
    assert stop["pre_update_kl"] > 0.02
    assert stop["pre_update_kl_exceeded"] and stop["kl_limit_enforced"]
    assert stop["last_verified_policy_kl"] == 0
    assert stop["pre_update_valid_chunks"] == 4
    assert stop["pre_update_weight_sum"] == 4
    assert result["guard/pre_stop_kl"] == stop["pre_update_kl"]
    assert result["guard/pre_stop_unweighted_kl"] == stop["pre_update_unweighted_kl"]
    assert result["guard/pre_stop_pass"] == stop["pass"]
    assert result["guard/pre_stop_minibatch"] == stop["minibatch"]
    assert result["guard/actor_updates"] == 2
    assert result["guard/critic_updates"] == 3


def test_current_batch_and_fixed_probe_share_loss_relative_length_weights(monkeypatch):
    worker = _configure_worker(monkeypatch, "libero_trajectory", 32)
    batch = process_nested_dict_for_train(
        copy.deepcopy(worker.rollout_batch), torch.arange(32)
    )
    with torch.no_grad():
        worker.model.logprobs.fill_(0.001)
    mask = batch["loss_mask"].reshape(-1)
    lengths = batch["loss_mask_sum"].reshape(-1)[mask].double()
    # Independent oracle using executed action counts, including partially
    # executed terminal chunks, short/long episodes and entirely padded slots.
    delta = batch["action_mask"].sum(-1)[mask].double() * 8 * 0.001
    k3 = delta.exp() - 1 - delta
    weights = 80 / lengths
    expected = float((weights * k3).sum() / weights.sum())
    expected_unweighted = float(k3.mean())
    assert abs(expected - expected_unweighted) > 1e-4
    direct = worker.train_micro_batch(
        batch,
        {},
        is_last=True,
        probability_only=True,
        loss_weighting="libero_trajectory",
    )
    probe = worker._measure_policy_probe(batch)
    for stats in (direct, probe):
        metrics = probability_stats_metrics(stats)
        assert metrics["kl"] == pytest.approx(expected, rel=2e-6)
        assert metrics["unweighted_kl"] == pytest.approx(expected_unweighted, rel=2e-6)
        assert metrics["weight_sum"] == pytest.approx(float(weights.sum()))
        assert metrics["valid_count"] == int(mask.sum())
        assert metrics["invalid_count"] == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1000.0])
def test_diagnostic_mode_still_aborts_invalid_probabilities_before_another_update(
    monkeypatch, invalid
):
    worker = make_worker(monkeypatch)
    worker.cfg.algorithm.ppo_update_guard.enforce_kl_limit = False
    step = worker.optimizer_step

    def corrupt_after_update(**kwargs):
        result = step(**kwargs)
        with torch.no_grad():
            worker.model.logprobs.fill_(invalid)
        return result

    worker.optimizer_step = corrupt_after_update
    result = worker.run_training()
    assert result["guard/abort"] == 1
    assert result["guard/invalid_probability"] == 1
    assert worker.optimizer_steps == 1


def test_diagnostic_mode_still_rejects_initial_old_new_probability_mismatch(
    monkeypatch,
):
    worker = make_worker(monkeypatch)
    worker.cfg.algorithm.ppo_update_guard.enforce_kl_limit = False
    with torch.no_grad():
        worker.model.logprobs.fill_(0.001)
    result = worker.run_training()
    assert result["guard/abort"] == result["guard/initial_probability_mismatch"] == 1
    assert worker.optimizer_steps == 0
