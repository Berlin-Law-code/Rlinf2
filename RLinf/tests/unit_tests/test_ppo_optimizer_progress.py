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

"""Optional rank-zero progress reports must not change guarded PPO updates."""

import json

import pytest
import torch
from test_ppo_update_guard import make_worker

import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module


def _capture(worker):
    events = []

    def log(message):
        prefix = "PPO_UPDATE_PROGRESS "
        assert message.startswith(prefix)
        events.append(json.loads(message.removeprefix(prefix)))

    worker.log_info = log
    return events


@pytest.mark.parametrize(("enabled", "rank"), [(None, 0), (False, 0), (True, 1)])
def test_progress_is_silent_by_default_and_on_other_ranks(monkeypatch, enabled, rank):
    worker = make_worker(monkeypatch)
    worker._rank = rank
    if enabled is not None:
        worker.cfg.actor.log_optimizer_progress = enabled
    events = _capture(worker)
    assert worker.run_training()["guard/abort"] == 0
    assert not events


def test_progress_records_completed_accept_rollback_and_critic_updates(monkeypatch):
    worker = make_worker(monkeypatch)
    worker.cfg.actor.log_optimizer_progress = True
    worker.cfg.actor.global_batch_size = 2
    worker.gradient_accumulation = 2
    worker.version = 7
    events = _capture(worker)
    result = worker.run_training()
    assert result["guard/abort"] == 0
    assert len(events) == 6
    assert [event["pass"] for event in events] == [1, 1, 2, 2, 3, 3]
    assert [event["minibatch"] for event in events] == [1, 2, 1, 2, 1, 2]
    assert [event["outcome"] for event in events] == [
        "accepted",
        "actor_rollback",
        "critic_only",
        "critic_only",
        "critic_only",
        "critic_only",
    ]
    assert [event["critic_updates"] for event in events] == list(range(1, 7))
    assert [event["optimizer_steps"] for event in events] == list(range(1, 7))
    assert [event["rejected_actor_updates"] for event in events] == [0, 1, 1, 1, 1, 1]
    for event in events:
        assert event["global_step"] == 7
        assert event["passes"] == 3 and event["minibatches"] == 2
        assert event["actor_updates"] == 1
        assert event["critic_lr"] == 0.01
        assert event["grad_norm"] > 0
        assert event["elapsed_training_seconds"] >= 0
        assert event["last_verified_policy_kl"] == events[0]["candidate_post_kl"]
    assert events[0]["actor_lr"] == 0.001
    assert all(event["actor_lr"] == 0 for event in events[1:])
    assert events[0]["candidate_post_kl"] <= 0.02 < events[1]["candidate_post_kl"]
    assert all(event["candidate_post_kl"] is None for event in events[2:])
    elapsed = [event["elapsed_training_seconds"] for event in events]
    assert elapsed == sorted(elapsed)


@pytest.mark.parametrize(("zero_reward", "warmup"), [(True, 0), (False, 3)])
def test_critic_only_progress_does_not_invent_a_post_update_probe(
    monkeypatch, zero_reward, warmup
):
    worker = make_worker(monkeypatch, zero_reward=zero_reward, warmup=warmup)
    worker.cfg.actor.log_optimizer_progress = True
    events = _capture(worker)
    worker.run_training()
    assert len(events) == 3
    for event in events:
        assert event["outcome"] == "critic_only"
        assert event["actor_updates"] == event["rejected_actor_updates"] == 0
        assert event["actor_lr"] == 0 and event["critic_lr"] == 0.01
        assert event["candidate_post_kl"] is None
        assert event["last_verified_policy_kl"] == 0


def test_nonfinite_gradient_reports_valid_json_before_abort(monkeypatch):
    worker = make_worker(monkeypatch)
    worker.cfg.actor.log_optimizer_progress = True
    worker._strategy.clip_grad_norm_ = lambda **kwargs: float("nan")
    events = _capture(worker)
    result = worker.run_training()
    assert result["guard/abort"] == result["guard/nonfinite_gradient"] == 1
    assert len(events) == 1
    assert events[0]["outcome"] == "nonfinite_gradient"
    assert events[0]["grad_norm"] is None
    assert events[0]["actor_updates"] == events[0]["critic_updates"] == 0
    assert events[0]["actor_lr"] == events[0]["critic_lr"] == 0
    assert events[0]["candidate_post_kl"] is None


def test_logging_changes_neither_updates_nor_forward_and_collective_counts(monkeypatch):
    states = []
    for enabled in (False, True):
        worker = make_worker(monkeypatch)
        worker.cfg.actor.log_optimizer_progress = enabled
        _capture(worker)
        calls = {"forward": 0, "reduce": 0}
        original_forward = worker.model.forward

        def forward(**kwargs):
            calls["forward"] += 1
            return original_forward(**kwargs)

        def reduce(values, **kwargs):
            calls["reduce"] += 1
            return values

        worker.model.forward = forward
        monkeypatch.setattr(actor_module, "all_reduce_dict", reduce)
        result = worker.run_training()
        states.append(
            {
                "result": result,
                "model": worker.model.state_dict(),
                "optimizer": worker.optimizer.state_dict(),
                "scheduler": worker.lr_scheduler.state_dict(),
                "calls": calls,
            }
        )
    baseline, logged = states
    assert baseline["result"] == pytest.approx(logged["result"], nan_ok=True)
    assert baseline["calls"] == logged["calls"]
    assert baseline["scheduler"] == logged["scheduler"]
    assert baseline["optimizer"]["param_groups"] == logged["optimizer"]["param_groups"]
    for name, tensor in baseline["model"].items():
        torch.testing.assert_close(logged["model"][name], tensor, rtol=0, atol=0)
    for key, values in baseline["optimizer"]["state"].items():
        for name, tensor in values.items():
            torch.testing.assert_close(
                logged["optimizer"]["state"][key][name], tensor, rtol=0, atol=0
            )
