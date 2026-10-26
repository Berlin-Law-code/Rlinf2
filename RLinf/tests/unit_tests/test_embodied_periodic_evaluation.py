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

"""Baseline and periodic evaluation use completed updates and independent saves."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_embodied_runner_stop_guard import (
    _assert_finished,
    _metrics,
)
from test_embodied_runner_stop_guard import (
    make_runner as _make_runner_fixture,
)

import rlinf.utils.timers as timers_module
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.runner_utils import check_progress

make_runner = _make_runner_fixture


@pytest.mark.parametrize(
    "max_steps, val_interval, before_train, expected_evaluations",
    [
        (1, 10, True, [0, 1]),
        (1, 1, True, [0, 1]),
        (3, 2, True, [0, 2, 3]),
        (1, 10, False, [1]),
        (0, 10, True, [0]),
    ],
)
def test_real_timer_consumes_pretrain_eval_before_training_and_final_eval(
    make_runner,
    monkeypatch,
    max_steps,
    val_interval,
    before_train,
    expected_evaluations,
):
    runner = make_runner(
        [_metrics(1)] * max_steps,
        max_steps=max_steps,
        save_interval=1,
        val_check_interval=val_interval,
        eval_before_train=before_train,
        log_eval_history=True,
    )
    # Keep the real duplicate-measurement and consume/reset contract. Only the
    # clock is fake, making timing separation exact without sleeping or CUDA.
    runner.timer = ScopedTimer(reduction="max", sync_cuda=False, trace=False)
    clock = [0.0]
    monkeypatch.setattr(timers_module, "time", SimpleNamespace(time=lambda: clock[0]))
    evaluations = []
    original_training = runner.actor.run_training

    def train():
        clock[0] += 3.0
        return original_training()

    def evaluate():
        evaluations.append(runner.global_step)
        clock[0] += 100.0 if runner.global_step == 0 else 7.0
        return {"success_once": 0.5, "num_trajectories": 32}

    runner.actor.run_training = train
    runner.evaluate = evaluate
    runner.run()

    _assert_finished(runner, max_steps)
    assert evaluations == expected_evaluations
    assert runner.actor.saved_steps == list(range(1, max_steps + 1))
    records = [
        json.loads(line)
        for line in (Path(runner.cfg.runner.logger.log_path) / "eval_history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["completed_rounds"] for record in records] == expected_evaluations
    time_events = [
        event
        for event in runner.test_events
        if event[0] == "metrics" and any(key.startswith("time/") for key in event[2])
    ]
    pretrain_events = [
        event for event in time_events if "time/eval_before_train" in event[2]
    ]
    assert pretrain_events == (
        [("metrics", 0, {"time/eval_before_train": 100.0})] if before_train else []
    )
    step_events = [event for event in time_events if "time/step" in event[2]]
    assert [event[1] for event in step_events] == list(range(max_steps))
    for _, step, metrics in step_events:
        evaluated = step + 1 in expected_evaluations
        assert metrics["time/step"] == 3.0 + (7.0 if evaluated else 0.0)
        assert metrics.get("time/eval", 0.0) == (7.0 if evaluated else 0.0)
        assert "time/eval_before_train" not in metrics
    assert runner.timer.consume_durations() == {}


def test_baseline_every_five_rounds_and_final_eval_keep_every_round_checkpoint(
    make_runner,
):
    runner = make_runner(
        [_metrics(1)] * 12,
        max_steps=12,
        save_interval=1,
        val_check_interval=5,
        eval_before_train=True,
        log_eval_history=True,
    )
    evaluations = []
    original_sync = runner.update_rollout_weights
    synced_after_updates = []

    def sync():
        synced_after_updates.append(runner.actor.training_calls)
        original_sync()

    def evaluate():
        completed = runner.actor.training_calls
        assert runner.global_step == completed
        assert synced_after_updates[-1] == completed
        # Every previous interact must be joined before an eval pool switch.
        assert all(
            handle.wait_count == 1
            for handle in runner.test_handles
            if handle.name == "env.interact"
        )
        evaluations.append(completed)
        return {"success_once": 0.5, "num_trajectories": 32}

    runner.update_rollout_weights = sync
    runner.evaluate = evaluate
    runner.run()
    _assert_finished(runner, 12)
    assert evaluations == [0, 5, 10, 12]
    assert runner.actor.saved_steps == list(range(1, 13))
    path = Path(runner.cfg.runner.logger.log_path) / "eval_history.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["completed_rounds"] for record in records] == evaluations
    assert [record["baseline"] for record in records] == [True, False, False, False]
    assert all(record["metrics"]["eval/num_trajectories"] == 32 for record in records)
    eval_events = [
        event
        for event in runner.test_events
        if event[0] == "metrics" and "eval/success_once" in event[2]
    ]
    assert [event[1] for event in eval_events] == [0, 5, 10, 12]


def test_aborted_actor_is_not_published_for_scheduled_evaluation(make_runner):
    metrics = _metrics(1, abort=1)
    metrics[0]["guard/invalid_probability"] = 1
    runner = make_runner(
        [metrics],
        max_steps=1,
        val_check_interval=5,
        eval_before_train=True,
    )
    evaluations = []
    runner.evaluate = lambda: evaluations.append(runner.global_step) or {}
    runner.run()
    assert evaluations == [0]
    _assert_finished(runner, 1)


def test_evaluation_failure_keeps_the_completed_round_checkpoint(make_runner):
    runner = make_runner(
        [_metrics(1)] * 5,
        max_steps=5,
        save_interval=1,
        val_check_interval=5,
    )

    def failed_evaluation():
        assert runner.actor.saved_steps[-1] == runner.global_step == 5
        raise RuntimeError("simulator evaluation failed")

    runner.evaluate = failed_evaluation
    with pytest.raises(RuntimeError, match="simulator evaluation failed"):
        runner.run()
    checkpoint = Path(runner._last_checkpoint_path)
    assert (checkpoint / ".checkpoint_complete").is_file()
    assert (checkpoint / "actor" / "weights.pt").is_file()


@pytest.mark.parametrize("flag", [1, "true", None])
def test_baseline_flag_must_be_boolean(make_runner, flag):
    with pytest.raises(ValueError, match="eval_before_train must be boolean"):
        make_runner([], eval_before_train=flag, val_check_interval=5)


def test_baseline_requires_evaluation_enabled(make_runner):
    with pytest.raises(ValueError, match="requires enabled evaluation"):
        make_runner([], eval_before_train=True, val_check_interval=-1)


def test_progress_allows_checkpoint_and_eval_at_independent_intervals():
    decisions = [check_progress(step, 12, 5, 1, 1.0) for step in range(1, 13)]
    assert [step + 1 for step, value in enumerate(decisions) if value[0]] == [5, 10, 12]
    assert all(value[1] for value in decisions)
