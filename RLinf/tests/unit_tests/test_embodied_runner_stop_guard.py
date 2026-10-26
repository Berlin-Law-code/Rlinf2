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

"""Exercise the synchronous runner loop without Ray, a model, or a simulator."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import rlinf.runners.embodied_runner as runner_module
from rlinf.runners.embodied_runner import EmbodiedRunner


class _Handle:
    def __init__(self, handles, name, result=None, callback=None):
        self.name = name
        self.result = result
        self.callback = callback
        self.wait_count = 0
        handles.append(self)

    def wait(self):
        self.wait_count += 1
        if self.callback is not None:
            self.callback()
        return self.result

    def consume_durations(self, *, return_per_rank):
        assert return_per_rank
        return {}, [{}]


class _Actor:
    worker_group_name = "Actor"

    def __init__(self, handles, rounds):
        self.handles = handles
        self.rounds = rounds
        self.training_calls = 0
        self.saved_steps = []

    def set_global_step(self, step):
        return _Handle(self.handles, "actor.set_step")

    def sync_model_to_rollout(self):
        return _Handle(self.handles, "actor.sync")

    def recv_rollout_trajectories(self, **kwargs):
        return _Handle(self.handles, "actor.recv")

    def compute_advantages_and_returns(self):
        return _Handle(self.handles, "actor.advantages", [{}])

    def run_training(self):
        metrics = self.rounds[self.training_calls]
        self.training_calls += 1
        return _Handle(self.handles, "actor.training", metrics)

    def save_checkpoint(self, path, step):
        def save():
            self.saved_steps.append(step)
            for name in ("weights.pt", "optimizer.pt", "scheduler.pt", "rng.pt"):
                (Path(path) / name).write_bytes(b"CPU fake checkpoint")

        return _Handle(self.handles, "actor.save", callback=save)


class _Rollout:
    worker_group_name = "Rollout"

    def __init__(self, handles):
        self.handles = handles

    def set_global_step(self, step):
        return _Handle(self.handles, "rollout.set_step")

    def sync_model_from_actor(self):
        return _Handle(self.handles, "rollout.sync")

    def generate(self, **kwargs):
        return _Handle(self.handles, "rollout.generate")


class _Env:
    worker_group_name = "Env"

    def __init__(self, handles):
        self.handles = handles

    def interact(self, **kwargs):
        return _Handle(self.handles, "env.interact", [])

    def prefetch_train_bootstrap(self, **kwargs):
        pytest.fail("Guarded stopping must not leave bootstrap work in flight")

    def evaluate(self, **kwargs):
        pytest.fail("Evaluation is disabled for these guarded training runs")


class _Reward:
    worker_group_name = "Reward"

    def __init__(self, handles):
        self.handles = handles

    def compute_rewards(self, **kwargs):
        return _Handle(self.handles, "reward.compute")


def _metrics(successes, *, abort=0, ranks=1):
    return [
        {
            "guard/positive_reward_count": float(successes),
            "guard/zero_reward_batch": float(successes == 0),
            "guard/abort": float(abort),
            "guard/actor_updates": float(successes > 0 and not abort),
            "guard/critic_updates": float(not abort),
        }
        for _ in range(ranks)
    ]


@pytest.fixture
def make_runner(tmp_path, monkeypatch):
    runners = []

    def make(
        rounds,
        *,
        threshold=3,
        enabled=True,
        save_interval=-1,
        max_steps=500,
        overlap=False,
        pipeline=False,
        reward=False,
        runner_cls=EmbodiedRunner,
        eval_before_train=False,
        val_check_interval=-1,
        log_eval_history=False,
    ):
        handles = []
        events = []
        cfg = OmegaConf.create(
            {
                "cluster": {},
                "algorithm": {"ppo_update_guard": {"enabled": enabled}},
                "runner": {
                    "max_epochs": max_steps,
                    "max_steps": max_steps,
                    "weight_sync_interval": 1,
                    "max_consecutive_zero_success": threshold,
                    "max_saved_checkpoints": 3,
                    "val_check_interval": val_check_interval,
                    "eval_before_train": eval_before_train,
                    "log_eval_history": log_eval_history,
                    "save_interval": save_interval,
                    "overlap_env_bootstrap": overlap,
                    "use_training_pipeline": pipeline,
                    "logger": {"log_path": str(tmp_path), "experiment_name": "run"},
                },
            }
        )
        monkeypatch.setattr(runner_module.Channel, "create", lambda name: name)
        monkeypatch.setattr(
            runner_module,
            "ScopedTimer",
            lambda **kwargs: SimpleNamespaceTimer(),
        )
        metrics_logger = SimpleNamespace(
            log_path=str(tmp_path),
            log=lambda data, step, **kwargs: events.append(("metrics", step, data)),
            finish=lambda: events.append(("finish",)),
        )
        monkeypatch.setattr(runner_module, "MetricLogger", lambda cfg: metrics_logger)
        monkeypatch.setattr(
            runner_module,
            "print_metrics_table",
            lambda step, *args: events.append(("table", step)),
        )
        actor = _Actor(handles, rounds)
        runner = runner_cls(
            cfg,
            actor,
            _Rollout(handles),
            _Env(handles),
            reward=_Reward(handles) if reward else None,
        )
        runner.test_handles = handles
        runner.test_events = events
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        if not runner.stop_logging:
            runner._finish_run()


class SimpleNamespaceTimer:
    def __call__(self, *args, **kwargs):
        return nullcontext()

    def consume_durations(self):
        return {}


def _assert_finished(runner, expected_rounds):
    assert runner.global_step == expected_rounds
    assert runner.actor.training_calls == expected_rounds
    assert all(handle.wait_count == 1 for handle in runner.test_handles)
    assert not runner.log_thread.is_alive()
    assert runner.log_queue.unfinished_tasks == 0
    assert runner.test_events[-1] == ("finish",)
    assert [event[1] for event in runner.test_events if event[0] == "table"] == list(
        range(expected_rounds)
    )


def _stop_record(runner):
    return json.loads(
        (Path(runner.cfg.runner.logger.log_path) / "stop_reason.json").read_text()
    )


def test_three_zero_success_rounds_save_full_checkpoint_and_drain_handles(make_runner):
    runner = make_runner([_metrics(0)] * 5, reward=True)
    runner.run()
    _assert_finished(runner, 3)
    assert runner.actor.saved_steps == [3]
    record = _stop_record(runner)
    assert record["reason"] == "max_consecutive_zero_success"
    assert record["positive_reward_count"] == 0
    assert record["max_consecutive_zero_success"] == 3
    assert record["global_step"] == record["consecutive_zero_success"] == 3
    assert not record["automatic_resume"]
    checkpoint = Path(record["checkpoint"])
    assert (checkpoint / ".checkpoint_complete").is_file()
    for name in ("weights.pt", "optimizer.pt", "scheduler.pt", "rng.pt"):
        assert (checkpoint / "actor" / name).is_file()


def test_positive_success_resets_streak_without_summing_global_rank_copies(make_runner):
    runner = make_runner([_metrics(n, ranks=2) for n in (0, 0, 2, 0, 0, 0, 4)])
    runner.run()
    _assert_finished(runner, 6)
    counters = [
        event[2]["train/guard/consecutive_zero_success"]
        for event in runner.test_events
        if event[0] == "metrics" and "train/guard/consecutive_zero_success" in event[2]
    ]
    assert counters == [1, 2, 0, 1, 2, 3]
    assert runner.actor.saved_steps == [6]


@pytest.mark.parametrize(
    "abort_reason",
    ["initial_probability_mismatch", "nonfinite_gradient", "invalid_probability"],
)
def test_any_rank_abort_saves_even_when_zero_success_threshold_is_disabled(
    make_runner, abort_reason
):
    round_metrics = _metrics(2, ranks=2)
    round_metrics[1]["guard/abort"] = 1.0
    round_metrics[1][f"guard/{abort_reason}"] = 1.0
    runner = make_runner([round_metrics], threshold=0)
    runner.run()
    _assert_finished(runner, 1)
    record = _stop_record(runner)
    assert record["reason"] == "actor_guard_abort"
    assert record["positive_reward_count"] == 2  # Already global, not 2 + 2.
    assert record["actor_abort_ranks"] == [1]
    assert record["actor_abort_reasons"] == {"1": [abort_reason]}
    assert runner.actor.saved_steps == [1]


def test_no_positive_rewards_counts_as_zero_success_even_with_negative_rewards(
    make_runner,
):
    metrics = _metrics(0)
    metrics[0]["guard/zero_reward_batch"] = 0.0
    runner = make_runner([metrics], threshold=1)
    runner.run()
    _assert_finished(runner, 1)
    record = _stop_record(runner)
    assert record["reason"] == "max_consecutive_zero_success"
    assert record["positive_reward_count"] == 0
    assert record["zero_reward_batch"] is False


def test_stop_record_retains_probability_cause_and_sanitizes_nonfinite(make_runner):
    metrics = _metrics(33, abort=1, ranks=2)
    for rank in metrics:
        rank.update(
            {
                "guard/invalid_probability": 1.0,
                "guard/abort_pass": 4.0,
                "guard/abort_minibatch": 7.0,
                "guard/abort_nonfinite_logratio_count": 0.0,
                "guard/abort_finite_out_of_range_logratio_count": 1.0,
                "guard/abort_max_abs_logratio_unfiltered": 21.5,
                "guard/abort_kl": float("nan"),
            }
        )
    runner = make_runner([metrics], threshold=0)
    runner.run()
    record = _stop_record(runner)
    cause = record["actor_abort_diagnostics"]["0"]
    assert cause["pass"] == 4
    assert cause["minibatch"] == 7
    assert cause["nonfinite_logratio_count"] == 0
    assert cause["finite_out_of_range_logratio_count"] == 1
    assert cause["max_abs_logratio_unfiltered"] == 21.5
    assert cause["kl"] is None
    assert record["actor_abort_diagnostics"]["1"] == cause
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_zero_success_threshold_runs_all_500_rounds(make_runner, enabled):
    runner = make_runner([_metrics(0)] * 500, threshold=0, enabled=enabled)
    runner.run()
    _assert_finished(runner, 500)
    assert runner.actor.saved_steps == []
    assert not (Path(runner.cfg.runner.logger.log_path) / "stop_reason.json").exists()


def test_periodic_save_at_trigger_round_is_reused(make_runner):
    runner = make_runner([_metrics(n) for n in (1, 1, 0, 0, 0)], save_interval=5)
    runner.run()
    _assert_finished(runner, 5)
    assert runner.actor.saved_steps == [5]
    record = _stop_record(runner)
    assert record["reused_checkpoint"]
    assert record["checkpoint"].endswith("global_step_5")


@pytest.mark.parametrize("corruption", ["missing", "nan", "disagreement"])
def test_invalid_or_inconsistent_guard_metrics_fail_closed(make_runner, corruption):
    metrics = _metrics(0, ranks=2)
    if corruption == "missing":
        del metrics[1]["guard/positive_reward_count"]
    elif corruption == "nan":
        metrics[1]["guard/positive_reward_count"] = float("nan")
    else:
        metrics[1]["guard/positive_reward_count"] = 1.0
    runner = make_runner([metrics])
    runner.run()
    _assert_finished(runner, 1)
    assert _stop_record(runner)["reason"] == "invalid_guard_metrics"
    assert _stop_record(runner)["errors"]
    assert runner.actor.saved_steps == [1]


@pytest.mark.parametrize("threshold", [-1, True, 3.0, "3"])
def test_threshold_rejects_invalid_values(make_runner, threshold):
    with pytest.raises(ValueError, match="nonnegative integer"):
        make_runner([], threshold=threshold)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"enabled": False}, "requires algorithm.ppo_update_guard"),
        ({"overlap": True}, "synchronous training"),
        ({"pipeline": True}, "synchronous training"),
        ({"enabled": "true"}, "must be a boolean"),
    ],
)
def test_guard_requires_explicit_boolean_and_synchronous_no_overlap(
    make_runner, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        make_runner([], **kwargs)


def test_logging_exception_is_drained_before_metric_logger_finishes(make_runner):
    runner = make_runner([], max_steps=0)

    def failing_log():
        runner.test_events.append(("failing_log",))
        raise RuntimeError("table output failed")

    runner.log_queue.put((failing_log, ()))
    runner.log_queue.put((runner.test_events.append, (("last_log",),)))
    runner.run()
    _assert_finished(runner, 0)
    assert runner.test_events == [("failing_log",), ("last_log",), ("finish",)]


def test_runner_with_different_loop_cannot_silently_bypass_guard(make_runner):
    class AsyncRunner(EmbodiedRunner):
        def run(self):
            raise AssertionError("An asynchronous loop must not be launched")

    with pytest.raises(ValueError, match="using EmbodiedRunner.run"):
        make_runner([], runner_cls=AsyncRunner)
