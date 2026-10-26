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

"""Worker initialization ordering without Ray, model allocation or simulation."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner


class _Handle:
    def __init__(self, events, name, on_wait=None):
        self.events = events
        self.name = name
        self.on_wait = on_wait

    def wait(self):
        self.events.append(f"wait:{self.name}")
        if self.on_wait is not None:
            self.on_wait()


class _Group:
    def __init__(self, name, events, initialized, fail_at=None):
        self.name = name
        self.events = events
        self.initialized = initialized
        self.fail_at = fail_at

    def init_worker(self):
        name = f"{self.name}.init"
        self.events.append(name)
        if self.fail_at == "call":
            raise RuntimeError("actor initialization failed")

        def finish():
            if self.fail_at == "wait":
                raise RuntimeError("actor initialization failed")
            self.initialized.add(self.name)

        return _Handle(self.events, name, finish)


def _runner(*, flag="absent", offload=True, reward=False, fail_at=None):
    events, initialized = [], set()
    cfg = {"runner": {}, "actor": {"enable_offload": offload}}
    if flag != "absent":
        cfg["runner"]["actor_first_initialization"] = flag
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(cfg)
    runner.actor = _Group("actor", events, initialized, fail_at)
    runner.rollout = _Group("rollout", events, initialized)
    runner.env = _Group("env", events, initialized)
    runner.reward = _Group("reward", events, initialized) if reward else None
    runner.global_step = 0
    runner.logger = SimpleNamespace(info=lambda message: None)
    return runner, events, initialized


@pytest.mark.parametrize("flag", ["absent", False])
@pytest.mark.parametrize("reward", [False, True])
def test_default_preserves_original_call_and_wait_order(flag, reward):
    runner, events, initialized = _runner(flag=flag, offload=False, reward=reward)
    runner.init_workers()
    expected = ["rollout.init", "env.init"]
    if reward:
        expected += ["reward.init", "wait:reward.init"]
    expected += ["wait:rollout.init", "wait:env.init", "actor.init", "wait:actor.init"]
    assert events == expected
    assert initialized == (
        {"actor", "env", "rollout", "reward"} if reward else {"actor", "env", "rollout"}
    )


@pytest.mark.parametrize("reward", [False, True])
def test_actor_finishes_before_any_other_group_starts_and_initializes_once(reward):
    runner, events, initialized = _runner(flag=True, reward=reward)
    runner.init_workers()
    expected = ["actor.init", "wait:actor.init", "rollout.init", "env.init"]
    if reward:
        expected += ["reward.init", "wait:reward.init"]
    expected += ["wait:rollout.init", "wait:env.init"]
    assert events == expected
    assert events.count("actor.init") == 1
    assert events.count("wait:actor.init") == 1
    assert {"actor", "env", "rollout"}.issubset(initialized)


@pytest.mark.parametrize("flag", [1, 0, "true", None])
def test_non_boolean_flag_is_rejected_before_any_worker_starts(flag):
    runner, events, _ = _runner(flag=flag)
    with pytest.raises(ValueError, match="must be a boolean"):
        runner.init_workers()
    assert events == []


@pytest.mark.parametrize("offload", [False, None, "true", 1])
def test_actor_first_requires_explicit_actor_offload(offload):
    runner, events, _ = _runner(flag=True, offload=offload)
    with pytest.raises(ValueError, match="requires actor.enable_offload=true"):
        runner.init_workers()
    assert events == []


def test_missing_actor_offload_is_rejected_before_any_worker_starts():
    runner, events, _ = _runner(flag=True)
    del runner.cfg.actor.enable_offload
    with pytest.raises(ValueError, match="requires actor.enable_offload=true"):
        runner.init_workers()
    assert events == []


@pytest.mark.parametrize("fail_at", ["call", "wait"])
def test_actor_failure_does_not_start_rollout_reward_or_simulators(fail_at):
    runner, events, initialized = _runner(flag=True, reward=True, fail_at=fail_at)
    with pytest.raises(RuntimeError, match="actor initialization failed"):
        runner.init_workers()
    assert events == (
        ["actor.init"] if fail_at == "call" else ["actor.init", "wait:actor.init"]
    )
    assert initialized == set()


@pytest.mark.parametrize("actor_first", [False, True])
def test_resume_runs_after_all_initialization_and_waits_for_actor_load(
    tmp_path, actor_first
):
    runner, events, initialized = _runner(flag=actor_first, reward=True)
    checkpoint = tmp_path / "global_step_5"
    actor_checkpoint = checkpoint / "actor"
    actor_checkpoint.mkdir(parents=True)
    runner.cfg.runner.resume_dir = str(checkpoint)
    checkpoint_loaded = []

    def load_checkpoint(path):
        assert initialized == {"actor", "env", "rollout", "reward"}
        assert path == str(actor_checkpoint)
        events.append("actor.load")
        return _Handle(events, "actor.load", lambda: checkpoint_loaded.append(True))

    def restore_rollout_progress(step):
        assert initialized == {"actor", "env", "rollout", "reward"}
        assert checkpoint_loaded == [True]
        assert step == 5
        events.append("env.restore")
        return _Handle(events, "env.restore")

    runner.actor.load_checkpoint = load_checkpoint
    runner.env.restore_rollout_progress = restore_rollout_progress
    runner.init_workers()
    assert events[-4:] == [
        "actor.load",
        "wait:actor.load",
        "env.restore",
        "wait:env.restore",
    ]
    assert events.count("actor.init") == 1
    assert runner.global_step == 5
