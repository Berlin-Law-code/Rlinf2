# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU fake-client contracts for releasing and restarting isolated Isaac pools."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_lbench_env import FakeVectorClient, env_cfg

from rlinf.envs.lbench import lbench_env as module
from rlinf.envs.lbench.lbench_env import LBenchEnv


class LifecycleClient(FakeVectorClient):
    def __init__(self, num_envs, *, tasks=None, list_error=False, reset_error=False):
        super().__init__(num_envs)
        self.tasks = tasks or ["task_a", "task_b"]
        self.list_error = list_error
        self.reset_error = reset_error
        self.list_calls = 0
        self.close_calls = 0

    def list_tasks(self):
        self.list_calls += 1
        if self.list_error:
            raise RuntimeError("task discovery failed")
        return self.tasks

    def reset(self, **kwargs):
        if self.reset_error:
            raise RuntimeError("worker reset failed")
        return super().reset(**kwargs)

    def close(self):
        self.close_calls += 1
        super().close()


@pytest.fixture(autouse=True)
def pure_protocol(monkeypatch):
    monkeypatch.setattr(
        module,
        "import_lbench_modules",
        lambda cfg: (None, None, None, SimpleNamespace, None),
    )
    monkeypatch.setattr(
        module,
        "map_action_batch",
        lambda actions, cfg: [{"joint_position": row.tolist()} for row in actions],
    )


def config(total=2):
    cfg = env_cfg(total)
    cfg.update(auto_reset=False, lazy_start=True, task_indices=[1])
    return cfg


def finish(env):
    env.chunk_step(np.zeros((env.num_envs, env.num_envs, 8), dtype=np.float32))
    assert not env._episode_active.any()


def install_factory(monkeypatch, clients):
    calls = []
    remaining = iter(clients)

    def factory(cfg, num_envs, worker_id_base):
        calls.append((copy.deepcopy(cfg), num_envs, worker_id_base))
        value = next(remaining)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(module, "create_vector_client", factory)
    return calls


def test_lazy_construction_and_init_offload_do_not_start_isaac(monkeypatch):
    client = LifecycleClient(2)
    calls = install_factory(monkeypatch, [client])
    env = LBenchEnv(config(), 2, 0, 1, None)
    assert calls == []
    assert env.seed == 10
    assert env.task_descriptions == ["", ""]
    env.offload()
    env.offload()
    assert calls == []
    assert env._all_task_names is None
    obs, _ = env.reset()
    assert len(calls) == client.list_calls == 1
    assert obs["states"].shape == (2, 8)
    assert env._task_indices == [1]
    env.close()


def test_default_constructor_still_starts_client_eagerly(monkeypatch):
    cfg = config()
    del cfg["lazy_start"]
    client = LifecycleClient(2)
    calls = install_factory(monkeypatch, [client])
    env = LBenchEnv(cfg, 2, 0, 1, None)
    assert len(calls) == client.list_calls == 1
    env.close()
    assert client.close_calls == 1


def test_restart_preserves_task_seed_counters_camera_and_worker_assignment(monkeypatch):
    first, second = LifecycleClient(2), LifecycleClient(2)
    calls = install_factory(monkeypatch, [first, second])
    cfg = config(4)
    cfg.update(isaac_cuda_visible_devices=[0, 1], isaac_gpu_assignment="env_rank")
    env = LBenchEnv(cfg, 2, 1, 2, None)
    before, _ = env.reset()
    assert [first.specs[i].seed for i in range(2)] == [12, 13]
    first_ids = [first.specs[i].episode_id for i in range(2)]
    finish(env)
    env.offload()
    env.offload()
    assert first.close_calls == 1
    assert env._client is None and env._current_raw_obs == [None, None]
    assert not env._closed
    np.testing.assert_array_equal(env._episode_counts, [1, 1])
    after, _ = env.reset()
    assert [second.specs[i].seed for i in range(2)] == [16, 17]
    assert [second.specs[i].task_index for i in range(2)] == [1, 1]
    assert all(second.specs[i].episode_id != first_ids[i] for i in range(2))
    assert calls[0] == calls[1]
    assert calls[0][1:] == (2, 2)
    for key in ("main_images", "wrist_images", "extra_view_images", "states"):
        if key in before:
            np.testing.assert_array_equal(before[key], after[key])
    env.close()
    assert second.close_calls == 1


def test_eval_fixed_seed_cycle_continues_across_offload_without_duplicate_ids(
    monkeypatch,
):
    first, second = LifecycleClient(2), LifecycleClient(2)
    install_factory(monkeypatch, [first, second])
    cfg = config()
    cfg["fixed_seeds"] = [0, 1, 2, 3]
    env = LBenchEnv(cfg, 2, 0, 1, None)
    env.reset()
    initial_ids = [first.specs[i].episode_id for i in range(2)]
    assert [first.specs[i].seed for i in range(2)] == [0, 1]
    finish(env)
    env.offload()
    env.reset()
    assert [second.specs[i].seed for i in range(2)] == [2, 3]
    finish(env)
    env.reset()
    assert [second.specs[i].seed for i in range(2)] == [0, 1]
    assert all(second.specs[i].episode_id != initial_ids[i] for i in range(2))
    env.close()


def test_lazy_restore_progress_does_not_launch_client_or_rewind_on_offload(monkeypatch):
    client = LifecycleClient(2)
    calls = install_factory(monkeypatch, [client])
    cfg = config()
    cfg.update(is_eval=False, rollout_epoch=4)
    env = LBenchEnv(cfg, 2, 0, 1, None)
    assert env.restore_rollout_progress(3) == 12
    env.offload()
    assert calls == []
    env.reset()
    assert [client.specs[i].seed for i in range(2)] == [34, 35]
    env.close()


def test_offload_refuses_active_episode_and_autoreset(monkeypatch):
    client = LifecycleClient(2)
    install_factory(monkeypatch, [client])
    env = LBenchEnv(config(), 2, 0, 1, None)
    env.reset()
    with pytest.raises(RuntimeError, match="active episodes"):
        env.offload()
    assert not client.closed
    finish(env)
    env.auto_reset = True
    with pytest.raises(RuntimeError, match="auto_reset=false"):
        env.offload()
    assert not client.closed
    env.close()


@pytest.mark.parametrize("start_first", [False, True])
def test_close_is_permanent_and_idempotent(monkeypatch, start_first):
    client = LifecycleClient(2)
    calls = install_factory(monkeypatch, [client])
    env = LBenchEnv(config(), 2, 0, 1, None)
    if start_first:
        env.reset()
    env.close()
    env.close()
    env.offload()
    assert len(calls) == int(start_first)
    assert client.close_calls == int(start_first)
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(np.zeros((2, 8)))
    with pytest.raises(RuntimeError, match="closed"):
        env.chunk_step(np.zeros((2, 1, 8)))


def test_step_cannot_restart_pool_or_reuse_observations_after_offload(monkeypatch):
    client = LifecycleClient(2)
    calls = install_factory(monkeypatch, [client])
    env = LBenchEnv(config(), 2, 0, 1, None)
    for method, actions in (
        (env.step, np.zeros((2, 8))),
        (env.chunk_step, np.zeros((2, 1, 8))),
    ):
        with pytest.raises(RuntimeError, match="Reset LBench"):
            method(actions)
    assert calls == []
    env.reset()
    finish(env)
    env.offload()
    with pytest.raises(RuntimeError, match="Reset LBench"):
        env.step(np.zeros((2, 8)))
    assert len(calls) == 1
    env.close()


@pytest.mark.parametrize("failure", ["create", "list", "reset"])
def test_failed_lazy_start_releases_client_and_can_retry_without_advancing(
    monkeypatch, failure
):
    bad = LifecycleClient(
        2, list_error=failure == "list", reset_error=failure == "reset"
    )
    good = LifecycleClient(2)
    install_factory(
        monkeypatch,
        [RuntimeError("create failed") if failure == "create" else bad, good],
    )
    env = LBenchEnv(config(), 2, 0, 1, None)
    with pytest.raises(RuntimeError, match="failed"):
        env.reset()
    assert bad.close_calls == int(failure != "create")
    assert env._client is None and env._current_raw_obs == [None, None]
    np.testing.assert_array_equal(env._episode_counts, [0, 0])
    env.reset()
    assert [good.specs[i].seed for i in range(2)] == [10, 11]
    env.close()


def test_restart_rejects_catalog_change_and_releases_new_pool(monkeypatch):
    first = LifecycleClient(2)
    changed = LifecycleClient(2, tasks=["task_b", "task_a"])
    install_factory(monkeypatch, [first, changed])
    env = LBenchEnv(config(), 2, 0, 1, None)
    env.reset()
    finish(env)
    env.offload()
    with pytest.raises(RuntimeError, match="catalog changed"):
        env.reset()
    assert changed.close_calls == 1
    assert env._all_task_names == ["task_a", "task_b"]
    np.testing.assert_array_equal(env._episode_counts, [1, 1])
    env.close()


def test_partial_restart_is_rejected_before_creating_pool(monkeypatch):
    calls = install_factory(monkeypatch, [])
    env = LBenchEnv(config(), 2, 0, 1, None)
    with pytest.raises(ValueError, match="reset all envs"):
        env.reset(env_ids=[0])
    assert calls == []
    env.close()


def test_invalid_lazy_flag_releases_injected_client():
    cfg = config()
    cfg["lazy_start"] = "false"
    client = LifecycleClient(2)
    with pytest.raises(TypeError, match="lazy_start must be boolean"):
        LBenchEnv(cfg, 2, 0, 1, None, vector_client=client)
    assert client.close_calls == 1
