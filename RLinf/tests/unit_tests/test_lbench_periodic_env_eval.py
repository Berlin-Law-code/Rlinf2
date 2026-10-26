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

"""CPU lifecycle checks for periodic evaluation sharing training GPU memory."""

import asyncio
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.workers.env import env_worker as module


def _config():
    return OmegaConf.create(
        {
            "runner": {"val_check_interval": 5, "only_eval": False},
            "algorithm": {},
            "env": {
                "train": {
                    "env_type": "lbench",
                    "total_num_envs": 16,
                    "rollout_epoch": 4,
                    "max_steps_per_rollout_epoch": 640,
                    "auto_reset": False,
                    "enable_offload": False,
                    "offload_before_eval": True,
                },
                "eval": {
                    "env_type": "lbench",
                    "total_num_envs": 4,
                    "rollout_epoch": 8,
                    "max_steps_per_rollout_epoch": 600,
                    "auto_reset": False,
                    "enable_offload": True,
                    "video_cfg": {"save_video": False},
                },
            },
            "rollout": {
                "group_name": "RolloutGroup",
                "pipeline_stage_num": 1,
                "model": {"num_action_chunks": 20},
            },
            "actor": {"model": {"num_action_chunks": 20}},
        }
    )


@pytest.fixture
def worker_factory(monkeypatch):
    def initialize_transport(worker):
        worker._world_size = 4
        worker._rank = 0
        worker._group_name = "EnvGroup"

    monkeypatch.setattr(module.Worker, "__init__", initialize_transport)
    monkeypatch.setattr(module, "Cluster", lambda: None)
    monkeypatch.setattr(
        module,
        "HybridComponentPlacement",
        lambda *_: SimpleNamespace(get_world_size=lambda _: 4),
    )
    monkeypatch.setattr(module, "get_env_cls", lambda *_: object)

    def make_worker(cfg=None):
        worker = object.__new__(module.EnvWorker)
        module.EnvWorker.__init__(worker, cfg if cfg is not None else _config())
        monkeypatch.setattr(worker, "broadcast", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(worker, "update_env_cfg", lambda: None)
        return worker

    return make_worker


class _FakeEnv:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.is_start = False

    def reset(self):
        self.events.append(f"{self.name}.reset")
        return {"states": torch.zeros(1, 8)}, {}

    def offload(self):
        self.events.append(f"{self.name}.offload")

    def update_reset_state_ids(self):
        self.events.append(f"{self.name}.finish")


def _install_pools(worker, monkeypatch, events, train_has_offload=True):
    def setup(*, env_cls, env_cfg, num_envs_per_stage):
        name = "eval" if env_cfg is worker.cfg.env.eval else "train"
        events.append(f"{name}.create")
        if name == "train" and not train_has_offload:
            return [SimpleNamespace()]
        return [_FakeEnv(name, events)]

    monkeypatch.setattr(worker, "_setup_env_and_wrappers", setup)
    worker.init_worker()


@pytest.mark.parametrize("value", [1, "true", None])
def test_eval_offload_flag_requires_boolean(worker_factory, value):
    cfg = _config()
    cfg.env.train.offload_before_eval = value
    with pytest.raises(ValueError, match="must be a boolean"):
        worker_factory(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("runner.val_check_interval", -1),
        ("runner.only_eval", True),
        ("env.train.auto_reset", True),
    ],
)
def test_eval_offload_requires_explicit_train_and_eval_resets(
    worker_factory, key, value
):
    cfg = _config()
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError, match="requires training and evaluation"):
        worker_factory(cfg)


def test_missing_train_offload_rejected_before_eval_pool_is_created(
    worker_factory, monkeypatch
):
    worker = worker_factory()
    events = []
    with pytest.raises(ValueError, match="offload method"):
        _install_pools(worker, monkeypatch, events, train_has_offload=False)
    assert events == ["train.create"]


def test_periodic_eval_releases_train_before_reset_and_eval_afterward(
    worker_factory, monkeypatch
):
    worker = worker_factory()
    events = []
    _install_pools(worker, monkeypatch, events)
    # Eval initialization can offload a lazy client without closing train.
    assert events == ["train.create", "eval.create", "eval.offload"]
    events.clear()
    worker.n_eval_chunk_steps = 2
    monkeypatch.setattr(worker, "send_to", lambda **_kwargs: None)
    monkeypatch.setattr(worker, "recv_from", lambda **_kwargs: np.zeros((1, 20, 8)))

    def eval_step(_actions, _stage_id):
        events.append("eval.step")
        metrics = (
            {"success_once": torch.tensor([True])}
            if events.count("eval.step") % 2 == 0
            else {}
        )
        return EnvOutput(obs={"states": torch.zeros(1, 8)}), metrics

    monkeypatch.setattr(worker, "env_evaluate_step", eval_step)
    result = inspect.unwrap(module.EnvWorker.evaluate)(worker, None, None)
    assert events == ["train.offload"] + [
        "eval.reset",
        "eval.step",
        "eval.step",
        "eval.finish",
    ] * 8 + ["eval.offload"]
    assert result["success_once"].shape == (8,)


def test_ordinary_training_keeps_pool_when_only_eval_offload_is_enabled(
    worker_factory, monkeypatch
):
    worker = worker_factory()
    events = []
    _install_pools(worker, monkeypatch, events)
    events.clear()

    async def run_interact(*_args, **_kwargs):
        events.append("train.interact")
        return {"success_once": torch.tensor([True])}

    monkeypatch.setattr(worker, "_run_interact_once", run_interact)
    result = asyncio.run(
        inspect.unwrap(module.EnvWorker.interact)(worker, None, None, None)
    )
    assert events == ["train.interact"]
    assert result["success_once"].tolist() == [True]


def test_default_does_not_require_or_close_train_offload(worker_factory, monkeypatch):
    cfg = _config()
    del cfg.env.train.offload_before_eval
    worker = worker_factory(cfg)
    events = []
    _install_pools(worker, monkeypatch, events, train_has_offload=False)
    assert worker.train_offload_before_eval is False
    events.clear()
    worker.n_eval_chunk_steps = 0
    monkeypatch.setattr(worker, "send_to", lambda **_kwargs: None)
    inspect.unwrap(module.EnvWorker.evaluate)(worker, None, None)
    assert events == ["eval.reset", "eval.finish"] * 8 + ["eval.offload"]


def test_prefetched_training_episode_blocks_eval_offload_before_any_mutation(
    worker_factory, monkeypatch
):
    worker = worker_factory()
    events = []
    _install_pools(worker, monkeypatch, events)
    events.clear()
    worker._prefetched_train_bootstrap = [EnvOutput(obs={})]
    with pytest.raises(RuntimeError, match="prefetching"):
        inspect.unwrap(module.EnvWorker.evaluate)(worker, None, None)
    assert not events
