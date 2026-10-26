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

"""CPU coverage for paired eval scheduling and the real sampler entrypoints."""

import asyncio
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.lbench import lbench_env
from rlinf.envs.lbench.scheduler import BalancedTaskScheduler
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.utils.eval_initial_noise import (
    LBenchEvalNoisePlan,
    build_eval_noise_plan,
    make_eval_initial_noise,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _config():
    return OmegaConf.create(
        {
            "runner": {"only_eval": True},
            "env": {
                "group_name": "EnvGroup",
                "eval": {
                    "env_type": "lbench",
                    "auto_reset": False,
                    "task_indices": [6],
                    "total_num_envs": 5,
                    "seed": 0,
                },
            },
            "rollout": {
                "eval_initial_noise_seed": 42,
                "pipeline_stage_num": 1,
                "model": {"model_type": "openpi", "openpi": {}},
            },
        }
    )


def test_default_is_off_even_for_other_training_backends():
    assert (
        build_eval_noise_plan(
            OmegaConf.create({"rollout": {}}), env_world_size=3, rollout_world_size=2
        )
        is None
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ("runner.only_eval", False),
        ("runner.enable_decoupled_mode", True),
        ("rollout.pipeline_stage_num", 2),
        ("rollout.enable_cuda_graph", True),
        ("rollout.model.model_type", "openpi_rlinf"),
        ("rollout.model.openpi.use_dsrl", True),
        ("rollout.model.openpi.use_rlt", True),
        ("rollout.model.openpi.rtc_enabled", True),
        ("env.eval.auto_reset", True),
        ("env.eval.task_indices", [6, 7]),
        ("env.eval.fixed_seeds", [1, 1]),
        ("rollout.eval_initial_noise_seed", -1),
    ],
)
def test_unsupported_config_is_rejected(path, value):
    cfg = _config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        build_eval_noise_plan(cfg, env_world_size=1, rollout_world_size=1)


def test_unequal_worker_groups_are_rejected():
    with pytest.raises(ValueError, match="equal env/rollout worker counts"):
        build_eval_noise_plan(_config(), env_world_size=2, rollout_world_size=1)


@pytest.mark.parametrize("fixed_seeds", [[], [13, 4, 81]])
def test_five_env_ten_epoch_keys_match_real_lbench_scheduler(monkeypatch, fixed_seeds):
    cfg = _config()
    cfg.env.eval.fixed_seeds = fixed_seeds
    plan = build_eval_noise_plan(cfg, env_world_size=1, rollout_world_size=1)
    env = object.__new__(lbench_env.LBenchEnv)
    env.cfg = cfg.env.eval
    env.num_envs = 5
    env._worker_id_base = 0
    env._episode_counts = np.zeros(5, dtype=np.int64)
    env._task_indices = [6]
    env._task_scheduler = BalancedTaskScheduler([6])
    env._all_task_names = [f"task_{index}" for index in range(7)]
    env._fixed_seeds = fixed_seeds
    env._base_seed = 0
    env._run_id = "test"
    env.is_eval = True
    env.seed_offset = 0
    monkeypatch.setattr(
        lbench_env,
        "import_lbench_modules",
        lambda _: (None, None, None, SimpleNamespace, None),
    )
    seen = {}
    for epoch in range(10):
        for slot in range(5):
            spec = env._episode_spec(slot)
            repeat = seen.get(spec.seed, 0)
            assert plan.episode_key(epoch, slot) == (spec.task_index, spec.seed, repeat)
            seen[spec.seed] = repeat + 1
            env._episode_counts[slot] += 1
    assert sum(seen.values()) == 50


def test_noise_is_nonzero_normal_reproducible_and_independent_of_global_rng():
    plan = LBenchEvalNoisePlan(42, 0, 6, 5)
    seeds = plan.batch_seeds(0, 0, 5)
    rng_state = torch.random.get_rng_state().clone()
    actual = make_eval_initial_noise(
        seeds, batch_size=5, action_horizon=50, action_dim=32
    )
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state, rtol=0, atol=0)
    expected = torch.randn((50, 32), generator=torch.Generator().manual_seed(seeds[0]))
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    torch.randn(731)
    again = make_eval_initial_noise(
        seeds, batch_size=5, action_horizon=50, action_dim=32
    )
    torch.testing.assert_close(actual, again, rtol=0, atol=0)
    assert torch.count_nonzero(actual) == actual.numel()
    # Noise of the same scene does not depend on which other slots still live.
    single = make_eval_initial_noise(
        [seeds[3]], batch_size=1, action_horizon=50, action_dim=32
    )
    torch.testing.assert_close(actual[3], single[0], rtol=0, atol=0)
    assert len(set(seeds + plan.batch_seeds(0, 1, 5) + plan.batch_seeds(1, 0, 5))) == 15
    repeated = LBenchEvalNoisePlan(42, 0, 6, 1, (7,))
    assert repeated.batch_seeds(0, 0, 1) != repeated.batch_seeds(1, 0, 1)
    assert (
        repeated.batch_seeds(0, 0, 1)
        == LBenchEvalNoisePlan(42, 0, 6, 5, (7,)).batch_seeds(0, 0, 5)[:1]
    )


def _tiny_sampler():
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.action_in_proj = torch.nn.Linear(32, 1, bias=False)
    model.config = OpenPi0Config(action_horizon=50, action_chunk=20, action_env_dim=8)
    model.obs_processor = lambda obs: obs
    model.input_transform = lambda obs, **kwargs: obs
    model.precision_processor = lambda obs: obs
    model.output_transform = lambda output: output
    calls = []

    def sample(observation, **kwargs):
        calls.append(kwargs)
        batch_size = observation.state.shape[0]
        return {
            "actions": kwargs.get("noise", torch.ones(batch_size, 50, 32)),
            "prev_logprobs": torch.zeros(batch_size, 20, 8),
            "prev_values": torch.zeros(batch_size),
            "chains": torch.zeros(batch_size, 1),
            "denoise_inds": torch.zeros(batch_size, 1, dtype=torch.int64),
        }

    model.sample_actions = sample
    return model, calls


def _obs():
    return {
        "state": torch.zeros(5, 32),
        "image": {},
        "image_mask": {},
        "tokenized_prompt": torch.ones(5, 2, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(5, 2, dtype=torch.bool),
    }


def test_real_model_entrypoint_forwards_noise_only_when_explicit_eval():
    model, calls = _tiny_sampler()
    seeds = LBenchEvalNoisePlan(42, 0, 6, 5).batch_seeds(0, 0, 5)
    actions, _ = model.predict_action_batch(_obs(), mode="eval", eval_noise_seeds=seeds)
    torch.testing.assert_close(
        actions,
        make_eval_initial_noise(seeds, batch_size=5, action_horizon=50, action_dim=32),
        rtol=0,
        atol=0,
    )
    model.predict_action_batch(_obs(), mode="train")
    model.predict_action_batch(_obs(), mode="eval")
    assert "noise" not in calls[1] and "noise" not in calls[2]
    with pytest.raises(ValueError, match="plain OpenPI evaluation"):
        model.predict_action_batch(_obs(), mode="train", eval_noise_seeds=seeds)


def test_real_worker_predict_passes_noise_to_model_and_preserves_training():
    calls = []

    def predict_action_batch(**kwargs):
        calls.append(kwargs)
        return torch.zeros(5, 20, 8), {}

    worker = SimpleNamespace(
        _train_sampling_params={},
        _eval_sampling_params={},
        model_cfg=SimpleNamespace(model_type="openpi"),
        enable_dagger=False,
        algorithm_cfg={},
        expert_model=None,
        hf_model=SimpleNamespace(predict_action_batch=predict_action_batch),
    )
    predict = inspect.unwrap(MultiStepRolloutWorker.predict)
    predict(worker, _obs(), mode="eval", eval_noise_seeds=[1, 2, 3, 4, 5])
    predict(worker, _obs(), mode="train")
    assert calls[0]["eval_noise_seeds"] == [1, 2, 3, 4, 5]
    assert "eval_noise_seeds" not in calls[1]


@pytest.mark.parametrize("enabled", [False, True])
def test_real_eval_loop_uses_epoch_slot_and_chunk_without_termination_rng(enabled):
    observed = []
    plan = LBenchEvalNoisePlan(42, 0, 6, 5) if enabled else None

    async def receive():
        return {"obs": {"states": torch.zeros(5, 8)}}

    def predict(env_obs, **kwargs):
        observed.append(kwargs)
        return torch.zeros(5, 20, 8), {}

    worker = SimpleNamespace(
        enable_offload=False,
        env_decoupled_mode=False,
        eval_rollout_epoch=10,
        _rank=0,
        n_eval_chunk_steps=45,
        num_pipeline_stages=1,
        cfg=_config(),
        eval_batch_size=5,
        eval_initial_noise_plan=plan,
        _eval_noise_epoch_offset=0,
        _merge_obs_batches=lambda batches: batches,
        _infer_env_batch_size=lambda obs: obs["states"].shape[0],
        recv_from=lambda **kwargs: SimpleNamespace(async_wait=receive),
        _predict_rollout_actions=predict,
        send_to=lambda **kwargs: None,
    )
    evaluate = inspect.unwrap(MultiStepRolloutWorker.evaluate)
    asyncio.run(evaluate(worker, None, None))
    assert len(observed) == 450 and worker._eval_noise_epoch_offset == 10
    for index, kwargs in enumerate(observed):
        if enabled:
            assert kwargs["eval_noise_seeds"] == plan.batch_seeds(
                index // 45, index % 45, 5
            )
        else:
            assert "eval_noise_seeds" not in kwargs
    # The standalone default keeps its historical continuous epoch schedule.
    asyncio.run(evaluate(worker, None, None))
    assert len(observed) == 900 and worker._eval_noise_epoch_offset == 20
    if enabled:
        assert observed[450]["eval_noise_seeds"] == plan.batch_seeds(10, 0, 5)
        assert observed[450]["eval_noise_seeds"] != observed[0]["eval_noise_seeds"]


def _periodic_config():
    cfg = _config()
    cfg.runner.only_eval = False
    cfg.runner.val_check_interval = 5
    cfg.actor = {"model": OmegaConf.to_container(cfg.rollout.model)}
    # Training rollout.model does not contain the model's architecture options.
    cfg.rollout.model = {"model_path": "unused-in-cpu-test"}
    cfg.rollout.eval_repeat_initial_noise = True
    cfg.env.eval.total_num_envs = 4
    cfg.env.eval.rollout_epoch = 8
    cfg.env.eval.fixed_seeds = list(range(32))
    return cfg


@pytest.mark.parametrize("total_num_envs,rollout_epochs", [(4, 8), (8, 8), (16, 2)])
def test_distributed_ordered_noise_matches_one_worker_and_lbench_scene_scheduler(
    monkeypatch, total_num_envs, rollout_epochs
):
    cfg = _periodic_config()
    cfg.env.eval.total_num_envs = total_num_envs
    cfg.env.eval.rollout_epoch = rollout_epochs
    cfg.env.eval.fixed_seeds = list(range(total_num_envs * rollout_epochs))
    plans = [
        build_eval_noise_plan(
            cfg, env_world_size=4, rollout_world_size=4, rollout_rank=rank
        )
        for rank in range(4)
    ]
    reference = build_eval_noise_plan(cfg, env_world_size=1, rollout_world_size=1)
    legacy_layout = LBenchEvalNoisePlan(42, 0, 6, 4, tuple(cfg.env.eval.fixed_seeds))
    local_num_envs = total_num_envs // 4
    envs = []
    monkeypatch.setattr(
        lbench_env,
        "import_lbench_modules",
        lambda _: (None, None, None, SimpleNamespace, None),
    )
    for rank in range(4):
        env = object.__new__(lbench_env.LBenchEnv)
        env.cfg = cfg.env.eval
        env.num_envs = local_num_envs
        env._worker_id_base = rank * local_num_envs
        env._episode_counts = np.zeros(local_num_envs, dtype=np.int64)
        env._task_indices = [6]
        env._task_scheduler = BalancedTaskScheduler([6])
        env._all_task_names = [f"task_{index}" for index in range(7)]
        env._fixed_seeds = list(cfg.env.eval.fixed_seeds)
        env._base_seed = 0
        env._run_id = "test"
        env.is_eval = True
        env.seed_offset = rank
        envs.append(env)
    observed_scene_seeds = []
    for epoch in range(rollout_epochs):
        for rank, (plan, env) in enumerate(zip(plans, envs)):
            for slot in range(local_num_envs):
                spec = env._episode_spec(slot)
                global_slot = rank * local_num_envs + slot
                assert plan.episode_key(epoch, global_slot) == (
                    spec.task_index,
                    spec.seed,
                    0,
                )
                observed_scene_seeds.append(spec.seed)
                env._episode_counts[slot] += 1
        for chunk in (0, 1, 44):
            distributed = [
                seed
                for plan in plans
                for seed in plan.batch_seeds(epoch, chunk, local_num_envs)
            ]
            assert distributed == reference.batch_seeds(epoch, chunk, total_num_envs)
            # The 16-env x 2-epoch layout must preserve each of the 32 case
            # keys from the original 4-env x 8-epoch layout, despite batching.
            expected_from_legacy_layout = [
                legacy_layout.batch_seeds(ordinal // 4, chunk, 4)[ordinal % 4]
                for ordinal in range(
                    epoch * total_num_envs, (epoch + 1) * total_num_envs
                )
            ]
            assert distributed == expected_from_legacy_layout
    assert observed_scene_seeds == list(range(total_num_envs * rollout_epochs))
    # After one complete cycle the environment scenes repeat even when its
    # episode counters continue. Resetting only the noise epoch now aligns.
    for rank, (plan, env) in enumerate(zip(plans, envs)):
        for slot in range(local_num_envs):
            global_slot = rank * local_num_envs + slot
            assert env._episode_spec(slot).seed == plan.episode_key(0, global_slot)[1]


@pytest.mark.parametrize("repeat_noise", [False, True])
@pytest.mark.parametrize("total_num_envs,rollout_epochs", [(4, 8), (16, 2)])
def test_periodic_eval_hook_repeats_noise_per_case_without_touching_global_rng(
    repeat_noise, total_num_envs, rollout_epochs
):
    cfg = _periodic_config()
    cfg.rollout.eval_repeat_initial_noise = repeat_noise
    cfg.env.eval.total_num_envs = total_num_envs
    cfg.env.eval.rollout_epoch = rollout_epochs
    local_num_envs = total_num_envs // 4
    calls_per_eval = rollout_epochs * 45
    reference = build_eval_noise_plan(cfg, env_world_size=1, rollout_world_size=1)
    legacy_layout = build_eval_noise_plan(
        _periodic_config(), env_world_size=1, rollout_world_size=1
    )
    all_ranks = []
    evaluate = inspect.unwrap(MultiStepRolloutWorker.evaluate)
    for rank in range(4):
        observed = []
        plan = build_eval_noise_plan(
            cfg, env_world_size=4, rollout_world_size=4, rollout_rank=rank
        )

        async def receive():
            return {"obs": {"states": torch.zeros(local_num_envs, 8)}}

        def predict(env_obs, **kwargs):
            seeds = kwargs["eval_noise_seeds"]
            observed.append(seeds)
            # Exercise the actual private generator, not a reseeding stub.
            noise = make_eval_initial_noise(
                seeds, batch_size=local_num_envs, action_horizon=50, action_dim=32
            )
            return noise[:, :20, :8], {}

        worker = SimpleNamespace(
            enable_offload=False,
            env_decoupled_mode=False,
            eval_rollout_epoch=rollout_epochs,
            _rank=rank,
            n_eval_chunk_steps=45,
            num_pipeline_stages=1,
            cfg=cfg,
            eval_batch_size=total_num_envs,
            eval_initial_noise_plan=plan,
            _eval_noise_epoch_offset=0,
            _merge_obs_batches=lambda batches: batches,
            _infer_env_batch_size=lambda obs: obs["states"].shape[0],
            recv_from=lambda **kwargs: SimpleNamespace(async_wait=receive),
            _predict_rollout_actions=predict,
            send_to=lambda **kwargs: None,
        )
        for call_index, version in enumerate((0, 5, 10)):
            worker.version = version
            before = torch.random.get_rng_state().clone()
            asyncio.run(evaluate(worker, None, None))
            torch.testing.assert_close(
                torch.random.get_rng_state(), before, rtol=0, atol=0
            )
            assert worker._eval_noise_epoch_offset == (
                rollout_epochs if repeat_noise else rollout_epochs * (call_index + 1)
            )
        assert len(observed) == 3 * calls_per_eval
        if repeat_noise:
            assert (
                observed[:calls_per_eval]
                == observed[calls_per_eval : 2 * calls_per_eval]
                == observed[2 * calls_per_eval :]
            )
        else:
            assert (
                observed[0] != observed[calls_per_eval] != observed[2 * calls_per_eval]
            )
        all_ranks.append(observed)
    for call_index in range(3):
        for epoch in range(rollout_epochs):
            for chunk in range(45):
                index = call_index * calls_per_eval + epoch * 45 + chunk
                actual = [
                    seed for observations in all_ranks for seed in observations[index]
                ]
                scheduled_epoch = (
                    epoch if repeat_noise else call_index * rollout_epochs + epoch
                )
                assert actual == reference.batch_seeds(
                    scheduled_epoch, chunk, total_num_envs
                )
                assert actual == [
                    legacy_layout.batch_seeds(ordinal // 4, chunk, 4)[ordinal % 4]
                    for ordinal in range(
                        scheduled_epoch * total_num_envs,
                        (scheduled_epoch + 1) * total_num_envs,
                    )
                ]


@pytest.mark.parametrize("rank", [-1, 4, True, 0.5])
def test_invalid_distributed_rank_is_rejected(rank):
    with pytest.raises(ValueError, match="rollout_rank"):
        build_eval_noise_plan(
            _periodic_config(),
            env_world_size=4,
            rollout_world_size=4,
            rollout_rank=rank,
        )


@pytest.mark.parametrize("world", [0, -1, True, 1.5])
def test_invalid_worker_counts_are_rejected(world):
    with pytest.raises(ValueError, match="positive integer"):
        build_eval_noise_plan(
            _periodic_config(), env_world_size=world, rollout_world_size=world
        )


def test_unequal_local_env_batch_and_nondivisible_shards_are_rejected():
    cfg = _periodic_config()
    plan = build_eval_noise_plan(
        cfg, env_world_size=4, rollout_world_size=4, rollout_rank=2
    )
    with pytest.raises(ValueError, match="ordered local env batch"):
        plan.batch_seeds(0, 0, 4)
    cfg.env.eval.total_num_envs = 5
    cfg.env.eval.fixed_seeds = list(range(40))
    with pytest.raises(ValueError, match="equally sized"):
        build_eval_noise_plan(cfg, env_world_size=4, rollout_world_size=4)


@pytest.mark.parametrize(
    "path,value",
    [
        ("runner.enable_decoupled_mode", True),
        ("runner.use_training_pipeline", True),
        ("runner.overlap_env_bootstrap", True),
        ("rollout.enable_cuda_graph", True),
        ("rollout.pipeline_stage_num", 2),
        ("actor.model.model_type", "unsupported"),
        ("env.eval.auto_reset", True),
        ("env.eval.fixed_seeds", []),
        ("env.eval.fixed_seeds", list(range(31))),
        ("env.eval.fixed_seeds", list(range(64))),
        ("env.eval.rollout_epoch", 0),
        ("env.eval.rollout_epoch", True),
        ("env.eval.rollout_epoch", 8.5),
        ("rollout.eval_repeat_initial_noise", 0),
        ("rollout.eval_repeat_initial_noise", "false"),
        ("rollout.eval_repeat_initial_noise", None),
        ("rollout.eval_initial_noise_seed", None),
    ],
)
def test_periodic_noise_rejects_unsafe_routing_and_incomplete_scene_cycles(path, value):
    cfg = _periodic_config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        build_eval_noise_plan(cfg, env_world_size=4, rollout_world_size=4)
