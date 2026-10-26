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

"""Common initial Gaussian noise for paired synchronous LBench evaluations."""

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class LBenchEvalNoisePlan:
    """Mirror ordered LBench slots across equal env and rollout worker groups."""

    noise_seed: int
    scene_seed: int
    task_index: int
    total_num_envs: int
    fixed_seeds: tuple[int, ...] = ()
    rollout_rank: int = 0
    rollout_world_size: int = 1

    def __post_init__(self):
        for name, value in (
            ("total_num_envs", self.total_num_envs),
            ("rollout_world_size", self.rollout_world_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Paired noise {name} must be a positive integer")
        if (
            isinstance(self.rollout_rank, bool)
            or not isinstance(self.rollout_rank, int)
            or not 0 <= self.rollout_rank < self.rollout_world_size
        ):
            raise ValueError("Paired noise rollout_rank is outside the worker group")
        if self.total_num_envs % self.rollout_world_size:
            raise ValueError("Paired noise requires equally sized ordered env shards")

    def episode_key(self, epoch: int, slot: int) -> tuple[int, int, int]:
        """Return task, scene seed and repeat index for an environment slot."""
        if epoch < 0 or not 0 <= slot < self.total_num_envs:
            raise ValueError("Evaluation epoch/slot is outside the noise schedule")
        ordinal = epoch * self.total_num_envs + slot
        if self.fixed_seeds:
            scene_seed = self.fixed_seeds[ordinal % len(self.fixed_seeds)]
            repeat = ordinal // len(self.fixed_seeds)
        else:
            scene_seed = self.scene_seed + ordinal
            repeat = 0
        return self.task_index, scene_seed, repeat

    def batch_seeds(self, epoch: int, chunk: int, batch_size: int) -> list[int]:
        """Derive this rank's scene/chunk seeds without consuming global RNG."""
        local_num_envs = self.total_num_envs // self.rollout_world_size
        if batch_size != local_num_envs or chunk < 0:
            raise ValueError("Paired noise requires one full, ordered local env batch")
        seeds = []
        slot_start = self.rollout_rank * local_num_envs
        for slot in range(slot_start, slot_start + local_num_envs):
            task, scene, repeat = self.episode_key(epoch, slot)
            key = f"lbench-eval-noise-v1:{self.noise_seed}:{task}:{scene}:{repeat}:{chunk}"
            digest = hashlib.blake2b(key.encode("ascii"), digest_size=8).digest()
            seeds.append(int.from_bytes(digest, "little") & ((1 << 63) - 1))
        return seeds


def build_eval_noise_plan(
    cfg: Any, *, env_world_size: int, rollout_world_size: int, rollout_rank: int = 0
) -> LBenchEvalNoisePlan | None:
    """Validate ordered, synchronous evaluation; the feature defaults to off.

    Repeated noise requires one complete fixed-scene cycle per evaluate call.
    This makes resetting the noise epoch offset agree with the environment's
    continuing episode counters at the next periodic evaluation.
    """
    noise_seed = cfg.rollout.get("eval_initial_noise_seed", None)
    repeat_noise = cfg.rollout.get("eval_repeat_initial_noise", False)
    if not isinstance(repeat_noise, bool):
        raise ValueError("rollout.eval_repeat_initial_noise must be boolean")
    if noise_seed is None:
        if repeat_noise:
            raise ValueError(
                "eval_repeat_initial_noise requires rollout.eval_initial_noise_seed"
            )
        return None
    if (
        isinstance(noise_seed, bool)
        or not isinstance(noise_seed, int)
        or noise_seed < 0
    ):
        raise ValueError(
            "rollout.eval_initial_noise_seed must be a nonnegative integer"
        )
    env = cfg.env.get("eval", {})
    only_eval = cfg.runner.get("only_eval", False)
    model_owner = cfg.rollout if only_eval else cfg.get("actor", {})
    model = model_owner.get("model", {})
    openpi = model.get("openpi", {})
    task_indices = env.get("task_indices", [])
    for name, value in (
        ("env_world_size", env_world_size),
        ("rollout_world_size", rollout_world_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Paired noise {name} must be a positive integer")
    supported = (
        not cfg.runner.get("enable_decoupled_mode", False)
        and not cfg.runner.get("use_training_pipeline", False)
        and not cfg.runner.get("overlap_env_bootstrap", False)
        and env_world_size == rollout_world_size
        and cfg.rollout.get("pipeline_stage_num", 1) == 1
        and not cfg.rollout.get("enable_cuda_graph", False)
        and env.get("env_type") == "lbench"
        and not env.get("auto_reset", True)
        and env.get("task_sampling", "balanced_round_robin") == "balanced_round_robin"
        and len(task_indices) == 1
        and not env.get("task_names", [])
        and model.get("model_type") == "openpi"
        and not openpi.get("use_dsrl", False)
        and not openpi.get("use_rlt", False)
        and not openpi.get("rtc_enabled", False)
    )
    if not supported:
        raise ValueError(
            "Paired initial noise requires synchronous OpenPI/LBench evaluation, "
            "equal env/rollout worker counts, one pipeline stage, one explicit "
            "task_index, balanced_round_robin, auto_reset=false, and no decoupled "
            "routing, training pipeline, overlapping env bootstrap, CUDA graphs, "
            "DSRL, RLT or RTC"
        )
    total_num_envs = env.get("total_num_envs", 0)
    scene_seed = env.get("seed", 0)
    fixed_seeds = tuple(env.get("fixed_seeds", []))
    for name, value in (
        ("total_num_envs", total_num_envs),
        ("seed", scene_seed),
        ("task_index", task_indices[0]),
        *(("fixed_seed", value) for value in fixed_seeds),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Paired evaluation {name} must be a nonnegative integer")
    if total_num_envs == 0 or len(set(fixed_seeds)) != len(fixed_seeds):
        raise ValueError(
            "Paired evaluation needs positive env count and unique fixed_seeds"
        )
    if repeat_noise:
        rollout_epoch = env.get("rollout_epoch", 0)
        if (
            isinstance(rollout_epoch, bool)
            or not isinstance(rollout_epoch, int)
            or rollout_epoch < 1
            or len(fixed_seeds) != total_num_envs * rollout_epoch
        ):
            raise ValueError(
                "eval_repeat_initial_noise requires exactly one unique fixed_seeds "
                "cycle per evaluation: len(fixed_seeds) == total_num_envs * "
                "rollout_epoch, with positive integer rollout_epoch"
            )
    return LBenchEvalNoisePlan(
        noise_seed=noise_seed,
        scene_seed=scene_seed,
        task_index=task_indices[0],
        total_num_envs=total_num_envs,
        fixed_seeds=fixed_seeds,
        rollout_rank=rollout_rank,
        rollout_world_size=rollout_world_size,
    )


def make_eval_initial_noise(
    seeds: Sequence[int], *, batch_size: int, action_horizon: int, action_dim: int
) -> torch.Tensor:
    """Draw CPU float32 N(0, I), independently for each scene/chunk key.

    A private CPU generator makes noise independent of model construction,
    CUDA device, other environment slots and their termination times.
    """
    if len(seeds) != batch_size or batch_size < 1:
        raise ValueError("Initial-noise seeds must match the observation batch size")
    if action_horizon < 1 or action_dim < 1:
        raise ValueError("Initial-noise action dimensions must be positive")
    samples = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("Initial-noise seeds must be integers in [0, 2**63)")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        samples.append(
            torch.randn(
                (action_horizon, action_dim),
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
        )
    return torch.stack(samples)
