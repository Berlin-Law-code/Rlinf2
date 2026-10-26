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

"""CPU oracles for weighted PPO diagnostics and optional finite KL enforcement."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.ppo_update_guard import (
    add_probability_stats,
    chunk_probability_stats,
    empty_probability_stats,
    probability_limit_exceeded,
    probability_stats_metrics,
    validate_ppo_update_guard_config,
)


def _stats(delta, weights=None, mask=None):
    new = torch.as_tensor(delta, dtype=torch.float32).reshape(-1, 1, 1)
    return chunk_probability_stats(
        new,
        torch.zeros_like(new),
        mask,
        0.2,
        0.2,
        sample_weights=weights,
    )


def test_short_and_long_trajectory_weighting_matches_independent_oracle():
    # One 20-action short trajectory and one 60-action long trajectory with
    # three full chunks. H/L assigns each trajectory total weight 45.
    horizon = 900
    lengths = torch.tensor([20, 60, 60, 60], dtype=torch.float64)
    delta = torch.tensor([math.log(2), 0.0, 0.0, 0.0])
    stats = _stats(delta, horizon / lengths)
    metrics = probability_stats_metrics(stats)
    short_delta = delta[0].item()  # PPO joint logprobs are FP32.
    short_ratio = math.exp(short_delta)
    short_kl = short_ratio - 1 - short_delta
    assert stats["count"] == 4
    assert stats["weight_sum"] == 90
    assert metrics["kl"] == pytest.approx(short_kl / 2)
    assert metrics["unweighted_kl"] == pytest.approx(short_kl / 4)
    assert metrics["ratio"] == pytest.approx((short_ratio + 1) / 2)
    assert metrics["sampled_negative_logratio"] == pytest.approx(-short_delta / 2)
    assert metrics["outside_clip_fraction"] == 0.5
    assert metrics["unweighted_outside_clip_fraction"] == 0.25
    assert metrics["kl"] == pytest.approx(
        metrics["sampled_negative_logratio"] + metrics["ratio"] - 1
    )
    assert metrics["unweighted_kl"] < 0.1 < metrics["kl"]
    assert probability_limit_exceeded(stats, 0.1)
    assert not probability_limit_exceeded(stats, 0.1, enforce_kl_limit=False)


def test_default_weights_reproduce_raw_stats_and_original_decision():
    values = torch.tensor([0.1, -0.3, 0.0, 100.0])
    mask = torch.tensor([[True], [True], [True], [False]])
    implicit = _stats(values, mask=mask)
    explicit = _stats(values, torch.ones(4, 1), mask)
    assert implicit == explicit
    metrics = probability_stats_metrics(implicit)
    expected_kl = sum(math.expm1(x) - x for x in values[:3].tolist()) / 3
    assert metrics["kl"] == pytest.approx(expected_kl)
    for key in ("kl", "ratio", "sampled_negative_logratio", "outside_clip_fraction"):
        assert metrics[key] == metrics["unweighted_" + key]
    assert metrics["weight_sum"] == metrics["valid_count"] == 3
    for target in (0.001, 0.02, 0.1):
        assert probability_limit_exceeded(implicit, target) == (expected_kl > target)


@pytest.mark.parametrize("scale", [1.0, 0.001, 100.0])
def test_positive_weight_rescaling_preserves_metrics_even_below_unit_total(scale):
    delta = [0.2, -0.1]
    metrics = probability_stats_metrics(_stats(delta, torch.tensor([3.0, 1.0]) * scale))
    baseline = probability_stats_metrics(_stats(delta, torch.tensor([3.0, 1.0])))
    for key in ("kl", "ratio", "sampled_negative_logratio", "outside_clip_fraction"):
        assert metrics[key] == pytest.approx(baseline[key])
    assert metrics["weight_sum"] == pytest.approx(4 * scale)


@pytest.mark.parametrize("padding_weight", [0.0, float("nan"), float("inf"), -10.0])
def test_padding_invalid_weights_and_probabilities_are_completely_ignored(
    padding_weight,
):
    stats = _stats(
        [0.1, float("nan")],
        torch.tensor([2.0, padding_weight]),
        torch.tensor([[True], [False]]),
    )
    baseline = _stats([0.1], torch.tensor([2.0]))
    assert stats == baseline
    assert stats["invalid_count"] == 0


@pytest.mark.parametrize("bad_weight", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_valid_weight_rejects_without_hiding_raw_probability(bad_weight):
    stats = _stats([0.3, 0.0], torch.tensor([bad_weight, 1.0]))
    assert stats["count"] == 2
    assert stats["invalid_count"] == 1
    assert stats["max_abs_logratio"] == pytest.approx(0.3)
    assert stats["kl_sum"] > 0
    assert stats["weighted_kl_sum"] == 0
    assert probability_limit_exceeded(stats, 0.02)
    assert probability_limit_exceeded(stats, 0.02, enforce_kl_limit=False)
    assert all(math.isfinite(x) for x in probability_stats_metrics(stats).values())


@pytest.mark.parametrize("invalid_delta", [21.0, float("nan"), float("inf")])
def test_invalid_probability_remains_fault_at_tiny_positive_weight(invalid_delta):
    stats = _stats([invalid_delta, 0.0], torch.tensor([1e-15, 1000.0]))
    assert stats["invalid_count"] == 1
    assert probability_limit_exceeded(stats, 0.02, enforce_kl_limit=False)


def test_same_sample_invalid_probability_and_weight_counted_once():
    stats = _stats([float("nan")], torch.tensor([float("nan")]))
    assert stats["count"] == 1
    assert stats["invalid_count"] == 1
    assert stats["weight_sum"] == 0


def test_all_padding_produces_zero_additive_stats_and_no_fault():
    stats = _stats(
        [float("nan"), float("inf")],
        torch.tensor([float("nan"), 0.0]),
        torch.zeros(2, 1, dtype=torch.bool),
    )
    assert stats == empty_probability_stats()
    assert all(value == 0 for value in probability_stats_metrics(stats).values())
    assert not probability_limit_exceeded(stats, 0.02)


@pytest.mark.parametrize("cuts", [[0, 7], [0, 1, 4, 7], [0, 2, 3, 4, 7]])
def test_split_and_additive_rank_reduction_matches_whole_batch_and_python_oracle(cuts):
    delta = torch.tensor([0.3, -0.4, 0.15, 1.0, float("nan"), 0.0, -0.2])
    weights = torch.tensor([45.0, 15.0, 3.0, 2.0, float("nan"), 0.0, 7.0])
    mask = torch.tensor([True, True, True, True, False, False, True]).reshape(-1, 1)
    whole = _stats(delta, weights, mask)
    total = empty_probability_stats()
    for begin, end in zip(cuts[:-1], cuts[1:]):
        add_probability_stats(
            total, _stats(delta[begin:end], weights[begin:end], mask[begin:end])
        )
    assert total == pytest.approx(whole, rel=1e-14, abs=1e-14)
    metrics = probability_stats_metrics(total)
    ids = (0, 1, 2, 3, 6)
    weight_sum = sum(weights[index].item() for index in ids)
    kl_sum = sum(
        weights[index].item() * (math.expm1(delta[index].item()) - delta[index].item())
        for index in ids
    )
    assert metrics["kl"] == pytest.approx(kl_sum / weight_sum)
    assert metrics["valid_count"] == len(ids)
    assert metrics["weight_sum"] == weight_sum


@pytest.mark.parametrize("shape", [(), (2, 2), (1, 2)])
def test_weights_require_one_scalar_per_chunk(shape):
    with pytest.raises(ValueError, match="one scalar per chunk"):
        _stats([0.1, 0.2], torch.ones(shape))


def test_reduction_overflow_is_a_fault_even_without_finite_kl_enforcement():
    stats = _stats([0.0], torch.tensor([1.0]))
    stats["weight_sum"] = float("inf")
    assert probability_limit_exceeded(stats, 0.02, enforce_kl_limit=False)


def _guard_config():
    return OmegaConf.create(
        {
            "runner": {},
            "algorithm": {
                "ppo_update_guard": {
                    "enabled": True,
                    "target_kl": 0.02,
                    "max_initial_logratio": 0.01,
                    "probe_batch_size": 64,
                    "skip_actor_on_zero_reward": True,
                },
                "loss_type": "actor_critic",
                "adv_type": "gae",
                "logprob_type": "chunk_level",
                "reward_type": "chunk_level",
            },
            "actor": {
                "micro_batch_size": 32,
                "model": {
                    "model_type": "openpi",
                    "openpi": {
                        "train_expert_only": True,
                        "value_after_vlm": True,
                        "detach_critic_input": True,
                        "strict_ppo_precision": True,
                    },
                },
                "fsdp_config": {
                    "sharding_strategy": "no_shard",
                    "use_orig_params": True,
                },
            },
            "env": {"train": {"auto_reset": False, "ignore_terminations": False}},
        }
    )


@pytest.mark.parametrize("setting", [None, True, False])
def test_kl_limit_enforcement_is_opt_out_and_does_not_mutate_config(setting):
    cfg = _guard_config()
    if setting is not None:
        cfg.algorithm.ppo_update_guard.enforce_kl_limit = setting
    original = OmegaConf.to_container(cfg)
    actual = validate_ppo_update_guard_config(cfg)
    assert actual["enforce_kl_limit"] is (True if setting is None else setting)
    assert actual["enabled"] is True
    assert OmegaConf.to_container(cfg) == original


@pytest.mark.parametrize("setting", [None, 0, 1, "false", "true", 0.1])
def test_kl_limit_enforcement_requires_an_actual_boolean(setting):
    cfg = _guard_config()
    cfg.algorithm.ppo_update_guard.enforce_kl_limit = setting
    with pytest.raises(ValueError, match="enforce_kl_limit must be boolean"):
        validate_ppo_update_guard_config(cfg)
