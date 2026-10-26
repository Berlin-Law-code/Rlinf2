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

"""Opt-in probability checks and reversible updates for synchronous OpenPI PPO."""

import copy
import math
from typing import Any

import torch


def validate_ppo_update_guard_config(cfg: Any) -> dict[str, Any]:
    """Validate the supported training path rather than silently ignore controls."""
    raw = cfg.algorithm.get("ppo_update_guard", {})
    if not raw.get("enabled", False):
        return {"enabled": False}
    allowed = {
        "enabled",
        "enforce_kl_limit",
        "target_kl",
        "max_initial_logratio",
        "probe_batch_size",
        "skip_actor_on_zero_reward",
        "loss_weighting",
    }
    if set(raw) - allowed:
        raise ValueError(f"Unknown PPO update guard fields: {set(raw) - allowed}")
    result = dict(raw)
    result.setdefault("enforce_kl_limit", True)
    if not isinstance(result["enforce_kl_limit"], bool):
        raise ValueError("ppo_update_guard.enforce_kl_limit must be boolean")
    for key in ("target_kl", "max_initial_logratio"):
        value = result.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value < 20
        ):
            raise ValueError(f"ppo_update_guard.{key} must be finite and in (0, 20)")
    count = result.get("probe_batch_size")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("ppo_update_guard.probe_batch_size must be a positive integer")
    if not isinstance(result.get("skip_actor_on_zero_reward"), bool):
        raise ValueError("ppo_update_guard.skip_actor_on_zero_reward must be boolean")
    result.setdefault("loss_weighting", "valid_chunk")
    if result["loss_weighting"] not in ("valid_chunk", "libero_trajectory"):
        raise ValueError(
            "ppo_update_guard.loss_weighting must be valid_chunk or libero_trajectory"
        )
    micro_batch_size = cfg.actor.micro_batch_size
    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size < 1
    ):
        raise ValueError(
            "PPO update guard requires a positive integer micro_batch_size"
        )
    if result["loss_weighting"] == "libero_trajectory":
        # The official actor/critic H/L objective is linear in the padded
        # minibatch slots. Its separate entropy masked-mean is not, so do not
        # silently promise microbatch invariance for a nonzero entropy bonus.
        if cfg.algorithm.entropy_bonus != 0:
            raise ValueError(
                "libero_trajectory loss weighting requires entropy_bonus=0"
            )
        horizon = cfg.env.train.max_episode_steps
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError(
                "libero_trajectory loss weighting requires positive integer "
                "max_episode_steps"
            )
    model = cfg.actor.model
    fsdp = cfg.actor.fsdp_config
    requirements = {
        "synchronous runner": not cfg.runner.get("use_training_pipeline", False),
        "no overlapping environment bootstrap": not cfg.runner.get(
            "overlap_env_bootstrap", False
        ),
        "every-round weight sync": cfg.runner.get("weight_sync_interval", 1) == 1,
        "OpenPI model": model.model_type == "openpi",
        "actor_critic loss": cfg.algorithm.loss_type == "actor_critic",
        "GAE": cfg.algorithm.adv_type == "gae",
        "chunk-level probabilities": cfg.algorithm.logprob_type == "chunk_level",
        "chunk-level rewards": cfg.algorithm.reward_type == "chunk_level",
        "non-autoreset episodes": not cfg.env.train.auto_reset,
        "native episode termination": not cfg.env.train.ignore_terminations,
        "FSDP NO_SHARD": fsdp.sharding_strategy == "no_shard",
        "original optimizer parameters": fsdp.use_orig_params,
        "expert-only policy": model.openpi.train_expert_only,
        "independent VLM critic": model.openpi.value_after_vlm
        and model.openpi.detach_critic_input,
        "strict FP32 computations": model.openpi.get("strict_ppo_precision", False),
        "no SFT co-training": not cfg.actor.get("enable_sft_co_train", False),
        "no reward filtering": not cfg.algorithm.get("filter_rewards", False),
    }
    for name, valid in requirements.items():
        if not valid:
            raise ValueError(f"PPO update guard requires {name}")
    return result


PROBABILITY_MAX_STAT_KEYS = frozenset(
    {"max_abs_logratio", "max_abs_logratio_unfiltered"}
)


def empty_probability_stats() -> dict[str, float]:
    """Return sufficient statistics; only the named maxima use MAX reduction."""
    return {
        "count": 0.0,
        "kl_sum": 0.0,
        "ratio_sum": 0.0,
        "logratio_sum": 0.0,
        "outside_clip_count": 0.0,
        "weight_sum": 0.0,
        "weighted_kl_sum": 0.0,
        "weighted_ratio_sum": 0.0,
        "weighted_logratio_sum": 0.0,
        "weighted_outside_clip_count": 0.0,
        "invalid_count": 0.0,
        "nonfinite_logratio_count": 0.0,
        "finite_out_of_range_logratio_count": 0.0,
        "extreme_positive_logratio_count": 0.0,
        "extreme_negative_logratio_count": 0.0,
        "invalid_weight_count": 0.0,
        "max_abs_logratio": 0.0,
        "max_abs_logratio_unfiltered": 0.0,
    }


def make_probability_probe(batch: dict, size: int) -> dict:
    """Select valid chunks and mask padding to keep rank forward counts equal.

    All ranks must supply equally sized rollout shards. Even a rank without
    valid chunks participates in each FSDP forward; its padded samples add
    nothing to the globally reduced probability statistics.
    """
    count = batch["prev_logprobs"].shape[0]
    if count < 1 or size < 1:
        raise ValueError(
            "Probability probes require a nonempty shard and positive size"
        )
    size = min(size, count)
    valid = batch["loss_mask"].reshape(count, -1).any(-1)
    selected = valid.nonzero(as_tuple=True)[0][:size]
    indices = torch.zeros(size, dtype=torch.long, device=selected.device)
    indices[: selected.numel()] = selected

    def select(value):
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        if torch.is_tensor(value):
            return value.index_select(0, indices.to(value.device))
        return value

    probe = select(batch)
    probe["loss_mask"][selected.numel() :] = False
    return probe


@torch.no_grad()
def chunk_probability_stats(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    loss_mask: torch.Tensor | None,
    clip_low: float,
    clip_high: float,
    *,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measure valid, already action-masked joint probabilities in FP64.

    The joint sums match the FP32 PPO loss. k3 = exp(logratio)-1-logratio
    is a nonnegative sampled estimate, not exact final-action or reference KL.
    ``sample_weights`` supplies one positive finite weight per valid chunk, for
    example the loss's episode horizon/length weight. Raw sums/counts remain
    available alongside additive weighted sums. Padding weights are ignored,
    including NaN or zero sentinels. Invalid valid weights or probabilities
    increment ``invalid_count`` for a collective decision by the caller.
    Diagnostic counts distinguish nonfinite deltas, finite deltas outside
    [-20, 20], and invalid weights; these causes may overlap on one chunk, so
    their sum need not equal the union ``invalid_count``. The unfiltered maximum
    includes every finite valid delta, even if its weight is invalid. The older
    ``max_abs_logratio`` continues to describe deltas inside [-20, 20].
    """
    batch = logprobs.shape[0]
    if old_logprobs.shape != logprobs.shape:
        raise ValueError("PPO new/old logprobs must have identical shapes")
    valid = torch.ones(batch, dtype=torch.bool, device=logprobs.device)
    if loss_mask is not None:
        valid = loss_mask.reshape(batch, -1).any(dim=-1)
    if sample_weights is None:
        weights = torch.ones(batch, dtype=torch.float64, device=logprobs.device)
    else:
        if (
            sample_weights.ndim < 1
            or sample_weights.shape[0] != batch
            or sample_weights.numel() != batch
        ):
            raise ValueError("PPO sample_weights must have one scalar per chunk")
        weights = sample_weights.to(
            device=logprobs.device, dtype=torch.float64
        ).reshape(batch)
    weights = weights[valid]
    delta = (
        logprobs.float().reshape(batch, -1).sum(-1)
        - old_logprobs.float().reshape(batch, -1).sum(-1)
    )[valid].double()
    result = empty_probability_stats()
    result["count"] = float(delta.numel())
    if not delta.numel():
        return result
    finite_weights = torch.isfinite(weights) & (weights > 0)
    result["weight_sum"] = float(weights[finite_weights].sum().item())
    finite_delta = torch.isfinite(delta)
    in_range = delta.abs() <= 20
    finite = finite_delta & in_range
    result["nonfinite_logratio_count"] = float((~finite_delta).sum().item())
    result["finite_out_of_range_logratio_count"] = float(
        (finite_delta & ~in_range).sum().item()
    )
    result["extreme_positive_logratio_count"] = float(
        (finite_delta & (delta > 20)).sum().item()
    )
    result["extreme_negative_logratio_count"] = float(
        (finite_delta & (delta < -20)).sum().item()
    )
    result["invalid_weight_count"] = float((~finite_weights).sum().item())
    if finite_delta.any():
        result["max_abs_logratio_unfiltered"] = float(
            delta[finite_delta].abs().max().item()
        )
    result["invalid_count"] = float((~(finite & finite_weights)).sum().item())
    weights = weights[finite]
    finite_weights = finite_weights[finite]
    delta = delta[finite]
    if delta.numel():
        ratio = delta.exp()
        kl = (torch.expm1(delta) - delta).clamp_min(0)
        outside_clip = (ratio < 1 - clip_low) | (ratio > 1 + clip_high)
        result.update(
            kl_sum=float(kl.sum().item()),
            ratio_sum=float(ratio.sum().item()),
            logratio_sum=float(delta.sum().item()),
            outside_clip_count=float(outside_clip.sum().item()),
            max_abs_logratio=float(delta.abs().max().item()),
        )
        weights = weights[finite_weights]
        for key, values in (
            ("weighted_kl_sum", kl),
            ("weighted_ratio_sum", ratio),
            ("weighted_logratio_sum", delta),
            ("weighted_outside_clip_count", outside_clip),
        ):
            result[key] = float((weights * values[finite_weights]).sum().item())
    return result


def add_probability_stats(total: dict, sample: dict) -> None:
    """Combine microbatches without averaging padding into valid samples."""
    for key, value in sample.items():
        if key in PROBABILITY_MAX_STAT_KEYS:
            total[key] = max(total[key], value)
        else:
            total[key] += value


def probability_stats_metrics(stats: dict[str, float]) -> dict[str, float]:
    """Normalize weighted primary metrics and retain raw chunk diagnostics.

    Sum these sufficient statistics across microbatches/ranks before calling.
    Do not average per-microbatch means when weights or valid counts differ.
    """
    denominator = max(stats["count"], 1.0)
    weight_denominator = stats["weight_sum"] if stats["weight_sum"] > 0 else 1.0
    return {
        "kl": stats["weighted_kl_sum"] / weight_denominator,
        "ratio": stats["weighted_ratio_sum"] / weight_denominator,
        "sampled_negative_logratio": -stats["weighted_logratio_sum"]
        / weight_denominator,
        "outside_clip_fraction": stats["weighted_outside_clip_count"]
        / weight_denominator,
        "unweighted_kl": stats["kl_sum"] / denominator,
        "unweighted_ratio": stats["ratio_sum"] / denominator,
        "unweighted_sampled_negative_logratio": -stats["logratio_sum"] / denominator,
        "unweighted_outside_clip_fraction": stats["outside_clip_count"] / denominator,
        "weight_sum": stats["weight_sum"],
        "valid_count": stats["count"],
        "invalid_count": stats["invalid_count"],
        "nonfinite_logratio_count": stats["nonfinite_logratio_count"],
        "finite_out_of_range_logratio_count": stats[
            "finite_out_of_range_logratio_count"
        ],
        "extreme_positive_logratio_count": stats["extreme_positive_logratio_count"],
        "extreme_negative_logratio_count": stats["extreme_negative_logratio_count"],
        "invalid_weight_count": stats["invalid_weight_count"],
        "max_abs_logratio": stats["max_abs_logratio"],
        "max_abs_logratio_unfiltered": stats["max_abs_logratio_unfiltered"],
    }


def probability_limit_exceeded(
    stats: dict[str, float], target_kl: float, *, enforce_kl_limit: bool = True
) -> bool:
    """Reject numerical faults; optionally enforce the weighted sampled KL limit.

    Disabling the finite KL limit is diagnostic mode, not a bypass of invalid
    probabilities or invalid weights. Reduction overflow is also a fault.
    """
    if stats["invalid_count"] > 0 or not all(
        math.isfinite(value) for value in stats.values()
    ):
        return True
    return enforce_kl_limit and (
        stats["weighted_kl_sum"] > target_kl * stats["weight_sum"]
    )


class ActorOptimizerSnapshot:
    """CPU snapshot of actor parameters and Adam state; critic stays independent.

    Snapshot only a live, NO_SHARD/use_orig_params optimizer after backward.
    Restore after a rejected update before publishing weights or saving a checkpoint.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.entries = []
        self.group_steps = []
        for group in optimizer.param_groups:
            if group.get("rlinf_role") != "actor":
                continue
            self.group_steps.append((group, group.get("rlinf_optimizer_steps", 0)))
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = optimizer.state.get(parameter, {})
                saved = {
                    key: value.detach().to("cpu", copy=True)
                    if torch.is_tensor(value)
                    else copy.deepcopy(value)
                    for key, value in state.items()
                }
                self.entries.append(
                    (parameter, parameter.detach().to("cpu", copy=True), saved)
                )
        if not self.group_steps:
            raise ValueError("Guarded PPO requires an optimizer actor parameter group")

    @torch.no_grad()
    def restore(self) -> None:
        """Undo actor parameter/state changes, including Adam's step counters."""
        for parameter, value, saved in self.entries:
            parameter.copy_(value)
            current = self.optimizer.state[parameter]
            for key in set(current) - set(saved):
                del current[key]
            for key, old in saved.items():
                if torch.is_tensor(old):
                    if key in current:
                        current[key].copy_(old)
                    else:
                        current[key] = old.to(parameter.device, copy=True)
                else:
                    current[key] = copy.deepcopy(old)
        for group, steps in self.group_steps:
            group["rlinf_optimizer_steps"] = steps
