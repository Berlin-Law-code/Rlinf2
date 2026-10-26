from __future__ import annotations

import math

import numpy as np
import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import compute_ppo_actor_loss, compute_ppo_critic_loss
from rlinf.envs.lbench.reward import RewardAdapter, RewardConfig
from rlinf.envs.lbench.scheduler import BalancedTaskScheduler
from rlinf.utils.metric_utils import compute_action_loss_mask


def _raw(progress: float, stage: int, success: bool = False, failure: bool = False):
    return {
        "robot": {
            "metric": {
                "progress": progress,
                "current_stage_id": stage,
                "episode_sr": float(success),
                "failure": failure,
            }
        }
    }


def test_progress_reward_is_delta_and_events_fire_once():
    adapter = RewardAdapter(
        1,
        RewardConfig(
            kind="progress_delta",
            progress_scale=2.0,
            stage_bonus=0.5,
            success_bonus=3.0,
            failure_penalty=4.0,
        ),
    )
    adapter.reset([0])
    reward, success, failure = adapter.adapt(
        [_raw(0.2, 0)], [{}], [99.0], [True]
    )
    np.testing.assert_allclose(reward, [0.4])
    assert not success.any() and not failure.any()

    reward, success, _ = adapter.adapt(
        [_raw(0.5, 1, success=True)], [{}], [99.0], [True]
    )
    np.testing.assert_allclose(reward, [4.1])  # delta .6 + stage .5 + success 3
    assert success.tolist() == [True]
    reward, success, _ = adapter.adapt(
        [_raw(0.5, 1, success=True)], [{}], [99.0], [True]
    )
    np.testing.assert_allclose(reward, [0.0])
    assert not success.any()


def test_progress_reward_is_primed_from_nonzero_reset_state():
    adapter = RewardAdapter(
        1,
        RewardConfig(
            kind="progress_delta",
            progress_scale=2.0,
            stage_bonus=0.5,
        ),
    )
    adapter.reset([0], [_raw(0.4, 2)])
    reward, success, failure = adapter.adapt(
        [_raw(0.5, 2)], [{}], [99.0], [True]
    )
    np.testing.assert_allclose(reward, [0.2])
    assert not success.any() and not failure.any()


def test_sparse_reward_and_inactive_slots_have_no_events():
    adapter = RewardAdapter(
        2,
        RewardConfig(kind="sparse_success", success_bonus=2, failure_penalty=3),
    )
    adapter.reset([0, 1])
    rewards, success, failure = adapter.adapt(
        [_raw(1, 1, success=True), _raw(0, 0, failure=True)],
        [{}, {}],
        [10, 10],
        [True, False],
    )
    assert rewards.tolist() == [2.0, 0.0]
    assert success.tolist() == [True, False]
    assert failure.tolist() == [False, False]


def test_libero_sparse_success_matches_rlinf_success_only_reward():
    adapter = RewardAdapter(
        3,
        RewardConfig(kind="libero_sparse_success", success_bonus=2.0),
    )
    adapter.reset([0, 1, 2])

    # Progress, stages, backend rewards, explicit failure and timeout do not
    # contribute reward. Only the first success transition does.
    rewards, success, adapter_failure = adapter.adapt(
        [
            _raw(0.8, 4, success=True),
            _raw(0.6, 3, failure=True),
            _raw(0.9, 5),
        ],
        [
            {"reward_components": {"success": 1.0}},
            {"reward_components": {"failure": 1.0}},
            {"truncated": True},
        ],
        [99.0, 99.0, 99.0],
        [True, True, True],
        timeout_mask=[False, False, True],
    )
    np.testing.assert_allclose(rewards, [2.0, 0.0, 0.0])
    assert success.tolist() == [True, False, False]
    assert adapter_failure.tolist() == [False, False, False]

    repeated, repeated_success, repeated_failure = adapter.adapt(
        [
            _raw(1.0, 5, success=True),
            _raw(0.7, 4, failure=True),
            _raw(1.0, 6),
        ],
        [{}, {}, {}],
        [-100.0, -100.0, -100.0],
        [True, True, True],
        timeout_mask=[False, False, True],
    )
    np.testing.assert_allclose(repeated, [0.0, 0.0, 0.0])
    assert not repeated_success.any()
    assert not repeated_failure.any()


def test_balanced_scheduler_differs_by_at_most_one_across_ranks():
    scheduler = BalancedTaskScheduler([3, 7, 11])
    assert [scheduler.task_for_ordinal(i) for i in range(6)] == [3, 7, 11, 3, 7, 11]
    counts = scheduler.counts(start=5, count=17)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_exact_action_mask_preserves_terminal_action_and_episode_count():
    action_mask = torch.tensor(
        [
            [[True, True, False], [True, True, True]],
            [[False, False, False], [True, False, False]],
        ]
    )
    loss_mask, loss_mask_sum = compute_action_loss_mask(
        action_mask, "chunk_level"
    )
    assert loss_mask.tolist() == [[[True], [True]], [[False], [True]]]
    assert loss_mask_sum[:, 0].unique().tolist() == [2]
    assert loss_mask_sum[:, 1].unique().tolist() == [4]


class TinyFlowPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(4, 3)
        self.value = torch.nn.Linear(4, 1)
        self.log_std = torch.nn.Parameter(torch.full((3,), -0.7))

    def forward(self, obs):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, self.value(obs).squeeze(-1)


def test_tiny_flow_ppo_masked_update_is_finite():
    torch.manual_seed(4)
    policy = TinyFlowPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    obs = torch.randn(6, 4)
    actions = torch.randn(6, 3)
    action_mask = torch.tensor([True, True, True, True, False, False])
    dones = torch.tensor([False, False, False, False, True, True, True])
    rewards = torch.tensor([0.1, 0.0, 0.2, 1.0, 0.0, 0.0])

    with torch.no_grad():
        old_mean, old_std, old_value = policy(obs)
        old_dist = torch.distributions.Normal(old_mean, old_std)
        old_logprob = old_dist.log_prob(actions).sum(-1).float()
        bootstrap_values = torch.cat([old_value, torch.zeros(1)])
        advantages, returns = compute_gae_advantages_and_returns(
            rewards,
            gamma=0.99,
            gae_lambda=0.95,
            values=bootstrap_values,
            normalize_advantages=True,
            loss_mask=action_mask,
            dones=dones,
        )

    mean, std, values = policy(obs)
    logprob = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1).float()
    actor_loss, actor_metrics = compute_ppo_actor_loss(
        logprobs=logprob,
        old_logprobs=old_logprob,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        advantages=advantages.float(),
        loss_mask=action_mask,
    )
    critic_loss, critic_metrics = compute_ppo_critic_loss(
        values=values.float(),
        returns=returns.float(),
        prev_values=old_value.float(),
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=action_mask,
    )
    total = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    all_values = [total, advantages, returns]
    all_values += list(actor_metrics.values()) + list(critic_metrics.values())
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in all_values)
    assert math.isclose(float(actor_metrics["actor/ratio"]), 1.0, rel_tol=1e-5)
    assert policy.actor.weight.grad is not None
    assert torch.isfinite(policy.actor.weight.grad).all()
