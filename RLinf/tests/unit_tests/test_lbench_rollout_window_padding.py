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

"""CPU contract for 900 physical steps inside a 960-step rollout window."""

import copy

import numpy as np
import pytest
import torch
from test_lbench_env import FakeVectorClient, env_cfg

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.envs.lbench.lbench_env import LBenchEnv
from rlinf.utils.metric_utils import compute_action_loss_mask


class _LongEpisodeClient(FakeVectorClient):
    """No success; optionally terminate a slot before the adapter's timeout."""

    def __init__(self, num_envs, terminal_steps=None):
        super().__init__(num_envs)
        self.terminal_steps = terminal_steps or {}

    def step(self, commands):
        self.commands.append(copy.deepcopy(commands))
        executed = self.active.copy()
        terminated = [False] * self.num_envs
        for env_id in range(self.num_envs):
            if not executed[env_id]:
                continue
            self.worker_step_counts[env_id] += 1
            self.steps[env_id] += 1
            terminal_step = self.terminal_steps.get(env_id)
            terminated[env_id] = (
                terminal_step is not None and self.steps[env_id] >= terminal_step
            )
        return (
            [{} for _ in range(self.num_envs)],
            [0.0] * self.num_envs,
            terminated,
            [False] * self.num_envs,
            [{"transition_executed": bool(value)} for value in executed],
            copy.deepcopy(self.raw),
        )


def _config(num_envs):
    cfg = env_cfg(num_envs)
    cfg.update(
        auto_reset=False,
        ignore_terminations=False,
        max_episode_steps=900,
        max_steps_per_rollout_epoch=960,
        rollout_epoch=8,
        task_indices=[0],
        seed=1000,
        is_eval=False,
    )
    return cfg


def _collect(env, chunk=10):
    masks, rewards = [], []
    dones = [torch.zeros((env.num_envs, chunk), dtype=torch.bool)]
    for _ in range(960 // chunk):
        _, reward, terminated, truncated, infos = env.chunk_step(
            np.zeros((env.num_envs, chunk, 8), dtype=np.float32)
        )
        masks.append(infos[-1]["action_mask"].clone())
        rewards.append(reward.clone())
        dones.append(terminated | truncated)
    return torch.stack(masks), torch.stack(rewards), torch.stack(dones)


@pytest.fixture(scope="module", params=[10, 20])
def padded_rollout(request):
    client = _LongEpisodeClient(2, terminal_steps={0: 300})
    env = LBenchEnv(_config(2), 2, 0, 1, None, vector_client=client)
    try:
        env.reset()
        masks, rewards, dones = _collect(env, request.param)
        return masks, rewards, dones, client
    finally:
        env.close()


def test_padding_has_no_backend_steps_rewards_or_valid_actions(padded_rollout):
    masks, rewards, dones, client = padded_rollout
    chunk = masks.shape[-1]
    cap_chunks = 900 // chunk
    assert len(client.commands) == 900
    assert client.worker_step_counts.tolist() == [300, 900]
    assert masks.sum(dim=(0, 2)).tolist() == [300, 900]
    assert masks[:cap_chunks, 1].all()
    assert not masks[cap_chunks:].any()
    assert not rewards[cap_chunks:].any()
    assert dones[300 // chunk, 0, -1] and dones[cap_chunks, 1, -1]
    assert dones.sum().item() == 2
    assert not client.active.any()


@pytest.mark.parametrize("env_id,valid_steps", [(0, 300), (1, 900)])
def test_terminal_prefix_gae_matches_without_padding(
    padded_rollout, env_id, valid_steps
):
    masks, rewards, dones, _ = padded_rollout
    valid_chunks = valid_steps // masks.shape[-1]
    chunk_rewards = rewards.sum(-1)[:, env_id : env_id + 1]
    chunk_dones = dones.any(-1)[:, env_id : env_id + 1]
    chunk_mask = masks.any(-1)[:, env_id : env_id + 1]
    # Arbitrary, large values in the unexecuted tail cannot cross the terminal.
    values = torch.linspace(-2.0, 100.0, masks.shape[0] + 1)[:, None]
    kwargs = {"gamma": 0.99, "gae_lambda": 0.95, "normalize_advantages": False}
    advantages, returns = compute_gae_advantages_and_returns(
        chunk_rewards,
        values=values,
        dones=chunk_dones,
        loss_mask=chunk_mask,
        **kwargs,
    )
    prefix_advantages, prefix_returns = compute_gae_advantages_and_returns(
        chunk_rewards[:valid_chunks],
        values=values[: valid_chunks + 1],
        dones=chunk_dones[: valid_chunks + 1],
        loss_mask=chunk_mask[:valid_chunks],
        **kwargs,
    )
    torch.testing.assert_close(
        advantages[:valid_chunks], prefix_advantages, rtol=0, atol=0
    )
    torch.testing.assert_close(returns[:valid_chunks], prefix_returns, rtol=0, atol=0)
    assert returns[valid_chunks - 1].item() == 0.0


def test_loss_lengths_count_executed_actions_and_keep_episode_cap_denominator(
    padded_rollout,
):
    masks, _, _, _ = padded_rollout
    chunk = masks.shape[-1]
    slots = 960 // chunk
    loss_mask, loss_mask_sum = compute_action_loss_mask(masks, "chunk_level")
    assert loss_mask.shape == loss_mask_sum.shape == (slots, 2, 1)
    assert torch.all(loss_mask_sum[:, 0] == 300)
    assert torch.all(loss_mask_sum[:, 1] == 900)
    assert loss_mask.sum(dim=(0, 2)).tolist() == [300 // chunk, 900 // chunk]
    logprobs = torch.zeros((slots, 2, 1), requires_grad=True)
    loss, _ = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.ones_like(logprobs),
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
        max_episode_steps=900,
    )
    loss.backward()
    # Official episode weighting uses all padded slots in the outer mean.
    # At fixed parameters, a single whole-rollout mean scales by 900/960.
    assert loss.item() == pytest.approx(-900 / 960)
    torch.testing.assert_close(
        logprobs.grad.sum(dim=(0, 2)), torch.full((2,), -900 / 960 / 2)
    )
    assert not logprobs.grad[~loss_mask].any()


@pytest.mark.parametrize("chunk,rollout_epochs", [(10, 8), (20, 4)])
def test_padding_does_not_advance_resumed_episode_cursor(chunk, rollout_epochs):
    cfg = _config(32)
    cfg["rollout_epoch"] = rollout_epochs
    client = _LongEpisodeClient(8)
    env = LBenchEnv(cfg, 8, 2, 4, None, vector_client=client)
    try:
        restored_episodes = 5 * rollout_epochs
        assert env.restore_rollout_progress(5) == restored_episodes
        env.reset()
        first_seeds = [client.specs[index].seed for index in range(8)]
        first_ids = [client.specs[index].episode_id for index in range(8)]
        first_seed = 1000 + restored_episodes * 32 + 2 * 8
        assert first_seeds == list(range(first_seed, first_seed + 8))
        _collect(env, chunk)
        assert env._episode_counts.tolist() == [restored_episodes + 1] * 8
        assert client.worker_step_counts.tolist() == [900] * 8
        assert [client.specs[index].episode_id for index in range(8)] == first_ids
        env.reset()
        assert [client.specs[index].seed for index in range(8)] == [
            seed + 32 for seed in first_seeds
        ]
        assert env._episode_counts.tolist() == [restored_episodes + 2] * 8
    finally:
        env.close()
