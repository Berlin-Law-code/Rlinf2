# Copyright 2025 The RLinf Authors.
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

import json
import math
import time

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.ppo_update_guard import (
    PROBABILITY_MAX_STAT_KEYS,
    ActorOptimizerSnapshot,
    add_probability_stats,
    chunk_probability_stats,
    empty_probability_stats,
    make_probability_probe,
    probability_limit_exceeded,
    probability_stats_metrics,
    validate_ppo_update_guard_config,
)
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_types import Trajectory, convert_trajectories_to_batch
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp import FSDP
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import (
    all_reduce_dict,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_action_loss_mask,
    compute_critic_explained_variance_from_stats,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import (
    flatten_nested_tensor_time_batch,
    process_nested_dict_for_adv,
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
    trim_nested_tensor_time_dim,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    masked_mean,
    reshape_entropy,
)


def _mask_unexecuted_action_logprobs(
    logprobs: torch.Tensor,
    action_mask: torch.Tensor | None,
    action_dim: int,
) -> torch.Tensor:
    """Zero log-prob terms for sub-actions the environment did not execute."""

    if action_mask is None:
        return logprobs
    if action_mask.ndim != 2:
        raise ValueError(
            f"micro-batch action_mask must be [B, C], got {action_mask.shape}"
        )
    batch_size, action_chunk = action_mask.shape
    if logprobs.shape[0] != batch_size:
        raise ValueError(
            f"action_mask/logprobs batch mismatch: {batch_size} != {logprobs.shape[0]}"
        )
    expected_elements = batch_size * action_chunk * int(action_dim)
    if logprobs.numel() != expected_elements:
        raise ValueError(
            "action_mask cannot be aligned to action logprobs: "
            f"mask={tuple(action_mask.shape)}, logprobs={tuple(logprobs.shape)}, "
            f"action_dim={action_dim}"
        )
    shaped = logprobs.reshape(batch_size, action_chunk, int(action_dim))
    return torch.where(
        action_mask.bool().unsqueeze(-1), shaped, torch.zeros_like(shaped)
    ).reshape_as(logprobs)


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        validate_ppo_update_guard_config(cfg)
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self._opd_teacher_model = None
        self.rollout_batch = None
        self._rollout_batch_in_use = False
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            if self.cfg.actor.get("rollout_state_dict_cpu_offload", False):
                # FSDP's first state-dict/forward initializes CUDA streams and
                # checks parameter devices. Complete that lazy initialization
                # before offloading; this context does not clone a state dict.
                with FSDP.summon_full_params(self.model, writeback=False):
                    pass
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        # With many resident Isaac environments, stage bucket-sync weights on
        # CPU instead of keeping a second complete model copy on the GPU.
        cpu_offload = self.cfg.actor.get("rollout_state_dict_cpu_offload", False)
        if cpu_offload and self.cfg.weight_syncer.type != "bucket":
            raise ValueError("CPU rollout state dict staging requires bucket sync")
        return self.get_model_state_dict(cpu_offload=cpu_offload, full_state_dict=False)

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        cpu_staging = self.cfg.actor.get("rollout_state_dict_cpu_offload", False)
        if cpu_staging:
            fsdp_config = self.cfg.actor.fsdp_config
            if not (
                self.enable_offload
                and self.cfg.weight_syncer.type == "bucket"
                and fsdp_config.get("strategy") == "fsdp"
                and fsdp_config.get("sharding_strategy") == "no_shard"
                and fsdp_config.get("use_orig_params", False)
            ):
                raise ValueError(
                    "CPU rollout staging requires actor offload, bucket sync, "
                    "fsdp, no_shard and use_orig_params"
                )
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()
            if not self.is_weight_offloaded:
                # Export from CPU views: NO_SHARD state_dict otherwise clones
                # the full model on GPU before copying it to CPU. In particular,
                # do not retain actor weights/cache while the colocated rollout
                # allocates its IPC receive buffers after an update.
                self.offload_param_and_grad(True)
            if not self._is_weight_sender:
                # NO_SHARD already has complete, synchronized parameters on
                # rank 0. Other ranks need only release their training memory;
                # exporting/cloning and packing a full CPU state dict here
                # creates unused copies and can exhaust host memory.
                return
        elif self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        # Drop the transfer copy before allocating CPU offload storage below.
        del state_dict

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad(True)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        if self._rollout_batch_in_use:
            raise RuntimeError("Cannot receive a rollout batch during actor training.")
        # Drop any previous batch before receiving and concatenating the next one.
        self.rollout_batch = None
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        rollout_batch = convert_trajectories_to_batch(
            recv_list, copy_single_trajectory=False
        )
        # The collocated layout receives one trajectory per actor. Reuse its
        # storage and release receive-side ownership before reordering fields.
        recv_list.clear()
        del trajectory
        self.rollout_batch = self._process_received_rollout_batch(rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.env.train.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(
            rollout_batch, rollout_epoch, consume=True
        )

        if rollout_batch.get("action_mask") is not None:
            loss_mask, loss_mask_sum = compute_action_loss_mask(
                rollout_batch["action_mask"], self.cfg.algorithm.reward_type
            )
            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum
        elif (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        if self.cfg.algorithm.adv_type == "opd":
            self.compute_opd_teacher_logprobs()

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "prev_logprobs": self.rollout_batch.get("prev_logprobs", None),
            "teacher_logprobs": self.rollout_batch.get("teacher_logprobs", None),
            "num_action_chunks": self.cfg.actor.model.num_action_chunks,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "advantage_mode": self.cfg.algorithm.get("advantage_mode", None),
        }

        if "normalize_advantages" in self.cfg.algorithm:
            kwargs["normalize_advantages"] = self.cfg.algorithm.normalize_advantages
        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        diagnostic_dir = self.cfg.actor.get("rollout_diagnostics_dir", None)
        if diagnostic_dir:
            from rlinf.envs.wrappers.episode_diagnostics import write_ppo_batch

            write_ppo_batch(
                diagnostic_dir, self.rollout_batch, int(self.version), self._rank
            )
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    @Worker.timer("actor/compute_opd_teacher_logprobs")
    def compute_opd_teacher_logprobs(self) -> None:
        assert self.rollout_batch.get("teacher_logprobs", None) is None, (
            "OPD teacher_logprobs must be computed after rollout on actor workers."
        )
        assert self.cfg.rollout.get("expert_model", None) is not None, (
            "OPD requires rollout.expert_model as teacher model config."
        )
        assert "forward_inputs" in self.rollout_batch, (
            "OPD teacher logprob computation requires rollout forward_inputs."
        )
        assert "prev_logprobs" in self.rollout_batch, (
            "OPD teacher logprob computation requires student prev_logprobs."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ], "OPD teacher logprob computation currently supports OpenVLA models."

        prev_logprobs = self.rollout_batch["prev_logprobs"]
        time_dim, batch_dim = prev_logprobs.shape[:2]
        flat_batch_size = time_dim * batch_dim

        assert self.enable_offload and self.is_weight_offloaded, (
            "OPD teacher logprob computation expects actor weights to be "
            "offloaded before moving the teacher model to GPU."
        )
        teacher_model = self._get_opd_teacher_model()
        teacher_model.to(self.device)

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            self.rollout_batch["forward_inputs"], ("forward_inputs",)
        )
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        teacher_logprobs = []
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    teacher_output = teacher_model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                teacher_logprobs.append(teacher_output["logprobs"].detach().cpu())

        teacher_logprobs = torch.cat(teacher_logprobs, dim=0)
        expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
        assert teacher_logprobs.shape == expected_shape, (
            f"teacher_logprobs shape {teacher_logprobs.shape} must match "
            f"flattened student logprobs shape {expected_shape}."
        )
        self.rollout_batch["teacher_logprobs"] = teacher_logprobs.reshape(
            time_dim, batch_dim, *teacher_logprobs.shape[1:]
        )

        teacher_model.to("cpu")
        clear_memory()

    def _get_opd_teacher_model(self):
        if self._opd_teacher_model is None:
            teacher_model_config = build_expert_model_config(
                self.cfg, self.cfg.actor.model
            )
            teacher_model = get_model(teacher_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                teacher_model_dict = torch.load(
                    self.cfg.runner.expert_ckpt_path, map_location="cpu"
                )
                teacher_model.load_state_dict(teacher_model_dict)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._opd_teacher_model = teacher_model
        return self._opd_teacher_model

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        sft_loss = self.model(
            data=(observation, actions),
            forward_type=ForwardType.SFT,
        )
        metrics_data["sft_loss"] = sft_loss.detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )
        return loss

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, float]:
        """Update on the received batch, releasing it after all epochs and metrics."""
        if self._rollout_batch_in_use:
            raise RuntimeError("Actor training is already using the rollout batch.")
        if self.rollout_batch is None:
            raise RuntimeError("Receive a rollout batch before actor training.")
        self._rollout_batch_in_use = True
        try:
            return self._run_training_on_rollout_batch()
        finally:
            # The helper's local minibatch views have also gone out of scope on
            # success. Keep cleanup out of train_micro_batch: pipeline workers
            # reuse those batches for later update epochs and rollout metrics.
            self.rollout_batch = None
            self._rollout_batch_in_use = False
            clear_memory(clear_host_cache=True)

    def _run_training_on_rollout_batch(self) -> dict[str, float]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        if self.cfg.algorithm.loss_type == "opd":
            target_steps = int(self.rollout_batch["advantages"].shape[0])
            for key in [
                "prev_logprobs",
                "forward_inputs",
                "loss_mask",
                "loss_mask_sum",
            ]:
                assert key in self.rollout_batch, f"OPD training requires {key}."
                self.rollout_batch[key] = trim_nested_tensor_time_dim(
                    self.rollout_batch[key], target_steps, (key,)
                )

        guard = validate_ppo_update_guard_config(self.cfg)
        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        shuffle_version = getattr(self, "version", 0) if guard["enabled"] else 0
        g.manual_seed(self.cfg.actor.seed + self._rank + shuffle_version)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id, consume=True
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        if guard["enabled"]:
            return self._run_guarded_updates(guard, rollout_size, batch_size_per_rank)

        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                    )
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        return mean_metric_dict

    def _reduce_probability_stats(self, stats: dict[str, float]) -> dict[str, float]:
        maximum = all_reduce_dict(
            {
                key: value
                for key, value in stats.items()
                if key in PROBABILITY_MAX_STAT_KEYS
            },
            dtype=torch.float64,
            op=torch.distributed.ReduceOp.MAX,
        )
        result = all_reduce_dict(
            {
                key: value
                for key, value in stats.items()
                if key not in PROBABILITY_MAX_STAT_KEYS
            },
            dtype=torch.float64,
            op=torch.distributed.ReduceOp.SUM,
        )
        return {**result, **maximum}

    @torch.no_grad()
    def _measure_policy_probe(self, probe: dict) -> dict[str, float]:
        """Recompute a fixed set of valid chunks, without retaining activations."""
        total = empty_probability_stats()
        for batch in split_dict_to_chunk(probe, probe["prev_logprobs"].shape[0]):
            result = self.train_micro_batch(
                batch,
                {},
                is_last=True,
                probability_only=True,
                loss_weighting=self.cfg.algorithm.get("ppo_update_guard", {}).get(
                    "loss_weighting", "valid_chunk"
                ),
            )
            add_probability_stats(total, result)
        return self._reduce_probability_stats(total)

    def _run_guarded_updates(
        self,
        guard: dict,
        rollout_size: int,
        batch_size_per_rank: int,
    ) -> dict[str, float]:
        """Update with microbatch-invariant losses and sampled KL checks.

        All decisions are reduced across actor ranks before changing optimizer
        state. The fixed probe is a sample of this rollout, not a guarantee on
        every possible state or on the final ODE action distribution.

        ``valid_chunk`` averages over globally valid chunks (the old default).
        ``libero_trajectory`` uses sum(mask * loss * H / L) / B, where H is the
        episode cap, L the episode's executed action count, and B the global
        padded minibatch size. This is the official actor/critic aggregation,
        not exact episode-equal weighting: terminal chunks can be partial and
        rollout-window padding remains in B. Both objectives are independent
        of microbatch partitioning, apart from floating-point roundoff.
        """
        log_progress = self._rank == 0 and self.cfg.actor.get(
            "log_optimizer_progress", False
        )
        training_started = time.monotonic() if log_progress else None
        loss_weighting = guard.get("loss_weighting", "valid_chunk")
        enforce_kl_limit = guard.get("enforce_kl_limit", True)
        masks = self.rollout_batch["loss_mask"].reshape(rollout_size, -1).any(-1)
        rewards = self.rollout_batch["rewards"]
        reward_mask = self.rollout_batch.get("action_mask")
        if reward_mask is None:
            reward_mask = masks.unsqueeze(-1).expand_as(rewards)
        valid_rewards = rewards[reward_mask.bool() & masks.unsqueeze(-1)]
        invalid_targets = sum(
            int((~torch.isfinite(self.rollout_batch[key][masks])).sum().item())
            for key in ("advantages", "returns", "prev_values")
        )
        invalid_episode_lengths = 0
        if loss_weighting == "libero_trajectory":
            lengths = self.rollout_batch.get("loss_mask_sum")
            if not torch.is_tensor(lengths) or lengths.numel() != rollout_size:
                invalid_episode_lengths = 1
            else:
                valid_lengths = lengths.reshape(-1)[masks]
                invalid_episode_lengths = int(
                    ((~torch.isfinite(valid_lengths)) | (valid_lengths <= 0))
                    .sum()
                    .item()
                )
        counts = all_reduce_dict(
            {
                "valid_chunks": float(masks.sum().item()),
                "positive_rewards": float((valid_rewards > 0).sum().item()),
                "nonzero_rewards": float((valid_rewards != 0).sum().item()),
                "invalid_rewards": float((~torch.isfinite(valid_rewards)).sum().item()),
                "invalid_targets": float(invalid_targets),
                "invalid_episode_lengths": float(invalid_episode_lengths),
            },
            dtype=torch.float64,
            op=torch.distributed.ReduceOp.SUM,
        )
        metrics = {
            "guard/valid_chunks": counts["valid_chunks"],
            "guard/positive_reward_count": counts["positive_rewards"],
            "guard/zero_reward_batch": float(counts["nonzero_rewards"] == 0),
            "guard/actor_updates": 0.0,
            "guard/critic_updates": 0.0,
            "guard/rejected_actor_updates": 0.0,
            "guard/skipped_empty_batches": 0.0,
            "guard/abort": 0.0,
            "guard/invalid_episode_length_count": counts["invalid_episode_lengths"],
            "guard/kl_limit_enforced": float(enforce_kl_limit),
            "guard/pre_update_kl_stop": 0.0,
            "guard/pre_update_kl_exceeded_batches": 0.0,
            "guard/max_pre_update_kl": 0.0,
            "guard/max_pre_update_unweighted_kl": 0.0,
        }

        def emit_probability_abort(
            reason: str,
            stage: str,
            stats: dict[str, float],
            pass_index: int | None = None,
            batch_index: int | None = None,
        ) -> None:
            """Record the already reduced cause without another model forward."""
            probability = probability_stats_metrics(stats)
            step = int(getattr(self, "version", 0))
            # Keep the first failure even if the final probe also fails.
            if "guard/abort_global_step" not in metrics:
                metrics["guard/abort_global_step"] = float(step)
                metrics["guard/abort_round"] = float(step + 1)
                metrics["guard/abort_pass"] = float(
                    pass_index + 1 if pass_index is not None else 0
                )
                metrics["guard/abort_minibatch"] = float(
                    batch_index + 1 if batch_index is not None else 0
                )
                metrics.update(
                    {f"guard/abort_{key}": value for key, value in probability.items()}
                )
            if self._rank != 0:
                return
            payload = {
                "reason": reason,
                "stage": stage,
                "global_step": step,
                "round": step + 1,
                "pass": pass_index + 1 if pass_index is not None else None,
                "minibatch": batch_index + 1 if batch_index is not None else None,
                "logging_rank": self._rank,
                "actor_world_size": self._world_size,
                "statistics_scope": "all_actor_ranks",
                "optimizer_steps": int(self.optimizer_steps),
                "actor_updates": int(metrics["guard/actor_updates"]),
                "critic_updates": int(metrics["guard/critic_updates"]),
                "kl_limit_enforced": enforce_kl_limit,
                "probability": {
                    key: value if math.isfinite(value) else None
                    for key, value in probability.items()
                },
                "nonfinite_statistics": sorted(
                    key for key, value in stats.items() if not math.isfinite(value)
                ),
            }
            self.log_info("PPO_GUARD_ABORT " + json.dumps(payload, allow_nan=False))

        if (
            counts["invalid_rewards"]
            or counts["invalid_targets"]
            or counts["invalid_episode_lengths"]
            or counts["valid_chunks"] == 0
        ):
            metrics["guard/abort"] = 1.0
            metrics["guard/invalid_rollout"] = 1.0
            return metrics

        # Equal forward counts avoid mismatched FSDP collectives when local
        # valid-chunk counts differ, including an entirely padded actor shard.
        probe = make_probability_probe(self.rollout_batch, guard["probe_batch_size"])
        initial = self._measure_policy_probe(probe)
        for key, value in probability_stats_metrics(initial).items():
            metrics[f"guard/initial_{key}"] = value
        if (
            probability_limit_exceeded(
                initial, guard["target_kl"], enforce_kl_limit=False
            )
            or initial["max_abs_logratio"] > guard["max_initial_logratio"]
        ):
            metrics["guard/abort"] = 1.0
            metrics["guard/initial_probability_mismatch"] = 1.0
            emit_probability_abort(
                "initial_probability_mismatch", "initial_probe", initial
            )
            return metrics

        skip_actor = bool(
            guard["skip_actor_on_zero_reward"] and counts["nonzero_rewards"] == 0
        )
        metrics["guard/actor_skipped_zero_reward"] = float(skip_actor)
        all_stats = empty_probability_stats()
        loss_metrics = {}
        applied_any = False
        minibatches_per_pass = rollout_size // batch_size_per_rank
        last_verified_policy_kl = metrics["guard/initial_kl"]

        def emit_progress(
            outcome: str,
            grad_norm: float,
            lr_list: list[float],
            candidate_post_kl: float | None = None,
            candidate_post_unweighted_kl: float | None = None,
        ) -> None:
            # Only serialize statistics already available on rank 0. No extra
            # probe forward, collective, or GPU synchronization is introduced.
            if not log_progress:
                return
            norm = float(grad_norm)
            payload = {
                "global_step": int(getattr(self, "version", 0)),
                "pass": pass_index + 1,
                "passes": self.cfg.algorithm.update_epoch,
                "minibatch": batch_index + 1,
                "minibatches": minibatches_per_pass,
                "outcome": outcome,
                "actor_updates": int(metrics["guard/actor_updates"]),
                "critic_updates": int(metrics["guard/critic_updates"]),
                "rejected_actor_updates": int(metrics["guard/rejected_actor_updates"]),
                "optimizer_steps": int(self.optimizer_steps),
                # Report rates of retained updates, not a rolled-back actor
                # candidate or an optimizer group skipped for critic warmup.
                "actor_lr": float(lr_list[0]) if outcome == "accepted" else 0.0,
                "critic_lr": float(lr_list[1]),
                "grad_norm": norm if math.isfinite(norm) else None,
                "candidate_post_kl": candidate_post_kl,
                "candidate_post_unweighted_kl": candidate_post_unweighted_kl,
                "last_verified_policy_kl": last_verified_policy_kl,
                "kl_limit_enforced": enforce_kl_limit,
                "pre_update_kl": current_metrics["kl"],
                "pre_update_unweighted_kl": current_metrics["unweighted_kl"],
                "pre_update_valid_chunks": current_metrics["valid_count"],
                "pre_update_weight_sum": current_metrics["weight_sum"],
                "pre_update_invalid_count": current_metrics["invalid_count"],
                "pre_update_kl_exceeded": pre_kl_exceeded,
                "pre_update_actor_stopped": pre_actor_stopped,
                "elapsed_training_seconds": round(
                    time.monotonic() - training_started, 3
                ),
            }
            self.log_info("PPO_UPDATE_PROGRESS " + json.dumps(payload, allow_nan=False))

        for pass_index in range(self.cfg.algorithm.update_epoch):
            for batch_index, global_batch in enumerate(
                split_dict_to_chunk(self.rollout_batch, minibatches_per_pass)
            ):
                valid_count = all_reduce_dict(
                    {
                        "count": float(
                            global_batch["loss_mask"]
                            .reshape(batch_size_per_rank, -1)
                            .any(-1)
                            .sum()
                            .item()
                        ),
                    },
                    dtype=torch.float64,
                    op=torch.distributed.ReduceOp.SUM,
                )["count"]
                if not valid_count:
                    metrics["guard/skipped_empty_batches"] += 1
                    continue
                warmup = self.optimizer_steps < self.critic_warmup_steps
                policy_disabled = skip_actor or warmup
                self.optimizer.zero_grad(set_to_none=True)
                current = empty_probability_stats()
                micro_batches = split_dict_to_chunk(
                    global_batch,
                    batch_size_per_rank // self.cfg.actor.micro_batch_size,
                )
                for index, batch in enumerate(micro_batches):
                    result = self.train_micro_batch(
                        batch,
                        loss_metrics,
                        is_last=index + 1 == self.gradient_accumulation,
                        guard_mode=True,
                        policy_update_disabled=policy_disabled,
                        loss_scale=(
                            self.cfg.actor.global_batch_size / valid_count
                            if loss_weighting == "valid_chunk"
                            else 1.0
                        ),
                        loss_weighting=loss_weighting,
                    )
                    add_probability_stats(current, result)
                    micro_batches[index] = None
                current = self._reduce_probability_stats(current)
                add_probability_stats(all_stats, current)
                current_metrics = probability_stats_metrics(current)
                pre_kl_exceeded = current_metrics["kl"] > guard["target_kl"]
                pre_actor_stopped = False
                for key, value in current_metrics.items():
                    metrics[f"guard/last_pre_update_{key}"] = value
                metrics["guard/pre_update_kl_exceeded_batches"] += float(
                    pre_kl_exceeded
                )
                for key in ("kl", "unweighted_kl"):
                    metrics[f"guard/max_pre_update_{key}"] = max(
                        metrics[f"guard/max_pre_update_{key}"], current_metrics[key]
                    )
                # Check the grad-enabled path too before the first actor update.
                if (
                    metrics["guard/actor_updates"] == 0
                    and not skip_actor
                    and (
                        current["invalid_count"]
                        or current["max_abs_logratio"] > guard["max_initial_logratio"]
                    )
                ):
                    self.optimizer.zero_grad(set_to_none=True)
                    metrics["guard/abort"] = 1.0
                    metrics["guard/initial_probability_mismatch"] = 1.0
                    emit_probability_abort(
                        "initial_probability_mismatch",
                        "pre_update",
                        current,
                        pass_index,
                        batch_index,
                    )
                    break
                if probability_limit_exceeded(
                    current, guard["target_kl"], enforce_kl_limit=False
                ):
                    self.optimizer.zero_grad(set_to_none=True)
                    metrics["guard/abort"] = 1.0
                    metrics["guard/invalid_probability"] = 1.0
                    emit_probability_abort(
                        "invalid_probability",
                        "pre_update",
                        current,
                        pass_index,
                        batch_index,
                    )
                    break
                if not policy_disabled and enforce_kl_limit and pre_kl_exceeded:
                    skip_actor = policy_disabled = True
                    pre_actor_stopped = True
                    metrics["guard/pre_update_kl_stop"] = 1.0
                    # Preserve the actual triggering minibatch, rather than
                    # overwriting it with subsequent critic-only statistics.
                    for key, value in current_metrics.items():
                        metrics[f"guard/pre_stop_{key}"] = value
                    metrics["guard/pre_stop_pass"] = float(pass_index + 1)
                    metrics["guard/pre_stop_minibatch"] = float(batch_index + 1)
                snapshot = (
                    ActorOptimizerSnapshot(self.optimizer)
                    if enforce_kl_limit and not policy_disabled
                    else None
                )
                grad_norm, lr_list = self.optimizer_step(
                    skip_actor_update=policy_disabled
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    self.optimizer.zero_grad(set_to_none=True)
                    metrics["guard/abort"] = 1.0
                    metrics["guard/nonfinite_gradient"] = 1.0
                    del snapshot
                    emit_progress("nonfinite_gradient", grad_norm, lr_list)
                    break
                step_applied = self.last_optimizer_step_applied
                applied_any |= step_applied
                metrics["guard/critic_updates"] += float(step_applied)
                append_to_dict(
                    loss_metrics,
                    {
                        "actor/grad_norm": grad_norm,
                        "actor/lr": lr_list[0],
                        "critic/lr": lr_list[1],
                    },
                )
                self.optimizer.zero_grad(set_to_none=True)
                outcome = "critic_only" if step_applied else "no_update"
                candidate_post_kl = None
                candidate_post_unweighted_kl = None
                if snapshot is not None and self.last_actor_optimizer_step_applied:
                    try:
                        post = self._measure_policy_probe(probe)
                    except BaseException:
                        snapshot.restore()
                        raise
                    for key, value in probability_stats_metrics(post).items():
                        metrics[f"guard/last_candidate_{key}"] = value
                    if not probability_limit_exceeded(
                        post, guard["target_kl"], enforce_kl_limit=False
                    ):
                        candidate_post_kl = metrics["guard/last_candidate_kl"]
                        candidate_post_unweighted_kl = metrics[
                            "guard/last_candidate_unweighted_kl"
                        ]
                    if probability_limit_exceeded(post, guard["target_kl"]):
                        snapshot.restore()
                        skip_actor = True
                        metrics["guard/rejected_actor_updates"] += 1
                        outcome = "actor_rollback"
                    else:
                        metrics["guard/actor_updates"] += 1
                        outcome = "accepted"
                        last_verified_policy_kl = candidate_post_kl
                elif self.last_actor_optimizer_step_applied:
                    metrics["guard/actor_updates"] += 1
                    outcome = "accepted"
                    # Diagnostic-only mode avoids CPU optimizer snapshots and
                    # a fixed-probe forward after every candidate. This policy
                    # is measured on the next minibatch and the final probe;
                    # an earlier probe is not a measurement of these weights.
                    last_verified_policy_kl = None
                del snapshot
                emit_progress(
                    outcome,
                    grad_norm,
                    lr_list,
                    candidate_post_kl,
                    candidate_post_unweighted_kl,
                )
            if metrics["guard/abort"]:
                break
        if applied_any:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        final = self._measure_policy_probe(probe)
        for key, value in probability_stats_metrics(final).items():
            metrics[f"guard/final_{key}"] = value
        if probability_limit_exceeded(
            final, guard["target_kl"], enforce_kl_limit=enforce_kl_limit
        ):
            metrics["guard/abort"] = 1.0
            metrics["guard/final_probability_mismatch"] = 1.0
            emit_probability_abort("final_probability_mismatch", "final_probe", final)
        explained = pop_critic_explained_variance_stats(loss_metrics)
        if loss_metrics:
            metrics.update(
                all_reduce_dict(
                    {key: float(np.mean(value)) for key, value in loss_metrics.items()},
                    op=torch.distributed.ReduceOp.AVG,
                )
            )
        # Expose valid-sample aggregates rather than zero-padding averages.
        valid_metrics = probability_stats_metrics(all_stats)
        metrics.update(
            {f"guard/update_{key}": value for key, value in valid_metrics.items()}
        )
        metrics["actor/ratio"] = valid_metrics["ratio"]
        metrics["actor/approx_kl"] = valid_metrics["sampled_negative_logratio"]
        metrics["actor/outside_clip_fraction"] = valid_metrics["outside_clip_fraction"]
        if explained:
            explained = all_reduce_dict(explained, op=torch.distributed.ReduceOp.SUM)
            metrics[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(explained).item()
            )
        return metrics

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
        guard_mode: bool = False,
        policy_update_disabled: bool = False,
        loss_scale: float = 1.0,
        loss_weighting: str = "valid_chunk",
        probability_only: bool = False,
    ) -> dict[str, float] | None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type == "gae"
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        # Chunk-level PPO sums action log-probabilities.  Mask both the current
        # and behavior-policy tensors first so a terminal chunk's unexecuted
        # tail contributes neither to its ratio nor to gradients.
        if self.cfg.algorithm.logprob_type == "chunk_level":
            action_mask = micro_batch.get("action_mask", None)
            action_dim = self.cfg.actor.model.get("action_dim", 7)
            output_dict["logprobs"] = _mask_unexecuted_action_logprobs(
                output_dict["logprobs"], action_mask, action_dim
            )
            prev_logprobs = _mask_unexecuted_action_logprobs(
                prev_logprobs, action_mask, action_dim
            )

        probability_stats = None
        if guard_mode or probability_only:
            sample_weights = None
            if loss_weighting == "libero_trajectory":
                # Match the loss's relative H/L weights for current batches
                # and fixed probes. Normalize only after the global reduction;
                # partial terminal chunks retain the same convention as loss.
                # Valid lengths were checked collectively before any forward.
                lengths = loss_mask_sum.reshape(-1).double()
                valid_chunks = loss_mask.reshape(lengths.shape[0], -1).any(-1)
                lengths = torch.where(valid_chunks, lengths, torch.ones_like(lengths))
                sample_weights = self.cfg.env.train.max_episode_steps / lengths
            probability_stats = chunk_probability_stats(
                output_dict["logprobs"],
                prev_logprobs,
                loss_mask,
                self.cfg.algorithm.clip_ratio_low,
                self.cfg.algorithm.clip_ratio_high,
                sample_weights=sample_weights,
            )
        if probability_only:
            return probability_stats

        use_episode_lengths = not guard_mode or loss_weighting == "libero_trajectory"
        if guard_mode and use_episode_lengths:
            # Valid lengths were checked collectively before any updates.
            # Padding can carry a zero/NaN sentinel; avoid 0/0 before masking,
            # while retaining a connected, zero-gradient graph on empty ranks.
            loss_mask_sum = torch.where(
                loss_mask.reshape(loss_mask_sum.shape).bool(),
                loss_mask_sum,
                torch.ones_like(loss_mask_sum),
            )

        loss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum if use_episode_lengths else None,
            "max_episode_steps": self.cfg.env.train.max_episode_steps
            if use_episode_lengths
            else None,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": policy_update_disabled
            or self.optimizer_steps < self.critic_warmup_steps,
        }

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
        ]:
            # OpenPI honors explicit PPO clipping settings; preserve GR00T's
            # existing dual-clip default and leave unconfigured OpenPI unchanged.
            default_dual_clip = (
                None
                if SupportedModel(self.cfg.actor.model.model_type)
                == SupportedModel.OPENPI
                else 3.0
            )
            loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get(
                "clip_ratio_c", default_dual_clip
            )
            if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                loss_kwargs["clip_log_ratio_min"] = (
                    self.cfg.algorithm.clip_log_ratio_min
                )
            if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                loss_kwargs["clip_log_ratio_max"] = (
                    self.cfg.algorithm.clip_log_ratio_max
                )

        loss, metrics_data = policy_loss(**loss_kwargs)
        entropy_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if self.cfg.algorithm.entropy_bonus > 0 and not loss_kwargs["critic_warmup"]:
            entropy = output_dict["entropy"]
            entropy = reshape_entropy(
                entropy,
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=self.cfg.actor.model.get("action_dim", 7),
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        if guard_mode and loss_weighting == "valid_chunk":
            # policy_loss is a local masked mean. Convert it to a mean over
            # this microbatch's padded slots before accumulation and DDP's
            # rank average. The caller's B / global_valid factor then gives
            # every valid chunk weight 1 / global_valid, including mixed and
            # entirely padded microbatches. For micro=1 this is the old loss.
            loss = loss * (loss_mask.count_nonzero().to(loss.dtype) / loss_mask.numel())
        loss *= loss_scale / self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()

        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)
        return probability_stats

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = {
            "actor/grad_norm": grad_norm,
            "actor/lr": lr_list[0],
        }
        if len(lr_list) > 1:
            metric_data["critic/lr"] = lr_list[1]
        append_to_dict(metrics, metric_data)
