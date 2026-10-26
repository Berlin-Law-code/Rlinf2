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

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.config import SupportedModel
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    model_type = SupportedModel(model_cfg.model_type)
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=getattr(model_cfg, "openpi_data", None),
    )
    # The model factory applies this override independently. The dataset must
    # request the same number of future actions, including at episode ends.
    action_horizon = model_cfg.openpi.get("action_horizon")
    if action_horizon is not None:
        if (
            isinstance(action_horizon, bool)
            or not isinstance(action_horizon, int)
            or action_horizon <= 0
        ):
            raise ValueError("openpi.action_horizon must be a positive integer")
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, action_horizon=action_horizon),
        )
    # Keep the RLinf SFT data controls effective for both the legacy OPENPI
    # model and the newer OPENPI_RLINF model. Previously the OPENPI path
    # silently ignored data.num_workers and actor.seed.
    num_workers = int(
        OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
    )
    if num_workers < 0:
        raise ValueError("data.num_workers must be non-negative")
    config = dataclasses.replace(
        config,
        num_workers=num_workers,
        seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
    )
    if model_type == SupportedModel.OPENPI_RLINF:
        _validate_openpi_rlinf_model_shape(model_cfg, config)

    data_loader = openpi_data_loader.create_data_loader(
        config,
        framework="pytorch",
        shuffle=not eval_dataset,
        # The compatible LBench factory supplies its own float32 quantile
        # normalizer before tokenization; do not normalize a second time.
        skip_norm_stats=bool(getattr(config.data, "lerobot_compatible", False)),
    )
    if str(model_cfg.openpi.config_name) == "pi05_lbench_joint":
        _validate_lbench_distributed_loader(
            data_loader,
            world_size=world_size,
            rank=rank,
            expected_local_batch_size=int(batch_size),
        )
    return data_loader, data_loader.data_config()


def _validate_lbench_distributed_loader(
    data_loader: Any,
    *,
    world_size: int,
    rank: int,
    expected_local_batch_size: int,
) -> None:
    """Fail fast if the installed OpenPI loader would duplicate SFT data.

    RLinf passes ``micro_batch_size * world_size`` as OpenPI's global batch.
    The RLinf OpenPI fork is expected to replace that with a per-rank batch and
    a ``DistributedSampler``. Older OpenPI builds lacked this PyTorch-DDP path;
    silently accepting one would make every FSDP rank train on the same data.
    """

    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None)
    if torch_loader is None:
        torch_loader = getattr(openpi_loader, "torch_loader", None)
    if torch_loader is None:
        raise TypeError(
            "LBench OpenPI loader does not expose its inner PyTorch DataLoader"
        )
    if int(getattr(torch_loader, "batch_size", -1)) != expected_local_batch_size:
        raise ValueError(
            "LBench OpenPI local batch mismatch: "
            f"expected={expected_local_batch_size}, "
            f"actual={getattr(torch_loader, 'batch_size', None)}"
        )
    if world_size == 1:
        return
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Multi-GPU LBench SFT constructed the OpenPI loader before the "
            "PyTorch distributed process group was initialized"
        )
    actual_world_size = torch.distributed.get_world_size()
    actual_rank = torch.distributed.get_rank()
    if actual_world_size != world_size or actual_rank != rank:
        raise ValueError(
            "OpenPI/RLinf distributed identity mismatch: "
            f"expected rank {rank}/{world_size}, got "
            f"{actual_rank}/{actual_world_size}"
        )
    sampler = getattr(torch_loader, "sampler", None)
    if not isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
        raise RuntimeError(
            "The installed OpenPI build did not create a DistributedSampler "
            "for multi-GPU LBench SFT; install the RLinf OpenPI fork"
        )
    if int(sampler.num_replicas) != world_size or int(sampler.rank) != rank:
        raise ValueError(
            "OpenPI DistributedSampler identity does not match the RLinf worker"
        )


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None)
    if torch_loader is None:
        torch_loader = getattr(openpi_loader, "torch_loader", None)
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    return len(torch_loader)


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return getattr(data_loader, "_data_loader", None) is not None


def _validate_openpi_rlinf_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config."""
    local_horizon = int(model_cfg.num_action_chunks)
    official_horizon = int(openpi_config.model.action_horizon)
    if local_horizon != official_horizon:
        raise ValueError(
            "openpi_rlinf SFT action horizon must match the official OpenPI "
            f"config: actor.model.num_action_chunks={local_horizon}, "
            f"{model_cfg.openpi.config_name}.model.action_horizon="
            f"{official_horizon}."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi_rlinf SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )
