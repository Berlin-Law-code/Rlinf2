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
"""Resolve the same checkpoint input contract for CFG training and inference."""

import dataclasses

from omegaconf import DictConfig, OmegaConf
from openpi.training.config import TrainConfig

from rlinf.models.embodiment.openpi.checkpoint_assets import (
    read_checkpoint_manifest,
    resolve_checkpoint_norm_stats,
)
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.precision import validate_parameter_precision


def resolve_cfg_contract(
    cfg: DictConfig, *, repo_id: str | None = None, batch_size: int | None = None
) -> tuple[TrainConfig, str]:
    """Return an independent OpenPI config and its parameter storage precision."""
    name = cfg.openpi.config_name
    manifest = read_checkpoint_manifest(cfg.model_path)
    compatibility = manifest.get("compatibility", {})
    data = cfg.get("openpi_data", {})
    if isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=True)
    data = dict(data or {})
    if compatibility.get("lerobot_compatible", False):
        if data.get("lerobot_compatible") is False:
            raise ValueError("Checkpoint requires LeRobot-compatible preprocessing")
        data["lerobot_compatible"] = True
    canonical_id = get_openpi_config(name).data.repo_id
    norm = resolve_checkpoint_norm_stats(
        cfg.model_path,
        data.get("norm_stats_path"),
        canonical_asset_id=canonical_id,
        allow_explicit_override=cfg.get("allow_norm_stats_override", False),
    )
    if norm is not None:
        data["norm_stats_path"] = str(norm)
    config = get_openpi_config(
        name,
        model_path=cfg.model_path,
        data_kwargs=data or None,
        repo_id=repo_id,
        batch_size=batch_size,
    )
    horizon = int(cfg.openpi.get("action_horizon", config.model.action_horizon))
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, action_horizon=horizon)
    )
    precision = validate_parameter_precision(
        str(
            cfg.openpi.get(
                "parameter_precision",
                compatibility.get("parameter_precision", "openpi"),
            )
        )
    )
    return config, precision
