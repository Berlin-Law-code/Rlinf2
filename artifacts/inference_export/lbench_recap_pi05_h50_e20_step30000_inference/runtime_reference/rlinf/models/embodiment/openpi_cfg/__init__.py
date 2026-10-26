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
import os

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors.torch
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi import _load_state_dict
    from rlinf.models.embodiment.openpi.checkpoint_assets import build_checkpoint_assets
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.precision import apply_parameter_precision
    from rlinf.models.embodiment.openpi_cfg.contract import resolve_cfg_contract
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
        OpenPi0Config,
        OpenPi0ForCFGActionPrediction,
    )

    config_name = getattr(cfg.openpi, "config_name", None)
    actor_train_config, parameter_precision = resolve_cfg_contract(cfg)
    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_config_kwargs = cfg.openpi
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            actor_model_config.__dict__[key] = val
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForCFGActionPrediction = OpenPi0ForCFGActionPrediction(
        actor_model_config
    )
    if parameter_precision == "float32":
        apply_parameter_precision(model, parameter_precision)
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    if os.path.exists(full_weights_path):
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        _load_state_dict(
            model, model_state_dict, config_name=config_name, add_value_head=False
        )
    elif os.path.exists(actor_full_weights_path):
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        _load_state_dict(
            model, model_state_dict, config_name=config_name, add_value_head=False
        )
    else:
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        model_state_dict = {}
        for weight_path in weight_paths:
            shard = safetensors.torch.load_file(weight_path, device="cpu")
            if model_state_dict.keys() & shard.keys():
                raise ValueError("Duplicate parameter keys in CFG safetensors shards")
            model_state_dict.update(shard)
        _load_state_dict(
            model, model_state_dict, config_name=config_name, add_value_head=False
        )

    apply_parameter_precision(model, parameter_precision)
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = data_config.norm_stats
    if norm_stats is None:
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    lerobot_compatible = bool(
        getattr(actor_train_config.data, "lerobot_compatible", False)
    )
    native_norm_stats = None if lerobot_compatible else norm_stats

    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                native_norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                native_norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    model._openpi_checkpoint_assets = build_checkpoint_assets(
        config_name=str(config_name),
        norm_stats=norm_stats,
        asset_ids=[data_config.asset_id, get_openpi_config(config_name).data.repo_id],
        parameter_precision=parameter_precision,
        lerobot_compatible=lerobot_compatible,
        action_horizon=int(actor_model_config.action_horizon),
        execute_horizon=min(
            int(actor_model_config.action_chunk), int(actor_model_config.action_horizon)
        ),
        action_dim=int(actor_model_config.action_env_dim),
    )
    return model
