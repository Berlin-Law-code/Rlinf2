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
# openpi model configs

import os
import pathlib

import torch
from omegaconf import DictConfig, OmegaConf


def _checkpoint_compatibility_settings(checkpoint_dir: str) -> dict:
    """Recover the input/precision contract bundled with a new SFT checkpoint."""
    from rlinf.models.embodiment.openpi.checkpoint_assets import (
        read_checkpoint_manifest,
    )

    return read_checkpoint_manifest(checkpoint_dir).get("compatibility", {})


def _validate_incompatible_keys(
    missing_keys,
    unexpected_keys,
    *,
    config_name: str,
    add_value_head: bool,
) -> None:
    """Apply a strict LBench-specific checkpoint compatibility check."""

    if config_name != "pi05_lbench_joint":
        return
    tied_lm_heads = {
        "paligemma_with_expert.paligemma.lm_head.weight",
        "paligemma_with_expert.gemma_expert.lm_head.weight",
    }
    allowed_missing = set(tied_lm_heads)
    if add_value_head:
        allowed_missing.update(
            key for key in missing_keys if key.startswith("value_head.")
        )
    disallowed_missing = sorted(set(missing_keys) - allowed_missing)
    # Loading an actor-only SFT checkpoint into PPO legitimately creates a new
    # critic head; loading a PPO checkpoint for pure evaluation legitimately
    # drops that critic head. Keep every non-critic tensor strict in both
    # directions.
    allowed_unexpected = {
        key
        for key in unexpected_keys
        if not add_value_head and key.startswith("value_head.")
    }
    unexpected = sorted(set(unexpected_keys) - allowed_unexpected)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "LBench Pi0.5 checkpoint is incompatible with the configured model: "
            f"missing={disallowed_missing[:20]}, unexpected={unexpected[:20]}"
        )


def _load_state_dict(
    model,
    state_dict: dict,
    *,
    config_name: str,
    add_value_head: bool,
) -> None:
    """Load one model state and validate its LBench compatibility."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    _validate_incompatible_keys(
        incompatible.missing_keys,
        incompatible.unexpected_keys,
        config_name=config_name,
        add_value_head=add_value_head,
    )


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors.torch as safetensors_torch
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.checkpoint_assets import (
        build_checkpoint_assets,
        resolve_checkpoint_norm_stats,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )
    from rlinf.models.embodiment.openpi.precision import (
        apply_parameter_precision,
        apply_strict_ppo_precision,
        validate_parameter_precision,
        validate_strict_ppo_precision_config,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    compatibility = _checkpoint_compatibility_settings(checkpoint_dir)
    if data_kwargs is not None:
        data_kwargs = (
            OmegaConf.to_container(data_kwargs, resolve=True)
            if isinstance(data_kwargs, DictConfig)
            else dict(data_kwargs)
        )
    if compatibility.get("lerobot_compatible", False):
        data_kwargs = dict(data_kwargs or {})
        if data_kwargs.get("lerobot_compatible") is False:
            raise ValueError("Checkpoint requires LeRobot-compatible preprocessing")
        data_kwargs["lerobot_compatible"] = True
    canonical_asset_id = get_openpi_config(config_name).data.repo_id
    norm_stats_file = resolve_checkpoint_norm_stats(
        checkpoint_dir,
        (data_kwargs or {}).get("norm_stats_path"),
        canonical_asset_id=canonical_asset_id,
        allow_explicit_override=cfg.get("allow_norm_stats_override", False),
    )
    if norm_stats_file is not None:
        data_kwargs = dict(data_kwargs or {})
        data_kwargs["norm_stats_path"] = str(norm_stats_file)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val
    parameter_precision = validate_parameter_precision(
        str(
            cfg.openpi.get(
                "parameter_precision",
                compatibility.get("parameter_precision", "openpi"),
            )
        )
    )
    actor_model_config.__dict__["parameter_precision"] = parameter_precision
    strict_ppo_precision = validate_strict_ppo_precision_config(
        cfg, parameter_precision=parameter_precision, torch_dtype=torch_dtype
    )
    actor_model_config.__dict__["strict_ppo_precision"] = strict_ppo_precision

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # Preserve original FP32 source weights during copy-based loading.
    if parameter_precision == "float32":
        apply_parameter_precision(model, parameter_precision)
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        _load_state_dict(
            model,
            model_state_dict,
            config_name=str(config_name),
            add_value_head=bool(getattr(actor_model_config, "add_value_head", False)),
        )
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        _load_state_dict(
            model,
            model_state_dict,
            config_name=str(config_name),
            add_value_head=bool(getattr(actor_model_config, "add_value_head", False)),
        )
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        add_value_head = bool(getattr(actor_model_config, "add_value_head", False))
        if len(weight_paths) == 1:
            # The converter uses save_model, which may omit aliases for tied
            # tensors. load_model reconstructs that sharing contract before we
            # inspect missing/unexpected keys.
            missing_keys, unexpected_keys = safetensors_torch.load_model(
                model, weight_paths[0], strict=False, device="cpu"
            )
            _validate_incompatible_keys(
                missing_keys,
                unexpected_keys,
                config_name=str(config_name),
                add_value_head=add_value_head,
            )
        else:
            all_state_dict = {}
            for weight_path in weight_paths:
                state_dict = safetensors_torch.load_file(weight_path, device="cpu")
                duplicate_keys = sorted(set(all_state_dict) & set(state_dict))
                if duplicate_keys:
                    raise RuntimeError(
                        "duplicate parameter keys across OpenPI safetensors files "
                        f"under {checkpoint_dir}: {duplicate_keys[:20]}"
                    )
                all_state_dict.update(state_dict)
            _load_state_dict(
                model,
                all_state_dict,
                config_name=str(config_name),
                add_value_head=add_value_head,
            )

    apply_parameter_precision(model, parameter_precision)
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    # This opt-in data factory already includes the exact LeRobot normalizer
    # and inverse in data_transforms, shared with the SFT data loader.
    native_norm_stats = (
        None
        if bool(getattr(actor_train_config.data, "lerobot_compatible", False))
        else norm_stats
    )
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

    # Keep the effective contract: PPO can inherit settings absent from Hydra.
    model._openpi_checkpoint_assets = build_checkpoint_assets(
        config_name=str(config_name),
        norm_stats=norm_stats,
        asset_ids=[data_config.asset_id, canonical_asset_id],
        parameter_precision=parameter_precision,
        lerobot_compatible=bool(
            getattr(actor_train_config.data, "lerobot_compatible", False)
        ),
        action_horizon=int(actor_model_config.action_horizon),
        execute_horizon=min(
            int(actor_model_config.action_chunk), int(actor_model_config.action_horizon)
        ),
        action_dim=int(actor_model_config.action_env_dim),
    )

    # PI0 construction resets PyTorch matmul precision to "high". Apply this
    # explicit policy last, identically for actor and rollout model loading.
    apply_strict_ppo_precision(model, enabled=strict_ppo_precision)
    return model
