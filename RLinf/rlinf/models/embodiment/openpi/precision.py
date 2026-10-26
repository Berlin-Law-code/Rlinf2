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

"""Explicit parameter storage and opt-in PPO computation precision for OpenPI."""

import logging
from collections.abc import Mapping

import torch


def validate_parameter_precision(parameter_precision: str) -> str:
    """Validate the opt-in parameter policy without silently accepting typos."""
    if parameter_precision not in ("openpi", "float32"):
        raise ValueError(
            "openpi.parameter_precision must be 'openpi' or 'float32', "
            f"got {parameter_precision!r}"
        )
    return parameter_precision


def apply_parameter_precision(
    model: torch.nn.Module, parameter_precision: str = "openpi"
) -> None:
    """Apply parameter storage precision; this does not enable AMP.

    For ``float32``, call before loading an FP32 state dict as well as after
    loading: copying FP32 source weights into BF16 destination parameters
    first would irreversibly round them. The legacy ``openpi`` policy keeps
    OpenPI's selected BF16 backbone and FP32 projections unchanged.
    """
    parameter_precision = validate_parameter_precision(parameter_precision)
    if parameter_precision == "float32":
        model.to(dtype=torch.float32)
    else:
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")


def validate_strict_ppo_precision_config(
    model_config: Mapping,
    *,
    parameter_precision: str | None = None,
    torch_dtype: torch.dtype | None = None,
    fsdp_config: Mapping | None = None,
) -> bool:
    """Validate explicit strict PPO precision without changing backend state.

    The model loader supplies checkpoint-resolved ``parameter_precision``.
    Launchers may additionally supply the actor's FSDP configuration to reject
    AMP and reduction/storage casts before creating workers. The loader only
    receives model configuration, so runtime checks also reject active AMP.
    A value head is not required: policy-only evaluation uses the same contract.
    """
    openpi = model_config.get("openpi", {}) or {}
    enabled = openpi.get("strict_ppo_precision", False)
    if not isinstance(enabled, bool):
        raise ValueError("openpi.strict_ppo_precision must be a boolean")
    if not enabled:
        return False

    parameter_precision = parameter_precision or openpi.get(
        "parameter_precision", "openpi"
    )
    if parameter_precision != "float32":
        raise ValueError(
            "openpi.strict_ppo_precision requires parameter_precision='float32'"
        )
    if model_config.get("precision") not in (None, "null", 32, "32", "fp32", "32-true"):
        raise ValueError(
            "openpi.strict_ppo_precision requires model.precision=null or FP32"
        )
    if torch_dtype not in (None, torch.float32):
        raise ValueError("openpi.strict_ppo_precision requires an FP32 torch_dtype")

    if fsdp_config is not None:
        if (fsdp_config.get("amp_autocast") or {}).get("enabled", False):
            raise ValueError("openpi.strict_ppo_precision requires AMP disabled")
        mixed = fsdp_config.get("mixed_precision") or {}
        for name in ("param_dtype", "reduce_dtype", "buffer_dtype"):
            dtype = mixed.get(name)
            if dtype not in (
                None,
                torch.float32,
                "float32",
                "fp32",
                32,
                "32",
                "32-true",
            ):
                raise ValueError(
                    "openpi.strict_ppo_precision requires FP32 FSDP "
                    f"mixed_precision.{name}, got {dtype!r}"
                )
    return True


def _compute_precision_state() -> dict[str, str | bool]:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
    }


def assert_strict_ppo_precision_runtime(model: torch.nn.Module | None = None) -> None:
    """Fail if AMP or another model has changed the opted-in process contract."""
    state = _compute_precision_state()
    if state["cuda_autocast_enabled"] or state["cpu_autocast_enabled"]:
        raise RuntimeError("openpi.strict_ppo_precision does not allow active AMP")
    if (
        state["float32_matmul_precision"] != "highest"
        or state["cuda_matmul_allow_tf32"]
        or state["cudnn_allow_tf32"]
    ):
        raise RuntimeError(
            "OpenPI strict PPO computation precision was changed after model "
            f"initialization: {state}. All actor/rollout models must keep "
            "highest matmul precision and TF32 disabled."
        )
    if model is not None:
        for name, parameter in model.named_parameters():
            if parameter.dtype != torch.float32:
                raise RuntimeError(
                    "OpenPI strict PPO parameters were cast after model loading: "
                    f"{name} has {parameter.dtype}; FSDP must retain FP32."
                )


def apply_strict_ppo_precision(
    model: torch.nn.Module, *, enabled: bool = False
) -> dict[str, str | bool] | None:
    """Set and log strict FP32 computation after OpenPI model construction.

    Upstream PI0 construction selects ``high`` precision, permitting TF32 even
    with FP32 parameters. These PyTorch switches are process-wide; an explicit
    opt-in must be applied after construction in both actor and rollout. The
    disabled branch does not inspect or reset global precision settings.
    """
    if not enabled:
        return None
    non_fp32 = [
        name
        for name, parameter in model.named_parameters()
        if parameter.dtype != torch.float32
    ]
    if non_fp32:
        raise ValueError(
            "openpi.strict_ppo_precision requires all parameters to be FP32; "
            f"non-FP32 parameters: {non_fp32[:10]}"
        )
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("openpi.strict_ppo_precision does not allow active AMP")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert_strict_ppo_precision_runtime()
    state = _compute_precision_state()
    model._openpi_strict_ppo_precision = dict(state)
    logging.getLogger(__name__).info("OpenPI strict PPO precision enabled: %s", state)
    return state
