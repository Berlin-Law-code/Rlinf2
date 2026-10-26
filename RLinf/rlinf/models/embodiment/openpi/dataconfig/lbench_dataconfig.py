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

"""OpenPI data configuration for the LBench joint feature schema."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import lbench_policy


@dataclasses.dataclass(frozen=True)
class LeRobotLBenchJointDataConfig(DataConfigFactory):
    """Exact feature schema shared by LBench's LeRobot writer and RLinf.

    This schema is independent of the LeRobot v2.1/v3.0 on-disk layout.  The
    installed LeRobot loader and dataset storage version must still match.
    """

    image_key: str = "observation.images.obs_camera"
    wrist_image_key: str = "observation.images.obs_camera_2"
    extra_view_image_key: str = "observation.images.realsense"
    state_key: str = "observation.state"
    action_key: str = "action"
    task_key: str | None = None
    # PromptFromLeRobotTask runs before repacking when prompt_from_task=True.
    prompt_key: str | None = "prompt"
    default_prompt: str | None = None
    state_dim: int = 8
    output_action_dim: int = 8
    # Opt-in because existing checkpoints were trained with native OpenPI
    # normalization and image resizing. This must match during SFT and eval.
    lerobot_compatible: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        if self.state_dim <= 0 or self.output_action_dim <= 0:
            raise ValueError("LBench state and action dimensions must be positive")
        if self.state_dim > model_config.action_dim:
            raise ValueError(
                f"LBench state_dim={self.state_dim} exceeds OpenPI "
                f"action_dim={model_config.action_dim}"
            )
        if self.output_action_dim > model_config.action_dim:
            raise ValueError(
                f"LBench output_action_dim={self.output_action_dim} exceeds OpenPI "
                f"action_dim={model_config.action_dim}"
            )

        base_config = self.create_base_config(assets_dirs, model_config)
        structure = {
            "observation/image": self.image_key,
            "observation/wrist_image": self.wrist_image_key,
            "observation/extra_view_image": self.extra_view_image_key,
            "observation/state": self.state_key,
            "actions": self.action_key,
        }
        # PromptFromLeRobotTask always creates the literal ``prompt`` key.
        if base_config.prompt_from_task:
            structure["prompt"] = "prompt"
        elif self.prompt_key is not None:
            structure["prompt"] = self.prompt_key
        elif self.task_key is not None:
            structure["task"] = self.task_key
        repack = _transforms.Group(inputs=[_transforms.RepackTransform(structure)])
        data = _transforms.Group(
            inputs=[
                lbench_policy.LBenchInputs(
                    model_type=model_config.model_type,
                    state_dim=self.state_dim,
                    action_horizon=model_config.action_horizon,
                    default_prompt=self.default_prompt,
                    preserve_float_images=self.lerobot_compatible,
                )
            ],
            outputs=[lbench_policy.LBenchOutputs(action_dim=self.output_action_dim)],
        )
        model_transforms = ModelTransformFactory()(model_config)
        if self.lerobot_compatible:
            from rlinf.models.embodiment.openpi.lerobot_transforms import (
                LeRobotNormalize,
                LeRobotResizeImages,
                LeRobotUnnormalize,
            )

            if base_config.norm_stats is None:
                raise ValueError("LeRobot-compatible LBench requires norm stats")
            if not base_config.use_quantile_norm:
                raise ValueError("LeRobot-compatible LBench requires quantile norm")
            # Normalization lives in data_transforms for this opt-in pipeline.
            # The loader and model wrapper skip the native OpenPI normalizer.
            data = _transforms.Group(
                inputs=[
                    *data.inputs,
                    LeRobotNormalize(base_config.norm_stats, use_quantiles=True),
                ],
                outputs=[
                    LeRobotUnnormalize(base_config.norm_stats, use_quantiles=True),
                    *data.outputs,
                ],
            )
            model_transforms = _transforms.Group(
                inputs=[
                    LeRobotResizeImages(transform.height, transform.width)
                    if isinstance(transform, _transforms.ResizeImages)
                    else transform
                    for transform in model_transforms.inputs
                ],
                outputs=model_transforms.outputs,
            )
        return dataclasses.replace(
            base_config,
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=model_transforms,
            action_sequence_keys=(self.action_key,),
        )


__all__ = ["LeRobotLBenchJointDataConfig"]
