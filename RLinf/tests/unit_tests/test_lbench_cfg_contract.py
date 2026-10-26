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
"""Regression coverage for H50/E20 LBench classifier-free guidance."""

from types import SimpleNamespace

import numpy as np
import torch
from openpi import transforms

from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    OpenPi0Config,
    OpenPi0ForCFGActionPrediction,
)


def test_cfg_loss_supervises_future_actions_and_excludes_padding():
    model = OpenPi0ForCFGActionPrediction.__new__(OpenPi0ForCFGActionPrediction)
    torch.nn.Module.__init__(model)
    model.config = OpenPi0Config(
        config_name="pi05_lbench_joint",
        pi05=True,
        action_horizon=50,
        action_chunk=20,
        action_env_dim=8,
    )
    model.predictions = torch.nn.Parameter(torch.ones(1, 50, 32))
    with torch.no_grad():
        model.predictions[:, 20:, :8] = 3
        model.predictions[:, :, 8:] = 100
    hidden = torch.zeros(1, 50, 2)
    masks = torch.ones(1, 50, dtype=torch.bool)
    model.embed_prefix = lambda *args: (hidden, masks, masks)
    model.embed_suffix = lambda *args: (hidden, masks, masks, None)
    model._prepare_attention_masks_4d = lambda x: x
    model._apply_checkpoint = lambda fn, *args: fn(*args)
    model.action_out_proj = lambda x: model.predictions
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(
            language_model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            q_proj=SimpleNamespace(weight=torch.zeros(1))
                        )
                    )
                ]
            )
        ),
        forward=lambda **kwargs: ((None, hidden), None),
    )
    loss, per_sample = model._compute_flow_losses(
        [],
        [],
        torch.zeros(1, 32),
        torch.zeros(1, 50, 32),
        torch.ones(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        "cpu",
        time=torch.ones(1),
        noise=torch.zeros(1, 50, 32),
    )
    torch.testing.assert_close(loss, torch.tensor(5.8))
    torch.testing.assert_close(per_sample, torch.tensor([5.8]))
    loss.backward()
    assert torch.all(model.predictions.grad[:, 20:, :8] != 0)
    assert torch.count_nonzero(model.predictions.grad[:, :, 8:]) == 0


def test_guidance_tokenizes_normalized_unpadded_state():
    observed = []

    class Tokenizer:
        def tokenize(self, prompt, state):
            observed.append((prompt, state.copy()))
            return np.ones(4, dtype=np.int32), np.ones(4, dtype=bool)

    def normalize(data):
        return {**data, "state": data["state"] * 2 - 1}

    model = OpenPi0ForCFGActionPrediction.__new__(OpenPi0ForCFGActionPrediction)
    torch.nn.Module.__init__(model)
    model.setup_wrappers(
        transforms=[
            normalize,
            transforms.TokenizePrompt(Tokenizer(), discrete_state_input=True),
            transforms.PadStatesAndActions(32),
        ]
    )
    result = model.input_transform(
        {
            "state": torch.full((1, 8), 0.75),
            "prompt": ["pick"],
            "positive_guidance_prompt": ["pick\nAdvantage: positive"],
            "negative_guidance_prompt": ["pick\nAdvantage: negative"],
        },
        transpose=False,
    )
    assert result["state"].shape == (1, 32)
    assert len(observed) == 3
    for _, state in observed:
        np.testing.assert_array_equal(state, np.full(8, 0.5))


def test_cfg_inference_preserves_third_camera():
    model = OpenPi0ForCFGActionPrediction.__new__(OpenPi0ForCFGActionPrediction)
    torch.nn.Module.__init__(model)
    model.config = OpenPi0Config(config_name="pi05_lbench_joint")
    extra = torch.zeros(1, 3, 224, 224)
    result = model.obs_processor(
        {
            "main_images": extra,
            "wrist_images": extra,
            "extra_view_images": extra,
            "states": torch.zeros(1, 8),
            "task_descriptions": ["pick"],
        }
    )
    assert result["observation/extra_view_image"] is extra
