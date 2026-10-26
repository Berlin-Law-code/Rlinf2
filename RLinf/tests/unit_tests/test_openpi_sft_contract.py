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

import pytest
import torch
from openpi.models import model as observation_model
from openpi.models_pytorch import pi0_pytorch

from rlinf.config import SupportedModel
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.workers.sft.fsdp_vla_sft_worker import _sft_forward_kwargs


@pytest.mark.parametrize("execution_horizon", [20, 50])
def test_offline_openpi_sft_trains_full_horizon_without_padded_dims(
    monkeypatch, execution_horizon
):
    model, observation = _small_sft_model(execution_horizon)

    def forward(self, observation, actions):
        return self.predictions.square()

    monkeypatch.setattr(pi0_pytorch.PI0Pytorch, "forward", forward)
    with torch.no_grad():
        model.predictions[:, :20, :8] = 1
        model.predictions[:, 20:, :8] = 3
        model.predictions[:, :, 8:] = 100
    loss = model.sft_forward(
        (observation, torch.zeros(1, 50, 32)),
        **_sft_forward_kwargs(SupportedModel.OPENPI),
    )
    torch.testing.assert_close(loss, torch.tensor(5.8))
    loss.backward()
    assert torch.all(model.predictions.grad[:, 20:, :8] != 0)
    assert torch.count_nonzero(model.predictions.grad[:, :, 8:]) == 0
    model._output_transform = lambda sample: {"actions": sample["actions"][..., :8]}
    executed = model.output_transform({"actions": model.predictions})["actions"]
    assert executed.shape == (1, execution_horizon, 8)
    torch.testing.assert_close(
        executed, model.predictions.detach()[:, :execution_horizon, :8]
    )


def test_dagger_chunk_loss_still_excludes_unexecuted_actions(monkeypatch):
    model, observation = _small_sft_model(20)

    def forward(self, observation, actions):
        return self.predictions.square()

    monkeypatch.setattr(pi0_pytorch.PI0Pytorch, "forward", forward)
    loss = model.sft_forward(
        (observation, torch.zeros(1, 50, 32)), use_action_chunk_loss=True
    )
    loss.backward()
    assert torch.all(model.predictions.grad[:, :20, :8] != 0)
    assert torch.count_nonzero(model.predictions.grad[:, 20:, :]) == 0
    assert torch.count_nonzero(model.predictions.grad[:, :, 8:]) == 0


def _small_sft_model(execution_horizon):
    # Exercise the real SFT wrapper without allocating PaliGemma weights.
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.predictions = torch.nn.Parameter(torch.ones(1, 50, 32))
    model.config = OpenPi0Config(
        pi05=True,
        action_horizon=50,
        action_chunk=execution_horizon,
        action_env_dim=8,
    )
    model.gradient_checkpointing_enabled = True
    observation = observation_model.Observation(
        images={"base_0_rgb": torch.zeros(1, 3, 4, 4)},
        image_masks={"base_0_rgb": torch.ones(1, dtype=torch.bool)},
        state=torch.zeros(1, 32),
        tokenized_prompt=torch.tensor([[1, 2]]),
        tokenized_prompt_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    return model, observation


def test_other_vla_sft_models_do_not_receive_openpi_only_argument():
    assert _sft_forward_kwargs(SupportedModel.OPENPI_RLINF) == {}
