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

from dataclasses import replace

import pytest
import torch
from openpi.models import model as observation_model
from openpi.models_pytorch import pi0_pytorch

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)


def _small_model(*, augmentation=True, use_rlt=False):
    # Exercise the wrapper without constructing PaliGemma or loading weights.
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.weight = torch.nn.Parameter(torch.tensor(2.0))
    model.config = OpenPi0Config(
        pi05=True,
        action_chunk=2,
        action_env_dim=1,
        sft_image_augmentation=augmentation,
        use_rlt=use_rlt,
    )
    model.gradient_checkpointing_enabled = True
    return model


def _observation():
    return observation_model.Observation(
        images={"base_0_rgb": torch.zeros(1, 4, 4, 3)},
        image_masks={"base_0_rgb": torch.ones(1, dtype=torch.bool)},
        state=torch.tensor([[0.2, -0.4]]),
        tokenized_prompt=torch.tensor([[1, 2]]),
        tokenized_prompt_mask=torch.ones(1, 2, dtype=torch.bool),
    )


def _record_preprocessor(monkeypatch):
    calls = []

    def preprocess(observation, *, train):
        calls.append(train)
        return replace(
            observation,
            images={
                key: image + float(train) for key, image in observation.images.items()
            },
        )

    monkeypatch.setattr(
        pi0_pytorch._preprocessing, "preprocess_observation_pytorch", preprocess
    )
    return calls


@pytest.mark.parametrize("augmentation", [False, True])
@pytest.mark.parametrize("use_rlt", [False, True])
def test_sft_augmentation_keeps_training_and_gradients(
    monkeypatch, augmentation, use_rlt
):
    model = _small_model(augmentation=augmentation, use_rlt=use_rlt)
    observation = _observation()
    calls = _record_preprocessor(monkeypatch)

    def forward(self, observation, actions):
        assert self.training
        assert torch.is_grad_enabled()
        images, _, tokens, token_masks, state = self._preprocess_observation(
            observation, train=True
        )
        torch.testing.assert_close(
            images[0], torch.full_like(images[0], float(augmentation))
        )
        torch.testing.assert_close(tokens, observation.tokenized_prompt)
        torch.testing.assert_close(token_masks, observation.tokenized_prompt_mask)
        torch.testing.assert_close(state, observation.state)
        return (actions + self.weight).square()

    if use_rlt:

        def rlt_prefix(self, observation, actions):
            loss = forward(self, observation, actions)
            return loss, self.weight.reshape(1, 1, 1), None

        class SmallRLT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, prefix, mask):
                return (prefix * self.weight).mean(), {}

        monkeypatch.setattr(
            OpenPi0ForRLActionPrediction, "_sft_forward_with_rlt_prefix", rlt_prefix
        )
        model.rlt_module = SmallRLT()
    else:
        monkeypatch.setattr(pi0_pytorch.PI0Pytorch, "forward", forward)

    result = model.sft_forward(
        (observation, torch.zeros(1, 2, 3)), use_action_chunk_loss=True
    )
    loss = result["loss"] if use_rlt else result
    loss.backward()
    assert model.weight.grad is not None
    assert model.weight.grad.abs().item() > 0
    assert model.training
    assert calls == [augmentation]

    # The SFT switch must not disable other training call sites or enable eval augmentation.
    model._preprocess_observation(observation, train=True)
    model._preprocess_observation(observation, train=False)
    assert calls == [augmentation, True, False]


def test_sft_augmentation_scope_restored_after_failure(monkeypatch):
    model = _small_model(augmentation=False)
    observation = _observation()
    calls = _record_preprocessor(monkeypatch)

    def failing_forward(self, observation, actions):
        self._preprocess_observation(observation, train=True)
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(pi0_pytorch.PI0Pytorch, "forward", failing_forward)
    with pytest.raises(RuntimeError, match="upstream failure"):
        model.sft_forward((observation, torch.zeros(1, 2, 3)))

    assert model.training
    model._preprocess_observation(observation, train=True)
    assert calls == [False, True]


def test_openpi_keeps_augmentation_by_default():
    assert OpenPi0Config().sft_image_augmentation is True
