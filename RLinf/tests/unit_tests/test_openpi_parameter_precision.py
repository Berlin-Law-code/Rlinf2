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

from rlinf.models.embodiment.openpi.precision import apply_parameter_precision


class _Backbone(torch.nn.Linear):
    def to_bfloat16_for_selected_params(self, precision):
        assert precision == "bfloat16"
        self.to(dtype=torch.bfloat16)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _Backbone(2, 1, bias=False).to(torch.bfloat16)
        self.action_in_proj = torch.nn.Linear(2, 1, bias=False)


def test_float32_before_load_preserves_weight_values_and_adam_state_precision():
    model = _Model()
    # These values contain low bits that copying into BF16 would discard.
    source = {
        "paligemma_with_expert.weight": torch.tensor([[1.0001, 1.0002]]),
        "action_in_proj.weight": torch.tensor([[0.0001234, -0.0002345]]),
    }
    apply_parameter_precision(model, "float32")
    model.load_state_dict(source)
    apply_parameter_precision(model, "float32")
    for key, value in model.state_dict().items():
        assert torch.equal(value, source[key])
        assert value.dtype == torch.float32

    optimizer = torch.optim.AdamW(model.parameters())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.paligemma_with_expert(torch.ones(1, 2)).float().square().sum()
    loss.backward()
    optimizer.step()
    state = optimizer.state[model.paligemma_with_expert.weight]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_legacy_policy_preserves_mixed_parameters_and_freeze_flags():
    model = _Model()
    model.paligemma_with_expert.weight.requires_grad_(False)
    apply_parameter_precision(model)
    assert model.paligemma_with_expert.weight.dtype == torch.bfloat16
    assert model.action_in_proj.weight.dtype == torch.float32
    assert not model.paligemma_with_expert.weight.requires_grad


def test_precision_policy_rejects_unknown_values():
    with pytest.raises(ValueError, match="parameter_precision"):
        apply_parameter_precision(_Model(), "fp23")
