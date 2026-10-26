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

"""Exercise real Critic construction and loading without the large backbone."""

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.openpi import _load_state_dict
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
    PI0Pytorch,
)


@pytest.fixture
def tiny_backbone(monkeypatch):
    """Replace only the expensive parent; keep production head and loader."""

    def initialize(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.action_in_proj = nn.Linear(8, 8)

    monkeypatch.setattr(PI0Pytorch, "__init__", initialize)


def make_model(config_name="pi05_lbench_joint", widths=None, add_value_head=True):
    """Construct the real OpenPI subclass with an inexpensive parent."""
    return OpenPi0ForRLActionPrediction(
        OpenPi0Config(
            config_name=config_name,
            pi05=True,
            action_horizon=20,
            action_chunk=20,
            add_value_head=add_value_head,
            value_after_vlm=True,
            value_head_hidden_sizes=widths,
        )
    )


@pytest.mark.parametrize(
    "name,widths,expected_parameters",
    [
        ("pi05_lbench_joint", None, 1213441),
        ("pi05_libero", None, 2754561),
        ("pi05_lbench_joint", OmegaConf.create([1024, 512, 256]), 2754561),
    ],
)
def test_task_defaults_and_hydra_override(
    tiny_backbone, name, widths, expected_parameters
):
    model = make_model(name, widths)
    assert sum(p.numel() for p in model.value_head.parameters()) == expected_parameters
    expected = [1024, 512, 256] if expected_parameters == 2754561 else [512, 256, 128]
    layers = [m for m in model.value_head.mlp if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in layers] == list(
        zip([2048, *expected], [*expected, 1])
    )
    assert model.config.config_name == name


def test_sft_load_and_wide_critic_checkpoint_roundtrip(tiny_backbone):
    sft = make_model(add_value_head=False)
    wide = make_model(widths=(1024, 512, 256))
    _load_state_dict(
        wide, sft.state_dict(), config_name="pi05_lbench_joint", add_value_head=True
    )
    assert torch.equal(wide.action_in_proj.weight, sft.action_in_proj.weight)
    inputs = torch.randn(4, 2048)
    optimizer = torch.optim.AdamW(wide.value_head.parameters(), lr=1e-4)
    before = wide.value_head.mlp[0].weight.detach().clone()
    loss = (wide.value_head(inputs) - 1).square().mean()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in wide.value_head.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, wide.value_head.mlp[0].weight)
    restored = make_model(widths=(1024, 512, 256))
    _load_state_dict(
        restored,
        wide.state_dict(),
        config_name="pi05_lbench_joint",
        add_value_head=True,
    )
    torch.testing.assert_close(
        restored.value_head(inputs), wide.value_head(inputs), rtol=0, atol=0
    )
    with pytest.raises(RuntimeError, match="size mismatch"):
        _load_state_dict(
            make_model(),
            wide.state_dict(),
            config_name="pi05_lbench_joint",
            add_value_head=True,
        )


@pytest.mark.parametrize("widths", [[], [0], [-1], [True], [1.5], "1024,512,256"])
def test_invalid_widths_rejected(tiny_backbone, widths):
    with pytest.raises(ValueError, match="positive integers"):
        make_model(widths=widths)
