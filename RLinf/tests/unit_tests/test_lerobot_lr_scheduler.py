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

from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

# Reference outputs from LeRobot's CosineDecayWithWarmupSchedulerConfig with
# peak_lr=2.5e-5, decay_lr=2.5e-6, actual_warmup_steps=1000 and decay_steps=30000.
# In particular, the cosine has already advanced at the end of warmup.
LEROBOT_MULTIPLIERS = {
    0: 0.000999000999000999,
    999: 0.999000999000999,
    1000: 0.997534852915723,
    15000: 0.55,
    30000: 0.1,
    31000: 0.1,
}


@pytest.mark.parametrize("minimum", [{"min_lr": 2.5e-6}, {"min_lr_rate": 0.1}])
def test_lerobot_schedule_matches_reference_optimizer_trajectory(minimum):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=2.5e-5)
    scheduler = get_lr_scheduler(
        "lerobot_cosine",
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=30000,
        **minimum,
    )
    for step in range(31001):
        if step in LEROBOT_MULTIPLIERS:
            assert optimizer.param_groups[0]["lr"] == pytest.approx(
                2.5e-5 * LEROBOT_MULTIPLIERS[step], rel=1e-12, abs=1e-18
            )
        optimizer.step()
        scheduler.step()


@pytest.mark.parametrize("name", ["openpi_cosine", "ref_warmup_cosine"])
def test_existing_openpi_schedule_still_restarts_cosine_after_warmup(name):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=2.5e-5)
    scheduler = get_lr_scheduler(
        name,
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=30000,
        min_lr=2.5e-6,
    )
    multiplier = scheduler.lr_lambdas[0]
    assert multiplier(0) == pytest.approx(1 / 1001)
    assert multiplier(1000) == 1.0
    assert multiplier(15500) == pytest.approx(0.55)
    assert multiplier(30000) == pytest.approx(0.1)
    assert multiplier(60000) == pytest.approx(0.1)


def test_lerobot_schedule_supports_zero_warmup_and_minimum_ratio_precedence():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=2.5e-5)
    scheduler = get_lr_scheduler(
        "lerobot_cosine",
        optimizer,
        num_warmup_steps=0,
        num_training_steps=30000,
        min_lr=2.5e-6,
        min_lr_rate=0.2,
    )
    multiplier = scheduler.lr_lambdas[0]
    assert multiplier(0) == 1.0
    assert multiplier(15000) == pytest.approx(0.6)
    assert multiplier(30000) == pytest.approx(0.2)
    assert multiplier(60000) == pytest.approx(0.2)


@pytest.mark.parametrize("warmup,duration", [(-1, 30000), (1000, 0)])
def test_lerobot_schedule_rejects_invalid_durations(warmup, duration):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=2.5e-5)
    with pytest.raises(ValueError, match="positive num_training_steps"):
        get_lr_scheduler("lerobot_cosine", optimizer, warmup, duration)
