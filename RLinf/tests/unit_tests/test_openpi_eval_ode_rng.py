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

"""Exercise the real denoising loop and mean/std formulas with a CPU velocity stub."""

import pytest
import torch

from rlinf.models.embodiment.openpi import openpi_action_model as module
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.utils.eval_initial_noise import make_eval_initial_noise


@pytest.fixture(autouse=True)
def preserve_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def sampler(noise_method="flow_sde"):
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.config = OpenPi0Config(
        action_horizon=4,
        action_chunk=2,
        action_env_dim=8,
        num_steps=4,
        noise_method=noise_method,
    )
    model.action_in_proj = torch.nn.Linear(32, 1, bias=False)
    model.use_vlm_value = False
    model.get_velocity = lambda state, x_t, *args: (torch.full_like(x_t, 0.1), None)
    calls = []
    original_noise = model.sample_noise

    def record_noise(shape, device):
        calls.append(tuple(shape))
        return original_noise(shape, device)

    model.sample_noise = record_noise
    return model, calls


def initial_noise():
    return make_eval_initial_noise(
        [42, 51], batch_size=2, action_horizon=4, action_dim=32
    )


def run(model, noise, mode):
    return model._sample_actions_with_prefix_cache(
        torch.zeros(2, 32), None, None, None, noise=noise, mode=mode
    )


def generator_from_global():
    generator = torch.Generator(device="cpu")
    generator.set_state(torch.random.get_rng_state())
    return generator


def reference(model, noise, generator, mode):
    """Previous loop: draw every denoising step, even when std is zero."""

    def draw(shape):
        return torch.normal(0.0, 1.0, size=shape, generator=generator)

    x_t = noise.clone() if noise is not None else draw((2, 4, 32))
    chains = [x_t]
    for index in range(model.config.num_steps):
        method = (
            model.config.noise_method if mode == "train" and index == 1 else "flow_ode"
        )
        mean, std, _, _ = model.sample_mean_var_val(
            x_t, index, torch.zeros(2, 32), None, None, method, model.config.num_steps
        )
        x_t = mean + draw(x_t.shape) * std
        chains.append(x_t)
    return torch.stack(chains, dim=1)


def test_paired_eval_actual_denoising_loop_does_not_advance_global_rng():
    model, calls = sampler()
    noise = initial_noise()
    old_formula = reference(model, noise, generator_from_global(), "eval")
    before = torch.random.get_rng_state().clone()
    result = run(model, noise, "eval")
    assert calls == []
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(result["chains"], old_formula, rtol=0, atol=0)
    repeated = run(model, noise, "eval")
    torch.testing.assert_close(repeated["actions"], result["actions"], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


def test_eval_without_explicit_noise_still_samples_initial_latent():
    model, calls = sampler()
    generator = generator_from_global()
    expected_initial = torch.normal(0.0, 1.0, size=(2, 4, 32), generator=generator)
    result = run(model, None, "eval")
    assert calls == [(2, 4, 32)]
    torch.testing.assert_close(result["chains"][:, 0], expected_initial, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.random.get_rng_state(), generator.get_state(), rtol=0, atol=0
    )


@pytest.mark.parametrize("noise_method", ["flow_sde", "flow_ode"])
@pytest.mark.parametrize("provide_initial", [True, False])
def test_training_keeps_every_draw_and_exact_previous_sampling_result(
    monkeypatch, noise_method, provide_initial
):
    model, calls = sampler(noise_method)
    monkeypatch.setattr(module.random, "randint", lambda *_: 1)
    noise = initial_noise() if provide_initial else None
    generator = generator_from_global()
    expected = reference(model, noise, generator, "train")
    result = run(model, noise, "train")
    assert len(calls) == model.config.num_steps + int(not provide_initial)
    torch.testing.assert_close(result["chains"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.random.get_rng_state(), generator.get_state(), rtol=0, atol=0
    )


def test_eval_preserves_sampling_if_ode_override_returns_nonzero_std():
    model, calls = sampler()
    original_mean_var = model.sample_mean_var_val

    def nonzero_std(*args, **kwargs):
        mean, std, value, velocity = original_mean_var(*args, **kwargs)
        std = std.clone()
        std[0, 0, 0] = 0.25
        return mean, std, value, velocity

    model.sample_mean_var_val = nonzero_std
    noise = initial_noise()
    generator = generator_from_global()
    expected = reference(model, noise, generator, "eval")
    result = run(model, noise, "eval")
    assert len(calls) == model.config.num_steps
    torch.testing.assert_close(result["chains"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.random.get_rng_state(), generator.get_state(), rtol=0, atol=0
    )
