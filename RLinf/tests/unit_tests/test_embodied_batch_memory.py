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

"""Value, ordering, ownership, and storage-lifetime checks for rollout batches."""

import weakref

import pytest
import torch

from rlinf.data.schema.embodied_types import Trajectory, convert_trajectories_to_batch
from rlinf.utils.nested_dict_process import (
    process_nested_dict_for_adv,
    process_nested_dict_for_train,
)


def trajectory(offset=0, batch=2):
    return Trajectory(
        rewards=torch.arange(12 * batch, dtype=torch.float32).reshape(12, batch, 1)
        + offset,
        dones=torch.zeros(15, batch, 1, dtype=torch.bool),
        prev_values=torch.arange(15 * batch, dtype=torch.float32).reshape(15, batch, 1),
        forward_inputs={
            "images": torch.arange(12 * batch * 9, dtype=torch.uint8).reshape(
                12, batch, 3, 3
            ),
            "chains": torch.arange(12 * batch * 5, dtype=torch.float32).reshape(
                12, batch, 5
            )
            + offset,
        },
    )


def equal_dict(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, dict):
            equal_dict(value, right[key])
        else:
            torch.testing.assert_close(value, right[key], rtol=0, atol=0)
            assert value.dtype == right[key].dtype


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("epochs", [1, 3])
def test_consumed_batch_matches_original_advantage_and_shuffle_order(count, epochs):
    source = [trajectory(offset=1000 * i, batch=i + 1) for i in range(count)]
    original = convert_trajectories_to_batch(source)
    expected_adv = process_nested_dict_for_adv(original, epochs)
    shared = convert_trajectories_to_batch(source, copy_single_trajectory=False)
    if count == 1:
        assert shared["rewards"] is source[0].rewards
        assert shared["forward_inputs"]["images"] is source[0].forward_inputs["images"]
        assert shared["forward_inputs"] is not source[0].forward_inputs
        assert original["rewards"].data_ptr() != source[0].rewards.data_ptr()
    source.clear()
    actual_adv = process_nested_dict_for_adv(shared, epochs, consume=True)
    assert shared == {}
    equal_dict(actual_adv, expected_adv)
    # prev_values/dones have one bootstrap step per epoch, as in real rollouts.
    if epochs == 1:
        actual_adv.pop("prev_values")
        actual_adv.pop("dones")
        expected_adv.pop("prev_values")
        expected_adv.pop("dones")
    n = actual_adv["rewards"].shape[0] * actual_adv["rewards"].shape[1]
    shuffle = torch.randperm(n, generator=torch.Generator().manual_seed(7))
    expected_train = process_nested_dict_for_train(expected_adv, shuffle)
    actual_train = process_nested_dict_for_train(actual_adv, shuffle, consume=True)
    assert actual_adv == {}
    equal_dict(actual_train, expected_train)
    assert original and expected_adv  # The default path retains its inputs.


@pytest.mark.parametrize("phase", ["adv", "train"])
def test_source_field_is_freed_before_reordering_the_next_field(monkeypatch, phase):
    first = torch.arange(48, dtype=torch.float32).reshape(12, 2, 2).clone()
    second = torch.arange(48, dtype=torch.float32).reshape(12, 2, 2).clone()
    first_ref = weakref.ref(first)
    second_id = id(second)
    data = {"nested": {"first": first, "second": second}}
    del first, second
    reshape = torch.Tensor.reshape
    reached_second = []

    def checked_reshape(self, *shape):
        if id(self) == second_id:
            assert first_ref() is None, (
                "Previous source tensor survived into the next field"
            )
            reached_second.append(True)
        return reshape(self, *shape)

    monkeypatch.setattr(torch.Tensor, "reshape", checked_reshape)
    if phase == "adv":
        output = process_nested_dict_for_adv(data, 3, consume=True)
    else:
        output = process_nested_dict_for_train(
            data, torch.arange(24).flip(0), consume=True
        )
    assert reached_second and data == {} and output["nested"]["first"].numel() == 48


def test_single_trajectory_reuse_preserves_empty_and_optional_fields():
    assert convert_trajectories_to_batch([], copy_single_trajectory=False) == {}
    assert (
        convert_trajectories_to_batch([Trajectory()], copy_single_trajectory=False)
        == {}
    )
    source = trajectory()
    source.next_obs = {"state": torch.zeros(12, 2, 8)}
    copied = convert_trajectories_to_batch([source])
    reused = convert_trajectories_to_batch([source], copy_single_trajectory=False)
    equal_dict(copied, reused)
    assert reused["next_obs"]["state"] is source.next_obs["state"]
