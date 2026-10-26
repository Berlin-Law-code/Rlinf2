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

"""Check diagnostic exports against physical transitions and real video files."""

import json

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from test_lbench_env import FakeVectorClient, env_cfg

from rlinf.envs.lbench.lbench_env import LBenchEnv
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.envs.wrappers.episode_diagnostics import write_episode, write_ppo_batch


class VideoClient(FakeVectorClient):
    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        return self.resize(result)

    def step(self, commands):
        result = super().step(commands)
        for i, raw in enumerate(result[-1]):
            if i == 1:
                raw["robot"]["metric"]["episode_sr"] = 0.0
                result[-2][i]["reward_components"]["success"] = 0.0
        return self.resize(result)

    @staticmethod
    def resize(result):
        for raw in result[-1]:
            for sensor in raw["robot"]["sensors"].values():
                sensor["rgb"] = np.full(
                    (16, 16, 3), sensor["rgb"][0, 0, 0], dtype=np.uint8
                )
        return result


def test_diagnostic_keeps_success_failure_terminal_alignment_and_video(tmp_path):
    cfg = env_cfg(2)
    cfg["auto_reset"] = False
    base = LBenchEnv(cfg, 2, 0, 1, None, vector_client=VideoClient(2))
    env = CollectEpisode(
        base,
        str(tmp_path),
        num_envs=2,
        export_format="diagnostic",
        fps=30,
        max_pending_exports=1,
        diagnostic_image_interval=1,
    )
    env.reset()
    actions = np.full((2, 4, 8), 0.7, dtype=np.float32)
    env.chunk_step(actions)
    env.flush_collection()
    paths = sorted(tmp_path.glob("rank_*"))
    assert len(paths) == 2
    for index, path in enumerate(paths):
        meta = json.loads((path / "metadata.json").read_text())
        assert meta["success"] is (index == 0)
        length = index + 1
        assert meta["transitions"] == length  # excludes padded sub-actions
        assert (path / ".complete").exists()
        with np.load(path / "transitions.npz") as a:
            assert a["observation.state"].shape == (length + 1, 8)
            assert a["reward"].shape == (length,)
            assert a["terminated"][-1]
            np.testing.assert_allclose(a["policy_action"], 0.7)
            np.testing.assert_equal(a["action"][:, -1], 1.0)
        for camera in ["obs_camera", "obs_camera_2", "realsense"]:
            with imageio.get_reader(path / f"{camera}.mp4") as video:
                assert video.count_frames() == length + 1
        with np.load(path / "raw_observation_frames.npz") as images:
            assert images["obs_camera"][0, 0, 0, 0] == index
            assert len(images["observation_indices"]) == length
    env.close()


def test_diagnostic_rejects_misaligned_buffers(tmp_path):
    with pytest.raises(ValueError, match="aligned"):
        write_episode(
            str(tmp_path / "bad"), {"actions": [1], "observations": []}, False, 30, 20
        )
    assert not (tmp_path / "bad/.complete").exists()


def test_ppo_export_does_not_mutate_or_export_images(tmp_path):
    values = torch.tensor([1.0, 2.0])
    batch = {
        "advantages": values,
        "loss_mask": torch.tensor([True, False]),
        "forward_inputs": {"image": torch.zeros(4, 4)},
    }
    write_ppo_batch(str(tmp_path), batch, 41, 0)
    with np.load(tmp_path / "step_000041_rank_000.npz") as a:
        assert set(a.files) == {"advantages", "loss_mask"}
        np.testing.assert_equal(a["advantages"], [1, 2])
    assert torch.equal(values, torch.tensor([1.0, 2.0]))
    with pytest.raises(FileExistsError):
        write_ppo_batch(str(tmp_path), batch, 41, 0)
