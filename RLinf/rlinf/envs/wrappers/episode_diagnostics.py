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

"""Non-training exports for comparing LBench rollouts with SFT demonstrations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def as_numpy(value: Any) -> np.ndarray:
    """Read a CPU array without modifying its tensor or array source."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def json_value(value: Any) -> Any:
    """Convert recorded scalar metadata, preserving nonfinite values as text."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return json_value(as_numpy(value).tolist())
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported diagnostic metadata type: {type(value)}")


def write_episode(
    destination: str,
    buffer: dict,
    success: bool,
    fps: int,
    image_interval: int,
) -> None:
    """Save T transitions, T+1 observations and three full-length videos.

    Video frame i and state i precede action i; frame T is the terminal
    observation. NPZ image samples retain lossless raw policy-input images.
    The .complete marker is written only after all video encoders close.
    """
    import imageio.v2 as imageio

    count = len(buffer["actions"])
    if count < 1 or fps < 1 or image_interval < 1:
        raise ValueError("Diagnostic episodes require transitions and positive rates")
    for key in ("observations", "infos", "rewards", "terminated", "truncated"):
        if len(buffer[key]) != count + 1:
            raise ValueError(f"Diagnostic {key} is not aligned to T+1 observations")
    if len(buffer["policy_actions"]) != count:
        raise ValueError("Diagnostic proposed policy actions must have T entries")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    partial = target.with_name(target.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    observations = buffer["observations"]
    arrays = {
        "observation.state": np.stack([as_numpy(x["states"]) for x in observations]),
        "action": np.stack([as_numpy(x) for x in buffer["actions"]]),
        "policy_action": np.stack([as_numpy(x) for x in buffer["policy_actions"]]),
        "reward": np.array(
            [as_numpy(x).item() for x in buffer["rewards"][1:]], dtype=np.float32
        ),
        "terminated": np.array(
            [as_numpy(x).item() for x in buffer["terminated"][1:]], dtype=bool
        ),
        "truncated": np.array(
            [as_numpy(x).item() for x in buffer["truncated"][1:]], dtype=bool
        ),
    }
    for key in ("observation.state", "action", "policy_action"):
        expected = (count + (key == "observation.state"), 8)
        if arrays[key].shape != expected:
            raise ValueError(
                f"LBench diagnostic {key}: expected {expected}, got {arrays[key].shape}"
            )
    np.savez_compressed(partial / "transitions.npz", **arrays)
    with (partial / "step_info.jsonl").open("w") as stream:
        for index, info in enumerate(buffer["infos"]):
            stream.write(
                json.dumps(
                    {"observation_index": index, "info": json_value(info)},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    indices = np.arange(0, count, image_interval)
    sampled = {"observation_indices": indices}
    camera_keys = {
        "obs_camera": "main_images",
        "obs_camera_2": "wrist_images",
        "realsense": "extra_view_images",
    }
    camera_shapes = {}
    for camera, key in camera_keys.items():
        first = as_numpy(observations[0][key])
        camera_shapes[camera] = list(first.shape)
        if first.dtype != np.uint8 or first.ndim != 3 or first.shape[-1] != 3:
            raise ValueError("Diagnostic images must be uint8 HWC RGB")
        with imageio.get_writer(
            str(partial / f"{camera}.mp4"),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-threads", "1", "-preset", "veryfast"],
        ) as writer:
            for obs in observations:
                writer.append_data(as_numpy(obs[key]))
        sampled[camera] = np.stack([as_numpy(observations[i][key]) for i in indices])
    np.savez_compressed(partial / "raw_observation_frames.npz", **sampled)
    metadata = {
        "schema_version": 1,
        "success": bool(success),
        "transitions": count,
        "video_frames": count + 1,
        "fps": fps,
        "video_is_lossy": True,
        "lossless_image_interval": image_interval,
        "camera_shapes": camera_shapes,
        "instruction": json_value(observations[0]["task_descriptions"]),
        "alignment": "state[i]/video[i] precede action[i]; reward[i] belongs to that transition; state[T] is terminal",
        "final_info": json_value(buffer["infos"][-1]),
    }
    (partial / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (partial / ".complete").write_text(f"{count}\n")
    partial.rename(target)


def write_ppo_batch(destination: str, batch: dict, step: int, rank: int) -> None:
    """Save pre-update PPO targets and masks, omitting large model inputs."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"step_{step:06d}_rank_{rank:03d}.npz"
    keys = (
        "rewards",
        "dones",
        "prev_values",
        "prev_logprobs",
        "advantages",
        "returns",
        "loss_mask",
        "loss_mask_sum",
        "action_mask",
        "versions",
    )
    arrays = {
        key: as_numpy(batch[key])
        for key in keys
        if isinstance(batch.get(key), torch.Tensor)
    }
    # Exclusive creation prevents accidental replacement of earlier evidence.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
