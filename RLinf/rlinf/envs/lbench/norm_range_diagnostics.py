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

"""Optional evaluation counters for state/action quantile-range coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .observation import _gripper_scalar


class NormRangeDiagnostics:
    """Count physical vectors against multiple reference normalization assets.

    Counters never modify observations, actions, or model statistics. Joint
    aggregates exclude the binary gripper, whose histogram endpoint can be
    slightly below one. Raw trajectories and images are not retained.
    """

    def __init__(self, stats_paths):
        self.references = {}
        self.assets = {}
        self.episodes = {}
        for label, path in stats_paths.items():
            path = Path(path).expanduser().resolve()
            content = path.read_bytes()
            stats = json.loads(content)["norm_stats"]
            self.references[str(label)] = {}
            for kind in ("state", "actions"):
                lo = np.asarray(stats[kind]["q01"], dtype=np.float32)
                hi = np.asarray(stats[kind]["q99"], dtype=np.float32)
                if lo.shape != (8,) or hi.shape != (8,):
                    raise ValueError("Range diagnostics require 8-D statistics")
                if not np.isfinite([lo, hi]).all() or (hi <= lo).any():
                    raise ValueError("Range diagnostics require finite positive ranges")
                self.references[str(label)][kind] = (lo, hi)
            self.assets[str(label)] = {
                "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
            }

    @staticmethod
    def state(raw, gripper_joint_threshold):
        """Match the numeric part of production map_observation_batch."""
        robot = raw["robot"]
        positions = np.asarray(robot["joints_state"]["positions"], dtype=np.float32)
        return np.r_[
            positions[:7],
            np.float32(
                _gripper_scalar(
                    robot, positions, joint_threshold=gripper_joint_threshold
                )
            ),
        ].astype(np.float32)

    def observe(self, episode_id, group, vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (8,) or not np.isfinite(vector).all():
            raise ValueError("Range diagnostics require finite 8-D vectors")
        kind = "state" if group.endswith("state") else "actions"
        episode = self.episodes.setdefault(episode_id, {})
        for label, reference in self.references.items():
            groups = episode.setdefault(label, {})
            if group not in groups:
                groups[group] = {
                    "vectors": 0,
                    "below_per_dim": np.zeros(8, dtype=np.int64),
                    "above_per_dim": np.zeros(8, dtype=np.int64),
                    "outside_gt_0_01_per_dim": np.zeros(8, dtype=np.int64),
                    "any_joint_outside_vectors": 0,
                    "low_token_saturated_per_dim": np.zeros(8, dtype=np.int64),
                    "high_token_saturated_per_dim": np.zeros(8, dtype=np.int64),
                    "normalized_min": np.full(8, np.inf),
                    "normalized_max": np.full(8, -np.inf),
                }
            counter = groups[group]
            lo, hi = reference[kind]
            # Use the same float32 arithmetic as LeRobotNormalize.
            normalized = np.float32(2) * (vector - lo) / (hi - lo) - np.float32(1)
            below, above = normalized < -1, normalized > 1
            counter["vectors"] += 1
            counter["below_per_dim"] += below
            counter["above_per_dim"] += above
            counter["outside_gt_0_01_per_dim"] += np.abs(normalized) > 1.01
            counter["any_joint_outside_vectors"] += int((below | above)[:7].any())
            if kind == "state":
                # np.digitize uses -1 below the first edge and 255 for every
                # value >= 1 - 2/256, including values beyond the upper range.
                counter["low_token_saturated_per_dim"] += normalized < -1
                counter["high_token_saturated_per_dim"] += normalized >= 1 - 2 / 256
            counter["normalized_min"] = np.minimum(
                counter["normalized_min"], normalized
            )
            counter["normalized_max"] = np.maximum(
                counter["normalized_max"], normalized
            )

    def finish(self, episode_id):
        groups = self.episodes.pop(episode_id, {})
        return {
            "schema_version": 1,
            "assets": self.assets,
            "references": {
                label: {
                    group: {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in counter.items()
                    }
                    for group, counter in counters.items()
                }
                for label, counters in groups.items()
            },
        }
