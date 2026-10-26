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

"""Convert LeRobot dataset statistics to OpenPI normalization assets."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_FIELDS = ("mean", "std", "q01", "q99")


def _feature_stats(
    source: Mapping[str, Any], feature: str, expected_dim: int
) -> dict[str, list[float]]:
    if feature not in source or not isinstance(source[feature], Mapping):
        raise ValueError(f"LeRobot stats are missing feature {feature!r}")

    converted: dict[str, list[float]] = {}
    for field in _FIELDS:
        values = source[feature].get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{feature}.{field} must be a numeric sequence")
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (expected_dim,):
            raise ValueError(
                f"{feature}.{field} must be {expected_dim}-D, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{feature}.{field} contains non-finite values")
        converted[field] = array.tolist()

    if np.any(np.asarray(converted["std"]) <= 0):
        raise ValueError(f"{feature}.std must be positive")
    if np.any(np.asarray(converted["q99"]) <= np.asarray(converted["q01"])):
        raise ValueError(f"{feature}.q99 must be greater than q01 in every dimension")
    return converted


def convert_lerobot_stats(
    source: Mapping[str, Any], expected_dim: int = 8
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Map LeRobot state/action stats to OpenPI's serialized key layout.

    Values are cast to float32, matching the tensors stored by LeRobot's
    normalization processor.
    """
    return {
        "norm_stats": {
            "state": _feature_stats(source, "observation.state", expected_dim),
            "actions": _feature_stats(source, "action", expected_dim),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-path",
        required=True,
        help="LeRobot meta/stats.json path or dataset root directory.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-dim", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats_path = pathlib.Path(args.stats_path).expanduser().resolve()
    if stats_path.is_dir():
        stats_path = stats_path / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"LeRobot stats file does not exist: {stats_path}")

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_path = output_dir / "norm_stats.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing OpenPI stats: {output_path}"
        )

    with stats_path.open(encoding="utf-8") as file:
        source = json.load(file)
    converted = convert_lerobot_stats(source, expected_dim=args.expected_dim)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote LeRobot-compatible OpenPI stats to: {output_path}")


if __name__ == "__main__":
    main()
