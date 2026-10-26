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

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "convert_lerobot_stats_to_openpi",
    ROOT / "toolkits/lerobot/convert_lerobot_stats_to_openpi.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_CONVERTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONVERTER)


def _source_stats() -> dict:
    return {
        feature: {
            "mean": [index + 0.123456789 for index in range(8)],
            "std": [1.0] * 8,
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
            "min": [-2.0] * 8,
            "max": [2.0] * 8,
        }
        for feature in ("observation.state", "action")
    }


def test_conversion_maps_keys_and_matches_lerobot_float32_stats():
    result = _CONVERTER.convert_lerobot_stats(_source_stats())
    assert set(result["norm_stats"]) == {"state", "actions"}
    assert set(result["norm_stats"]["state"]) == {"mean", "std", "q01", "q99"}
    assert (
        result["norm_stats"]["state"]["mean"]
        == np.asarray(
            _source_stats()["observation.state"]["mean"], dtype=np.float32
        ).tolist()
    )
    assert result["norm_stats"]["actions"]["q99"] == [1.0] * 8


@pytest.mark.parametrize("field", ["mean", "std", "q01", "q99"])
def test_conversion_rejects_wrong_dimensions(field):
    source = _source_stats()
    source["action"][field] = [0.0] * 7
    with pytest.raises(ValueError, match="must be 8-D"):
        _CONVERTER.convert_lerobot_stats(source)


def test_conversion_rejects_invalid_quantile_order():
    source = _source_stats()
    source["action"]["q99"][2] = -1.0
    with pytest.raises(ValueError, match="greater than q01"):
        _CONVERTER.convert_lerobot_stats(source)
