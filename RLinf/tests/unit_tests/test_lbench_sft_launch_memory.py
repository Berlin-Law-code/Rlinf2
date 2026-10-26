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

"""Verify allocator controls reach both preflight and the training process."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("threshold", [None, "262144"])
def test_aligned_launch_inherits_allocator_thresholds(tmp_path, threshold):
    root = Path(__file__).resolve().parents[2]
    calls = tmp_path / "calls.jsonl"
    python = tmp_path / "record_python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(calls)!r}, 'a') as stream:\n"
        "    stream.write(json.dumps({'args': sys.argv[1:], "
        "'mmap': os.environ['MALLOC_MMAP_THRESHOLD_'], "
        "'trim': os.environ['MALLOC_TRIM_THRESHOLD_']}) + '\\n')\n"
    )
    python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        RLINF_ROOT=str(root),
        RLINF_PYTHON=str(python),
        RLINF_GPUS="0,1,2,3",
        LBENCH_SFT_OUTPUT=str(tmp_path / "output"),
        # Stale generic exports must not silently defeat the aligned recipe.
        MALLOC_MMAP_THRESHOLD_="33554432",
        MALLOC_TRIM_THRESHOLD_="33554432",
    )
    for name in (
        "LBENCH_ALIGNED_MALLOC_MMAP_THRESHOLD",
        "LBENCH_ALIGNED_MALLOC_TRIM_THRESHOLD",
    ):
        env.pop(name, None)
        if threshold is not None:
            env[name] = threshold
    subprocess.run(
        [
            "bash",
            str(root / "scripts/lbench/remote/06_sft_lerobot_aligned.sh"),
            "runner.max_steps=5",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(recorded) == 2
    assert recorded[0]["args"] == ["-"]
    assert recorded[1]["args"][0] == "examples/sft/train_vla_sft.py"
    assert "runner.max_steps=5" in recorded[1]["args"]
    for call in recorded:
        assert call["mmap"] == (threshold or "131072")
        assert call["trim"] == (threshold or "131072")
