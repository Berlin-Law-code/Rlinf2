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
"""Stream process/GPU memory snapshots without retaining history in memory."""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import psutil


def cgroup_memory(root: Path = Path("/sys/fs/cgroup")) -> dict:
    """Read container limits and distinguish anonymous memory from file cache."""
    result = {}
    for filename in ("memory.current", "memory.max", "memory.peak"):
        path = root / filename
        if path.is_file():
            value = path.read_text().strip()
            result[filename] = int(value) if value != "max" else None
    for filename in ("memory.stat", "memory.events"):
        path = root / filename
        if path.is_file():
            result[filename] = {
                key: int(value)
                for key, value in (
                    line.split() for line in path.read_text().splitlines()
                )
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    parent = psutil.Process(args.parent_pid)
    birth = parent.create_time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", buffering=1) as output:
        while parent.is_running() and parent.create_time() == birth:
            processes = []
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    memory = process.memory_full_info()
                    processes.append(
                        {
                            "pid": process.pid,
                            "name": process.name(),
                            "rss_gib": memory.rss / 2**30,
                            "uss_gib": memory.uss / 2**30,
                            "pss_gib": memory.pss / 2**30,
                            "threads": process.num_threads(),
                            "open_fds": process.num_fds(),
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            record = {
                "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "host_available_gib": psutil.virtual_memory().available / 2**30,
                "cgroup": cgroup_memory(),
                "processes": processes,
            }
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                record["gpu_index_used_mib_total_mib_util_percent"] = (
                    result.stdout.strip().splitlines()
                )
            except (OSError, subprocess.SubprocessError) as error:
                record["gpu_query_error"] = type(error).__name__
            output.write(json.dumps(record) + "\n")
            # Do not accumulate samples or process objects between snapshots.
            del processes, record
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except psutil.NoSuchProcess:
        pass
