"""CPU-only LBench adapter dry run using the protocol fake worker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make direct ``python toolkits/...py`` invocation work without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.envs.lbench.lbench_env import LBenchEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lbench-root",
        default="/home/berlin/flywheel/LBench_multi_env",
    )
    parser.add_argument("--num-envs", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {
        "lbench_root": str(Path(args.lbench_root).resolve()),
        "isaac_python": sys.executable,
        "backend_factory_path": (
            "tests.fakes.fake_simbox_worker:create_lbench_eval_fake_backend"
        ),
        "backend_config": {},
        "total_num_envs": args.num_envs,
        "group_size": 1,
        "auto_reset": True,
        "max_episode_steps": 4,
        "is_eval": True,
        "camera_mapping": {
            "main": "obs_camera",
            "wrist": "obs_camera_2",
            "additional": "realsense",
        },
    }
    env = LBenchEnv(
        cfg=cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, _ = env.reset()
        chunk = np.zeros((args.num_envs, 2, 8), dtype=np.float32)
        chunk[..., -1] = 1.0
        _, rewards, terminations, truncations, infos_list = env.chunk_step(chunk)
        infos = infos_list[-1]
        result = {
            "num_envs": args.num_envs,
            "observation_shapes": {
                key: list(value.shape)
                for key, value in obs.items()
                if hasattr(value, "shape")
            },
            "task_descriptions": obs["task_descriptions"],
            "reward_shape": list(rewards.shape),
            "terminations": terminations.tolist(),
            "truncations": truncations.tolist(),
            "has_lbench_metadata": "lbench_metadata" in infos["final_info"],
        }
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
