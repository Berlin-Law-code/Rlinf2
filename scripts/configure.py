"""Create relocated RECAP configurations without modifying recorded evidence."""

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OLD_OUTPUT = "/root/autodl-tmp/outputs/lbench_recap_round1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/datasets"))
    parser.add_argument("--models-root", type=Path, default=Path("/root/autodl-tmp/models"))
    parser.add_argument("--initial-policy", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/recap_round1")
    parser.add_argument("--value-checkpoint", type=Path)
    parser.add_argument("--destination", type=Path, default=ROOT / "configs/local")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    mapping = {
        "/root/autodl-tmp/datasets": str(args.data_root.resolve()),
        "/root/autodl-tmp/models": str(args.models_root.resolve()),
        OLD_OUTPUT: str(args.output_root.resolve()),
        "/root/autodl-tmp/Berlin/RLinf": str(ROOT / "RLinf"),
    }

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            for old, new in mapping.items():
                if value == old or value.startswith(old + "/"):
                    return new + value[len(old):]
        return value

    for stage in ["returns", "value", "advantages", "policy"]:
        if stage == "value":
            source = ROOT / "configs/recorded/value_final.json"
        elif stage == "policy":
            source = ROOT / "configs/recorded/policy_final.yaml"
        else:
            source = ROOT / f"configs/current/{stage}.json"
        config = relocate(yaml.safe_load(source.read_text()))
        if stage == "policy" and args.initial_policy:
            config["actor"]["model"]["model_path"] = str(args.initial_policy.resolve())
        if stage == "advantages":
            checkpoint = args.value_checkpoint or args.output_root / "value/checkpoints/global_step_8000"
            config["advantage"]["value_checkpoint"] = str(checkpoint.resolve())
        config["hydra"] = {"run": {"dir": "."}, "output_subdir": None, "job": {"chdir": False}}
        target = args.destination / f"{stage}.yaml"
        target.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        print(json.dumps({"stage": stage, "source": str(source), "config": str(target)}))


if __name__ == "__main__":
    main()

