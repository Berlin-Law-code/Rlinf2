"""Load this inference bundle using the matching RLinf/OpenPI environment."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def load_policy(device: str = "cuda:0", bundle_dir: str | Path | None = None):
    """Load weights, CFG settings, bundled normalization and local tokenizer."""
    root = Path(bundle_dir or Path(__file__).parent).resolve()
    os.environ["OPENPI_DATA_HOME"] = str(root / "assets/openpi_cache")
    cfg = OmegaConf.load(root / "inference_config.yaml").model
    cfg.model_path = str(root)
    from rlinf.models.embodiment.openpi_cfg import get_model

    model = get_model(cfg).to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = load_policy(args.device)
    report = {"checkpoint_step": 30000, "model_loaded": True}
    if args.smoke:
        # Exercise all three cameras without a robot connection.
        image = torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
        observation = {
            "main_images": image,
            "wrist_images": image.clone(),
            "extra_view_images": image.clone(),
            "states": torch.zeros((1, 8), dtype=torch.float32),
            "task_descriptions": ["Stack the bowls."],
        }
        device = torch.device(args.device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ),
        ):
            actions, _ = model.predict_action_batch(observation, mode="eval")
        assert actions.shape == (1, 20, 8), actions.shape
        assert np.isfinite(actions).all()
        report.update(
            synthetic_inference_passed=True,
            returned_action_shape=list(actions.shape),
            action_horizon=model.config.action_horizon,
            execute_horizon=model.config.action_chunk,
            guidance_type=model.config.guidance_type,
            guidance_scale=model.config.cfgrl_guidance_scale,
            finite_actions=True,
            note="Synthetic input validates loading and inference, not task success.",
        )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
