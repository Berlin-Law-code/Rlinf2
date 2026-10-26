"""Export full-run TensorBoard losses and plot raw plus trailing moving averages."""

import argparse
import csv
import json
from pathlib import Path
import sys
import types

# TensorBoard's lightweight reader needs no TensorFlow runtime or GPU.
sys.modules["tensorboard.compat.notf"] = types.ModuleType("tensorboard.compat.notf")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "artifacts/lbench_recap_round1/tensorboard")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/loss_plots")
    parser.add_argument("--window", type=int, default=200)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    candidates = {"Value": [], "Policy (H50 / E20)": []}
    inventory = []
    for path in sorted(args.input_dir.glob("events.out.tfevents.*")):
        reader = EventAccumulator(str(path), size_guidance={"scalars": 0})
        reader.Reload()
        tags = reader.Tags()["scalars"]
        if "train/loss" not in tags:
            inventory.append({"file": path.name, "points": 0})
            continue
        name = "Value" if "train/cat_acc_best" in tags else "Policy (H50 / E20)"
        points = reader.Scalars("train/loss")
        inventory.append({"file": path.name, "run": name, "points": len(points), "last_step": points[-1].step})
        candidates[name].append((points[-1].step, len(points), points[-1].wall_time, path, reader))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    rows, summary = [], {}
    for ax, (name, runs), color in zip(axes, candidates.items(), ["#2563eb", "#e87524"]):
        if not runs:
            raise ValueError(f"No loss events for {name}")
        _, _, _, path, reader = max(runs, key=lambda item: item[:3])
        points = reader.Scalars("train/loss")
        x = np.array([point.step for point in points])
        y = np.array([point.value for point in points])
        if len(y) < args.window or not np.isfinite(y).all() or not (np.diff(x) > 0).all():
            raise ValueError(f"Invalid or too short series: {path}")
        average = np.convolve(y, np.ones(args.window) / args.window, mode="valid")
        ax.plot(x, y, color=color, alpha=0.22, lw=0.65, label="Raw loss")
        ax.plot(x[args.window-1:], average, color=color, lw=2.2, label=f"{args.window}-step moving average")
        ax.set_title(f"{name}  |  {len(y):,} steps", loc="left", pad=15, fontweight="bold")
        ax.set_xlabel("Global step (TensorBoard)")
        ax.set_ylabel("Training loss")
        ax.set_xlim(0, x[-1])
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", frameon=False)
        ax.text(0.98, 0.74, f"Last {args.window} mean: {y[-args.window:].mean():.4f}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#4b5563")
        summary[name] = {"source": path.name, "points": len(y), "first_step": int(x[0]), "last_step": int(x[-1]), "first_loss": float(y[0]), "last_loss": float(y[-1]), "first_200_mean": float(y[:200].mean()), "last_200_mean": float(y[-200:].mean()), "min": float(y.min()), "max": float(y.max()), "nonfinite_count": int((~np.isfinite(y)).sum())}
        extra = {}
        for tag in reader.Tags()["scalars"]:
            if "loss" in tag or tag in {"train/cat_acc_best", "train/cat_acc_neighbor", "train/value_spearman", "train/mae"}:
                tail = reader.Scalars(tag)[-200:]
                extra[tag] = {"points": len(reader.Scalars(tag)), "last_200_mean": sum(p.value for p in tail) / len(tail)}
        summary[name]["metrics"] = extra
        rows.extend((name, int(step), float(loss)) for step, loss in zip(x, y))
    fig.suptitle("lbench_recap_round1 | Training loss", fontsize=18, fontweight="bold", x=0.07, ha="left", y=0.98)
    fig.text(0.07, 0.02, "Source: TensorBoard train/loss. Complete runs shown; earlier aborted runs excluded. Moving average is trailing.", fontsize=9, color="#6b7280")
    fig.tight_layout(rect=[0.015, 0.065, 1, 0.91])
    fig.savefig(args.output_dir / "loss_curves.png", dpi=160, facecolor="white")
    plt.close(fig)
    with (args.output_dir / "loss_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "global_step", "loss"])
        writer.writerows(rows)
    (args.output_dir / "loss_curves_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "event_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

