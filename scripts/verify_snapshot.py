"""Verify copied source/artifact hashes and the complete publication manifest."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def included(path):
    relative = path.relative_to(ROOT)
    return not any(part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "runs", "validation_local"} for part in relative.parts) and not str(relative).startswith("configs/local/") and relative.as_posix() != "SHA256SUMS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    copied = json.loads((ROOT / "provenance/copied_files.json").read_text())
    for entry in copied:
        path = ROOT / entry["path"]
        assert digest(path) == entry["sha256"], entry["path"]
        if entry["compression"] == "gzip":
            assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == entry["source_sha256"], entry["path"]
    preparation = json.loads((ROOT / "assets/preparation_manifest.json").read_text())
    assert digest(ROOT / "assets/normalization/lerobot_episode_stats/norm_stats.json") == preparation["norm_stats_sha256"]
    counts = {}
    with (ROOT / "artifacts/lbench_recap_round1/loss_curves.csv").open() as stream:
        for row in csv.DictReader(stream):
            counts[row["run"]] = counts.get(row["run"], 0) + 1
    assert counts == {"Value": 8000, "Policy (H50 / E20)": 30000}, counts
    manifest = ROOT / "SHA256SUMS"
    if args.write_manifest:
        paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
        manifest.write_text("".join(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths))
    lines = manifest.read_text().splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        assert digest(path) == expected, relative
    print(json.dumps({"copied_files_verified": len(copied), "publication_files_verified": len(lines), "loss_points": counts, "normalization_verified": True}, indent=2))


if __name__ == "__main__":
    main()
