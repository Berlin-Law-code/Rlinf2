"""Export and verify a ZIP64 inference bundle using bounded buffers."""

import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

from build_archive import BLOCK, ROOT, HashingStream

DEST = Path("/root/autodl-fs") / f"{ROOT.name}.zip"
PARTIAL = DEST.with_suffix(".zip.partial")


class ZipWriter(HashingStream):
    def tell(self):
        return self.count

    def flush(self):
        self.stream.flush()


def digest_stream(stream, label):
    digest = hashlib.sha256()
    count = 0
    last_report = time.monotonic()
    while block := stream.read(BLOCK):
        digest.update(block)
        count += len(block)
        if time.monotonic() - last_report >= 20:
            print(f"{label}: {count / 2**30:.2f} GiB", flush=True)
            last_report = time.monotonic()
    return digest.hexdigest()


def main():
    validation = json.loads((ROOT / "inference_validation.json").read_text())
    assert validation["model_loaded"] and validation["synthetic_inference_passed"]
    assert not DEST.exists() and not PARTIAL.exists(), "Refusing to overwrite an export"
    files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts
    )
    expected = {}
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        with path.open("rb") as stream:
            expected[name] = digest_stream(stream, f"Source {name}")
    checksum_file = ROOT / "SHA256SUMS"
    checksum_file.write_text("".join(f"{digest}  {name}\n" for name, digest in expected.items()))
    expected["SHA256SUMS"] = hashlib.sha256(checksum_file.read_bytes()).hexdigest()
    files.append(checksum_file)
    # ZIP_STORED preserves the original weights and avoids spending CPU time
    # compressing mostly high-entropy floating-point arrays. ZIP64 supports >4GB.
    with PARTIAL.open("xb", buffering=BLOCK) as raw:
        writer = ZipWriter(raw, "Write ZIP")
        with zipfile.ZipFile(writer, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in files:
                name = f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}"
                with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target, length=BLOCK)
        raw.flush()
        os.fsync(raw.fileno())
        written_sha = writer.digest.hexdigest()
    print("ZIP written; verifying all members against source SHA-256", flush=True)
    seen = set()
    with zipfile.ZipFile(PARTIAL, "r") as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename).relative_to(ROOT.name).as_posix()
            assert relative in expected and relative not in seen, relative
            with archive.open(entry) as stream:
                assert digest_stream(stream, f"Verify {relative}") == expected[relative], relative
            seen.add(relative)
    assert seen == set(expected)
    with PARTIAL.open("rb", buffering=BLOCK) as stream:
        readback_sha = digest_stream(stream, "Verify ZIP checksum")
    assert readback_sha == written_sha
    assert not DEST.exists()
    PARTIAL.rename(DEST)
    with DEST.with_suffix(".zip.sha256").open("x") as stream:
        stream.write(f"{readback_sha}  {DEST.name}\n")
    report = {
        "archive": str(DEST), "format": "ZIP64", "compression": "stored",
        "bytes": DEST.stat().st_size, "sha256": readback_sha,
        "files_verified": len(seen), "every_file_matches_source": True,
        "archive_checksum_verified_after_readback": True,
        "inference_validation": validation,
    }
    with DEST.with_suffix(".verification.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
