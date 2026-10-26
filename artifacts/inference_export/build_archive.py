"""Create and read back the inference archive with bounded streaming buffers."""

import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).parent / "lbench_recap_pi05_h50_e20_step30000_inference"
DEST = Path("/root/autodl-fs") / f"{ROOT.name}.tar"
PARTIAL = DEST.with_suffix(".tar.partial")
BLOCK = 8 * 1024**2


def digest_file(stream):
    digest = hashlib.sha256()
    while block := stream.read(BLOCK):
        digest.update(block)
    return digest.hexdigest()


class HashingStream:
    def __init__(self, stream, mode):
        self.stream = stream
        self.mode = mode
        self.digest = hashlib.sha256()
        self.count = 0
        self.last_report = time.monotonic()

    def _record(self, data):
        self.digest.update(data)
        self.count += len(data)
        if time.monotonic() - self.last_report >= 20:
            print(f"{self.mode}: {self.count / 2**30:.2f} GiB", flush=True)
            self.last_report = time.monotonic()

    def write(self, data):
        count = self.stream.write(data)
        if count != len(data):
            raise OSError("Incomplete archive write")
        self._record(data)
        return count

    def read(self, size=-1):
        data = self.stream.read(size)
        self._record(data)
        return data


def main():
    validation = json.loads((ROOT / "inference_validation.json").read_text())
    assert validation["model_loaded"] and validation["synthetic_inference_passed"]
    assert not DEST.exists() and not PARTIAL.exists(), "Refusing to overwrite an export"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts)
    expected = {}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        with path.open("rb") as stream:
            expected[relative] = digest_file(stream)
        print(f"Hashed {relative}", flush=True)
    (ROOT / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in expected.items()))
    checksum_file = ROOT / "SHA256SUMS"
    expected["SHA256SUMS"] = hashlib.sha256(checksum_file.read_bytes()).hexdigest()
    files.append(checksum_file)
    with PARTIAL.open("xb", buffering=BLOCK) as raw:
        writer = HashingStream(raw, "Write")
        with tarfile.open(fileobj=writer, mode="w|", bufsize=BLOCK) as archive:
            archive.add(ROOT, arcname=ROOT.name, recursive=False)
            for path in files:
                archive.add(path, arcname=f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}", recursive=False)
        raw.flush()
        os.fsync(raw.fileno())
        written_sha = writer.digest.hexdigest()
    print("Archive written; reading every member back for verification", flush=True)
    seen = set()
    with PARTIAL.open("rb", buffering=BLOCK) as raw:
        reader = HashingStream(raw, "Verify")
        with tarfile.open(fileobj=reader, mode="r|", bufsize=BLOCK) as archive:
            for member in archive:
                if member.isdir():
                    continue
                assert member.isfile(), member.name
                relative = Path(member.name).relative_to(ROOT.name).as_posix()
                assert relative in expected and relative not in seen, relative
                stream = archive.extractfile(member)
                assert stream is not None
                assert digest_file(stream) == expected[relative], relative
                seen.add(relative)
        while reader.read(BLOCK):
            pass
        assert reader.digest.hexdigest() == written_sha, "Archive checksum differs after readback"
    assert seen == set(expected)
    assert not DEST.exists()
    PARTIAL.rename(DEST)
    with DEST.with_suffix(".tar.sha256").open("x") as stream:
        stream.write(f"{written_sha}  {DEST.name}\n")
    report = {
        "archive": str(DEST), "bytes": DEST.stat().st_size,
        "sha256": written_sha, "files_verified": len(seen),
        "every_file_matches_source": True,
        "archive_checksum_verified_after_readback": True,
        "inference_validation": validation,
    }
    with DEST.with_suffix(".verification.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
