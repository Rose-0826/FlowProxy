"""Restore hash-verified assets from an extracted complete delivery folder.
No network, training, CFD, or inference is performed. Existing mismatches fail.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True,
                        help="Extracted complete package folder containing model_payload/")
    args = parser.parse_args()
    source = args.package.expanduser().resolve()
    records = json.loads((ROOT / "assets_manifest.json").read_text(encoding="utf-8"))
    copied = 0
    for record in records:
        rel = record["path"]
        target = ROOT / rel
        candidate = target if target.is_file() else source / rel
        if not candidate.is_file():
            raise FileNotFoundError(f"Required asset missing: {rel}")
        if candidate.stat().st_size != record["bytes"] or digest(candidate) != record["sha256"]:
            raise ValueError(f"Asset SHA256/size mismatch: {rel}")
        if candidate != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, target)
            if digest(target) != record["sha256"]:
                raise ValueError(f"Copied asset failed verification: {rel}")
            copied += 1
    print(f"ASSETS_VERIFIED {len(records)}; copied={copied}")


if __name__ == "__main__":
    main()
