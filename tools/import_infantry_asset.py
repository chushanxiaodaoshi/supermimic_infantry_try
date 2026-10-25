"""Import the existing mjlab infantry asset into the ignored local data tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        help="mjlab src/mjlab/asset_zoo/robots/infantry directory",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "infantry_description",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    xml = source / "Fudan-2026RoboMaster-Balance.xml"
    if not xml.is_file():
        raise FileNotFoundError(f"source does not contain {xml.name}: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str | int]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".xml", ".stl"}:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    manifest = {
        "source": str(source),
        "destination": str(destination),
        "files": copied,
    }
    (destination / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Imported {len(copied)} files ({sum(int(item['bytes']) for item in copied)} bytes)"
    )


if __name__ == "__main__":
    main()
