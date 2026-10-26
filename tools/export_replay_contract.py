"""Export the controller contract used by the standalone ONNX replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from supermimic.infantry.replay_contract import ReplayContract, validate_onnx_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", action="append", default=[], metavar="SKILL=ONNX")
    args = parser.parse_args()

    contract = ReplayContract.from_replay(args.replay)
    teachers: dict[str, dict] = {}
    for value in args.teacher:
        if "=" not in value:
            parser.error(f"--teacher must be SKILL=ONNX, got {value!r}")
        skill, raw_path = value.split("=", 1)
        if not skill or skill in teachers:
            parser.error(f"invalid or repeated skill {skill!r}")
        teachers[skill] = validate_onnx_contract(raw_path, contract)
    payload = contract.to_dict() | {"teachers": teachers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote replay contract with {len(teachers)} Teacher(s): {args.output}")


if __name__ == "__main__":
    main()
