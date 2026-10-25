"""Print a machine-readable audit of the real infantry MJCF."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from supermimic.infantry.mjcf import audit_mjcf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mjcf", help="Path to Fudan-2026RoboMaster-Balance.xml")
    args = parser.parse_args()
    print(json.dumps(audit_mjcf(args.mjcf).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
