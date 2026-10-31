"""Command-line entry point for the first-stage PACE identification tools."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from . import __version__
from .audit import audit_csv, format_audit_report
from .core import Mapping, create_output_dir, write_json


ROOT = Path(__file__).resolve().parents[1]


def _output_path(value: Path | None, csv_path: Path, command: str) -> Path:
  if value is not None:
    return value
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  return Path("pace_results") / f"{csv_path.stem}_{command}_{timestamp}"


def _plot_workpoints(value: str) -> set[int]:
  if not value.strip():
    return set()
  try:
    return {int(item.strip()) for item in value.split(",")}
  except ValueError as error:
    raise argparse.ArgumentTypeError(
      "plot workpoints must be comma-separated integers"
    ) from error


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Audit PACE logs and replay total motor commands in MuJoCo."
  )
  parser.add_argument("--version", action="version", version=__version__)
  subparsers = parser.add_subparsers(dest="command", required=True)

  audit = subparsers.add_parser("audit", help="streaming CSV integrity audit")
  audit.add_argument("--csv", type=Path, required=True)
  audit.add_argument("--out", type=Path)
  audit.add_argument("--expected-workpoints", type=int, default=30)

  replay = subparsers.add_parser(
    "replay", help="audit, validate the model, then run baseline replay"
  )
  replay.add_argument("--csv", type=Path, required=True)
  replay.add_argument("--xml", type=Path, default=ROOT / "mjcf/scene.xml")
  replay.add_argument("--out", type=Path)
  replay.add_argument("--expected-workpoints", type=int, default=30)
  replay.add_argument("--window-s", type=float, default=5.0)
  replay.add_argument("--window-stride-s", type=float, default=5.0)
  replay.add_argument(
    "--plot-workpoints",
    type=_plot_workpoints,
    default={0, 14, 15, 29},
    help="comma-separated workpoint IDs; empty string disables plots",
  )

  diagnose = subparsers.add_parser(
    "diagnose",
    help="short-window error growth and timestep convergence checks",
  )
  diagnose.add_argument("--csv", type=Path, required=True)
  diagnose.add_argument("--xml", type=Path, default=ROOT / "mjcf/scene.xml")
  diagnose.add_argument("--baseline", type=Path, required=True)
  diagnose.add_argument("--out", type=Path)
  diagnose.add_argument("--expected-workpoints", type=int, default=30)
  return parser.parse_args()


def _write_audit(output_dir: Path, report: dict[str, object]) -> None:
  write_json(output_dir / "audit.json", report)
  (output_dir / "audit.txt").write_text(
    format_audit_report(report), encoding="utf-8"
  )


def _fit_interface() -> dict[str, object]:
  return {
    "schema": "pace-fit-interface-v1",
    "implemented": False,
    "stage_one_only": True,
    "global_across_workpoints": True,
    "split_policy": "hold out complete workpoints, never adjacent random rows",
    "initially_frozen": [
      "geometry",
      "encoder_zero",
      "link_mass_and_inertia",
      "motor_torque_scale",
      "input_delay",
    ],
    "first_candidate_parameters": [
      "small shared or per-motor equivalent viscous friction",
      "small shared or per-motor equivalent Coulomb friction",
    ],
    "conditional_later_parameter": (
      "gas-spring correction only after pose-structured residual evidence"
    ),
    "prohibited_first_fit": (
      "do not jointly free torque scale, inertia, spring, friction, and delay"
    ),
  }


def main() -> int:
  args = parse_args()
  csv_path = args.csv.resolve()
  if not csv_path.is_file():
    raise FileNotFoundError(csv_path)
  output_dir = create_output_dir(
    _output_path(args.out, csv_path, args.command)
  )
  audit = audit_csv(csv_path, expected_workpoints=args.expected_workpoints)
  _write_audit(output_dir, audit)
  write_json(output_dir / "mapping.json", Mapping().as_dict())
  write_json(output_dir / "fit_interface.json", _fit_interface())
  if args.command == "audit":
    print(format_audit_report(audit), end="")
    print(f"audit output: {output_dir}")
    return 0 if audit["accepted_for_baseline_replay"] else 2
  if not audit["accepted_for_baseline_replay"]:
    print("Replay refused because the audit did not pass.", file=sys.stderr)
    return 2
  if args.command == "diagnose":
    from .diagnose import format_diagnosis_report, run_diagnosis

    diagnosis = run_diagnosis(
      csv_path=csv_path,
      xml_path=args.xml.resolve(),
      output_dir=output_dir,
      baseline_dir=args.baseline.resolve(),
    )
    write_json(output_dir / "diagnosis.json", diagnosis)
    report = format_diagnosis_report(diagnosis)
    (output_dir / "diagnosis.txt").write_text(report, encoding="utf-8")
    print("audit: accepted; 60 sweep segments expected and checked")
    print(report, end="")
    return 0
  from .replay import format_replay_report, run_replay

  replay = run_replay(
    csv_path=csv_path,
    xml_path=args.xml.resolve(),
    output_dir=output_dir,
    window_s=args.window_s,
    window_stride_s=args.window_stride_s,
    plot_workpoints=args.plot_workpoints,
  )
  write_json(output_dir / "replay.json", replay)
  (output_dir / "replay.txt").write_text(
    format_replay_report(replay), encoding="utf-8"
  )
  print(format_replay_report(replay), end="")
  print(f"replay output: {output_dir}")
  return 0 if replay["baseline_valid"] else 3


if __name__ == "__main__":
  raise SystemExit(main())
