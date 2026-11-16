from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entanglement_data.dataset.generator import load_config
from entanglement_data.dataset.production import generate_parallel_production_dataset, generate_production_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot-gated N=16 production dataset generation")
    parser.add_argument("--config", default=str(ROOT / "configs" / "production_n16_v1.yaml"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--skip-pilot", action="store_true", help="Run production directly; use only after inspecting a saved pilot")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.pilot_only and args.skip_pilot:
        parser.error("--pilot-only and --skip-pilot are mutually exclusive")
    config = load_config(args.config)
    output = Path(args.output or config["dataset"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    if args.skip_pilot:
        pilot_report_path = output / "pilot_report.json"
        if not pilot_report_path.exists():
            raise SystemExit("--skip-pilot requires an existing pilot_report.json")
        saved_pilot = json.loads(pilot_report_path.read_text(encoding="utf-8"))
        if not saved_pilot.get("gate", {}).get("passed", False):
            raise SystemExit("saved pilot gate did not pass")
    else:
        pilot_config = copy.deepcopy(config)
        pilot_config["dataset"]["num_separable"] = int(config["dataset"]["pilot_num_separable"])
        pilot_config["dataset"]["num_entangled"] = int(config["dataset"]["pilot_num_entangled"])
        pilot_report = generate_production_dataset(pilot_config, output / "pilot")
        gate = {
            "passed": bool(pilot_report["validation"]["valid"] and pilot_report["shortcut_diagnostics"]["passed"]),
            "validation_passed": bool(pilot_report["validation"]["valid"]),
            "shortcut_gate_passed": bool(pilot_report["shortcut_diagnostics"]["passed"]),
            "failures": pilot_report["validation"]["errors"] + pilot_report["shortcut_diagnostics"]["failures"],
        }
        combined = {"gate": gate, "pilot": pilot_report}
        (output / "pilot_report.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(combined, ensure_ascii=False, indent=2))
        if not gate["passed"]:
            print("Pilot gate failed; production generation was not started.", file=sys.stderr)
            raise SystemExit(2)
        if args.pilot_only:
            return

    workers = args.workers or int(config["runtime"].get("parallel_workers", 1))
    production_report = generate_parallel_production_dataset(config, output, workers=workers)
    print(json.dumps(production_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
