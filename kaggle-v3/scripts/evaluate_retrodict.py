"""Score duck-retrodict on recorded transition traces and apply its gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ouro3.promotion import evaluate_retrodict_offline_promotion
from ouro3.retrodict_eval import (
    attach_generated_python_predictions,
    evaluate_recorded_transitions,
    load_recorded_transitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--minimum-coverage", type=float, default=0.60)
    parser.add_argument("--maximum-p95-ms", type=float, default=5.0)
    parser.add_argument(
        "--generated-python-traces",
        type=Path,
        nargs="+",
        help="optional old simulator predictions keyed by game_id and index",
    )
    args = parser.parse_args()

    transitions = load_recorded_transitions(args.traces)
    if args.generated_python_traces:
        transitions = attach_generated_python_predictions(
            transitions,
            args.generated_python_traces,
        )
    report = evaluate_recorded_transitions(
        transitions,
        train_fraction=args.train_fraction,
    )
    decision = evaluate_retrodict_offline_promotion(
        report,
        minimum_precision=args.minimum_precision,
        minimum_coverage=args.minimum_coverage,
        maximum_p95_ms=args.maximum_p95_ms,
    )
    payload = {**report, "promotion": asdict(decision)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not decision.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
