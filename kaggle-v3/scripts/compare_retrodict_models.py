"""Compare paired 27B control and 35B-A3B challenger public artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ouro3.model_ab import compare_retrodict_model_runs, decision_payload


def _load(paths: list[Path]) -> list[dict[str, object]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, nargs=2, required=True)
    parser.add_argument("--challenger", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = compare_retrodict_model_runs(
        _load(args.control),
        _load(args.challenger),
    )
    payload = decision_payload(decision)
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
