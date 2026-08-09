"""Command line entrypoint for local, rehearsal, and Kaggle runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ouro3.config import HarnessConfig, RuntimeProfile
from ouro3.runner import (
    run_110_rehearsal,
    run_hidden_submission,
    run_public,
    run_public_multiseed,
    write_smoke_submission,
)


def _default_environment_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "kaggle" / "environment_files"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--experiment", default=None)
    value.add_argument("--seed", type=int, default=None)
    sub = value.add_subparsers(dest="command", required=True)

    public = sub.add_parser("public")
    public.add_argument("--fold", choices=["dev", "test", "quarantine", "public"], default="public")
    public.add_argument("--environments-dir", type=Path, default=_default_environment_dir())
    public.add_argument("--max-actions", type=int)
    public.add_argument("--scripted", action="store_true")
    public.add_argument("--local", action="store_true")
    public.add_argument(
        "--seeds",
        default=None,
        help="comma-separated consecutive seeds; runs one concurrent multi-seed benchmark",
    )

    rehearsal = sub.add_parser("rehearse-110")
    rehearsal.add_argument("--environments-dir", type=Path, default=_default_environment_dir())
    rehearsal.add_argument("--max-actions", type=int, default=1)

    sub.add_parser("hidden")
    sub.add_parser("notebook-smoke")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.config:
        config = HarnessConfig.from_json(args.config)
    elif getattr(args, "local", False):
        config = HarnessConfig.local(seed=args.seed or 0)
    elif args.command == "rehearse-110":
        config = HarnessConfig.scripted(seed=args.seed or 0)
    else:
        config = HarnessConfig.reference(seed=args.seed)
    overrides = {}
    if args.experiment is not None:
        overrides["experiment"] = args.experiment
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        config = config.with_overrides(**overrides)

    if args.command == "public":
        if args.seeds:
            seeds = tuple(int(value) for value in args.seeds.split(","))
            metrics = run_public_multiseed(
                config=config,
                environments_dir=args.environments_dir,
                fold=args.fold,
                output_path=args.output,
                seeds=seeds,
            )
        else:
            metrics = run_public(
                config=config,
                environments_dir=args.environments_dir,
                fold=args.fold,
                output_path=args.output,
                max_actions=args.max_actions,
                scripted=args.scripted,
            )
    elif args.command == "rehearse-110":
        metrics = run_110_rehearsal(
            config=config,
            environments_dir=args.environments_dir,
            output_path=args.output,
            max_actions=args.max_actions,
        )
    elif args.command == "hidden":
        metrics = run_hidden_submission(config=config, output_path=args.output)
    else:
        rerun = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").lower() in {"1", "true", "yes"}
        if rerun:
            metrics = run_hidden_submission(config=config, output_path=args.output)
        else:
            write_smoke_submission(
                Path("/kaggle/working/submission.parquet"),
                message=f"save-run smoke {config.config_hash[:12]}",
            )
            metrics = {
                "smoke": True,
                "profile": config.profile.value,
                "config_hash": config.config_hash,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
