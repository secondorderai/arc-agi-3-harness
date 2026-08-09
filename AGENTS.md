# AGENTS.md

## Read this first

This repository contains three generations of ARC-AGI-3 experiments. `kaggle-v3`
is the primary maintained harness; `kaggle-v2` and `kaggle` are separate legacy
trees and must not be treated as shared implementation dependencies.

Before changing code:

1. Identify the generation and execution mode being changed.
2. Read that generation's README or operating guide and the relevant tests.
3. Make the smallest isolated change, preserving experiment attribution and
   the existing reference behavior.
4. Run the narrowest relevant tests, then the full generation test suite when
   practical.

Do not commit secrets, model weights, generated notebooks, run results, logs,
virtual environments, or Kaggle credentials. Do not publish or submit to
Kaggle unless the user explicitly asks and the repository's validation gates
have passed.

## Repository map

- `kaggle-v3/` — maintained Duck/TAAF harness, local integrations, notebook
  generation, validation, promotion, and submission pipeline.
  - `src/ouro3/` — hybrid orchestration, configuration, modes, runner,
    perception, verification, metrics, scheduling, and promotion gates.
  - `src/duck_*/` — isolated Duck experiment agents and solvers.
  - `src/inference/`, `src/taaf/` — attributed Duck/TAAF framework code.
  - `scripts/` — packaging, notebook generation, evaluation, router training,
    and Kaggle pipeline tooling.
  - `tests/` — pytest coverage for modes, contracts, sandboxing, artifacts,
    metrics, and transport behavior.
  - `configs/` — checked-in local, validation, and experiment profiles.
- `kaggle-v2/` — ouro2 symbolic world-model/director harness with its own
  `Makefile`, `pytest.ini`, `ouro2/`, `tests/`, and holdout workflow.
- `kaggle/` — v1 deterministic/Qwen harness with its own `Makefile`,
  `ouro_arc/`, `tests/`, model packaging, and GPU validation workflow.
- `README.md` — repository overview and setup summary.

For architecture and operating detail, consult these only when needed:

- v3: [`kaggle-v3/README.md`](kaggle-v3/README.md),
  [`kaggle-v3/HOW-IT-WORKS.md`](kaggle-v3/HOW-IT-WORKS.md),
  [`kaggle-v3/OPERATING-GUIDE.md`](kaggle-v3/OPERATING-GUIDE.md), and
  [`kaggle-v3/SCORECARD.md`](kaggle-v3/SCORECARD.md).
- v2: [`kaggle-v2/HOW-IT-WORKS.md`](kaggle-v2/HOW-IT-WORKS.md).
- v1: [`kaggle/README.md`](kaggle/README.md) and `kaggle/docs/`.

## v3 development workflow

Run v3 commands from `kaggle-v3/` with Python 3.12 and the editable project
environment:

```sh
cd kaggle-v3
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

For a root-directory invocation, use the equivalent paths:

```sh
kaggle-v3/.venv/bin/python -m pytest kaggle-v3/tests
```

Useful local-only checks:

```sh
.venv/bin/python scripts/package_source.py
.venv/bin/python scripts/build_notebooks.py
.venv/bin/python scripts/validate_notebooks.py
.venv/bin/python -m ouro3.cli \
  --config configs/local-mlx.json \
  --output results/local-integration.json \
  public --fold public --environments-dir ../kaggle/environment_files
```

Use the mode-specific config in `configs/` for experiments. The reference
profile is intentionally strict: `HarnessConfig.validate()` protects runtime
pins and Stock Duck fidelity. Keep `duck-reference` changes separate from
experimental modes, and update the corresponding mode tests, fingerprints,
metrics, or promotion logic when a mode's behavior or contract changes.

Retrodict and portfolio work has additional offline gates:

```sh
.venv/bin/python scripts/evaluate_retrodict.py TRACE.jsonl.gz \
  --output results/retrodict-offline-gate.json
.venv/bin/python scripts/build_portfolio_router.py
```

The first command may exit nonzero when the precision, coverage, or latency
gate fails; that is an expected fail-closed result, not a reason to weaken the
gate without an explicit experiment decision.

## v3 implementation rules

- Preserve mode isolation. A new behavior belongs in its own `duck_*` agent and
  solver, mode-specific config, and tests unless the change is intentionally a
  shared framework fix.
- Keep prompts, tool contracts, action normalization, and history policies
  attributable. Do not silently change Stock Duck behavior while implementing
  an ablation.
- Treat `HarnessConfig` and generated artifact fingerprints as compatibility
  boundaries. Runtime pins currently include vLLM `0.19.0`, Torch `2.10.0`,
  FlashInfer `0.6.6`, the RTX Pro 6000 profile, 30-second Python tools, and
  1,024-token tool output.
- Keep the Python tool sandbox restricted. It intentionally denies filesystem,
  environment, network, child-process, native-module, and unapproved-package
  access; update sandbox tests alongside any capability change.
- Use generic visual/transition evidence for gameplay logic. Do not add
  hard-coded game IDs, coordinates, board hashes, or public-game rules to
  solver behavior. Preserve the holdout and anti-overfit discipline.
- Keep deterministic artifacts reproducible. If a router, predictor, manifest,
  or packaged source changes, record the source/config/training-artifact cause
  and verify the resulting hash or parity test.
- Avoid editing `src/inference/` or `src/taaf/` casually: these are attributed
  framework sources. Confirm whether a change is a framework fix or an
  experiment-specific adapter before modifying them.

## Validation and publishing boundaries

Local gameplay, notebook dry-runs, source packaging, and unit tests are safe
development checks. Kaggle pipeline commands can consume GPU quota, mutate
remote datasets/kernels, submit competition runs, and write the local ledger.

The v3 pipeline is in `kaggle-v3/scripts/kaggle_pipeline.py`. Before using it:

- verify the exact `--mode`, config, seed, generated notebook metadata, and
  expected output paths;
- run the relevant local public integration and `rehearse-110` transport check;
- preserve the fail-closed score, breadth, telemetry, source-hash, config-hash,
  and daily-quota gates;
- use `--seed0-only` for a new candidate unless the mode's documented gate
  explicitly requires multiple seeds;
- never use `--submit` as a shortcut around validation or promotion.

Generated v3 paths are ignored by design: `notebooks/`, `results/`, `logs/`,
`dist/`, `.venv/`, `.kaggle/`, and `*.egg-info/`. Inspect generated artifacts
when debugging, but do not add them to commits unless the task specifically
requires a tracked baseline or model artifact.

## Legacy generation workflows

The legacy trees have independent dependencies and test commands. Do not import
their modules into v3.

For v2, run from `kaggle-v2/`:

```sh
make setup
make test
make play-local ARGS='--fold dev'
make holdout
make notebook ARGS='...'
```

`make holdout` and `make notebook` enforce the v2 holdout gate. Read
`kaggle-v2/Makefile` before changing a publishing target.

For v1, run from `kaggle/`:

```sh
make setup
make test
make verify-local
make score-local-deterministic
make holdout
make overfit-lint
```

Use `make` targets for v1 model packaging, GPU validation, promotion, and
submission; they contain credential checks and model/runtime staging rules.
The v1 test suite uses `python3 -m unittest discover -s tests`.

## Change handoff

In the final handoff, state the generation/mode changed, files changed, tests
run, and any checks not run because they require a model server, local ARC
environment files, Kaggle credentials, GPU quota, or remote execution. Mention
whether generated artifacts were intentionally left ignored.
