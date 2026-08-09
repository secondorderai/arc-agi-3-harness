# `duck-portfolio-v1` operating guide

Updated: 2026-08-03 09:13 AEST (Brisbane)

This lane runs one persistent Qwen/Stock-Duck agent per game. It does not run
four models in parallel. Stock controls the first eight executed actions; the
committed deterministic router then activates Stock, Audit, Deliberate, or
Contract Repair from generic frame and transition statistics.

## Rebuild and inspect the router

```sh
cd kaggle-v3
.venv/bin/python scripts/build_portfolio_router.py
```

The generated `src/duck_portfolio/router_model.json` records feature
normalization, coefficients, leave-one-game-out RMSE, training-artifact
SHA-256 values, and the offline gate. A rebuild must be byte-equivalent unless
one declared training artifact or router setting changed. Game identifiers,
coordinates, board hashes, and public-game rules are forbidden inputs.

## Local prerequisites

```sh
.venv/bin/python -m ouro3.cli \
  --config configs/duck-portfolio-local-mlx.json \
  --output results/duck-portfolio-local-public-25.json \
  public --fold public \
  --environments-dir ../kaggle/environment_files

.venv/bin/python -m ouro3.cli \
  --config configs/duck-portfolio-v1.json \
  --output results/duck-portfolio-rehearsal-110.json \
  rehearse-110 \
  --environments-dir ../kaggle/environment_files
```

The local score is integration evidence only. Both artifacts must have zero
infrastructure failures; the rehearsal must expose 110 unique IDs over the
competition HTTP transport and project below 8h40.

## Kaggle public and gated hidden flow

```sh
.venv/bin/python scripts/kaggle_pipeline.py \
  --mode duck-portfolio \
  --gpu-hours-remaining HOURS \
  --submit
```

The pipeline fails closed unless at least 4.5 GPU-hours are confirmed before
the two public kernels. Seed 0 must score at least 2.5631, complete 18 levels,
score on 15 games, and exceed 0.9370812 after removing its three highest
scores. If any condition fails, seed 1 is not launched. Submission then
requires seed 1 floors of 10 levels and 9 nonzero games, two-seed mean at
least 1.4081755, two-seed trimmed mean above 0.6060513, clean telemetry, and
the daily submission allowance.

The exact Git SHA, source manifest, prompt, per-seed config hashes, router
hash, kernel versions, metrics paths, submission reference, and score are
written to `results/duck-portfolio-progress.json` and
`submission-ledger.json`. The account's hidden gateway run is treated as
GPU-quota-neutral, as confirmed by the user; the daily submission limit still
applies.

## Failure iteration order

Change exactly one variable per public candidate: warm-up action count, Stock
margin, uncertainty penalty, switch threshold, then candidate set. Poetiq may
enter only after a new artifact proves a unique positive per-game contribution.
