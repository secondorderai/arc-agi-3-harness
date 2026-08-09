# ARC-AGI-3 Kaggle Harness

ARC-AGI-3 Kaggle Harness is my experiments applying world models, LLM-based
approaches, neuro-symbolic AI, solvers, and deterministic models to the
[ARC Prize 2026 ARC-AGI-3 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3).
It contains local experiment code, scripts to generate the Jupyter notebooks
published to Kaggle for public-game runs on GPU, and the submission pipelines
for both public and private (hidden) game runs.

The experiments went through a number of iterations, from v1 through v3, and
are by no means complete — this is still an ongoing, evolving set of
experiments. The scorecard tracking local and Kaggle run results lives at
[kaggle-v3/SCORECARD.md](kaggle-v3/SCORECARD.md).

These experiments were conducted with help and healthy debate with OpenAI
Codex on the directions, hypotheses, and implementation along the way.

This repository was extracted from Ouroboros so the submission experiments,
model packaging, notebook generation, and result artifacts can evolve
independently from the Ouroboros agent.

## Layout

- `kaggle/`: legacy deterministic and Qwen local/Kaggle harness (v1).
- `kaggle-v2/`: the ouro2 exploration, induction, and director harness (v2).
- `kaggle-v3/`: the attributed Duck/TAAF hybrid harness with guarded Kaggle
  publishing and submission pipelines (v3).

Each generation has its own README, tests, configuration, and setup scripts.
Generated notebooks, model data, run results, logs, virtual environments,
vendored dependencies, and Kaggle credentials are intentionally ignored.

## Quick start

```bash
git clone https://github.com/secondorderai/arc-agi-3-harness.git
cd arc-agi-3-harness

cd kaggle-v3
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

See the generation-specific README files for local gameplay, notebook
generation, validation gates, and Kaggle submission commands.

Never commit Kaggle credentials. Use the local ignored `.kaggle/access_token`
file or the Kaggle CLI's supported environment-based authentication.
