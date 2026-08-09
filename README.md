# ARC-AGI-3 Kaggle Harness

Standalone harnesses for ARC Prize 2026 ARC-AGI-3 Kaggle submissions and
local validation. This repository was extracted from Ouroboros so the
submission experiments, model packaging, notebook generation, and result
artifacts can evolve independently from the Ouroboros agent.

## Layout

- `kaggle/`: legacy deterministic and Qwen local/Kaggle harness.
- `kaggle-v2/`: the ouro2 exploration, induction, and director harness.
- `kaggle-v3/`: the attributed Duck/TAAF hybrid harness with guarded Kaggle
  publishing and submission pipelines.

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
