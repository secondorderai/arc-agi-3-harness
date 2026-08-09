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

## Attributions and influences

This is an independent research and experimentation repository. It combines
original Ouroboros code with adapted open-source code, public benchmark
infrastructure, model/runtime dependencies, and ideas from public research.

### Adapted code and benchmark infrastructure

- **Duck / TAAF:** `kaggle-v3/src/inference/` and `kaggle-v3/src/taaf/` are
  adapted from Tufa Labs' public MIT-licensed [Duck ARC-AGI-3 inference
  harness](https://github.com/Tufalabs/duck-harness), including the Tufa ARC-AGI
  Framework (TAAF). The v3 experiment lanes and Ouroboros additions are
  separately developed; `kaggle-v3/THIRD_PARTY_NOTICES.md` records the audited
  source commit, dataset provenance, and contributor credit.
- **ARC Prize Foundation:** the benchmark and environment tooling come from
  [ARC-AGI-3](https://arcprize.org/) and the [ARC-AGI-3-Agents
  framework](https://github.com/arcprize/ARC-AGI-3-Agents). The v1 and v2 setup
  workflows download that framework, while v3 uses the
  [`arc-agi`](https://github.com/arcprize/arc-agi) and `arcengine` packages.

### Models and runtimes

- **Qwen:** local and Kaggle experiments use Qwen3.5/Qwen3.6 models from the
  [Qwen team](https://github.com/QwenLM/Qwen3.5). Model weights and model-card
  terms remain subject to the applicable Qwen and checkpoint licenses.
- **vLLM, Ollama, and MLX:** the inference adapters support the
  [vLLM](https://github.com/vllm-project/vllm),
  [Ollama](https://github.com/ollama/ollama), and
  [MLX](https://github.com/ml-explore/mlx) ecosystems. These are runtime
  dependencies, not source code copied into this repository.
- **Kaggle:** public validation and hidden-game submission use the
  [Kaggle](https://www.kaggle.com/) platform and its CLI; Kaggle's own terms
  govern remote kernels, datasets, credentials, and competition submissions.

### Research influences

- **Retained reasoning and compaction:** the `duck-memory` and
  `duck-reasoning` experiments were informed by OpenAI's
  [ARC-AGI-3 harness analysis](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/).
  The local implementation is an independent Qwen/vLLM experiment and does
  not claim to reproduce OpenAI's model or score.
- **Adaptive, self-auditing harnesses:** the `duck-poetiq` lane is inspired by
  [Poetiq's public ARC research](https://poetiq.ai/posts/arcagi_announcement/),
  especially iterative feedback, self-auditing, selective computation, and
  model-agnostic orchestration. No Poetiq source code is vendored here.
- **World models and verification:** the v2 ouro2 lane develops explicit rule
  induction, replay/backtesting, planning, and reality-over-model verification
  for unfamiliar games. These ideas are documented in
  [`kaggle-v2/HOW-IT-WORKS.md`](kaggle-v2/HOW-IT-WORKS.md) and are this
  repository's own implementation experiments rather than a copied external
  solver.

See [`kaggle-v3/THIRD_PARTY_NOTICES.md`](kaggle-v3/THIRD_PARTY_NOTICES.md) and
[`kaggle-v3/LICENSE`](kaggle-v3/LICENSE) for the detailed redistribution notice.

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
