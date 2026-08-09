# ARC-AGI-3 `kaggle-v3`

`kaggle-v3` contains isolated, attributed execution modes built from the
public MIT-licensed [Duck/TAAF
harness](https://github.com/Tufalabs/duck-harness):

- `duck-reference` locks the published Duck agent behavior for score
  reproduction.
- `duck-memory` keeps Stock Duck's prompt, tools, action policy, sampling,
  concurrency, and time limits while retaining Qwen private reasoning and
  replacing rolling history deletion with semantic compaction.
- `duck-reasoning` is the clean retained-reasoning ablation: it keeps Stock
  Duck's history eviction, prompt, tools, action policy, sampling, concurrency,
  and time limits, changing only the Qwen reasoning wire field. It makes no
  auxiliary compactor calls.
- `duck-robust` keeps stock Duck until a conservative stagnation gate starts
  one verified recovery fork per level. Its first experiment is retained as a
  retired negative result.
- `duck-audit` keeps Stock Duck unchanged except for a sparse self-audit note
  injected after generic repeated-action or unchanged-frame signals. It never
  adds a mandatory expectation contract or a second model request.
- `duck-hierarchy` keeps Stock Duck unchanged except for a bounded candidate
  ranking prompt at new-level or unchanged-frame boundaries; it compares at
  most three generic hypotheses against the next frame.
- `duck-diversity` keeps Stock Duck unchanged except for one alternate sampling
  seed after an unchanged-frame trigger, then returns to the primary seed.
- `duck-poetiq` combines those five harness ideas in one compact protocol:
  event-triggered audit, one discriminating information request, at most three
  hypotheses, one optional verified probe, one alternate seed, and a guarded
  pre-progress stalled yield. Normal turns remain Stock Duck turns.
- `duck-portfolio` keeps one Stock Duck/Qwen conversation per game, observes
  exactly eight Stock actions, then deterministically selects Stock, Audit,
  Deliberate, or Contract Repair from nine generic visual/transition features.
  It never forks the game or uses game IDs, fingerprints, fixed coordinates,
  or public-game rules.
- `duck-retrodict` keeps the Stock Duck actor but moves evidence and execution
  authority into a persistent host-owned version space. It replays a bounded
  typed rule language over the complete per-game transition log, keeps
  competing 4/8-connected object ontologies, chooses information-gain probes,
  and automatically executes only one exact or replay-certified action before
  observing and replanning. A history-conditioned clone graph remains dormant
  until incompatible successors reveal temporal aliasing.
- `ouro-hybrid` adds Ouroboros perception, evidence memory,
  prediction-checked plans, adaptive scheduling, and deterministic recovery.

Nothing under `kaggle-v2` is imported, rewritten, or packaged.

## Reference profile

The reference path uses `duck_reference.DuckReferenceToolAgent` and
`DuckReferenceHarnessSolver`; it cannot construct the hybrid agent or
scheduler. It reproduces the audited Kaggle settings:

- 4× current-grid image and segmentation-first prompt
- 32,768-token active context
- temperature 0.6, top-p 0.95, top-k 20
- 28 concurrent sessions
- one 25-game wave per kernel version
- 7,920-second per-game cap and 900-second analyzer timeout
- unlimited model output/tool turns, 30-second Python tools, and 1,024-token
  tool output
- thinking, prefix caching, Qwen tool/reasoning parsers, and
  `ONLY_RESET_LEVELS=true`
- Qwen3.6-27B-FP8 served by vLLM 0.19.0
- Torch 2.10.0 and FlashInfer 0.6.6
- one `NvidiaRtxPro6000`, no internet

The fidelity kernel omits the model seed. Seeds 0–4 then run as five
independent full-budget kernel versions and are aggregated from
`mean_engine_score`. Tufa Labs measured a 1.6002 public-game mean with
0.4475 population standard deviation in its separate 20-pass research run,
so 1.2 is a distribution gate, not a guaranteed single-run score.

## Layout

- `src/duck_reference/`: locked reference solver/agent import path.
- `src/duck_memory/`: Stock Duck reasoning adapter, semantic compaction, and
  memory audit telemetry.
- `src/duck_reasoning/`: Stock Duck reasoning-only agent and solver; no
  semantic compaction or Ouroboros behavior is reachable.
- `src/duck_robust/`: confidence-gated recovery agent and solver.
- `src/duck_audit/`: event-triggered self-audit sidecar with Stock-compatible
  tools, action policy, and history behavior.
- `src/duck_hierarchy/`: bounded candidate-ranking sidecar with a cheap
  frame-delta verifier prompt and isolated telemetry.
- `src/duck_diversity/`: bounded alternate-seed sampling sidecar with
  stall-trigger telemetry.
- `src/duck_poetiq/`: the isolated composite Poetiq protocol and solver.
- `src/duck_portfolio/`: the persistent routed agent plus the transparent,
  source-trained ridge-router JSON artifact.
- `src/duck_retrodict/`: the typed verifier-controlled actor and solver.
- `src/inference/`, `src/taaf/`: attributed MIT-licensed Duck/TAAF source.
- `src/ouro3/`: Ouroboros perception, ledger, verification, fallback,
  scheduler, metrics, promotion gates, and runners.
- `scripts/package_source.py`: private Kaggle dataset builder with SHA-256
  manifest; rejects opaque pickle inputs.
- `scripts/build_notebooks.py`: generates public-validation and hidden
  submission notebooks with pinned RTX metadata.
- `scripts/evaluate_retrodict.py`: chronological transition holdout evaluator;
  requires 95% precision, 60% coverage, and a 5ms p95 prediction latency by
  default before public gameplay.
- `scripts/compare_retrodict_models.py`: paired two-seed 27B/challenger model
  gate; accepts a challenger only for a material score lift or a score-safe
  elapsed-time improvement.
- `scripts/kaggle_pipeline.py`: dataset/kernel publishing, validation gate,
  exact-version submission, waiting, and ledger recording.
- `tests/`: synthetic, sandbox, scheduler, promotion, and artifact tests.
- `HOW-IT-WORKS.md`, `architecture.drawio`: architecture and operating model.

## Local commands

Use the existing ARC Python environment:

```sh
export PYTHONPATH="$PWD/kaggle-v3/src"
PY=kaggle-v2/.venv/bin/python

$PY -m pytest kaggle-v3/tests
$PY kaggle-v3/scripts/package_source.py
$PY kaggle-v3/scripts/build_notebooks.py
```

Run the retrodictive lane locally, then score its emitted validation traces:

```sh
$PY -m ouro3.cli \
  --config kaggle-v3/configs/duck-retrodict-local-mlx.json \
  --output kaggle-v3/results/duck-retrodict-local.json \
  public --fold dev

$PY kaggle-v3/scripts/evaluate_retrodict.py \
  kaggle-v3/results/duck-retrodict-local-retrodict-trace.jsonl.gz \
  --output kaggle-v3/results/duck-retrodict-offline-gate.json
```

Build the separate 35B-A3B arm only after attaching its Kaggle snapshot:

```sh
$PY kaggle-v3/scripts/build_notebooks.py \
  --mode duck-retrodict --validation-seed 0 \
  --model-id vrfai/Qwen3.6-35B-A3B-FP8 \
  --model-dataset OWNER/ATTACHED-SNAPSHOT \
  --output kaggle-v3/notebooks/retrodict-35b-a3b
```

Generate a seed-0 robust candidate without changing the default reference
notebooks:

```sh
$PY kaggle-v3/scripts/build_notebooks.py \
  --mode duck-robust --validation-seed 0 \
  --output kaggle-v3/notebooks/robust
```

Publish and run exactly that seed-0 candidate through the guarded Kaggle
pipeline:

```sh
$PY kaggle-v3/scripts/kaggle_pipeline.py \
  --mode duck-robust --seed0-only
```

This path writes isolated `duck-robust` results, verifies the notebook and
returned metrics mode, and rejects `--submit`. Promotion remains a separate
reviewed operation.

Generate and run the retained-reasoning candidate:

```sh
$PY kaggle-v3/scripts/build_notebooks.py \
  --mode duck-memory --validation-seed 0 \
  --output kaggle-v3/notebooks/memory

$PY kaggle-v3/scripts/kaggle_pipeline.py \
  --mode duck-memory --seed0-only \
  --gpu-hours-remaining HOURS
```

The memory pipeline requires a verified value of at least 12 remaining weekly
GPU hours before it starts. Add `--submit` to authorize the exact gated hidden
notebook version. Submission remains fail-closed unless the public result is at
least 1.20, all 25 games complete cleanly, the tokenizer reasoning sentinel
passes, and the memory audit reports no lost reasoning or emergency trimming.

The clean reasoning-only ablation is intentionally separate:

```sh
$PY kaggle-v3/scripts/build_notebooks.py \
  --mode duck-reasoning --validation-seed 0 \
  --output kaggle-v3/notebooks/reasoning

$PY kaggle-v3/scripts/kaggle_pipeline.py \
  --mode duck-reasoning --seed0-only \
  --gpu-hours-remaining HOURS
```

Its gate requires the same 1.20 public score and clean 25-game runtime, a
passing tokenizer sentinel, and zero compaction calls. Stock Duck context
evictions are reported separately as part of the ablation telemetry.

`src/ouro3/recovery_predictor.json` is reproducibly trained from Tufa's 500
published `benchmark.json` trajectories. Metadata-only leave-one-game-out
performance is intentionally recorded as insufficient, so the probability
cannot trigger a recovery without an independently observed repeated
state-action cycle or contradicted hypothesis.

Real local-model integration:

```sh
$PY -m ouro3.cli \
  --config kaggle-v3/configs/duck-memory-local-mlx.json \
  --output kaggle-v3/results/duck-memory-local-public-25.json \
  public --fold public \
  --environments-dir kaggle/environment_files
```

Use `configs/duck-reasoning-local-mlx.json` with the same command to run the
reasoning-only local integration. The local profile uses a 20-minute cap per
game so the 4B MLX model has time to finish an analysis turn; its score is an
interface check, not a proxy for the RTX/Qwen submission.

The composite candidate uses the six-minute local integration profile:

```sh
$PY -m ouro3.cli \
  --config kaggle-v3/configs/duck-poetiq-local-mlx.json \
  --output kaggle-v3/results/duck-poetiq-local-public-25.json \
  public --fold public \
  --environments-dir kaggle/environment_files
```

Run the one-game `ft09` smoke before publishing its public kernel; the exact
recorded artifact is `results/duck-audit-local-ft09-v2.json`. For the next
stage, generate a seed-0 public notebook with `--mode duck-audit` and use the
existing Kaggle pipeline only after that smoke has no infrastructure failure.

The corrected wire-preserving verified-action ablation is also public-only:

```sh
$PY scripts/build_notebooks.py \
  --mode duck-contract-repair --validation-seed 0 \
  --output notebooks/contract-repair

$PY scripts/kaggle_pipeline.py \
  --mode duck-contract-repair --seed0-only \
  --gpu-hours-remaining HOURS
```

Its public artifact is kept separate from the earlier pre-wirefix result; the
pipeline records model-authored proposals, repairs, prediction matches, and
mismatches so the transport fix is directly attributable.

Fast 110-game competition-transport rehearsal:

```sh
$PY -m ouro3.cli \
  --config kaggle-v3/configs/duck-poetiq-v1.json \
  --output kaggle-v3/results/duck-poetiq-rehearsal-110.json \
  rehearse-110 \
  --environments-dir kaggle/environment_files
```

The localhost rehearsal may require permission to bind a loopback port.

Portfolio integration and rehearsal use the same interfaces:

```sh
$PY -m ouro3.cli \
  --config kaggle-v3/configs/duck-portfolio-local-mlx.json \
  --output kaggle-v3/results/duck-portfolio-local-public-25.json \
  public --fold public \
  --environments-dir kaggle/environment_files

$PY -m ouro3.cli \
  --config kaggle-v3/configs/duck-portfolio-v1.json \
  --output kaggle-v3/results/duck-portfolio-rehearsal-110.json \
  rehearse-110 \
  --environments-dir kaggle/environment_files
```

After both artifacts pass, the pipeline runs public seed 0, stops immediately
if its hard score/level/breadth/trimmed gate fails, and otherwise runs seed 1:

```sh
$PY scripts/kaggle_pipeline.py \
  --mode duck-portfolio --gpu-hours-remaining HOURS --submit
```

`--submit` remains dormant until both public seeds pass. The exact source,
prompt, configuration, and router hashes are recorded in the progress file
and submission ledger.

Retrodict uses the same staged flow, plus a chronological transition replay
gate before any Kaggle GPU is spent:

```sh
$PY -m ouro3.cli \
  --config kaggle-v3/configs/duck-retrodict-local-mlx.json \
  --output kaggle-v3/results/duck-retrodict-local-public-25.json \
  public --fold public \
  --environments-dir kaggle/environment_files

$PY scripts/evaluate_retrodict.py \
  results/duck-retrodict-local-public-25-retrodict-trace.jsonl.gz \
  --output results/duck-retrodict-offline-gate.json

$PY -m ouro3.cli \
  --config kaggle-v3/configs/duck-retrodict-v1.json \
  --output kaggle-v3/results/duck-retrodict-rehearsal-110.json \
  rehearse-110 \
  --environments-dir kaggle/environment_files

$PY scripts/kaggle_pipeline.py \
  --mode duck-retrodict --gpu-hours-remaining HOURS --submit
```

The pipeline refuses to publish public kernels unless typed prediction has at
least 95% precision, 60% coverage, ten held-out transitions, and p95 latency
at or below 5ms. It then requires two independent 25-game seeds to beat the
current leaderboard by 0.01, retain a top-three-trimmed mean of at least 1.0,
and score on at least 12 games per seed. Submission rechecks the exact source,
prompt, and configuration hashes.

For a measurement-only public run while the offline gate is still failing,
use `--experimental-public` without `--submit`. This still requires the full
25-game local integration artifact and clean 110-game rehearsal, runs both
public seeds, records the ordinary gate failure, and is structurally forbidden
from submitting a private-games kernel:

```sh
$PY scripts/kaggle_pipeline.py \
  --mode duck-retrodict --experimental-public \
  --gpu-hours-remaining HOURS
```

## Kaggle flow

The pipeline first runs an unseeded 25-game fidelity notebook. That run is
gated on exact runtime fingerprints, all 25 games, zero infrastructure
failures, and completion before 8h40—not on score. It then generates and
pushes five more notebook versions, one for each seed 0–4. Only their
aggregate is gated at mean engine score 1.20.

After the gate passes:

```sh
$PY kaggle-v3/scripts/kaggle_pipeline.py --submit
```

This is a long sequential operation because it performs six full validation
kernels. Exact cached versions can be resumed with `--fidelity-version` and
`--seed-versions 0:V0,1:V1,2:V2,3:V3,4:V4`.
To stop after the first health/fingerprint check, use `--fidelity-only`;
resume its exact cached version in the full command afterward.

After the five-seed gate passes, the pipeline refreshes the leaderboard,
enforces one submission per day, submits the exact completed notebook
version, waits for completion, and appends the result to
`submission-ledger.json`.

The composite Poetiq lane is independently gated on a clean local 25-game
artifact, the 110-game HTTP rehearsal, two independent public seeds, and at
least 4.5 reported weekly GPU-hours before launch. The current competition
account reports hidden gateway reruns as quota-neutral, so the public reserve
is not duplicated for the later private submission. It stops after seed 0 if
its hard breadth/level floor fails; `--submit` is accepted only after both
seeds pass the composite gate:

```sh
$PY kaggle-v3/scripts/kaggle_pipeline.py \
  --mode duck-poetiq --gpu-hours-remaining HOURS --submit
```

`duck-memory-v1` follows a deliberately narrower path: one seed-0 public
kernel, followed by one hidden submission only if its score and memory
integrity gate both pass. The hidden run must use the exact source manifest
and configuration hash pulled from that public kernel.

When `--gpu-hours-remaining` is omitted, the pipeline queries the authenticated
Kaggle CLI quota endpoint itself and still fails closed if the response is
missing or malformed. Supplying a freshly checked value remains useful for an
auditable launch record.

## Verification

Because this repository is a standalone Python harness, verification requires:

```sh
$PY -m pytest kaggle-v3/tests
```

Desktop E2E is intentionally skipped.
