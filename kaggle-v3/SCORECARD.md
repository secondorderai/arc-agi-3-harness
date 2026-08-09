# ARC-AGI-3 `kaggle-v3` scorecard

Last updated: 2026-08-03

All timestamps below are Brisbane local time (AEST, UTC+10), converted from the artifact `created_at` field unless noted otherwise; the source JSON retains UTC values for auditability. For validation artifacts this is the metrics-artifact
creation time (normally at run completion), not a separately recorded kernel
launch time. The submission timestamps come from the tracked Kaggle ledger.

This document records the scores produced by the local integrations, Kaggle
public-game validation kernels, and the hidden-game submission ledger. Scores
are copied from the machine-readable artifacts; they are not estimates.

`mean_engine_score` is the harness score averaged over the games in that
artifact. `mean_completed_levels` is reported separately because it is a more
stable progress measure. A blank Kaggle `privateScore` means Kaggle did not
return a private-game score in the recorded response; it must not be treated as
zero.

## At a glance

| Lane | Run | Brisbane time (AEST) | Games | Mean engine score | Mean completed levels | Total completed levels | Result |
|---|---|---|---:|---:|---:|---:|---|
| Local | `local-qwen-smoke` | 2026-07-28 22:46:45 | 3 | 0.0000 | 0.000 | 0 | Integration smoke |
| Local | `local-public-25` | 2026-07-29 00:11:30 | 25 | 0.0000 | 0.000 | 0 | Completed local public integration |
| Local | `duck-memory-v1-local` smoke | 2026-07-31 21:53:20 | 3 | 0.0000 | 0.000 | 0 | Integration smoke |
| Local | `duck-memory-v1-local` public | 2026-08-01 10:03:57 | 25 | 0.0000 | 0.000 | 0 | Completed local public integration |
| Local | `duck-reasoning-v1-local`, `ft09`, 1,200-second cap | 2026-08-01 15:57:26 | 1 | 0.0000 | 0.000 | 0 | Gave up at cap; 40 actions |
| Local | `duck-deliberate-v1-local`, `ft09`, stopped smoke | 2026-08-01 21:30:40 | 1 | 0.0000 | 0.000 | 0 | 10 model turns, 0 invoked actions; stopped before finalization; no infrastructure failure |
| Local | `duck-contract-v1-local`, `ft09`, strict contract | 2026-08-02 00:20:04 | 1 | 0.0000 | 0.000 | 0 | 360-second smoke; 0 actions; precise contract errors were surfaced; no infrastructure failure |
| Local | `duck-contract-repair-v1-local`, `ft09` | 2026-08-02 00:30:04 | 1 | 0.0000 | 0.000 | 0 | 1 repaired proposal, no executed action; no infrastructure failure |
| Local | `duck-contract-repair-v1-local`, `cn04` | 2026-08-02 00:37:41 | 1 | 0.0000 | 0.000 | 0 | 7 actions, 7 repaired proposals, 7 prediction matches; no infrastructure failure |
| Local | `duck-contract-repair-v1-wirefix`, `cn04` | 2026-08-02 01:41:29 | 1 | 0.0000 | 0.000 | 0 | 8 model-authored proposals, 8 prediction matches, 0 repairs; one local-model timeout, no infrastructure crash |
| Local | `duck-audit-v1-local`, `ft09`, v2 | 2026-08-02 03:50:00 | 1 | 0.0000 | 0.000 | 0 | 7 actions, one no-change audit trigger, no infrastructure failures; local 4B integration only |
| Local | `duck-information-v1-local`, `ft09` | 2026-08-02 06:28:06 | 1 | 0.0000 | 0.000 | 0 | 3 actions, two targeted-information triggers, two tolerated request failures, no infrastructure failures; local 4B integration only |
| Local | `duck-hierarchy-v1-local`, `ft09` | 2026-08-02 07:06:05 | 1 | 0.0000 | 0.000 | 0 | 1 action before the 4B model gave up; 3 request failures and 1 timeout tolerated; no infrastructure failures; local integration only |
| Local | `duck-diversity-v1-local`, `ft09` | 2026-08-02 07:23:25 | 1 | 0.0000 | 0.000 | 0 | 2 actions, one alternate-seed trigger/use, 3 request failures and 1 timeout tolerated; no infrastructure failures; local integration only |
| Local | `duck-poetiq-v1-local`, public | 2026-08-02 23:04:19 | 25 | 0.0000 | 0.000 | 0 | Completed 25/25 with zero infrastructure failures; 16 tolerated model request timeouts; local 4B integration only |
| Local | `duck-portfolio-v1-local`, public | 2026-08-03 21:50:48 | 25 | 0.0000 | 0.000 | 0 | Completed 25/25 with zero infrastructure failures; 165 analyzer requests, 23 request failures, 20 timeouts; local 4B integration only |
| Kaggle public | Prior v3 aggregate (`validation-v4`) | 2026-07-29 08:08:10 | 125 | 0.3873 | 0.168 | 21 | Five-seed geometry was not faithful |
| Kaggle public | Stock Duck, unseeded fidelity, v5 | 2026-07-29 11:34:20 | 25 | 1.7662 | 0.760 | 19 | Reference run |
| Kaggle public | Stock Duck, seed 0, v6 | 2026-07-29 14:02:04 | 25 | **1.7816** | 0.720 | 18 | Best internal public validation |
| Kaggle public | Stock Duck, seed 1, v7 | 2026-07-29 17:12:53 | 25 | 0.8347 | 0.400 | 10 | Stochastic reference run |
| Kaggle public | Stock Duck two-seed aggregate | 2026-07-30 05:34:06 | 50 | 1.3082 | — | — | Seeds 0 and 1 only; seeds 2–4 deferred |
| Kaggle public | Duck-robust seed 0 | 2026-07-30 09:48:53 | 25 | 1.1216 | 0.480 | 12 | Retired negative experiment |
| Kaggle public | Duck-memory v1 | 2026-08-01 12:33:46 | 25 | 0.8884 | 0.520 | 13 | Retired compactor experiment |
| Kaggle public | Duck-reasoning v1, seed 0, v10 | 2026-08-01 18:34:00 | 25 | **0.8973** | 0.560 | 14 | Latest retained-reasoning run; gate failed |
| Kaggle public | Duck-deliberate v1, seed 0, v11 | 2026-08-01 23:49:07 | 25 | 1.1328 | 0.600 | 15 | Clean run; below 1.20 gate and Stock Duck seed-0 reference |
| Kaggle public | Duck-contract-repair v1, seed 0, v12 | 2026-08-02 03:00:22 | 25 | 1.0037 | 0.520 | 13 | Complete, no infrastructure failures; pre-wirefix repair ablation |
| Candidate | `duck-poetiq-v1` | 2026-08-02 23:04:19 | local complete → two public seeds → gated hidden run | 0.0000 local | 0.000 | 0 | Local/rehearsal prerequisites passed; public launch now authorized with 13.25 GPU-hours because hidden rerun is quota-neutral; daily submission remains separate |
| Offline candidate | `duck-portfolio-v1` router | 2026-08-03 09:13:55 | 25 leave-one-game-out folds | **2.1269 clipped estimate** | — | — | Passed offline gate: +0.6483 vs Stock target, breadth 14 vs 13, three non-Stock policies selected; not a gameplay/Kaggle score |
| Kaggle public | `duck-portfolio-v1`, seed 0, v17 | 2026-08-04 00:20:24 | 25 | 1.5765 | 0.560 | 14 | Clean RTX run; 11 nonzero games, trimmed mean 0.6304; portfolio hard gate failed, seed 1 not started |
| Offline candidate | `duck-portfolio-parity-v1` | 2026-08-08 12:24:24 | 25 leave-one-game-out folds | **1.9201 clipped estimate** | — | — | Runtime/CV parity plus Stock-relative confidence guardrail; +0.4415 vs Stock target, 13 nonzero games, 19/25 folds select Stock |
| Kaggle public | `duck-portfolio-parity-v1`, seed 0, v18 | 2026-08-08 14:59:19 | 25 | **1.1394757** | 13 | 0.6988093 trimmed | Complete; 15 nonzero games, 0 infrastructure failures, 1,454 context evictions, 22 request timeouts; below Stock seed 0 (1.7816224) and prior portfolio v17 (1.5765409) |
| Kaggle submission | `gateway-repair-v2`, submission 55079344 | 2026-07-29 20:37:37 | 110 hidden gateway games | Not returned | Not returned | Not returned | Visible public score **0.80**; private score blank |
| Kaggle submission | `duck-audit-v1`, submission 55174267 | 2026-08-02 11:02:43 | 110 hidden gateway games | **0.72** | Not returned | Not returned | Complete; private score blank |

The published Duck research oracle is **1.6002035** overall mean. It is a
diagnostic reference, not one of our runs: [duck-public-oracle.json](baselines/duck-public-oracle.json).

The portfolio's 2.1269 value is also not a run score. It is a held-out routing
estimate over historical artifacts with a 10-point clip and uncertainty
penalty. Likewise, summing the best historical harness result for each game is
a retrospective oracle because it selects after outcomes are known. Only a
future local or Kaggle artifact belongs in the gameplay tables below.

## Local runs

### Local gameplay integrations

The local profiles use the 4B MLX model and are integration checks, not proxies
for the RTX/Qwen score.

The composite Poetiq local run completed at **2026-08-02 23:04:19 AEST**. Its
structural prerequisite passed: all 25 public IDs ran, the artifact contains
Poetiq diagnostics for every game, and `infrastructure_failures` is empty. The
4B endpoint nevertheless produced a 0.0000 mean engine score and no completed
levels, so this result does not authorize gameplay promotion or substitute for
RTX validation. The 110-game competition-HTTP rehearsal also passed with 110
unique IDs and no infrastructure failures. Kaggle validation is deferred until
the account has at least 14.0 GPU-hours remaining; the live quota check
reported 13.25 hours on 2026-08-02.

| Artifact | Brisbane time (AEST) | Mode/model | Games | Mean score | Levels | Notes |
|---|---|---|---:|---:|---:|---|
| [local-qwen-smoke.json](results/local-qwen-smoke.json) | 2026-07-28 22:46:45 | Local Qwen smoke | 3 | 0.0000 | 0 | All games scored zero |
| [local-public-25.json](results/local-public-25.json) | 2026-07-29 00:11:30 | Local public, 25 games | 25 | 0.0000 | 0 | All 25 game scores were zero |
| [duck-memory-local-smoke.json](results/duck-memory-local-smoke.json) | 2026-07-31 21:53:20 | Duck-memory local smoke | 3 | 0.0000 | 0 | All games scored zero |
| [duck-memory-local-public-25.json](results/duck-memory-local-public-25.json) | 2026-08-01 10:03:57 | Duck-memory local public | 25 | 0.0000 | 0 | All 25 game scores were zero |
| [duck-reasoning-local-ft09-timeout180.json](results/duck-reasoning-local-ft09-timeout180.json) | 2026-08-01 15:57:26 | Duck-reasoning v1, `qwen3.5:4b-mlx` | 1 | 0.0000 | 0 | `ft09`; 40 actions; 1,200 seconds; one request timeout |
| [duck-deliberate-local-ft09.json](results/duck-deliberate-local-ft09.json) | 2026-08-01 21:30:40 | Duck-deliberate v1, `qwen3.5:4b-mlx` | 1 | 0.0000 | 0 | `ft09`; 10 analysis turns; model assigned an action list but invoked no action; stopped before finalization |
| [duck-contract-local-ft09-v2.json](results/duck-contract-local-ft09-v2.json) | 2026-08-02 00:20:04 | Duck-contract v1, `qwen3.5:4b-mlx` | 1 | 0.0000 | 0 | `ft09`; strict one-step contract; 0 accepted proposals/actions; 1 request timeout; no infrastructure failure |
| [duck-contract-repair-local-ft09.json](results/duck-contract-repair-local-ft09.json) | 2026-08-02 00:30:04 | Duck-contract-repair v1, `qwen3.5:4b-mlx` | 1 | 0.0000 | 0 | `ft09`; 1 repaired proposal; no executed action; no infrastructure failure |
| [duck-contract-repair-local-cn04.json](results/duck-contract-repair-local-cn04.json) | 2026-08-02 00:37:41 | Duck-contract-repair v1, `qwen3.5:4b-mlx` | 1 | 0.0000 | 0 | `cn04`; 7 actions; 7 repaired proposals; 7 prediction matches; no infrastructure failure |
| [duck-contract-repair-local-cn04-wirefix.json](results/duck-contract-repair-local-cn04-wirefix.json) | 2026-08-02 01:41:29 | Duck-contract-repair wire-preserving correction, `qwen3.5:4b-mlx` | 1 | 0.0000 | 0 | `cn04`; 8 model-authored proposals; 8 prediction matches; 0 repairs; one request timeout; no infrastructure crash |
| [duck-poetiq-local-public-25.json](results/duck-poetiq-local-public-25.json) | 2026-08-02 23:04:19 | Duck-Poetiq v1, `qwen3.5:4b-mlx` | 25 | 0.0000 | 0 | All 25 games completed with zero infrastructure failures; 216 model requests, 17 request failures, 16 timeouts; integration-only result |
| [duck-portfolio-local-public-25.json](results/duck-portfolio-local-public-25.json) | 2026-08-03 21:50:48 | Duck-Portfolio v1, `qwen3.5:4b-mlx` | 25 | 0.0000 | 0 | All 25 games completed with zero infrastructure failures; 165 model requests, 23 request failures, 20 timeouts; integration-only result |

The later 25-game Duck-reasoning local run was intentionally stopped before it
produced a final artifact. It is therefore not counted as a score.

### Transport rehearsals

The 110-game rehearsal artifacts use scripted actions to validate unique IDs,
gateway HTTP transport, concurrency, cancellation, and artifact teardown. Their
recorded score is 0.0 because no gameplay model is used; this is not a gameplay
benchmark.

| Artifact | Brisbane time (AEST) | Games | Unique IDs | Transport | Infrastructure failures | Recorded score |
|---|---|---:|---:|---|---:|---:|
| [rehearsal-110.json](results/rehearsal-110.json) | 2026-07-28 22:33:24 | 110 | 110 | competition HTTP | 0 | 0.0000 (transport-only) |
| [duck-memory-rehearsal-110.json](results/duck-memory-rehearsal-110.json) | 2026-07-31 21:43:19 | 110 | 110 | competition HTTP | 0 | 0.0000 (transport-only) |
| [duck-reasoning-rehearsal-110.json](results/duck-reasoning-rehearsal-110.json) | 2026-08-01 16:02:53 | 110 | 110 | competition HTTP | 0 | 0.0000 (transport-only) |
| [duck-poetiq-rehearsal-110.json](results/duck-poetiq-rehearsal-110.json) | 2026-08-02 21:25:06 | 110 | 110 | competition HTTP | 0 | 0.0000 (transport-only) |

## Kaggle public validation history

| Artifact | Brisbane time (AEST) | Mode | Seed | Games | Mean score | Mean levels | Total levels | Infrastructure failures |
|---|---|---|---:|---:|---:|---:|---:|---:|
| [validation-v4](results/validation-v4/validation_metrics.json) | 2026-07-29 08:08:10 | Prior reference aggregate | 0 | 125 | 0.3873 | 0.168 | 21 | 0 |
| [duck-reference-unseeded-v5](results/duck-reference-unseeded-v5/validation_metrics.json) | 2026-07-29 11:34:20 | Stock Duck | — | 25 | 1.7662 | 0.760 | 19 | 0 |
| [duck-reference-seed-0-v6](results/duck-reference-seed-0-v6/validation_metrics.json) | 2026-07-29 14:02:04 | Stock Duck | 0 | 25 | **1.7816** | 0.720 | 18 | 0 |
| [duck-reference-seed-1-v7](results/duck-reference-seed-1-v7/validation_metrics.json) | 2026-07-29 17:12:53 | Stock Duck | 1 | 25 | 0.8347 | 0.400 | 10 | 0 |
| [duck-robust-latest](results/duck-robust-latest/validation_metrics.json) | 2026-07-30 09:48:53 | Duck-robust | 0 | 25 | 1.1216 | 0.480 | 12 | 0 |
| [duck-memory-public-validation-current](results/duck-memory-public-validation-current/validation_metrics.json) | 2026-08-01 12:33:46 | Duck-memory v1 | 0 | 25 | 0.8884 | 0.520 | 13 | 0 |
| [duck-reasoning-seed-0-v10](results/duck-reasoning-seed-0-v10/validation_metrics.json) | 2026-08-01 18:34:00 | Duck-reasoning v1 | 0 | 25 | **0.8973** | 0.560 | 14 | 0 |
| [duck-deliberate-seed-0-v11](results/duck-deliberate-seed-0-v11/validation_metrics.json) | 2026-08-01 23:49:07 | Duck-deliberate v1 | 0 | 25 | 1.1328 | 0.600 | 15 | 0 |
| [duck-contract-repair-seed-0-v12](results/duck-contract-repair-seed-0-v12/validation_metrics.json) | 2026-08-02 03:00:22 | Duck-contract-repair v1 | 0 | 25 | 1.0037 | 0.520 | 13 | 0 |
| [duck-portfolio-seed-0-v17](results/duck-portfolio-seed-0-v17/validation_metrics.json) | 2026-08-04 00:20:24 | Duck-portfolio v1 | 0 | 25 | 1.5765 | 0.560 | 14 | 0 |

The `duck-deliberate-v1` seed-0 public kernel was launched at 2026-08-01
11:26:22 AEST and completed as kernel version 11. Its exact machine-readable
artifact is linked above.

No Kaggle public kernel was launched for `duck-contract-v1`: its required
one-game local smoke did not produce an accepted proposal, so the loop stopped
before publishing. The precise rejection text was added to the contract lane
for the next repair experiment.

### Latest Duck-reasoning v1 per-game scores

Source: [duck-reasoning-seed-0-v10/validation_metrics.json](results/duck-reasoning-seed-0-v10/validation_metrics.json).

| Game | Score | Levels completed |
|---|---:|---:|
| `tn36-ef4dde99` | 0.0000 | 0 |
| `lf52-271a04aa` | 0.0000 | 0 |
| `cn04-2fe56bfb` | 0.0000 | 0 |
| `bp35-0a0ad940` | 0.2722 | 1 |
| `wa30-ee6fef47` | 0.0000 | 0 |
| `lp85-305b61c3` | 2.7778 | 1 |
| `r11l-495a7899` | 0.0826 | 1 |
| `tu93-0768757b` | 0.8709 | 2 |
| `sp80-589a99af` | 0.9353 | 1 |
| `m0r0-492f87ba` | 0.0000 | 0 |
| `vc33-5430563c` | 3.7085 | 2 |
| `ar25-0c556536` | 0.0000 | 0 |
| `ka59-38d34dbb` | 3.5714 | 1 |
| `sc25-635fd71a` | 0.1399 | 1 |
| `sk48-d8078629` | 0.0000 | 0 |
| `dc22-fdcac232` | 0.0000 | 0 |
| `cd82-fb555c5d` | 0.0000 | 0 |
| `ft09-0d8bbf25` | 0.0000 | 0 |
| `g50t-5849a774` | 0.0000 | 0 |
| `ls20-9607627b` | 0.0000 | 0 |
| `re86-8af5384d` | 2.7778 | 1 |
| `s5i5-18d95033` | 2.2957 | 1 |
| `sb26-7fbdac44` | 2.7778 | 1 |
| `su15-1944f8ab` | 2.2222 | 1 |
| `tr87-cd924810` | 0.0000 | 0 |

For this run, 12 of 25 games completed at least one level, with 14 levels in
total. The runtime fingerprint confirmed Qwen3.6-27B-FP8, vLLM 0.19.0,
Torch 2.10.0, FlashInfer 0.6.6, RTX Pro 6000, 32K active context, 4× images,
and `analyzer_timeout_s=900`. The reasoning sentinel passed for all 25 games.
There were 31 tolerated model request timeouts and 1,619 context evictions;
there was no compaction and no infrastructure crash.

### Duck-deliberate v1 result and diagnosis

Source: [duck-deliberate-seed-0-v11/validation_metrics.json](results/duck-deliberate-seed-0-v11/validation_metrics.json).

| Metric | Deliberate v1 | Stock Duck seed 0 | Delta |
|---|---:|---:|---:|
| Mean engine score | 1.1328 | 1.7816 | -0.6488 |
| Mean completed levels | 0.600 | 0.720 | -0.120 |
| Total completed levels | 15 | 18 | -3 |
| Actions | 3,358 | 3,989 | -631 |
| Requests | 2,091 | 1,954 | +137 |
| Request timeouts | 29 | 25 | +4 |
| Context evictions | 1,258 | 1,242 | +16 |
| Infrastructure failures | 0 | 0 | 0 |

The verifier was wired correctly but not exercised: the model emitted zero
structured `expect` proposals, with zero prediction matches, mismatches, or
hypothesis revisions. The score therefore measures the prompt-only ablation,
not the complete proposal/verification mechanism. It improved over
Duck-reasoning v1 by 0.2356, but remains 0.0672 below the 1.20 public gate.
Largest gains versus Stock Duck seed 0 were `ka59` (+3.4032), `re86` (+2.7778),
`cn04` (+2.6315), and `vc33` (+1.6468); largest losses were `ft09` (-14.2857),
`ar25` (-6.6161), `sp80` (-4.0078), and `tu93` (-1.9767). All 25 sessions
completed without infrastructure failures.

The next controlled experiment should keep this mode isolated but make the
action contract executable: provide a canonical `result = action([...])`
template and require one-step `expect` objects only when the model can state a
testable board/level outcome. First validate that proposal telemetry is
non-zero on a one-game smoke, then run one public seed before adding any other
memory or perception feature.

The strict contract smoke did not pass: the local Qwen 4B model attempted
actions but omitted `expect` on every accepted call, even after the host
returned the precise missing-expectation error. The next experiment should
therefore test a generic, explicitly logged contract repair that supplies a
one-step probe expectation only after the model has selected an action; it must
separate repaired proposals from model-authored proposals in telemetry.

### Duck-contract-repair v1 local smoke

The repair lane passed the integration criterion on `cn04-2fe56bfb` at
2026-08-02 00:37:41 AEST: seven one-step actions executed, seven repaired
proposals, and seven prediction matches, with zero infrastructure failures.
The local engine score was 0.0 and no level completed, so this is not an RTX
score proxy. The companion `ft09` smoke at 14:30:04 AEST recorded one repair
but no executed action. The clean `cn04` smoke authorized the 25-game RTX
public run; no hidden submission is authorized by this result.

The wire-preserving correction passed the same interface smoke on
`cn04-2fe56bfb` at 2026-08-02 01:41:29 AEST: eight model-authored one-step
proposals executed, eight prediction matches, and zero repairs. The local
Qwen/MLX server timed out once and the game completed with score 0.0 and no
level completion, so this remains an integration pass rather than a gameplay
score proxy. It verifies that the model-authored expectation survives the
sandbox transport.

The corrected source/notebook package was rebuilt and published to the stable
source dataset at 2026-08-02 02:03:11 AEST. Its manifest SHA-256 is
`75dd564e146dc801f8aa83f12d5d79241f15791341d7ab2e63b7f87a3754159e`. A new
seed-0 public kernel is running from that package and is tracked in
`results/duck-contract-repair-progress.json`; its exact metrics remain
pending until the Kaggle worker finishes.

### Duck-audit v1 local smoke

Source: [duck-audit-local-ft09-v2.json](results/duck-audit-local-ft09-v2.json).

The one-game local smoke completed at **2026-08-02 03:50:00 AEST** on `ft09`
with the `qwen3.5:4b-mlx` integration model. It executed seven actions and
completed no levels before the six-minute local cap. The score was 0.0, which
is an integration result rather than an RTX/Qwen capability estimate. The
sidecar triggered once on an unchanged gameplay frame, with zero
infrastructure failures, one tolerated request timeout, and one context
eviction. The required Stock-compatible interface smoke therefore passed;
the local score did not.

The preliminary attempt reached the same gameplay cap but could not write its
artifact because optional image diagnostics require an unavailable
local `imageio` dependency. It was not used as the recorded result. The v2
rerun used minimal diagnostics and produced the exact artifact above.

### Duck-audit v1 public validation

Source: [duck-audit-seed-0-v14/validation_metrics.json](results/duck-audit-seed-0-v14/validation_metrics.json).

The seed-0 public kernel completed at **2026-08-02 07:55:07 AEST** with all 25
games and no infrastructure failures. It scored **2.5530541**, a **+0.7714317**
delta over Stock Duck seed 0 (**1.7816224**), with 18 total completed levels,
median completed levels 1, 3,455 actions, and 1,468,508 generated tokens.

| Metric | Duck-audit v1 | Stock Duck seed 0 | Delta |
|---|---:|---:|---:|
| Mean engine score | 2.5531 | 1.7816 | +0.7714 |
| Mean completed levels | 0.720 | 0.720 | 0.000 |
| Median completed levels | 1 | 1 | 0 |
| Total completed levels | 18 | 18 | 0 |
| Actions | 3,455 | 3,989 | -534 |
| Generated tokens | 1,468,508 | 1,652,946 | -184,438 |
| Context evictions | 1,315 | 1,242 | +73 |
| Request failures / timeouts | 25 / 25 | 25 / 25 | 0 / 0 |
| Infrastructure failures | 0 | 0 | 0 |

The score lift is concentrated: `ft09` **+14.2857**, `vc33` **+9.8043**, and
`tn36` **+1.6846**; losses are `ar25` **-3.1167**, `r11l` **-2.3636**, and
`s5i5` **-0.9612**. Five games improved, six regressed, and fourteen were
unchanged. Three games gained a level and three lost one, so the strict
promotion gate is **not passed** despite the attractive mean-score lift:
median levels did not improve and per-game non-regression was violated.

Audit telemetry recorded 132 sparse triggers (113 repeated-action, 19
unchanged-frame), zero tool-parse failures, and no overflow recovery. The
candidate remains a strong diagnostic result, but the next public ablation
should target its concentrated regressions rather than stack more global
prompting.

### Duck-information v1 local smoke

Source: [duck-information-local-ft09.json](results/duck-information-local-ft09.json).

The one-game local smoke completed at **2026-08-02 06:28:06 AEST** on `ft09`
with `qwen3.5:4b-mlx`. It executed three actions, completed no levels, and
scored 0.0 before the six-minute local cap. The sidecar triggered twice on
unchanged gameplay frames and recorded two tolerated request failures and
three context evictions. There were no infrastructure failures. This passes
the interface/integration criterion but is not a score proxy for RTX/Qwen;
the public run remains gated on the final `duck-audit-v1` comparison.

### Duck-hierarchy v1 local smoke

Source: [duck-hierarchy-local-ft09.json](results/duck-hierarchy-local-ft09.json).

The one-game local smoke completed at **2026-08-02 07:06:05 AEST** on `ft09`
with `qwen3.5:4b-mlx`. It executed one action, completed no levels, and scored
0.0 before the six-minute cap. The local model produced malformed/tool-timeout
responses and gave up; the harness recorded three request failures, one
timeout, and one context eviction, but no infrastructure failure. The hierarchy
trigger did not fire because the model produced only one accepted action, so
this is an interface smoke rather than evidence about candidate-search quality.
The public run is intentionally held until the active audit comparison is
complete.

### Duck-diversity v1 local smoke

Source: [duck-diversity-local-ft09.json](results/duck-diversity-local-ft09.json).

The one-game local smoke completed at **2026-08-02 07:23:25 AEST** on `ft09`
with `qwen3.5:4b-mlx`. It executed two actions, completed no levels, and scored
0.0 before the six-minute cap. After two unchanged gameplay transitions, the
agent used the alternate sampling seed once and returned to the normal path.
The local model then produced malformed/timeout responses and gave up; the
harness recorded three request failures, one timeout, and one context eviction,
but no infrastructure failure. This validates the diversity wire and
telemetry, not gameplay quality. Public publication remains queued behind the
active audit comparison.

### Duck-contract-repair v1 public result and diagnosis

Source: [duck-contract-repair-seed-0-v12/validation_metrics.json](results/duck-contract-repair-seed-0-v12/validation_metrics.json).

The seed-0 public run completed at 2026-08-02 03:00:22 AEST with all 25 games
started and no infrastructure failures. It scored **1.0036923**, with 13 total
levels completed (mean 0.52; median 0), 2,544 actions, and 1,809 model
requests. Relative to Stock Duck seed 0 (1.7816224), the delta is **-0.7779299**;
relative to Duck-deliberate v1 (1.1328436), it is **-0.1291513**.

| Metric | Contract-repair v1 | Stock Duck seed 0 | Delta |
|---|---:|---:|---:|
| Mean engine score | 1.0037 | 1.7816 | -0.7779 |
| Mean completed levels | 0.520 | 0.720 | -0.200 |
| Total completed levels | 13 | 18 | -5 |
| Actions | 2,544 | 3,989 | -1,445 |
| Contract repairs | 2,673 | — | — |
| Prediction matches / mismatches | 2,294 / 224 | — | — |
| Context evictions | 1,487 | 1,242 | +245 |
| Request failures / timeouts | 30 / 30 | 25 / 25 | +5 / +5 |
| Infrastructure failures | 0 | 0 | 0 |

The result is a deliberately attributable **pre-wirefix ablation**. Kernel v12
was launched before the corrected source dataset was published, so the
sandbox stripped model-authored `expect` payloads and every accepted proposal
entered the generic repair path. The telemetry therefore shows 2,673 repairs
and 2,294 prediction matches, rather than measuring the corrected
wire-preserving lane. The local wirefix smoke had already shown eight
model-authored expectations surviving transport with zero repairs. A corrected
public validation is required before judging the contract idea itself.

The run's strongest games were `vc33` (5.5840), `tn36` (3.5714), `lp85`
(2.7778), `sk48` (2.7778), `sb26` (2.7778), and `su15` (2.2222). Its largest
losses versus Stock Duck seed 0 were `ft09` (-14.2857), `ar25` (-7.0192),
`sp80` (-4.5083), `r11l` (-3.4554), and `tu93` (-2.0990). Thus the aggregate
loss is concentrated in several high-value games rather than an infrastructure
collapse. The generic repair path is not promoted, but the corrected
wire-preserving candidate remains a justified next public ablation because it
fixes a proven transport defect without changing Stock Duck's model or prompt.

## Kaggle submission / private games

The completed v3 submissions are recorded in
[submission-ledger.json](submission-ledger.json):

| Field | Recorded value |
|---|---|
| Experiment | `gateway-repair-v2` / `duck-reference-early-private-baseline` |
| Submission created (AEST) | 2026-07-29 20:37:37 |
| Submission verified (AEST) | 2026-07-30 05:34:06 |
| Submission kernel | `kinwochan/ouroboros-arc-agi-3-v3`, version 2 |
| Submission reference | `55079344` |
| Hidden gateway workload | 110 games (55 public leaderboard + 55 private) |
| Public validation mean | 1.3082 (two reference seeds: 1.7816 and 0.8347) |
| Visible public leaderboard score | **0.80** |
| Private-game score | **Not returned; `privateScore` is blank** |
| Submission status | Complete |
| Leaderboard best before submission | 1.86 |

### Duck-audit v1 submission

| Field | Recorded value |
|---|---|
| Experiment | `duck-audit-v1` |
| Submission created (AEST) | 2026-08-02 11:02:43 |
| Submission kernel | `ouroboros-arc-agi-3-v3`, version 3 |
| Submission reference | `55174267` |
| Source manifest | `6e49cea924ec...` (exact audit public-run source) |
| Hidden gateway workload | 110 games (55 public leaderboard + 55 private) |
| Public validation mean | 2.5531 |
| Visible public leaderboard score | **0.72** |
| Private-game score | **Not returned; `privateScore` is blank** |
| Submission status | Complete |

Two earlier submission references (`55075061`, `55077185`) are recorded as
failed gateway attempts and have no score artifact. Duck-memory v1 and
Duck-reasoning v1 have public validation artifacts but no hidden submission.

## Interpretation and promotion status

- The best internal public validation remains Stock Duck seed 0 at **1.7816**.
- The latest Duck-reasoning v1 score, **0.8973**, is **0.8843 below** that
  reference and **0.3027 below** the 1.20 candidate gate.
- Duck-deliberate v1 scored **1.1328** with no infrastructure failures: it is
  **0.6488 below** Stock Duck seed 0 and **0.0672 below** the 1.20 gate. Its
  generic verifier saw zero structured proposals, so the next experiment must
  first fix proposal emission before adding another capability.
- The latest visible submitted score is **0.72** from Duck-audit v1; its
  private score is still unavailable. The earlier gateway-repair submission
  remains at **0.80**.
- No current candidate is authorized for another hidden submission until a new
  public run passes the score and infrastructure gates.
