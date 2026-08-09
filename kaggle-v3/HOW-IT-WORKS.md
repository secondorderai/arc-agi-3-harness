# How `kaggle-v3` works

## Isolated execution lanes

`duck-reference` is the score-recovery lane. The runner imports
`DuckReferenceToolAgent` and `DuckReferenceHarnessSolver` from
`src/duck_reference/`. Its model-facing prompt and Python state match stock
Duck: no Ouroboros ledger, rich transition perception, prediction checking,
fallback, strategic reset, or adaptive budget is reachable.

`duck-memory` is the retained-reasoning experiment. It starts from the same
Stock Duck prompt, Python tool, action policy, sampling, concurrency, and
fixed game budget. It changes only conversation memory:

- vLLM's `reasoning` response field is canonicalized and replayed as
  `reasoning_content`, which is the field rendered by the bundled Qwen3.6
  tokenizer template;
- `enable_thinking=true` and `preserve_thinking=true` are explicit on gameplay
  requests;
- a two-turn tokenizer sentinel fails closed unless historical private
  reasoning is visibly placed inside the next `<think>` block;
- the 30-assistant-turn deletion rule is replaced by structured semantic
  compaction while the newest eight assistant turns and complete
  tool-call/result pairs remain verbatim.

Compaction begins at an estimated 24,576 input tokens and targets at most
16,384. Qwen3.6 performs it with thinking disabled, temperature 0.2, a
2,048-token output cap, a 300-second timeout, and no more than four concurrent
compactions. Its validated summary records level state, mechanics, action
effects, objects and coordinates, evidence and contradictions, successful and
failed experiments, the current plan, unresolved questions, and cross-level
knowledge. One smaller-prefix retry is allowed. Stock Duck emergency trimming
keeps a session alive if both attempts fail, but that loss blocks promotion.
This is an offline Qwen/vLLM analogue of the retained-reasoning and compaction
mechanisms in OpenAI's
[ARC-AGI-3 write-up](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/).
Its reported 13.3%→38.3% improvement is evidence for the mechanism, not an
expected score transfer to Qwen or Duck.

`ouro-hybrid` is the experiment lane described below. It stays disabled until
the five-seed reference distribution is recovered. This separation prevents a
promising hybrid idea from silently changing the baseline it is measured
against.

`duck-robust` is the low-risk candidate lane. It uses the stock Duck prompt,
sampling, and compact state until a session has spent at least 30 minutes and
64 actions without level progress. Recovery additionally requires two
consecutive low-success estimates and either a repeated state-action cycle or
a contradicted hypothesis. Each level gets at most one recovery.

At recovery, verified evidence is compacted, old conversational turns are
discarded, and a deterministic alternate seed gets one temperature-0.8 turn
to propose two competing explanations plus a discriminating predicted action.
Subsequent execution uses temperature 0.2, requires predictions for every
action, caps batches at eight, and aborts on the first mismatch. A reset is
allowed only for a contradicted hypothesis with no more than two meaningful
changes in the last 16 actions.

The probability signal is trained reproducibly from Tufa's 500 published
trajectories. The public JSON lacks frames and hypothesis consistency, and its
leave-one-game-out AUC is only about 0.49. This limitation is preserved in the
artifact and makes the gate fail closed: probability alone is never enough
to fork a trajectory. `duck-robust-v1` underperformed Stock Duck and is kept
as a negative result rather than promoted.

`duck-reasoning` is the follow-up isolation lane. It retains the useful part of
the memory experiment—the Qwen reasoning adapter—but abandons its semantic
compactor entirely. The agent is a thin `ToolAgent` subclass that:

- canonicalizes vLLM `message.reasoning` to Qwen's historical
  `reasoning_content` field;
- explicitly requests `enable_thinking=true` and
  `preserve_thinking=true` through the existing Stock Duck request path;
- preserves Stock Duck's normal prompt, tools, action batching, 30-turn
  history policy, context budgeting, concurrency, and time limits; and
- records retained, evicted, and reasoning-bearing turn counts without making
  auxiliary model calls or injecting a generated summary.

This lane is the causal ablation for retained reasoning. If it improves over
Stock Duck, compaction can be reconsidered as a separate experiment. If it
does not, the reasoning transport itself—not compaction overhead—becomes the
next variable to investigate.

`duck-contract` and `duck-contract-repair` are the next narrow control-loop
ablations. They keep Stock Duck's perception, sampling, context policy, and
game budget, but make the falsification step executable: one action per turn,
an observable `expect` object, and transition verification before the next
plan. The strict lane rejects missing expectations. The repair lane preserves
the action, injects a generic `{"board_changed": true}` probe when necessary,
truncates an oversized batch to its first action, and records both repairs and
model-authored proposals separately. The sandbox explicitly preserves
`expect` for these modes; otherwise a valid model proposal would be lost before
the host verifier sees it. Neither lane contains game-specific rules or
coordinates.

`duck-audit` is the first Poetiq-inspired self-auditing candidate. It imports
the Stock Duck `ToolAgent` directly and leaves its system prompt, Python tool,
action normalization, history eviction, sampling, and runtime limits intact.
The only change is a sparse user-prompt addendum after a generic signal: a
short trailing run of the same action or multiple transitions whose gameplay
frame is unchanged. The addendum asks the model to inspect the newest evidence
and choose `continue`, `inspect`, or `replan`; it does not force an `expect`
object, add a second model request, or reset a level. Trigger counts and the
two generic signal types are recorded separately. This is intentionally
minimal: the earlier broad deliberation and contract lanes increased context
churn and reduced score, so self-auditing is gated by observed stagnation
rather than paid on every turn.

`duck-poetiq-v1` is the composite experiment built from the five Poetiq
recommendations. It keeps Stock Duck's model, Python surface, 30-turn history,
sampling, and 7,920-second public cap, but adds one persistent protocol and no
auxiliary model call. A generic stall (four repeated actions or three unchanged
gameplay transitions) starts one intervention: audit the latest evidence, ask
for the smallest discriminating observation, rank at most three hypotheses,
and execute one low-risk action with an optional model-authored prediction.
The shared verifier aborts on a mismatch. A second failed intervention uses
the same model and history with seed `primary + 17`; a 12-action cooldown and
two-attempt-per-level limit prevent an ensemble or prompt storm. A game that
has completed no level may yield only after both interventions fail, 64 actions
and 30 minutes have elapsed, and the last 16 transitions are unchanged. Games
that have made level progress are never yielded early.

The composite lane records triggers, candidate sets, information requests,
prediction omissions/matches/mismatches, alternate-seed use, intervention
outcomes, and stalled yields. Its quota-aware gate uses two independent public
seeds: mean score at least 1.4082, seed-level floors of 18/10 completed levels
and 15/9 nonzero games, and a top-three-trimmed mean above the Stock baseline
0.6061. This combines the five mechanisms for the first experiment while
keeping the telemetry needed for later one-variable ablations.

`duck-portfolio-v1` is a deterministic policy router rather than a parallel
ensemble. Every game owns one `DuckPortfolioToolAgent`, one Qwen conversation,
one Python history, and one live trajectory. The first eight executed actions
are unchanged Stock Duck actions. If they complete a level, Stock is locked
for the game. Otherwise the host extracts nine normalized, generic features:
color/component counts, repeated-shape fraction, maximum axial symmetry,
mouse/change/HUD-only/repeated-action fractions, and changed-area fraction.
Committed ridge scorers select Stock, Audit, Deliberate, or Contract Repair
without another model call. Policy activation changes only the future control
protocol; accumulated reasoning, messages, tool results, and world-model text
remain in the same agent.

The transparent router artifact is trained from Stock medians (unseeded,
seed 0, and seed 1), Audit seed 0, Deliberate seed 0, and the median of two
Contract Repair seed-0 artifacts. Its inputs come only from the first eight
Stock seed-0 transitions. Scores are clipped at 10, ridge alpha is 10, the
leave-one-game-out RMSE penalty is `0.5 × RMSE`, and a non-Stock policy must
clear Stock by 0.25. Ties resolve Stock → Audit → Deliberate → Contract Repair.
The committed leave-one-game-out estimate improves clipped mean from 1.4786
to 2.1269 (+0.6483), preserves/increases nonzero breadth from 13 to 14, and
selects all three non-Stock policies. This is an offline generalization check,
not a claimed Kaggle score.

At most one policy switch is possible. It requires no completed level, 64
actions since selection, zero gameplay changes in the latest 16 transitions,
and at least 1,800 seconds remaining. The next-highest original router score
is activated without resetting or forking. Full validation diagnostics record
features, raw/adjusted scores, margin, route/switch events, per-policy action
counts, and policy-specific telemetry. Hidden runs retain aggregate routing
telemetry only.

The per-game maximum across historical harness results is an unattainable
retrospective oracle: it chooses after seeing each result. The portfolio is
the achievable counterpart. It must choose prospectively after eight generic
transitions, is penalized by held-out error, and cannot see an identifier or
public-game lookup. Therefore the oracle sum is a ceiling for diagnosis, not
the router's expected or promoted score.

### Persistent retrodictive lane

`duck-retrodict-v1` reuses the stable worker, runtime-state, transition
callback, failure-floor, and notebook infrastructure, but does not reuse the
old generated-Python simulator contract. Each game owns one
`RetrodictiveWorldModel` for its entire session. The model records immutable
before/action/after evidence, exact level-scoped edges, and a bounded typed
version space (`noop`, color map, rigid translation, and object click-recolor
in v1). After every observation it executes every applicable rule over the
complete log. Two supporting transitions and zero contradictions are required
for general rule certification; a contradiction immediately removes that
certification.

Perception retains three simultaneous interpretations: color components under
4-connectivity, color components under 8-connectivity, and an all-color view.
Rules carry their ontology, so evidence—not a hardcoded background choice—can
decide which view survives. Candidate probes are ranked by predicted-outcome
entropy, novelty, and observed game-over risk. A compact history-conditioned
clone graph activates only after the same level/observation/action has
incompatible successors; its activation is diagnostic in v1 and prevents the
ambiguous exact edge from entering a plan.

CPU search uses deterministic exact edges and fully certified typed
predictions. The host may execute the first action of that plan without an LLM
request, but the action batch limit is always one. The real frame is observed
and the whole version space is replayed before another action can run. If no
plan is certified, the single Stock Duck actor receives the verifier summary
and low-risk probe recommendation. Model prose and temporary Python never gain
execution authority.

Public evaluation is fail-closed. Validation traces are emitted only outside
the hidden-submission profile and are scored chronologically: calibrate on the
prefix, predict each held-out transition before observing its answer, then add
it to evidence. The default offline gate requires at least ten holdout
transitions, 95% precision, 60% useful coverage, and 5ms p95 prediction time.
If old generated-Python predictions are included in the records, typed-rule
precision may not regress them. The later two-seed public gate requires a mean
above the supplied leaderboard target plus margin, a trimmed-mean and breadth
floor, no infrastructure failures, and an on-time 110-game rehearsal.

The Qwen3.6-35B-A3B arm is a separate A/B, not an ensemble. Notebook generation
requires an explicit model ID and attached Kaggle dataset. Paired seed-0/1
artifacts must contain identical games. The challenger is accepted only if it
avoids material score regression and produces either a 0.10 score lift or at
least a 15% elapsed-time improvement. No second end-to-end policy or per-game
training is enabled.

## The central bet

ARC-AGI-3 games reward flexible inference more than a large catalog of
handwritten game rules. The public Duck result supports that view: a capable
multimodal coding model, allowed to inspect a scene and write small searches,
substantially outperformed rigid solvers. `kaggle-v3` preserves that
flexibility and strengthens the control loop around it.

Four ideas carry most of the design:

1. **Coding-agent flexibility.** Qwen can write temporary Python to inspect
   objects, search paths, score candidate sequences, and act. Helpers are
   generic—there are no game IDs, fixed coordinates, or public-game rules.
2. **Compact evidence memory.** A `HypothesisLedger` carries the world, goal,
   and action models; supporting evidence; contradictions; open questions;
   the active plan; and cross-level knowledge. It survives context eviction,
   while old conversational turns do not.
3. **Multimodal perception.** Every frame exposes a 4× image, crop-based ASCII,
   4-connected segmentation, changed regions, gameplay-versus-HUD
   classification, object tracks, and an animation summary.
4. **Verified execution.** A planned action may include a prediction. The
   executor compares it with the real transition and aborts the remaining
   batch on the first mismatch. Evidence wins over a stale plan.

The detailed two-page flow is in
[`architecture.drawio`](architecture.drawio).

## One game session

The runner controls `arc_agi.Arcade` directly through TAAF's `GameAPI`; it
does not force Qwen through the competition's one-action template agent.

Each session repeats:

```text
observe → update hypotheses → inspect/search in Python → act → verify → revise
```

### Stable model-facing state

`current_frame` and `previous_frame` expose:

- `image`
- `ascii` and `ascii_crop(row0, row1, col0, col1)`
- `segmentation`
- `changed_regions`
- `object_tracks`
- `animation_summary`
- `step`, `level`, and `shape`

`transitions` expose the action, before/after frames, gameplay and HUD
changes, terminal state, and reward. `hypothesis_ledger` is a compact,
read-only view of the durable evidence ledger.

`action(...)` accepts a named action, a mouse object such as
`{"action":"MOUSE","row":12,"col":28}`, or an ordered list. The executor
stops a batch immediately on level completion, game over, run completion,
invalid action, action error, or prediction mismatch.

### Constrained Python

Model-written Python runs in an isolated `python -I -S` subprocess with
resource limits and a JSON-lines action RPC. Its builtins and imports are
allowlisted. It cannot:

- open files
- read environment variables
- access the network
- load native modules
- create child processes
- import unapproved packages

It can use connected components, compact frame diffs, object tracks,
shortest-path search, candidate-sequence scoring, and `action(...)`.

### Failure handling

Three consecutive model failures activate a deterministic explorer. The same
floor handles an unavailable model server. It normalizes engine action names,
cycles movement actions deterministically, and selects generic component
centroids for mouse games.

Game-over reset is automatic and separate. A strategic reset is allowed at
most once per level and only when all three conditions hold:

- at least 64 actions since level progress
- no more than two gameplay-changing transitions in the last 16 actions
- the active hypothesis has a recorded contradiction

## Global scheduling (`ouro-hybrid` only)

The hybrid RTX profile runs 28 concurrent sessions. A global scheduler reserves 20
minutes for setup/teardown and treats 8h40 as the soft gameplay deadline.
When a session starts, its budget is recomputed from remaining wall time,
remaining games, concurrency, and remaining waves. Model reasoning stays
parallel; the brief shared competition HTTP call is serialized because
`requests.Session` is not thread-safe.

Cancellation drains active tasks, marks unfinished games, closes the shared
scorecard, and still writes the final metrics artifact.

The Duck reference does not use this adaptive scheduler. Each of its 25 games
gets the audited 7,920-second cap, with 28 concurrent slots and a 900-second
analyzer timeout.

`duck-robust` also retains the fixed reference game budget. Adaptive global
allocation remains a later, independently measured experiment.

### Hidden-submission wall-clock budget

The competition rerun is a separate timing envelope from public validation. It
plays 110 gateway games with 28 workers, which is four worst-case waves. The
public 7,920-second cap would require 31,680 seconds (8h48) of gameplay before
notebook setup and teardown, so the nine-hour Kaggle limit can truncate the
last wave. The submission path now derives a four-wave cap of 7,200 seconds
(120 minutes) per game, for 28,800 seconds of worst-case gameplay. This leaves
the configured setup/teardown reserve and a launch-skew cushion while leaving
the public 132-minute Duck reference unchanged. The hidden metrics record the
calculated waves, cap, and worst-case gameplay budget for post-run auditing.

## Evaluation protocol

The public split is frozen:

- DEV: 13 games
- TEST: 9 games
- QUARANTINE: 3 games

One change is screened on DEV seed 0. Finalists run against the reference on
TEST seeds 0–4. QUARANTINE opens once for the selected candidate, then
freezes.

A hybrid candidate is promoted only when:

- mean TEST engine score improves by at least 0.10
- median completed-level total improves by at least one
- no game loses a median completed level
- no model-load, OOM, sandbox, transport, or game-session crash occurs
- the 110-game rehearsal projects completion before 8h40

Reference recovery is a separate sequence:

1. One unseeded 25-game fidelity kernel must have the exact runtime
   fingerprint, start all games, finish normally, and have zero infrastructure
   failures. Its score alone does not gate.
2. Seeds 0–4 run in five independent full-budget kernel versions.
3. Their `mean_engine_score` values are aggregated. The gate requires a mean
   of at least 1.20, no infrastructure failures, and every kernel finishing
   before 8h40.

The prior `0.387329` result is retained as diagnostic evidence only: it was a
125-session hybrid job. The `0.514362` value in its log was seed 4, not the
aggregate.

Local `qwen3.5:4b-mlx` results are integration evidence only. They validate
the same interfaces on a 16-GB M1 with two workers and six minutes per game;
they are not a score proxy for Qwen3.6-27B-FP8.

The first `duck-memory-v1` promotion gate is intentionally one seed-0 public
run rather than a five-seed study because Kaggle GPU quota must reserve the
hidden run. Promotion requires all 25 games to start and finish, engine score
at least 1.20, runtime below 8h40, no infrastructure failure, a passing
reasoning sentinel, complete raw-or-compacted coverage for every
reasoning-bearing turn, and zero emergency trims, ordinary evictions, or
unrecovered compaction failures.

## Packaging and notebooks

The source dataset contains Python and documentation only. The builder omits
`.pkl` and `.pickle`, writes `OURO3_SOURCE_BUNDLE.json`, and produces a
SHA-256 manifest for every packaged file.

Both notebooks are generated with `nbformat` and executed top to bottom with
`nbclient` in dry-run validation:

- **Validation:** verifies source hashes and runtime pins, then runs exactly
  25 public sessions. The embedded seed is either omitted for fidelity or one
  of 0–4; each seed is a separate Kaggle kernel version.
- **Submission:** on Save & Run, verifies RTX hardware and performs a real
  Qwen/vLLM API smoke test; on competition rerun, discovers and plays the 110
  hidden gateway IDs.

Both pin `NvidiaRtxPro6000`, disable internet, attach the private source,
wheelhouse, model snapshot, and competition, and fail closed on a runtime or
hardware mismatch.

The notebook builder defaults to `duck-reference`. Passing
`--mode duck-robust --validation-seed 0` creates a separate candidate pair;
it never silently rewrites the reference lane.

Passing `--mode duck-memory --validation-seed 0` creates the memory candidate.
Before gameplay its validation notebook loads the attached model tokenizer,
renders the sentinel conversation, and records the exact rendered-history
result in the runtime fingerprint. Validation saves a compressed full memory
trace; hidden submission keeps reasoning in process but emits only aggregate
memory telemetry.

The narrower Kaggle operation is `--mode duck-robust --seed0-only`. It checks
the embedded mode and seed before upload, checks the returned runtime
fingerprint and recovery diagnostics, and cannot create a competition
submission. The first candidate therefore consumes one 25-game GPU kernel
and requires an explicit review before promotion.

`--mode duck-memory --seed0-only` additionally verifies the local 110-game
transport rehearsal, local 25-game integration artifact, weekly GPU reserve,
reasoning coverage, compaction diagnostics, exact source manifest, and exact
configuration hash. With `--submit`, only that successfully validated source
and configuration can be published to the hidden notebook.

The composite `duck-poetiq-v1` lane runs two independent 25-game public
kernels. Because the current competition account reports hidden gateway
reruns as quota-neutral, its preflight reserves 4.5 GPU-hours for those public
kernels and does not reserve GPU hours for the later private submission. The
private run remains gated by the public promotion result and the daily
submission limit.

The portfolio lane has the same quota geometry but a stricter seed-0 gate:
score at least 2.5631, at least 18 levels and 15 nonzero games, and a
top-three-trimmed mean above 0.9370812. Only then may seed 1 run. Hidden
submission additionally requires seed 1 floors of 10 levels and 9 nonzero
games, two-seed mean at least 1.4081755, two-seed trimmed mean above 0.6060513,
clean infrastructure/routing telemetry, and an under-8h40 rehearsal.

## Submission and iteration

The publishing script versions the private source dataset, runs and pulls the
unseeded fidelity kernel, then runs and pulls the five seed kernels
sequentially. It verifies source, prompt, and runtime hashes; request failures,
timeouts, tool-call parse failures, context evictions, actions, tokens, and
per-game scores are retained in every artifact. Only after the aggregate gate
passes does it version the final kernel, refresh the leaderboard, check the
daily quota, and submit that exact version. The message contains the Git SHA,
config hash, runtime/model profile, validation mean, and experiment.

Every result is appended to `submission-ledger.json`. Only the 55-game public
leaderboard half is visible during the competition; the other 55 games
remain private until final evaluation.

Controlled experiments are attempted in this order:

1. evidence-ledger compaction
2. changed regions, object tracking, and animation perception
3. verified plans and deterministic fallback
4. adaptive budgeting and strategic restart
5. image scale, context, and sampling

At most one competition submission is made per day. The stop target is
`max(1.86, refreshed leaderboard best) + 0.01`.

## Attribution

The direct Arcade runner, TAAF orchestration, tool-agent loop, sandbox
foundation, prompt structure, image handling, and vLLM setup are adapted from
the public MIT-licensed [Duck/TAAF
harness](https://github.com/Tufalabs/duck-harness) by Tufa Labs. The linked
Kaggle notebook is a wrapper around the same public source. No opaque pickle
from that dataset is copied into v3.

See `THIRD_PARTY_NOTICES.md` for the retained notice and links.
