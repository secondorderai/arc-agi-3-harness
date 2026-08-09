# Poetiq-inspired harness recommendations for ARC-AGI-3

Updated: 2026-08-03
Scope: Stock Duck-compatible experiments, with the first composite candidate
implemented as `duck-poetiq-v1`

This note translates the public Poetiq research posts into testable harness
changes for the ARC-AGI-3 game loop. The recommendations are hypotheses, not
promises of a score transfer. Every candidate must retain Stock Duck as its
control, pass a one-game local interface smoke, and receive at most one
25-game public validation before another change is selected.

## Current evidence snapshot (updated 2026-08-02)

The completed `duck-audit-v1` RTX public kernel (artifact:
`results/duck-audit-seed-0-v14/validation_metrics.json`) scored **2.5531** on
all 25 games, versus Stock Duck seed 0 at **1.7816**. This is a strong
**+0.7714** mean-score lift with no infrastructure failures, but it did not
pass the strict promotion gate: mean and median completed levels were both
unchanged at 0.72 and 1, while three games lost a level relative to the
control. The lift is concentrated in `ft09` (+14.2857), `vc33` (+9.8043), and
`tn36` (+1.6846); `ar25` (-3.1167), `r11l` (-2.3636), and `s5i5` (-0.9612)
regressed. Telemetry recorded 132 sparse triggers, 1,315 context evictions,
and 25 request timeouts. The result supports event-triggered auditing as a
capability hypothesis, but not as a frozen promotion candidate. A corrected
wire-preserving `duck-contract-repair-v1` public run is currently in progress;
the older 1.0037 artifact is explicitly a pre-wirefix ablation and must not be
used to judge the corrected contract.

The completed ablations explain why the harness should stay small and
event-driven: Stock Duck seed 0 scored **1.7816**, while contract-repair
(**1.0037** pre-wirefix), deliberate (**1.1328**), memory (**0.8884**),
reasoning (**0.8973**), and robust (**1.1216**) variants all lost substantial
score. The Stock Duck seed spread (seed 0 **1.7816** versus seed 1 **0.8347**)
is large enough that one public run is not evidence of a reliable lift.

The five recommendations are now integrated into an isolated composite lane.
`duck-poetiq-v1` keeps Stock Duck's model and tool/history surface and uses one
compact persistent protocol. It activates only on generic stalls, asks for a
small discriminating inspection, ranks at most three hypotheses, executes one
optional-prediction probe, verifies the transition, and uses one alternate
seed (`primary + 17`) only after the first intervention fails. It may yield a
game only before any level progress and only after two failed interventions,
64 actions, 30 minutes, and 16 unchanged recent transitions. This is an
implementation and test candidate; no score or Kaggle promotion is implied
until its local rehearsal and two-seed gate pass.

The next composite experiment is `duck-portfolio-v1` (implemented 2026-08-03
09:13 AEST). It operationalizes the strongest lesson from the ablations:
different Stock-compatible control protocols have complementary per-game
strengths, but an after-the-fact per-game maximum is not deployable. The
portfolio observes eight unchanged Stock actions and uses a small regularized
router to select Stock, Audit, Deliberate, or Contract Repair prospectively.
It excludes Poetiq because the completed evidence showed no unique positive
per-game maximum. One persistent conversation avoids paying for parallel
agents or discarding accumulated reasoning.

The router's leave-one-game-out check passes: clipped mean 2.1269 versus the
Stock training target 1.4786 (+0.6483), breadth 14 versus 13, with Audit chosen
8 times, Contract Repair 3, Deliberate 4, and Stock 10. These numbers estimate
route selection on the existing 25-game evidence and are neither a Kaggle
validation score nor the historical best-per-game oracle. The latter chooses
with outcome knowledge; the router sees only generic first-eight-transition
features and pays a `0.5 × RMSE` uncertainty penalty plus a 0.25 Stock margin.

The composite local 25-game integration completed at 2026-08-02 23:04:19
AEST with all 25 sessions finalized and zero infrastructure failures. It
scored 0.0000 with the local `qwen3.5:4b-mlx` endpoint and completed no levels;
this is an integration result, not an RTX capability estimate. Its paired
110-game competition-HTTP rehearsal passed with 110 unique IDs and no
infrastructure failures. The live Kaggle quota check then reported 13.25 GPU
hours remaining, below the 14-hour public-seed reserve, so no public kernel was
launched and no submission was attempted. The competition account policy has
since been clarified: the hidden gateway rerun is quota-neutral, so the
Poetiq lane now reserves 4.5 hours for its two public kernels and does not
reserve GPU hours for the later private submission.

## What the Poetiq reports imply

Poetiq describes the prompt as an interface rather than the intelligence: the
system iterates between a proposed solution, tool/environment feedback, and a
revised solution. It also emphasizes self-auditing, adaptive information
acquisition, selective tool/code use, hierarchical decomposition, and
model-agnostic orchestration. These are the mechanisms relevant to a stateful
game, where the expensive failure is not a bad first thought but continuing a
wrong plan after the board has falsified it.

## Top five recommendations

### 1. Event-triggered self-audit (implemented: `duck-audit-v1`)

Add a short audit instruction only after generic no-progress evidence: repeated
identical actions or unchanged gameplay frames. The host should not inject an
extra turn on every step, reset the game, or change the Stock Duck tool surface.
The audit must choose one of continue, inspect, or replan and then return to the
normal loop.

Why first: it is the smallest change that tests Poetiq's self-monitoring claim,
with a clear trigger count and no prompt/history/model confound. The local
smoke passed the interface criterion on `ft09`; the RTX public result is the
active run and will determine whether the idea survives.

Promotion signal: no infrastructure failures, no increase in context
evictions, and a public mean improvement over the Stock Duck seed-0 control.
If the score is neutral or worse, retire the sidecar rather than increasing
its trigger frequency.

### 2. Active information acquisition

When the model is uncertain or the last action produces no new information,
ask for the smallest discriminating inspection (for example, one coordinate,
one object relation, or one candidate action outcome) before asking it to
restate the entire plan. The host should log the question, the predicted
information gain, and whether the next frame resolved the uncertainty.

This is the preferred next candidate if sparse auditing does not improve the
score: it tests Poetiq's claim that the key problem is discovering what to ask
next, not merely producing a longer chain of thought. It should be an
event-triggered addendum, not a permanent verbose prompt. The isolated
`duck-information-v1` implementation now passes unit/config/notebook checks
and its one-game `ft09` local smoke (2026-08-02 06:28:06 AEST) completed with
no infrastructure failures; its 0.0 local score is not an RTX proxy. It is
intentionally not published until the active audit comparison is final.

### 3. Verified one-step hypotheses

Have the model optionally state one falsifiable prediction alongside one action
(expected changed region, object movement, or level transition). Execute the
action, compare the observed frame with the prediction, and feed the mismatch
back into the next normal turn. Do not auto-correct the action or invent a
prediction when the model omitted one; omission must remain visible telemetry.

The earlier contract-repair public result is not evidence against this idea:
that kernel ran before the wire-preserving fix and repaired nearly every
proposal. A corrected transport smoke must precede any public ablation.
That smoke is now recorded (`duck-contract-repair-local-cn04-wirefix.json`),
and the public-only pipeline path is implemented; it remains queued until the
active audit kernel has completed.

### 4. Hierarchical candidate search with a cheap verifier

Separate a game into a small hierarchy: identify the current level objective,
generate two or three candidate mechanics, test each with one low-risk action,
then commit to the best-supported plan. A verifier scores candidates from
observed transitions, not from the model's confidence. Keep the branch budget
bounded and abort a branch immediately after a contradiction.

This maps to Poetiq's hierarchical harnesses and programmatic solution
selection. It is intentionally different from a long retained-memory summary:
the benefit comes from comparing alternatives against evidence.

Implementation status: `duck-hierarchy-v1` is now isolated in its own mode. It
adds a bounded (maximum three) candidate-ranking prompt only at a new-level or
unchanged-frame trigger, keeps Stock Duck's tool/history/action surface, and
records trigger telemetry. Its one-game `ft09` local smoke completed on
2026-08-02 07:06:05 AEST with no infrastructure failures. The local model gave
up after one action, so the smoke is an interface pass rather than a score
proxy; public publication remains queued behind the active audit result.

### 5. Controlled diversity and budget allocation

Use diversity only where it buys information: vary one proposal/model sampling
path for a hard level, then let a common verifier select the action. Allocate
more requests to games with unresolved hypotheses and stop early on solved or
stalled games. Do not run an always-on ensemble, which would reduce the
28-session throughput and confound attribution.

This follows Poetiq's model-agnostic and multi-model observations while
respecting Kaggle's fixed GPU budget. The first implementation should use the
same Qwen model with two sampling paths; a different model is a later
experiment, not part of the first ablation.

Implementation status: `duck-diversity-v1` is now isolated. After two
unchanged gameplay transitions it uses one alternate request seed (`primary +
17`) for the next analysis turn, then restores Stock Duck's primary seed. The
one-game `ft09` smoke completed at 2026-08-02 07:23:25 AEST, exercised one
alternate-seed trigger/use, and had no infrastructure failures. Its 0.0 local
score is not a capability proxy; public publication remains queued behind the
active audit comparison.

## Experiment order

1. Run the composite `duck-poetiq-v1` local smoke, 25-game local integration,
   and 110-clone HTTP rehearsal. **Completed:** the integration and rehearsal
   artifacts pass their structural/infrastructure checks.
2. When the account reports at least 4.5 GPU-hours remaining, run independent
   Kaggle public seeds 0 and 1 with exact source/config
   hashes. Require mean ≥ **1.4082**, seed floors 18/10 completed levels and
   15/9 nonzero games, trimmed mean > **0.6061**, and zero infrastructure
   failures.
3. If the gate passes and quota remains, submit the exact validated hidden
   kernel once; record the private result and stop the initial candidate after
   a visible score above **0.80**.
4. If it fails, keep the composite isolated and change one variable per
   iteration in this order: stall thresholds, intervention cooldown,
   candidate/verifier wording, diversity offset, then stalled-yield thresholds.
5. Preserve the five individual lanes as diagnostic ablations, but do not
   stack their prompts into the composite or silently alter `duck-reference`.

## Portfolio follow-on

1. Run targeted route/contract/switch tests, the local 25-game MLX integration,
   and the scripted 110-game gateway rehearsal.
2. Require the offline leave-one-game-out lift, breadth, and multi-policy gate
   before publishing seed 0.
3. Require seed 0 score ≥ 2.5631, 18 levels, 15 nonzero games, and trimmed mean
   > 0.9370812 before spending quota on seed 1.
4. Submit only if seed 1 also reaches 10 levels/9 nonzero games, the two-seed
   mean is ≥ 1.4081755, trimmed mean > 0.6060513, and all hashes/telemetry agree.
5. If it fails, vary one item at a time: warm-up length, Stock margin,
   uncertainty penalty, switch threshold, then candidate set.

Every result belongs in `SCORECARD.md` with Brisbane local artifact time (AEST), kernel version,
source/config hashes, score, completed levels, request failures, context
evictions, and the reason for promotion or retirement.

## Sources

- [Traversing the Frontier of Superintelligence](https://poetiq.ai/posts/arcagi_announcement/)
- [ARC-AGI-2 SOTA at Half the Cost](https://poetiq.ai/posts/arcagi_verified/)
- [Raising the Bar on HLE and SimpleQA](https://poetiq.ai/posts/raising_the_bar_hle_simpleqa/)
- [Benchmarks Are Dead (for us)](https://poetiq.ai/posts/benchmarks_are_dead/)
