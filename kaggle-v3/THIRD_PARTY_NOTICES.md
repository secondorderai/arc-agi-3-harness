# Third-party notices

## Duck harness / ARC3 Inference / TAAF

Source under `src/inference/` and `src/taaf/` is adapted from the public
Duck/TAAF ARC-AGI-3 harness published by Tufa Labs under the MIT license.
`src/duck_reference/` is the isolated selector for the stock behavior.
`src/duck_robust/` adds a selective recovery controller and uses the published
500-run `example-run/benchmark.json` as an attributed, SHA-256-pinned training
source for its diagnostic probability signal.

- Repository: <https://github.com/Tufalabs/duck-harness>
- Audited public source commit:
  `dd50b552a3b83af6339d1ee0c95f04f5af05ff07`
- Public source dataset: `jeroencottaar/taaf-kaggle-source-share`
- Referenced notebook:
  <https://www.kaggle.com/code/kaiwalyaatulraut/arc-agi-3-solution>
- Technical write-up:
  <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133>

The public milestone notebook credits the Tufa Labs team, alphabetically, as
Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar,
and Stefano Viel.

The original source is redistributed and modified under the MIT license
included in `LICENSE`. Material v3 modifications include richer frame state,
explicit evidence/contradiction memory, prediction verification, strategic
reset gating, deadline-aware budgets, deterministic model-failure recovery,
manifested source packaging, and generated Kaggle notebooks.

The broad hybrid additions are enabled only by `ouro-hybrid`. `duck-reference`
retains Duck’s prompt/tool contract and disables the evidence ledger,
perception additions, prediction checks, deterministic fallback, strategic
reset, and adaptive game budgets.

`duck-robust` remains on that same reference contract until its recovery gate
fires; richer perception and prediction checks are scoped to the recovery
fork only.

Opaque pickle artifacts from the public Kaggle dataset are deliberately not
included.
