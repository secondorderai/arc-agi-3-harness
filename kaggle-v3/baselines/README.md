# Frozen reference

`duck-public-oracle.json` is a compact, SHA-attributed diagnostic projection
of Tufa Labs' published 20-pass `score.json`. It is never treated as an exact
per-game gate because that run used different hardware and timing.

`reference.json` is written only after the five independent seed 0–4 RTX
kernel artifacts pass the engine-score gate. It retains the exact kernel,
config, prompt, runtime, and metrics hashes plus per-seed/per-game comparison
data while omitting large action traces.

Do not overwrite it during DEV screening. A deliberate reference replacement
must pass the same public mean, infrastructure, and deadline gates.
