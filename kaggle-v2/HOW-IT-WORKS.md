# How ouro2 Works — in Plain English

This is the ARC-AGI-3 Kaggle agent (`kaggle-v2`, codename **ouro2**). It plays
video games it has *never seen before* and figures out the rules on its own,
just by pressing buttons and watching what changes on screen.

This document explains, in simple language, how it does that — and how three
big ideas fit together inside it: a **World Model**, **Symbolic AI**, and the
connection to **AGI** (Artificial General Intelligence).

> The companion picture is [architecture.drawio](architecture.drawio). This
> doc is the words version of that diagram.

---

## 1. The problem it's solving

ARC-AGI-3 hands the agent a series of tiny games. Each game is a 64×64 grid of
colored squares. The agent can do a handful of things:

- Press one of a few buttons (actions 1–5 and 7)
- Click on a square (action 6)

That's it. The agent is **never told the rules**. It doesn't know which color is
the player, what the goal is, what kills it, or what the buttons do. It has to
work all of that out by *playing* — and it only gets a limited number of moves,
so it can't just try everything randomly.

A human is great at this. You watch the screen for a few seconds, notice "oh,
the green blob moves when I press up," and you're off. The whole point of
ARC-AGI-3 is to see whether a machine can do that same kind of **fast, from-
scratch learning**. That is what makes it an *AGI* benchmark rather than an
ordinary game AI.

---

## 2. The one big idea

Most game-playing AI either (a) is trained for millions of games until it
memorizes one specific game, or (b) asks a giant language model "what should I
do?" on every single move. ouro2 does **neither**.

Instead it works the way a curious scientist does:

> **Watch what happens → guess the rules → test the guess → use the rules that
> survive testing to plan ahead → act → repeat.**

The clever part is that once the agent has *guessed the rules*, it can **plan
inside its own head** — trying out moves on an imagined copy of the game —
before spending any of its precious real moves. That imagined copy of the game
is the **World Model**. Building it out of explicit, checkable rules is
**Symbolic AI**. Doing all of this on a brand-new game with no training is the
**AGI** ambition.

Let's take those three ideas one at a time.

---

## 3. The World Model — a copy of the game in the agent's head

A **World Model** is exactly what it sounds like: the agent's internal *model*
of how the world (the game) behaves. If you give the model the current screen
and a button press, it predicts the *next* screen — without touching the real
game.

Why is this so valuable? Because a real move is expensive and can't be taken
back. If the agent has a good-enough model, it can *simulate* thousands of
possible move sequences privately, find one that reaches the goal, and only then
play those moves for real. It's the difference between rehearsing a chess line
in your mind versus moving the pieces and hoping.

In ouro2 the World Model is literally a small set of **rules**, each of which is
a tiny predictor:

| Rule type   | What it predicts (in plain words)                                  |
|-------------|--------------------------------------------------------------------|
| `MoveRule`  | "When you press *up*, the player square moves one cell up (and pushes/collects/dies on contact with X)." |
| `ClickRule` | "When you click a square of color X, it turns into color Y."        |
| `TickRule`  | "Every turn, thing X drifts by one cell on its own."               |
| `HazardRule`| "Touching color X ends the game."                                  |
| `Goal`      | "You win when *this* condition is true on the screen."             |

Put those rules together and you have a working miniature of the game. The code
that runs them is a small, fixed **interpreter** — its own source file even
opens with the line *"The world model: rules as data, run by a fixed
interpreter"* ([ouro2/rules.py](ouro2/rules.py)).

Two things make this World Model trustworthy:

1. **It's always checked against reality.** Every rule is *backtested* by
   replaying the entire history of what actually happened and seeing whether the
   rule would have predicted it. A rule that mispredicts even once is thrown out
   or narrowed. (See [ouro2/induce.py](ouro2/induce.py).)
2. **Reality always wins.** When the agent executes a planned move and the real
   screen doesn't match what the model predicted, it stops trusting the plan and
   feeds the surprise back in as a *counterexample* to fix the rules. The
   executor's rule is literally "reality outranks the model."

So the World Model is never blindly trusted — it's a hypothesis that is
constantly re-earned.

---

## 4. Symbolic AI — rules you can read, not weights you can't

There are, broadly, two families of AI:

- **Neural / "connectionist" AI** (like large language models): knowledge is
  spread across billions of numeric weights. It's powerful but a black box — you
  can't point at where "the player moves up" is stored, and you can't easily
  guarantee it.
- **Symbolic AI**: knowledge is stored as explicit **symbols and rules** you can
  read, print, and check — closer to logic or algebra than to a neural net.

ouro2 is, at its heart, **Symbolic AI**. Everything important is a rule written
in a small made-up language (a *DSL*, domain-specific language) of `move`,
`click_effect`, `tick_move`, `hazard`, and `goal`. Those rules are **data** —
just structured facts — and a single fixed interpreter runs them. Crucially:

- **A rule is either right or wrong, and you can prove which.** "Pressing up
  moves the player up" is a claim you can test against every frame in history.
- **The knowledge is legible.** You can dump the current rule set and read, in
  plain terms, exactly what the agent believes about the game.
- **It's deterministic and needs no training.** The winning submission runs on a
  **CPU with zero AI-model calls** — no GPU, no neural network in the loop
  ([architecture.drawio](architecture.drawio), CPU kernel = "the product").

The rules aren't hand-written by a person, and they aren't written by a language
model either. The agent **induces** them itself — *induction* meaning "infer the
general rule from specific examples." It watches a single before/after screen
pair, notices "every green cell moved exactly one step right," and proposes the
matching rule. Then backtesting decides whether that guess survives.

This is the old dream of Symbolic AI — *learning explicit, checkable rules from
observation* — but applied to messy pixels in real time.

### Where the neural net does show up (and why it's optional)

There *is* an optional small language model, the **Oracle** (a 4-billion-
parameter Qwen model). But look at how tightly it's boxed in:

- It **never writes rules**. It only picks one answer from a
  multiple-choice list the symbolic code already generated — e.g. "which of
  these three is more likely the goal?"
- It runs at temperature 0 (no randomness) and must answer in strict JSON.
- If it's slow, missing, or gives a bad answer, the system **falls back to the
  CPU's default and keeps going**. It is a *tie-breaker, not a dependency*
  ([ouro2/oracle.py](ouro2/oracle.py)).

So the neural net is a small advisor bolted onto a symbolic core — not the
brain. The brain is the rules.

---

## 5. How a single turn actually flows

Here's the loop the agent runs on every turn. The components in **bold** are the
boxes in the architecture diagram.

1. **Game environment** sends the current 64×64 frame.
2. **Director** records what the *last* action actually did into the
   **Timeline** — an append-only, never-edited log that is the *ground truth*.
3. Every so often (roughly every 16 recorded steps), **Induction** re-reads the
   Timeline and re-derives the rules: which color is the player, what each
   button does, what's a hazard, what the goal is.
4. Those rule guesses are **backtested** against the whole Timeline. Only rules
   that fit the evidence get **certified** into the **Model** (the World Model).
5. If the Model is healthy, the **Planner** searches *inside* it — a
   breadth-first search that imagines move sequences until it finds a path to
   the goal, spending zero real moves.
6. The **Executor** plays that plan one move at a time, checking after each move
   that reality matches the prediction. If it doesn't, it **aborts** and sends
   the mismatch back to Induction as a counterexample.
7. If there's no trustworthy plan yet, the **Explorer floor** takes over: try
   untried buttons, then promising clicks, then rotate through options — a
   principled way to gather new evidence (and it remembers which actions did
   nothing, so it never wastes moves repeating them).
8. Repeat until the game is won or the move budget runs out.

The short version: **explore to gather evidence → induce rules from the
evidence → plan inside the rules → execute carefully → correct the rules when
surprised.** That closing-the-loop-on-itself quality is why the project is
codenamed *ouroboros* (the snake eating its own tail).

---

## 6. The AGI connection — why this is more than a game bot

**AGI** — Artificial General Intelligence — means an AI that can handle *new*
problems it wasn't specifically built for, the way a person can. The ARC family
of benchmarks was designed by François Chollet to measure exactly this:
**skill-acquisition efficiency**, i.e. how quickly a system learns a *brand-new*
skill from very few examples, rather than how much it has memorized.

A narrow game AI that's been trained on a million rounds of *one* game is
impressive but not general — move it to a new game and it's helpless. ouro2 is
built to be the opposite:

- **No pre-training on the games.** It walks in blind and learns each game's
  mechanics live, from a handful of frames.
- **It builds understanding, not reflexes.** The output isn't "press up now" —
  it's an explicit, inspectable theory of how the game works, which it then
  reasons over.
- **It's actively defended against cheating.** A whole third of the system
  (Lane C in the diagram) exists to stop the agent from secretly memorizing
  specific games. It keeps a hidden **holdout** set of games it's never allowed
  to tune against, forbids hard-coded game IDs or coordinates in the code, and
  only lets a change ship if it improves scores on games it hasn't seen. This
  *anti-overfit discipline* is the practical test of whether the intelligence is
  general or just memorized.

That combination — **learn the rules of an unseen world from scratch, form an
explicit model of it, and plan using that model** — is a small, concrete step in
the direction of general intelligence. ouro2 doesn't claim to *be* AGI. It's a
demonstration of one ingredient of it: **turning raw observation into
understanding, fast, without being told the answer.**

---

## 7. Glossary

- **World Model** — the agent's internal, runnable copy of the game, used to
  predict outcomes and plan without spending real moves. Here it's a set of
  symbolic rules.
- **Symbolic AI** — AI whose knowledge is explicit, human-readable rules and
  symbols (as opposed to opaque neural-network weights).
- **Induction** — inferring general rules from specific observed examples.
- **Backtest** — replaying the full history to check whether a candidate rule
  would have correctly predicted what actually happened.
- **DSL** — a tiny domain-specific language; here, the vocabulary of rule types
  (`move`, `click_effect`, `tick_move`, `hazard`, `goal`).
- **Oracle** — the optional 4B neural model used only as a multiple-choice
  tie-breaker; the product runs fine without it.
- **Holdout / anti-overfit** — games deliberately hidden from tuning, used to
  prove the agent generalizes instead of memorizing.
- **AGI** — Artificial General Intelligence: an AI that can learn and solve
  problems it wasn't specifically built for.

---

## 8. Where to look in the code

| Concept          | File                                             |
|------------------|--------------------------------------------------|
| World Model + interpreter | [ouro2/rules.py](ouro2/rules.py)        |
| Rule induction (learning) | [ouro2/induce.py](ouro2/induce.py)      |
| Ground-truth history      | [ouro2/timeline.py](ouro2/timeline.py)  |
| Planning inside the model | [ouro2/plan.py](ouro2/plan.py)          |
| Per-turn orchestration    | [ouro2/director.py](ouro2/director.py)  |
| Fallback exploration      | [ouro2/explore.py](ouro2/explore.py)    |
| Optional neural tie-break | [ouro2/oracle.py](ouro2/oracle.py)      |
| The big picture           | [architecture.drawio](architecture.drawio) |
