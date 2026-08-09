"""Typed, retrodictive world models for sparse interactive grid evidence.

The engine deliberately owns its evidence and executable hypotheses.  A model
may propose actions or explanations, but only rules represented here can
authorize automatic execution.  Every candidate is replayed against the full
per-game transition log after each observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ouro3.perception import normalize_grid, segment_grid, segment_grid_multi

Grid = tuple[tuple[int, ...], ...]

_MOUSE_PATTERN = re.compile(
    r"(?:MOUSE|ACTION6)\s*\(\s*(?:row\s*=\s*)?(-?\d+)\s*,\s*"
    r"(?:col\s*=\s*)?(-?\d+)\s*\)",
    re.IGNORECASE,
)
_MODEL_ACTIONS = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION6": "MOUSE",
}


def grid_hash(grid: Sequence[Sequence[int]]) -> str:
    normalized = normalize_grid(grid)
    return hashlib.blake2b(
        repr(normalized).encode("ascii"), digest_size=16
    ).hexdigest()


def action_key(action: Mapping[str, Any]) -> str:
    name = str(action.get("action", "")).strip().upper()
    if name == "MOUSE":
        return f"MOUSE:{int(action.get('row', -1))}:{int(action.get('col', -1))}"
    return name


def normalize_action(
    raw: str | Mapping[str, Any], payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Normalize solver displays and model-facing actions to one schema."""

    if isinstance(raw, Mapping):
        value = dict(raw)
        name = str(value.get("action", "")).strip().upper()
        name = _MODEL_ACTIONS.get(name, name)
        result: dict[str, Any] = {"action": name}
        if name == "MOUSE":
            result["row"] = int(value.get("row", -1))
            result["col"] = int(value.get("col", -1))
        return result

    details = dict(payload or {})
    engine_name = str(details.get("action_name", "")).strip().upper()
    display = str(raw or "").strip().upper()
    name = _MODEL_ACTIONS.get(engine_name, _MODEL_ACTIONS.get(display, display))
    data = details.get("action_data")
    if name == "MOUSE" and isinstance(data, Mapping):
        return {
            "action": "MOUSE",
            "row": int(data.get("row", data.get("y", -1))),
            "col": int(data.get("col", data.get("x", -1))),
        }
    mouse = _MOUSE_PATTERN.search(display)
    if mouse:
        return {
            "action": "MOUSE",
            "row": int(mouse.group(1)),
            "col": int(mouse.group(2)),
        }
    return {"action": name}


@dataclass(frozen=True)
class TransitionEvidence:
    index: int
    level: int
    action: dict[str, Any]
    before: Grid
    after: Grid
    level_completed: bool = False
    game_over: bool = False
    run_complete: bool = False

    @property
    def before_hash(self) -> str:
        return grid_hash(self.before)

    @property
    def after_hash(self) -> str:
        return grid_hash(self.after)

    @property
    def action_key(self) -> str:
        return action_key(self.action)


@dataclass
class TypedRule:
    """A bounded executable rule plus evidence obtained by full-log replay."""

    kind: str
    action: str
    parameters: dict[str, Any]
    ontology: str = "pixel"
    support: tuple[int, ...] = ()
    contradictions: tuple[int, ...] = ()
    applicable: tuple[int, ...] = ()
    goal_support: int = 0
    source_index: int = -1

    @property
    def rule_id(self) -> str:
        payload = json.dumps(
            {
                "kind": self.kind,
                "action": self.action,
                "parameters": self.parameters,
                "ontology": self.ontology,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def certified(self) -> bool:
        minimum_support = 3 if self.kind in {"noop", "object-translation"} else 2
        return len(self.support) >= minimum_support and not self.contradictions

    @property
    def confidence(self) -> float:
        return (len(self.support) + 1.0) / (
            len(self.support) + len(self.contradictions) + 2.0
        )

    @property
    def complexity(self) -> int:
        return 1 + len(self.parameters) + int(self.ontology != "pixel")

    @property
    def posterior_weight(self) -> float:
        evidence = min(1.0, len(self.support) / 2.0)
        return self.confidence * evidence / self.complexity

    def predicts_goal(self) -> float:
        return self.goal_support / max(1, len(self.support))


@dataclass(frozen=True)
class RulePrediction:
    grid: Grid
    confidence: float
    rule_ids: tuple[str, ...]
    certified: bool
    goal_probability: float = 0.0


@dataclass(frozen=True)
class Plan:
    actions: tuple[dict[str, Any], ...]
    state_hashes: tuple[str, ...]
    confidence: float
    source: str
    expanded: int


@dataclass(frozen=True)
class ProbeRecommendation:
    action: dict[str, Any]
    information_gain: float
    risk: float
    novelty: float
    predicted_outcomes: int


@dataclass
class CloneStateGraph:
    """A bounded history-conditioned graph for diagnosed perceptual aliasing.

    It is intentionally dormant until the same observation/action pair has
    incompatible successors.  Clones are keyed by the immediately preceding
    observation and action, a conservative finite-context approximation to a
    full clone-structured cognitive graph.
    """

    transitions: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    clone_transitions: dict[tuple[str, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    conflicts: set[tuple[str, str]] = field(default_factory=set)
    previous_observation: str | None = None
    previous_action: str | None = None

    def observe(self, record: TransitionEvidence) -> None:
        scoped_before = f"{record.level}:{record.before_hash}"
        scoped_after = f"{record.level}:{record.after_hash}"
        base = (scoped_before, record.action_key)
        self.transitions[base][record.after_hash] += 1
        if len(self.transitions[base]) > 1:
            self.conflicts.add(base)
        clone = self.clone_key(
            scoped_before,
            previous_observation=self.previous_observation,
            previous_action=self.previous_action,
        )
        self.clone_transitions[(clone, record.action_key)][record.after_hash] += 1
        self.previous_observation = scoped_after
        self.previous_action = record.action_key

    def clone_key(
        self,
        observation: str,
        *,
        previous_observation: str | None,
        previous_action: str | None,
    ) -> str:
        if not any(key[0] == observation for key in self.conflicts):
            return observation
        context = f"{previous_observation or '-'}:{previous_action or '-'}"
        suffix = hashlib.sha1(context.encode("utf-8")).hexdigest()[:8]
        return f"{observation}@{suffix}"

    @property
    def active(self) -> bool:
        return bool(self.conflicts)


class RetrodictiveWorldModel:
    """Persistent typed version space with replay, probes and safe planning."""

    def __init__(
        self,
        *,
        max_rules: int = 256,
        prediction_threshold: float = 0.90,
    ) -> None:
        self.max_rules = max(8, int(max_rules))
        self.prediction_threshold = max(0.5, min(1.0, prediction_threshold))
        self.timeline: list[TransitionEvidence] = []
        self.rules: dict[str, TypedRule] = {}
        self.exact_edges: dict[
            tuple[int, str, str], list[tuple[str, bool, dict[str, Any]]]
        ] = defaultdict(list)
        self.exact_counts: dict[tuple[int, str, str], Counter[str]] = defaultdict(
            Counter
        )
        self.states: dict[str, Grid] = {}
        self._objects_by_state: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self.prediction_attempts = 0
        self.prediction_matches = 0
        self.prediction_mismatches = 0
        self.last_prediction_mismatch = ""
        self.plan_requests = 0
        self.plan_successes = 0
        self.plan_aborts = 0
        self.probe_requests = 0
        self.clone_graph = CloneStateGraph()

    def observe(
        self,
        *,
        level: int,
        action: str | Mapping[str, Any],
        before: Sequence[Sequence[int]],
        after: Sequence[Sequence[int]],
        payload: Mapping[str, Any] | None = None,
    ) -> TransitionEvidence:
        details = dict(payload or {})
        normalized_action = normalize_action(action, details)
        prior_prediction = self.predict(
            before,
            normalized_action,
            level=level,
        )
        after_grid = normalize_grid(after)
        if prior_prediction is not None:
            self.prediction_attempts += 1
            if prior_prediction.grid == after_grid:
                self.prediction_matches += 1
                self.last_prediction_mismatch = ""
            else:
                self.prediction_mismatches += 1
                self.plan_aborts += 1
                self.last_prediction_mismatch = (
                    f"predicted {grid_hash(prior_prediction.grid)} but observed "
                    f"{grid_hash(after_grid)}"
                )
        before_grid = normalize_grid(before)
        before_hash = grid_hash(before_grid)
        after_hash = grid_hash(after_grid)
        canonical_before = self.states.setdefault(before_hash, before_grid)
        canonical_after = self.states.setdefault(after_hash, after_grid)
        record = TransitionEvidence(
            index=len(self.timeline),
            level=int(level),
            action=normalized_action,
            before=canonical_before,
            after=canonical_after,
            level_completed=bool(details.get("level_completed")),
            game_over=bool(details.get("game_over")),
            run_complete=bool(details.get("run_complete")),
        )
        self.timeline.append(record)
        edge = (record.after_hash, record.level_completed or record.run_complete, record.action)
        exact_key = (record.level, record.before_hash, record.action_key)
        self.exact_counts[exact_key][record.after_hash] += 1
        if edge not in self.exact_edges[exact_key]:
            self.exact_edges[exact_key].append(edge)
        self.clone_graph.observe(record)
        object_cache = self._objects_by_state.setdefault(record.before_hash, {})
        for rule in self.rules.values():
            self._extend_rule(rule, record, object_cache=object_cache)
        for rule in self._induce(record):
            if rule.rule_id in self.rules:
                continue
            self._replay_rule(rule)
            self.rules[rule.rule_id] = rule
        self._prune_rules()
        return record

    def _induce(self, record: TransitionEvidence) -> list[TypedRule]:
        rules: list[TypedRule] = []
        name = str(record.action.get("action", ""))
        if record.before == record.after:
            rules.append(
                TypedRule(
                    "noop",
                    name,
                    {"level": record.level},
                    source_index=record.index,
                )
            )

            if name == "MOUSE":
                click_noop = _infer_click_noop(record)
                if click_noop is not None:
                    rules.append(click_noop)

            for prior in self.rules.values():
                if prior.kind != "object-translation" or prior.action != name:
                    continue
                parameters = {
                    key: value
                    for key, value in prior.parameters.items()
                    if key != "border_steps"
                }
                regular = _apply_object_translation(record.before, parameters)
                blocked = _apply_object_translation(
                    record.before,
                    parameters,
                    blocked_as_noop=True,
                )
                if regular is None and blocked == record.before:
                    rules.append(
                        TypedRule(
                            "object-blocked-noop",
                            name,
                            parameters,
                            ontology=prior.ontology,
                            source_index=record.index,
                        )
                    )

        color_map = _infer_color_map(record.before, record.after)
        if color_map:
            rules.append(
                TypedRule(
                    "color-map",
                    name,
                    {
                        "level": record.level,
                        "mapping": [[old, new] for old, new in color_map],
                    },
                    source_index=record.index,
                )
            )

        border_steps = _infer_border_steps(record.before, record.after)
        if border_steps:
            rules.append(
                TypedRule(
                    "border-steps",
                    name,
                    {"level": record.level, "steps": border_steps},
                    source_index=record.index,
                )
            )

        for background in _background_candidates(record.before):
            shift = _infer_rigid_translation(record.before, record.after, background)
            if shift is not None:
                rules.append(
                    TypedRule(
                        "rigid-translation",
                        name,
                        {
                            "level": record.level,
                            "background": background,
                            "delta": list(shift),
                        },
                        source_index=record.index,
                    )
                )

        rules.extend(_infer_object_translations(record))

        if name == "MOUSE":
            row = int(record.action.get("row", -1))
            col = int(record.action.get("col", -1))
            for ontology, objects in segment_grid_multi(record.before).items():
                if ontology == "color-4-all":
                    continue
                rule = _infer_click_recolor(
                    record, row=row, col=col, ontology=ontology, objects=objects
                )
                if rule is not None:
                    rules.append(rule)
        return rules

    def _replay_rule(self, rule: TypedRule) -> None:
        rule.applicable = ()
        rule.support = ()
        rule.contradictions = ()
        rule.goal_support = 0
        for record in self.timeline:
            self._extend_rule(
                rule,
                record,
                object_cache=self._objects_by_state.setdefault(
                    record.before_hash,
                    {},
                ),
            )

    @staticmethod
    def _extend_rule(
        rule: TypedRule,
        record: TransitionEvidence,
        *,
        object_cache: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        predicted = apply_rule(
            rule,
            record.before,
            record.action,
            _level=record.level,
            _object_cache=object_cache,
        )
        if predicted is None:
            return
        rule.applicable = (*rule.applicable, record.index)
        if predicted == record.after:
            rule.support = (*rule.support, record.index)
            if record.level_completed or record.run_complete:
                rule.goal_support += 1
        else:
            rule.contradictions = (*rule.contradictions, record.index)

    def _prune_rules(self) -> None:
        if len(self.rules) <= self.max_rules:
            return
        ranked = sorted(
            self.rules.values(),
            key=lambda rule: (
                rule.certified,
                rule.posterior_weight,
                len(rule.support),
                -len(rule.contradictions),
                -rule.complexity,
                rule.rule_id,
            ),
            reverse=True,
        )[: self.max_rules]
        self.rules = {rule.rule_id: rule for rule in ranked}

    def predict(
        self,
        grid: Sequence[Sequence[int]],
        action: str | Mapping[str, Any],
        *,
        level: int | None = None,
        include_uncertified: bool = False,
    ) -> RulePrediction | None:
        state = normalize_grid(grid)
        normalized_action = normalize_action(action)
        state_hash = grid_hash(state)
        normalized_key = action_key(normalized_action)
        exact = (
            self.exact_edges.get((int(level), state_hash, normalized_key), [])
            if level is not None
            else [
                edge
                for (edge_level, before_hash, edge_action), edges in self.exact_edges.items()
                if before_hash == state_hash and edge_action == normalized_key
                for edge in edges
            ]
        )
        exact_outcomes = {item[0] for item in exact}
        exact_support = (
            self.exact_counts.get((int(level), state_hash, normalized_key), Counter())
            if level is not None
            else Counter(
                {
                    outcome: sum(
                        counts[outcome]
                        for (edge_level, before_hash, edge_action), counts
                        in self.exact_counts.items()
                        if before_hash == state_hash and edge_action == normalized_key
                    )
                    for outcome in exact_outcomes
                }
            )
        )
        if (
            len(exact_outcomes) == 1
            and sum(exact_support.values()) >= 2
        ):
            after_hash, goal, _ = exact[0]
            return RulePrediction(
                grid=self.states[after_hash],
                confidence=1.0,
                rule_ids=(
                    f"exact:{level if level is not None else '*'}:"
                    f"{state_hash}:{normalized_key}",
                ),
                certified=True,
                goal_probability=float(any(item[1] for item in exact)),
            )

        votes: dict[str, dict[str, Any]] = {}
        object_cache: dict[int, list[dict[str, Any]]] = {}
        for rule in self.rules.values():
            if not include_uncertified and not rule.certified:
                continue
            predicted = apply_rule(
                rule,
                state,
                normalized_action,
                _level=level,
                _object_cache=object_cache,
            )
            if predicted is None:
                continue
            digest = grid_hash(predicted)
            bucket = votes.setdefault(
                digest,
                {"grid": predicted, "weight": 0.0, "rules": [], "goal": 0.0},
            )
            weight = max(1e-9, rule.posterior_weight)
            bucket["weight"] += weight
            bucket["rules"].append(rule.rule_id)
            bucket["goal"] += weight * self._rule_goal_probability(
                rule,
                level=level,
            )
        if not votes:
            return None
        total = sum(float(bucket["weight"]) for bucket in votes.values())
        winner = max(votes.values(), key=lambda bucket: float(bucket["weight"]))
        confidence = float(winner["weight"]) / max(1e-9, total)
        if not include_uncertified and confidence < self.prediction_threshold:
            return None
        return RulePrediction(
            grid=winner["grid"],
            confidence=confidence,
            rule_ids=tuple(sorted(winner["rules"])),
            certified=all(self.rules[rule_id].certified for rule_id in winner["rules"]),
            goal_probability=float(winner["goal"]) / max(
                1e-9, float(winner["weight"])
            ),
        )

    def plan(
        self,
        grid: Sequence[Sequence[int]],
        *,
        level: int | None = None,
        valid_actions: Sequence[str] = (),
        max_depth: int = 32,
        max_expanded: int = 5_000,
    ) -> Plan | None:
        """Find a path using only exact or fully replay-certified transitions."""

        self.plan_requests += 1
        start = grid_hash(grid)
        self.states.setdefault(start, normalize_grid(grid))
        queue: deque[
            tuple[str, tuple[dict[str, Any], ...], tuple[str, ...], bool]
        ] = deque(
            [(start, (), (start,), False)]
        )
        visited = {start}
        expanded = 0
        by_state: dict[
            str, list[tuple[str, str, bool, dict[str, Any], bool]]
        ] = defaultdict(list)
        for (
            edge_level,
            before_hash,
            edge_action,
        ), edges in self.exact_edges.items():
            if level is not None and edge_level != int(level):
                continue
            if len({edge[0] for edge in edges}) != 1:
                continue
            after_hash, goal, action = edges[0]
            support = self.exact_counts[
                (edge_level, before_hash, edge_action)
            ][after_hash]
            if support < 2:
                continue
            by_state[before_hash].append(
                (edge_action, after_hash, goal, action, True)
            )
        while queue and expanded < max_expanded:
            current, actions, hashes, used_typed = queue.popleft()
            expanded += 1
            if len(actions) >= max_depth:
                continue
            transitions = list(by_state.get(current, []))
            state = self.states.get(current)
            if state is not None and valid_actions:
                exact_action_keys = {item[0] for item in transitions}
                for candidate in _candidate_actions(state, valid_actions):
                    if action_key(candidate) in exact_action_keys:
                        continue
                    prediction = self.predict(
                        state,
                        candidate,
                        level=level,
                    )
                    if prediction is None or not prediction.certified:
                        continue
                    successor = grid_hash(prediction.grid)
                    self.states.setdefault(successor, prediction.grid)
                    transitions.append(
                        (
                            action_key(candidate),
                            successor,
                            prediction.goal_probability >= 0.5,
                            candidate,
                            False,
                        )
                    )
            for _edge_action, successor, goal, action, is_exact in sorted(
                transitions, key=lambda item: item[0]
            ):
                next_actions = (*actions, dict(action))
                next_hashes = (*hashes, successor)
                next_used_typed = used_typed or not is_exact
                if goal:
                    self.plan_successes += 1
                    return Plan(
                        actions=next_actions,
                        state_hashes=next_hashes,
                        confidence=(
                            self.prediction_threshold if next_used_typed else 1.0
                        ),
                        source=(
                            "typed-certified-search"
                            if next_used_typed
                            else "exact-replay"
                        ),
                        expanded=expanded,
                    )
                if successor not in visited:
                    visited.add(successor)
                    queue.append(
                        (successor, next_actions, next_hashes, next_used_typed)
                    )
        return None

    def select_probe(
        self,
        grid: Sequence[Sequence[int]],
        valid_actions: Sequence[str],
        *,
        level: int | None = None,
    ) -> ProbeRecommendation | None:
        self.probe_requests += 1
        candidates = _candidate_actions(grid, valid_actions)
        if not candidates:
            return None
        ranked: list[ProbeRecommendation] = []
        object_cache: dict[int, list[dict[str, Any]]] = {}
        for candidate in candidates:
            outcomes: Counter[str] = Counter()
            for rule in self.rules.values():
                predicted = apply_rule(
                    rule,
                    normalize_grid(grid),
                    candidate,
                    _level=level,
                    _object_cache=object_cache,
                )
                if predicted is not None:
                    outcomes[grid_hash(predicted)] += max(
                        1, int(round(rule.posterior_weight * 1000))
                    )
            total = sum(outcomes.values())
            entropy = 0.0
            if total:
                for count in outcomes.values():
                    probability = count / total
                    entropy -= probability * math.log2(probability)
            state_hash = grid_hash(grid)
            candidate_key = action_key(candidate)
            known = (
                (int(level), state_hash, candidate_key) in self.exact_edges
                if level is not None
                else any(
                    before_hash == state_hash and edge_action == candidate_key
                    for _edge_level, before_hash, edge_action in self.exact_edges
                )
            )
            novelty = 0.0 if known else 1.0
            prior = [
                record
                for record in self.timeline
                if record.action.get("action") == candidate.get("action")
            ]
            risk = (
                sum(record.game_over for record in prior) / len(prior)
                if prior
                else 0.0
            )
            ranked.append(
                ProbeRecommendation(
                    action=candidate,
                    information_gain=entropy,
                    risk=risk,
                    novelty=novelty,
                    predicted_outcomes=len(outcomes),
                )
            )
        return max(
            ranked,
            key=lambda item: (
                item.information_gain + 0.25 * item.novelty - 2.0 * item.risk,
                -int(item.action.get("row", -1)),
                -int(item.action.get("col", -1)),
                action_key(item.action),
            ),
        )

    def _rule_goal_probability(
        self,
        rule: TypedRule,
        *,
        level: int | None,
    ) -> float:
        if level is None:
            return rule.predicts_goal()
        support = [
            self.timeline[index]
            for index in rule.support
            if 0 <= index < len(self.timeline)
            and self.timeline[index].level == int(level)
        ]
        if not support:
            return 0.0
        goals = sum(
            record.level_completed or record.run_complete for record in support
        )
        return goals / len(support)

    def diagnostics(self) -> dict[str, Any]:
        certified = [rule for rule in self.rules.values() if rule.certified]
        explained = {
            index
            for rule in certified
            for index in rule.support
        }
        ontology_support: Counter[str] = Counter()
        for rule in certified:
            ontology_support[rule.ontology] += len(rule.support)
        return {
            "schema_version": 1,
            "transitions": len(self.timeline),
            "states": len(self.states),
            "rule_candidates": len(self.rules),
            "certified_rules": len(certified),
            "general_replay_coverage": (
                len(explained) / len(self.timeline) if self.timeline else 0.0
            ),
            "prediction_attempts": self.prediction_attempts,
            "prediction_matches": self.prediction_matches,
            "prediction_mismatches": self.prediction_mismatches,
            "prediction_precision": (
                self.prediction_matches / self.prediction_attempts
                if self.prediction_attempts
                else 0.0
            ),
            "last_prediction_mismatch": self.last_prediction_mismatch,
            "plan_requests": self.plan_requests,
            "plan_successes": self.plan_successes,
            "plan_aborts": self.plan_aborts,
            "probe_requests": self.probe_requests,
            "alias_conflicts": len(self.clone_graph.conflicts),
            "cscg_active": self.clone_graph.active,
            "ontology_support": dict(sorted(ontology_support.items())),
            "top_rules": [
                {
                    "rule_id": rule.rule_id,
                    "kind": rule.kind,
                    "ontology": rule.ontology,
                    "support": len(rule.support),
                    "contradictions": len(rule.contradictions),
                    "confidence": round(rule.confidence, 6),
                }
                for rule in sorted(
                    self.rules.values(),
                    key=lambda item: (
                        item.certified,
                        item.posterior_weight,
                        len(item.support),
                    ),
                    reverse=True,
                )[:12]
            ],
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "timeline": [
                {
                    **asdict(record),
                    "before": [list(row) for row in record.before],
                    "after": [list(row) for row in record.after],
                }
                for record in self.timeline
            ],
            "rules": [asdict(rule) for rule in self.rules.values()],
            "diagnostics": self.diagnostics(),
        }


def apply_rule(
    rule: TypedRule,
    grid: Sequence[Sequence[int]],
    action: str | Mapping[str, Any],
    *,
    _level: int | None = None,
    _object_cache: dict[int, list[dict[str, Any]]] | None = None,
) -> Grid | None:
    state = normalize_grid(grid)
    normalized_action = normalize_action(action)
    if normalized_action.get("action") != rule.action:
        return None
    scoped_level = rule.parameters.get("level")
    if (
        scoped_level is not None
        and _level is not None
        and int(scoped_level) != int(_level)
    ):
        return None
    if rule.kind == "noop":
        return state
    if rule.kind == "color-map":
        mapping = {
            int(old): int(new) for old, new in rule.parameters.get("mapping", [])
        }
        if not mapping or not all(cell in mapping for row in state for cell in row):
            return None
        return tuple(tuple(mapping[cell] for cell in row) for row in state)
    if rule.kind == "rigid-translation":
        delta = rule.parameters.get("delta", [0, 0])
        return _translate_grid(
            state,
            int(delta[0]),
            int(delta[1]),
            int(rule.parameters.get("background", 0)),
        )
    if rule.kind == "border-step":
        return _apply_border_step(state, rule.parameters)
    if rule.kind == "border-steps":
        return _apply_border_steps(state, rule.parameters.get("steps", []))
    if rule.kind == "object-translation":
        return _apply_object_translation(
            state,
            rule.parameters,
            object_cache=_object_cache,
        )
    if rule.kind == "object-blocked-noop":
        return _apply_object_translation(
            state,
            rule.parameters,
            object_cache=_object_cache,
            blocked_as_noop=True,
        )
    if rule.kind == "click-recolor":
        if normalized_action.get("action") != "MOUSE":
            return None
        connectivity = 8 if rule.ontology == "color-8" else 4
        row = int(normalized_action.get("row", -1))
        col = int(normalized_action.get("col", -1))
        if _object_cache is not None:
            objects = _object_cache.get(connectivity)
            if objects is None:
                objects = segment_grid(state, connectivity=connectivity)
                _object_cache[connectivity] = objects
        else:
            objects = segment_grid(state, connectivity=connectivity)
        source = int(rule.parameters["source_color"])
        shape = str(rule.parameters["shape_hash"])
        target = int(rule.parameters["target_color"])
        matched = [
            item
            for item in objects
            if item["color"] == source
            and item["shape_hash"] == shape
            and item["bbox"][0] <= row <= item["bbox"][2]
            and item["bbox"][1] <= col <= item["bbox"][3]
        ]
        if len(matched) != 1:
            return None
        obj = matched[0]
        output = [list(values) for values in state]
        row0, col0, row1, col1 = [int(value) for value in obj["bbox"]]
        representative = next(
            (
                (cell_row, cell_col)
                for cell_row in range(row0, row1 + 1)
                for cell_col in range(col0, col1 + 1)
                if state[cell_row][cell_col] == source
            ),
            None,
        )
        if representative is None:
            return None
        cells = _component_at(
            state,
            representative[0],
            representative[1],
            color=source,
            connectivity=connectivity,
        )
        if not cells:
            return None
        if _shape_hash(cells) != obj["shape_hash"]:
            return None
        for cell_row, cell_col in cells:
            output[cell_row][cell_col] = target
        return normalize_grid(output)
    if rule.kind == "click-noop":
        if normalized_action.get("action") != "MOUSE":
            return None
        row = int(normalized_action.get("row", -1))
        col = int(normalized_action.get("col", -1))
        profile = _click_profile(state, row, col)
        if profile is None:
            return None
        expected = {
            key: rule.parameters.get(key)
            for key in (
                "source_color",
                "local_same_mask",
                "component_signature",
            )
        }
        return state if profile == expected else None
    return None


def _infer_color_map(before: Grid, after: Grid) -> tuple[tuple[int, int], ...]:
    if len(before) != len(after) or any(
        len(first) != len(second) for first, second in zip(before, after)
    ):
        return ()
    mapping: dict[int, set[int]] = defaultdict(set)
    changed = False
    for first_row, second_row in zip(before, after):
        for first, second in zip(first_row, second_row):
            mapping[first].add(second)
            changed = changed or first != second
    if not changed or any(len(values) != 1 for values in mapping.values()):
        return ()
    return tuple(sorted((color, next(iter(values))) for color, values in mapping.items()))


def _background_candidates(grid: Grid) -> tuple[int, ...]:
    counts = Counter(cell for row in grid for cell in row)
    return tuple(color for color, _count in counts.most_common(2))


def _translate_grid(grid: Grid, dr: int, dc: int, background: int) -> Grid | None:
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    if not rows or not cols:
        return None
    output = [[background for _ in range(cols)] for _ in range(rows)]
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            if value == background:
                continue
            target_row = row + dr
            target_col = col + dc
            if not (0 <= target_row < rows and 0 <= target_col < cols):
                return None
            output[target_row][target_col] = value
    return normalize_grid(output)


def _infer_rigid_translation(
    before: Grid, after: Grid, background: int
) -> tuple[int, int] | None:
    if before == after:
        return None
    for distance in range(1, 5):
        for dr, dc in (
            (-distance, 0),
            (distance, 0),
            (0, -distance),
            (0, distance),
        ):
            if _translate_grid(before, dr, dc, background) == after:
                return dr, dc
    return None


def _edge_cells(edge: str, rows: int, cols: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, col) for col in range(cols)]
    if edge == "bottom":
        return [(rows - 1, col) for col in range(cols)]
    if edge == "left":
        return [(row, 0) for row in range(rows)]
    if edge == "right":
        return [(row, cols - 1) for row in range(rows)]
    return []


def _infer_border_step(before: Grid, after: Grid) -> dict[str, Any] | None:
    changed = [
        (row, col)
        for row, (first_row, second_row) in enumerate(zip(before, after))
        for col, (first, second) in enumerate(zip(first_row, second_row))
        if first != second
    ]
    if len(changed) != 1:
        return None
    rows = len(before)
    cols = max((len(row) for row in before), default=0)
    row, col = changed[0]
    source = before[row][col]
    target = after[row][col]
    candidates: list[dict[str, Any]] = []
    for edge in ("top", "bottom", "left", "right"):
        cells = _edge_cells(edge, rows, cols)
        if (row, col) not in cells:
            continue
        index = cells.index((row, col))
        for offset in (-1, 1):
            neighbor = index + offset
            if 0 <= neighbor < len(cells):
                other_row, other_col = cells[neighbor]
                if before[other_row][other_col] == target:
                    candidates.append(
                        {
                            "edge": edge,
                            "source_color": source,
                            "target_color": target,
                            "target_neighbor_offset": offset,
                        }
                    )
    return candidates[0] if len(candidates) == 1 else None


def _apply_border_step(
    grid: Grid,
    parameters: Mapping[str, Any],
) -> Grid | None:
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    cells = _edge_cells(str(parameters.get("edge", "")), rows, cols)
    source = int(parameters.get("source_color", -1))
    target = int(parameters.get("target_color", -1))
    offset = int(parameters.get("target_neighbor_offset", 0))
    candidates: list[tuple[int, int]] = []
    for index, (row, col) in enumerate(cells):
        neighbor = index + offset
        if not 0 <= neighbor < len(cells):
            continue
        other_row, other_col = cells[neighbor]
        if grid[row][col] == source and grid[other_row][other_col] == target:
            candidates.append((row, col))
    if len(candidates) != 1:
        return None
    output = [list(row) for row in grid]
    row, col = candidates[0]
    output[row][col] = target
    return normalize_grid(output)


def _infer_border_steps(
    before: Grid,
    after: Grid,
    *,
    maximum_steps: int = 4,
) -> list[dict[str, Any]]:
    """Infer a short sequence of deterministic edge-counter updates."""

    if before == after:
        return []

    def search(
        current: Grid,
        chosen: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if current == after:
            return chosen
        if len(chosen) >= maximum_steps:
            return None
        differences = [
            (row, col)
            for row, (first_row, second_row) in enumerate(zip(current, after))
            for col, (first, second) in enumerate(zip(first_row, second_row))
            if first != second
        ]
        rows = len(current)
        cols = max((len(row) for row in current), default=0)
        candidates: list[dict[str, Any]] = []
        for edge in ("top", "bottom", "left", "right"):
            cells = _edge_cells(edge, rows, cols)
            positions = {cell: index for index, cell in enumerate(cells)}
            for row, col in differences:
                index = positions.get((row, col))
                if index is None:
                    continue
                source = current[row][col]
                target = after[row][col]
                for offset in (-1, 1):
                    neighbor = index + offset
                    if not 0 <= neighbor < len(cells):
                        continue
                    other_row, other_col = cells[neighbor]
                    if current[other_row][other_col] != target:
                        continue
                    candidate = {
                        "edge": edge,
                        "source_color": source,
                        "target_color": target,
                        "target_neighbor_offset": offset,
                    }
                    if candidate not in candidates:
                        candidates.append(candidate)
        for candidate in candidates:
            predicted = _apply_border_step(current, candidate)
            if predicted is None or predicted == current:
                continue
            if any(
                predicted[row][col] != after[row][col]
                and predicted[row][col] != current[row][col]
                for row in range(len(current))
                for col in range(len(current[row]))
            ):
                continue
            result = search(predicted, [*chosen, candidate])
            if result is not None:
                return result
        return None

    return search(before, []) or []


def _apply_border_steps(
    grid: Grid,
    steps: Sequence[Mapping[str, Any]],
) -> Grid | None:
    if not steps:
        return None
    current = grid
    for step in steps:
        predicted = _apply_border_step(current, step)
        if predicted is None:
            return None
        current = predicted
    return current


def _infer_object_translations(record: TransitionEvidence) -> list[TypedRule]:
    if record.before == record.after:
        return []
    direction = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }.get(str(record.action.get("action", "")))
    if direction is None:
        return []
    backgrounds = _background_candidates(record.before)
    if not backgrounds:
        return []
    background = backgrounds[0]
    rules: list[TypedRule] = []
    for ontology, objects in segment_grid_multi(record.before).items():
        if ontology == "color-4-all":
            continue
        connectivity = 8 if ontology == "color-8" else 4
        object_cache = {connectivity: list(objects)}
        after_objects = segment_grid(record.after, connectivity=connectivity)
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        grouped_after: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for item in objects:
            grouped[(int(item["color"]), str(item["shape_hash"]))].append(item)
        for item in after_objects:
            grouped_after[(int(item["color"]), str(item["shape_hash"]))].append(item)
        for (parent_color, parent_shape), before_group in grouped.items():
            after_group = grouped_after.get((parent_color, parent_shape), [])
            if len(before_group) < 2 or len(after_group) != len(before_group):
                continue
            ordered_before = sorted(before_group, key=lambda item: tuple(item["bbox"]))
            ordered_after = sorted(after_group, key=lambda item: tuple(item["bbox"]))
            deltas = [
                [
                    int(after_item["bbox"][0]) - int(before_item["bbox"][0]),
                    int(after_item["bbox"][1]) - int(before_item["bbox"][1]),
                ]
                for before_item, after_item in zip(ordered_before, ordered_after)
            ]
            if (
                not any(dr or dc for dr, dc in deltas)
                or any(dr and dc for dr, dc in deltas)
                or any(max(abs(dr), abs(dc)) > 8 for dr, dc in deltas)
            ):
                continue
            row0, col0, row1, col1 = [
                int(value) for value in ordered_before[0]["bbox"]
            ]
            pattern = [
                [row - row0, col - col0, record.before[row][col]]
                for row in range(row0, row1 + 1)
                for col in range(col0, col1 + 1)
                if record.before[row][col] != background
            ]
            parameters = {
                "level": record.level,
                "background": background,
                "connectivity": connectivity,
                "parent_color": parent_color,
                "parent_shape_hash": parent_shape,
                "parent_count": len(ordered_before),
                "pattern": pattern,
                "deltas": deltas,
            }
            predicted = _apply_object_translation(
                record.before,
                parameters,
                object_cache=object_cache,
            )
            if predicted == record.after:
                rules.append(
                    TypedRule(
                        "object-translation",
                        str(record.action.get("action", "")),
                        parameters,
                        ontology=ontology,
                        source_index=record.index,
                    )
                )
            elif predicted is not None:
                ticker = _infer_border_steps(predicted, record.after)
                if ticker:
                    composite = {**parameters, "border_steps": ticker}
                    if _apply_object_translation(
                        record.before,
                        composite,
                        object_cache=object_cache,
                    ) == record.after:
                        rules.append(
                            TypedRule(
                                "object-translation",
                                str(record.action.get("action", "")),
                                composite,
                                ontology=ontology,
                                source_index=record.index,
                            )
                        )
        for obj in objects[:64]:
            box = [int(value) for value in obj.get("bbox", [])]
            if len(box) != 4 or int(obj.get("area", 0)) > 2_048:
                continue
            row0, col0, row1, col1 = box
            pattern = [
                [row - row0, col - col0, record.before[row][col]]
                for row in range(row0, row1 + 1)
                for col in range(col0, col1 + 1)
                if record.before[row][col] != background
            ]
            if not pattern:
                continue
            for distance in range(1, 9):
                dr = direction[0] * distance
                dc = direction[1] * distance
                parameters: dict[str, Any] = {
                    "level": record.level,
                    "background": background,
                    "connectivity": connectivity,
                    "parent_color": int(obj["color"]),
                    "parent_shape_hash": str(obj["shape_hash"]),
                    "parent_count": sum(
                        int(other.get("color", -1)) == int(obj["color"])
                        and str(other.get("shape_hash", ""))
                        == str(obj["shape_hash"])
                        for other in objects
                    ),
                    "pattern": pattern,
                    "delta": [dr, dc],
                }
                predicted = _apply_object_translation(
                    record.before,
                    parameters,
                    object_cache=object_cache,
                )
                if predicted == record.after:
                    rules.append(
                        TypedRule(
                            "object-translation",
                            str(record.action.get("action", "")),
                            parameters,
                            ontology=ontology,
                            source_index=record.index,
                        )
                    )
                    continue
                if predicted is None:
                    continue
                ticker = _infer_border_steps(predicted, record.after)
                if not ticker:
                    continue
                composite = {**parameters, "border_steps": ticker}
                if _apply_object_translation(
                    record.before,
                    composite,
                    object_cache=object_cache,
                ) == record.after:
                    rules.append(
                        TypedRule(
                            "object-translation",
                            str(record.action.get("action", "")),
                            composite,
                            ontology=ontology,
                            source_index=record.index,
                        )
                    )
    return rules


def _apply_object_translation(
    grid: Grid,
    parameters: Mapping[str, Any],
    *,
    object_cache: dict[int, list[dict[str, Any]]] | None = None,
    blocked_as_noop: bool = False,
) -> Grid | None:
    connectivity = int(parameters.get("connectivity", 4))
    if object_cache is not None:
        objects = object_cache.get(connectivity)
        if objects is None:
            objects = segment_grid(grid, connectivity=connectivity)
            object_cache[connectivity] = objects
    else:
        objects = segment_grid(grid, connectivity=connectivity)
    parent_color = int(parameters.get("parent_color", -1))
    parent_shape = str(parameters.get("parent_shape_hash", ""))
    matched = [
        obj
        for obj in objects
        if int(obj.get("color", -1)) == parent_color
        and str(obj.get("shape_hash", "")) == parent_shape
    ]
    expected_count = int(parameters.get("parent_count", 1))
    if len(matched) != expected_count:
        return None
    pattern = [
        (int(row), int(col), int(color))
        for row, col, color in parameters.get("pattern", [])
    ]
    if not pattern:
        return None
    ordered_matched = sorted(matched, key=lambda item: tuple(item["bbox"]))
    source_by_object: list[dict[tuple[int, int], int]] = []
    source_cells: dict[tuple[int, int], int] = {}
    for obj in ordered_matched:
        box = [int(value) for value in obj.get("bbox", [])]
        if len(box) != 4:
            return None
        row0, col0 = box[:2]
        object_cells: dict[tuple[int, int], int] = {}
        for row, col, color in pattern:
            object_cells[(row0 + row, col0 + col)] = color
        source_cells.update(object_cells)
        source_by_object.append(object_cells)
    if any(
        not (0 <= row < len(grid) and 0 <= col < len(grid[row]))
        or grid[row][col] != color
        for (row, col), color in source_cells.items()
    ):
        return None
    raw_deltas = parameters.get("deltas")
    if isinstance(raw_deltas, Sequence) and len(raw_deltas) == len(source_by_object):
        deltas = [
            (int(delta[0]), int(delta[1]))
            for delta in raw_deltas
        ]
    else:
        delta = parameters.get("delta", [0, 0])
        deltas = [(int(delta[0]), int(delta[1]))] * len(source_by_object)
    background = int(parameters.get("background", 0))
    targets: dict[tuple[int, int], int] = {}
    for object_cells, (dr, dc) in zip(source_by_object, deltas):
        for (row, col), color in object_cells.items():
            target = (row + dr, col + dc)
            prior = targets.get(target)
            if prior is not None and prior != color:
                return None
            targets[target] = color
    blocked = any(
        not (0 <= row < len(grid) and 0 <= col < len(grid[row]))
        or ((row, col) not in source_cells and grid[row][col] != background)
        for row, col in targets
    )
    # A rule induced from a successful motion does not claim to model blocked
    # motion.  Treat collision as non-applicability so failed moves cannot
    # contradict (or be authorized by) the successful-transition rule.
    if blocked:
        return grid if blocked_as_noop else None
    output = [list(row) for row in grid]
    for row, col in source_cells:
        output[row][col] = background
    for (row, col), color in targets.items():
        output[row][col] = color
    result = normalize_grid(output)
    ticker = parameters.get("border_steps")
    if isinstance(ticker, Sequence):
        return _apply_border_steps(result, ticker)
    return result


def _component_at(
    grid: Grid,
    row: int,
    col: int,
    *,
    color: int,
    connectivity: int,
) -> set[tuple[int, int]]:
    if not (0 <= row < len(grid) and 0 <= col < len(grid[row])):
        return set()
    if grid[row][col] != color:
        return set()
    cells = {(row, col)}
    queue = deque([(row, col)])
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    while queue:
        current_row, current_col = queue.popleft()
        for dr, dc in offsets:
            other_row = current_row + dr
            other_col = current_col + dc
            if (
                0 <= other_row < len(grid)
                and 0 <= other_col < len(grid[other_row])
                and grid[other_row][other_col] == color
                and (other_row, other_col) not in cells
            ):
                cells.add((other_row, other_col))
                queue.append((other_row, other_col))
    return cells


def _shape_hash(cells: Iterable[tuple[int, int]]) -> str:
    values = sorted(cells)
    row0 = min(row for row, _ in values)
    col0 = min(col for _, col in values)
    canonical = ";".join(
        f"{row-row0},{col-col0}" for row, col in values
    )
    return hashlib.sha1(canonical.encode("ascii")).hexdigest()[:12]


def _click_profile(
    grid: Grid,
    row: int,
    col: int,
) -> dict[str, Any] | None:
    if not (0 <= row < len(grid) and 0 <= col < len(grid[row])):
        return None
    source = grid[row][col]
    local_same_mask = [
        (
            -1
            if not (
                0 <= other_row < len(grid)
                and 0 <= other_col < len(grid[other_row])
            )
            else int(grid[other_row][other_col] == source)
        )
        for other_row in range(row - 1, row + 2)
        for other_col in range(col - 1, col + 2)
    ]
    cells = _component_at(
        grid,
        row,
        col,
        color=source,
        connectivity=4,
    )
    if len(cells) <= 256:
        component_signature = f"shape:{_shape_hash(cells)}"
    else:
        component_signature = f"large:{(len(cells) + 15) // 16}"
    return {
        "source_color": source,
        "local_same_mask": local_same_mask,
        "component_signature": component_signature,
    }


def _infer_click_noop(record: TransitionEvidence) -> TypedRule | None:
    row = int(record.action.get("row", -1))
    col = int(record.action.get("col", -1))
    profile = _click_profile(record.before, row, col)
    if profile is None:
        return None
    return TypedRule(
        "click-noop",
        "MOUSE",
        {"level": record.level, **profile},
        source_index=record.index,
    )


def _infer_click_recolor(
    record: TransitionEvidence,
    *,
    row: int,
    col: int,
    ontology: str,
    objects: Sequence[Mapping[str, Any]],
) -> TypedRule | None:
    connectivity = 8 if ontology == "color-8" else 4
    if not (0 <= row < len(record.before) and 0 <= col < len(record.before[row])):
        return None
    changed = {
        (cell_row, cell_col)
        for cell_row, (first_row, second_row) in enumerate(
            zip(record.before, record.after)
        )
        for cell_col, (first, second) in enumerate(zip(first_row, second_row))
        if first != second
    }
    if not changed:
        return None
    sources = {record.before[cell_row][cell_col] for cell_row, cell_col in changed}
    if len(sources) != 1:
        return None
    source = next(iter(sources))
    first_row, first_col = next(iter(changed))
    cells = _component_at(
        record.before,
        first_row,
        first_col,
        color=source,
        connectivity=connectivity,
    )
    if changed != cells:
        return None
    targets = {record.after[cell_row][cell_col] for cell_row, cell_col in cells}
    if len(targets) != 1:
        return None
    shape = _shape_hash(cells)
    matched = [
        item
        for item in objects
        if str(item.get("shape_hash")) == shape
        and int(item.get("color", -1)) == source
        and int(item["bbox"][0]) <= row <= int(item["bbox"][2])
        and int(item["bbox"][1]) <= col <= int(item["bbox"][3])
    ]
    if len(matched) != 1:
        return None
    return TypedRule(
        "click-recolor",
        "MOUSE",
        {
            "level": record.level,
            "source_color": source,
            "target_color": next(iter(targets)),
            "shape_hash": shape,
        },
        ontology=ontology,
        source_index=record.index,
    )


def _candidate_actions(
    grid: Sequence[Sequence[int]], valid_actions: Sequence[str]
) -> list[dict[str, Any]]:
    names = []
    for raw in valid_actions:
        value = _MODEL_ACTIONS.get(str(raw).strip().upper(), str(raw).strip().upper())
        if value and value not in names:
            names.append(value)
    candidates = [
        {"action": name} for name in names if name not in {"MOUSE", "RESET"}
    ]
    if "MOUSE" in names:
        centroids: set[tuple[int, int]] = set()
        for objects in segment_grid_multi(grid).values():
            for item in objects[:128]:
                centroid = item.get("centroid", [-1, -1])
                centroids.add(
                    (int(round(float(centroid[0]))), int(round(float(centroid[1]))))
                )
        candidates.extend(
            {"action": "MOUSE", "row": row, "col": col}
            for row, col in sorted(centroids)
            if row >= 0 and col >= 0
        )
    return candidates[:256]
