"""Compact evidence ledger that survives model-context eviction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


def _compact_lines(values: Iterable[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = " ".join(str(raw).strip().split())
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value[:360])
    return out[-limit:]


@dataclass
class HypothesisLedger:
    world_model: str = ""
    goal_model: str = ""
    action_model: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    current_plan: list[dict[str, Any]] = field(default_factory=list)
    cross_level_knowledge: list[str] = field(default_factory=list)
    revision: int = 0

    @property
    def has_active_contradiction(self) -> bool:
        return bool(self.contradictions and (self.world_model or self.goal_model or self.action_model))

    def add_evidence(self, statement: str) -> None:
        self.supporting_evidence = _compact_lines(
            [*self.supporting_evidence, statement], limit=12
        )
        self.revision += 1

    def contradict(self, statement: str) -> None:
        self.contradictions = _compact_lines([*self.contradictions, statement], limit=8)
        self.current_plan = []
        self.revision += 1

    def compact(self) -> "HypothesisLedger":
        self.world_model = " ".join(self.world_model.split())[:900]
        self.goal_model = " ".join(self.goal_model.split())[:600]
        self.action_model = " ".join(self.action_model.split())[:900]
        self.supporting_evidence = _compact_lines(self.supporting_evidence, limit=12)
        self.contradictions = _compact_lines(self.contradictions, limit=8)
        self.open_questions = _compact_lines(self.open_questions, limit=8)
        self.cross_level_knowledge = _compact_lines(self.cross_level_knowledge, limit=10)
        self.current_plan = [
            item
            for item in self.current_plan[-12:]
            if isinstance(item, dict) and str(item.get("action", "")).strip()
        ]
        return self

    def to_payload(self) -> dict[str, Any]:
        self.compact()
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> "HypothesisLedger":
        if not isinstance(payload, dict):
            return cls()
        ledger = cls(
            world_model=str(payload.get("world_model", "")),
            goal_model=str(payload.get("goal_model", "")),
            action_model=str(payload.get("action_model", "")),
            supporting_evidence=list(payload.get("supporting_evidence") or []),
            contradictions=list(payload.get("contradictions") or []),
            open_questions=list(payload.get("open_questions") or []),
            current_plan=list(payload.get("current_plan") or []),
            cross_level_knowledge=list(payload.get("cross_level_knowledge") or []),
            revision=int(payload.get("revision", 0) or 0),
        )
        return ledger.compact()

    @classmethod
    def from_duck_memory(cls, memory: dict[str, Any] | None) -> "HypothesisLedger":
        value = memory or {}
        findings = str(
            value.get("supporting_evidence") or value.get("recent_findings", "")
        ).strip()
        questions = str(value.get("open_questions", "")).strip()
        plan = str(value.get("current_plan", "")).strip()
        return cls(
            world_model=str(value.get("world_model", "")),
            goal_model=str(value.get("goal_model", "")),
            action_model=str(value.get("action_model", "")),
            supporting_evidence=[findings] if findings else [],
            contradictions=_as_lines(value.get("contradictions")),
            open_questions=[questions] if questions else [],
            current_plan=[{"action": plan}] if plan else [],
            cross_level_knowledge=_as_lines(value.get("cross_level_notes")),
        ).compact()

    def to_duck_memory(self) -> dict[str, str]:
        return {
            "world_model": self.world_model,
            "goal_model": self.goal_model,
            "action_model": self.action_model,
            "recent_findings": " | ".join(self.supporting_evidence),
            "supporting_evidence": " | ".join(self.supporting_evidence),
            "contradictions": " | ".join(self.contradictions),
            "open_questions": " | ".join(self.open_questions),
            "current_plan": json.dumps(self.current_plan, separators=(",", ":"))
            if self.current_plan
            else "",
            "cross_level_notes": " | ".join(self.cross_level_knowledge),
        }


def _as_lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    return [text] if text else []
