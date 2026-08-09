from __future__ import annotations

from ouro3.ledger import HypothesisLedger
from ouro3.fallback import DeterministicExplorer
from ouro3.verification import verify_prediction


def test_ledger_compacts_and_aborts_plan_on_contradiction() -> None:
    ledger = HypothesisLedger(
        world_model="player moves",
        supporting_evidence=["same"] * 20,
        current_plan=[{"action": "RIGHT"}],
    )
    ledger.contradict("RIGHT did not move the player")
    assert ledger.has_active_contradiction
    assert ledger.current_plan == []
    assert len(ledger.compact().supporting_evidence) == 1
    round_trip = HypothesisLedger.from_payload(ledger.to_payload())
    assert round_trip.contradictions == ["RIGHT did not move the player"]


def test_prediction_verifier_handles_multilevel_and_game_over() -> None:
    assert verify_prediction(
        {"level_completed": True, "level": 2},
        {"level_completed": True, "level": 2, "game_over": False},
    ).matched
    mismatch = verify_prediction(
        {"game_over": False},
        {"game_over": True},
    )
    assert not mismatch.matched
    assert "game_over" in mismatch.reason


def test_fallback_normalizes_engine_mouse_action() -> None:
    action = DeterministicExplorer("mouse-only").choose(
        grid=[[0, 0], [0, 1]],
        valid_actions=["ACTION6"],
    )
    assert action["action"] == "MOUSE"
    assert {"row", "col"} <= set(action)
