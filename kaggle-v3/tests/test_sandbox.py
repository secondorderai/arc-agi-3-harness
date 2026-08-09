from __future__ import annotations

from inference.agent.python_tool_sandbox import run_sandboxed_python


def _state(*, mode: str = "ouro-hybrid") -> dict:
    frame = {
        "ascii": "00\n01",
        "step": 0,
        "level": 1,
        "shape": [2, 2],
        "grid": [[0, 0], [0, 1]],
        "image": "data:image/png;base64,AA==",
        "segmentation": [{"id": 0}],
        "changed_regions": [],
        "object_tracks": [],
        "animation_summary": {"detected": False},
    }
    return {
        "harness_mode": mode,
        "current_frame": frame,
        "history": [{"action": "", "frame": frame}],
        "hypothesis_ledger": {"world_model": "test", "revision": 1},
        "valid_actions": ["RIGHT", "MOUSE"],
        "last_action_result": {},
    }


def test_sandbox_exposes_stable_views_and_helpers() -> None:
    result = run_sandboxed_python(
        code=(
            "result = {'crop': current_frame.ascii_crop(0,1,0,2), "
            "'segments': current_frame.segmentation, "
            "'ledger': hypothesis_ledger.world_model, "
            "'path': shortest_path((0,0), {(1,1)}, 2, 2)}"
        ),
        timeout_seconds=3,
        initial_state=_state(),
        action_handler=lambda actions: {"action_result": {}, "state": _state()},
    )
    assert not result["error"]
    assert result["result"]["crop"] == "00"
    assert result["result"]["ledger"] == "test"
    assert result["result"]["path"][-1] == [1, 1]


def test_duck_reference_sandbox_hides_hybrid_helpers_and_expectations() -> None:
    received = []

    def handler(actions):
        received.extend(actions)
        return {"action_result": {}, "state": _state(mode="duck-reference")}

    hidden = run_sandboxed_python(
        code="result = hypothesis_ledger",
        timeout_seconds=3,
        initial_state=_state(mode="duck-reference"),
        action_handler=handler,
    )
    assert "NameError" in hidden["error"]

    result = run_sandboxed_python(
        code=(
            "result = {"
            "'has_crop': hasattr(current_frame, 'image'),"
            "'action': action({'action':'RIGHT','expect':{'gameplay_changed':True}})"
            "}"
        ),
        timeout_seconds=3,
        initial_state=_state(mode="duck-reference"),
        action_handler=handler,
    )
    assert not result["error"]
    assert result["result"]["has_crop"] is False
    assert received == [{"action": "RIGHT"}]


def test_sandbox_denies_filesystem_environment_network_and_children() -> None:
    for code in (
        "open('/tmp/nope', 'w')",
        "import os",
        "__import__('socket')",
        "__import__('subprocess')",
    ):
        result = run_sandboxed_python(
            code=code,
            timeout_seconds=3,
            initial_state=_state(),
            action_handler=lambda actions: {"action_result": {}, "state": _state()},
        )
        assert result["error"]


def test_sandbox_handles_malformed_code_timeout_and_batch_stop() -> None:
    malformed = run_sandboxed_python(
        code="if:",
        timeout_seconds=2,
        initial_state=_state(),
        action_handler=lambda actions: {"action_result": {}, "state": _state()},
    )
    assert "SyntaxError" in malformed["error"]

    timed_out = run_sandboxed_python(
        code="while True: pass",
        timeout_seconds=1,
        initial_state=_state(),
        action_handler=lambda actions: {"action_result": {}, "state": _state()},
    )
    assert "timed out" in timed_out["error"]

    received = []

    def handler(actions):
        received.extend(actions)
        return {
            "action_result": {
                "executed": True,
                "requested_count": len(actions),
                "executed_count": 1,
                "stopped_early": True,
                "level_completed": True,
                "stop_reason": "level_completed",
            },
            "state": _state(),
        }

    batch = run_sandboxed_python(
        code=(
            "result = action(["
            "{'action':'RIGHT','expect':{'gameplay_changed':True}},"
            "{'action':'RIGHT'}])"
        ),
        timeout_seconds=3,
        initial_state=_state(),
        action_handler=handler,
    )
    assert not batch["error"]
    assert received[0]["expect"] == {"gameplay_changed": True}
    assert batch["result"]["stopped_early"]
