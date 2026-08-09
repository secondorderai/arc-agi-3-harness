from __future__ import annotations

from types import SimpleNamespace

from ouro3.metrics import summarize_runs


def test_retrodict_metrics_preserve_host_diagnostics() -> None:
    diagnostics = {
        "mode": "duck-retrodict",
        "world_model": {"certified_rules": 2},
    }
    run = SimpleNamespace(
        game_id="game",
        state="gave_up",
        levels_completed=0,
        number_of_levels=1,
        actions_per_level=[1],
        history=[],
        final_generated_tokens=0,
        final_uncached_input_tokens=0,
        final_wallclock_seconds=1.0,
        final_score=0.0,
        solver_note="",
        solver_telemetry={"retrodict_certified_rules": 2},
        solver_diagnostics=diagnostics,
    )
    metrics = summarize_runs(
        [run],
        experiment="duck-retrodict-v1",
        seed=0,
        config_hash="a" * 64,
        elapsed_seconds=1.0,
        mode="duck-retrodict",
    )
    assert metrics["telemetry"]["retrodict_certified_rules"] == 2
    assert metrics["games"][0]["retrodict_diagnostics"] == diagnostics
    assert metrics["retrodict_diagnostics"]["game"] == diagnostics
