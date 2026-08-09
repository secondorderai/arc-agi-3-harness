from __future__ import annotations

from types import SimpleNamespace

from ouro3 import runner
from ouro3.config import HarnessConfig, RuntimeProfile


def test_hidden_submission_environment_matches_competition_gateway(
    monkeypatch,
) -> None:
    for name in (
        "ARC_API_KEY",
        "ARC_BASE_URL",
        "RECORDINGS_DIR",
        "MPLBACKEND",
        "TAAF_RUN_AS_SUBMISSION",
        "TAAF_MINIMAL_DIAGNOSTICS",
    ):
        monkeypatch.delenv(name, raising=False)

    runner.configure_hidden_submission_environment()

    assert runner.os.environ["ARC_API_KEY"] == "test-key-123"
    assert runner.os.environ["ARC_BASE_URL"] == "http://gateway:8001/"
    assert runner.os.environ["RECORDINGS_DIR"] == "/kaggle/working/server_recording"
    assert runner.os.environ["MPLBACKEND"] == "Agg"
    assert runner.os.environ["TAAF_RUN_AS_SUBMISSION"] == "1"
    assert runner.os.environ["TAAF_MINIMAL_DIAGNOSTICS"] == "1"


def test_hidden_gateway_discovery_retries_until_all_unique_games_arrive(
    monkeypatch,
    tmp_path,
) -> None:
    calls = 0

    class FakeArcade:
        def __init__(self, **_kwargs) -> None:
            nonlocal calls
            calls += 1

        def get_environments(self):
            if calls == 1:
                raise ConnectionError("gateway is booting")
            return [SimpleNamespace(game_id=f"hidden-{index:03d}") for index in range(110)]

    monkeypatch.setenv("ARC_BASE_URL", "http://gateway.test:8001")
    monkeypatch.setenv("OURO3_EMPTY_ENVIRONMENTS", str(tmp_path / "empty"))
    monkeypatch.setattr(runner.arc_agi, "Arcade", FakeArcade)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    spec, game_ids = runner.discover_hidden_gateway(
        timeout_s=1,
        poll_interval_s=0.01,
    )

    assert calls == 2
    assert len(game_ids) == len(set(game_ids)) == 110
    assert spec.arc_base_url == "http://gateway.test:8001"


def test_hidden_submission_wires_the_safe_four_wave_budget(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}
    game_ids = [f"hidden-{index:03d}" for index in range(110)]

    monkeypatch.setattr(
        runner,
        "discover_hidden_gateway",
        lambda: (SimpleNamespace(arc_base_url="http://gateway.test:8001"), game_ids),
    )
    monkeypatch.setattr(
        runner,
        "make_solver",
        lambda config: captured.setdefault("solver_config", config) or object(),
    )
    monkeypatch.setattr(
        runner,
        "build_competition_benchmark",
        lambda **kwargs: captured.setdefault("benchmark", kwargs) or object(),
    )

    async def fake_run_benchmark(*_args, config, **_kwargs):
        captured["run_config"] = config
        return {"mean_engine_score": 0.0}

    monkeypatch.setattr(runner, "run_benchmark", fake_run_benchmark)

    config = HarnessConfig.audit(seed=0).with_overrides(
        profile=RuntimeProfile.KAGGLE_SUBMISSION,
    )
    output_path = tmp_path / "hidden-metrics.json"
    metrics = runner.run_hidden_submission(config=config, output_path=output_path)

    assert metrics["submission_budget"]["total_games"] == 110
    assert metrics["submission_budget"]["waves"] == 4
    assert metrics["submission_budget"]["per_game_cap_s"] == 7_200
    assert metrics["submission_budget"]["worst_case_gameplay_s"] == 28_800
    assert captured["solver_config"].reference_game_cap_s == 7_200
    assert captured["run_config"].reference_game_cap_s == 7_200
    assert output_path.exists()
