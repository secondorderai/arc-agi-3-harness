from ouro2.config import Config


def test_deterministic_submission_budget_defaults(monkeypatch):
    monkeypatch.delenv("OURO2_MAX_ACTIONS", raising=False)
    monkeypatch.delenv("OURO2_TIME_BUDGET_S", raising=False)
    monkeypatch.delenv("OURO2_DISABLE_MODEL", raising=False)

    config = Config.from_env()

    assert config.max_actions == 640
    assert config.time_budget_s == 1200.0
    assert config.disable_model is True
