from __future__ import annotations

import os

import pytest

from duck_reference.solver import DuckReferenceHarnessSolver
from inference.agent.tool_agent import ToolAgent, python_tool_description
from inference.utils.openai_compat import build_chat_payload
from ouro3.config import HarnessConfig
from ouro3.mode import HarnessMode
from ouro3.runner import make_solver
from ouro3.solver import HybridHarnessSolver


def test_reference_runtime_constants_and_seed_omission(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_ANALYZER_SEED", "999")
    config = HarnessConfig.reference()
    config.apply_environment()

    assert config.mode == HarnessMode.DUCK_REFERENCE
    assert config.model_id == "vrfai/Qwen3.6-27B-FP8"
    assert config.max_model_len == 65_536
    assert config.context_window == 32_768
    assert config.image_scale == 4
    assert config.temperature == 0.6
    assert config.top_p == 0.95
    assert config.top_k == 20
    assert config.concurrency == 28
    assert config.reference_game_cap_s == 7_920
    assert config.analyzer_timeout_s == 900
    assert config.python_timeout_s == 30
    assert config.python_output_tokens == 1_024
    assert os.environ["LOCAL_ANALYZER_TOOL_STEPS"] == "0"
    assert os.environ["LOCAL_ANALYZER_MAX_OUTPUT"] == "0"
    assert os.environ["LOCAL_ANALYZER_YIELD_SECONDS"] == "60"
    assert os.environ["ONLY_RESET_LEVELS"] == "true"
    assert "LOCAL_ANALYZER_SEED" not in os.environ
    payload = build_chat_payload(
        provider="vllm",
        model=config.model_id,
        messages=[],
        max_tokens=None,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        thinking=config.enable_thinking,
        seed=-1,
    )
    assert "seed" not in payload
    assert "max_tokens" not in payload
    with pytest.raises(ValueError, match="duck-reference runtime changed"):
        config.with_overrides(analyzer_timeout_s=120)


def test_reference_and_hybrid_select_disjoint_solver_types(monkeypatch) -> None:
    reference = HarnessConfig.reference(seed=2)
    reference.apply_environment()
    reference_solver = make_solver(reference)
    assert type(reference_solver) is DuckReferenceHarnessSolver
    assert reference_solver.max_runtime_s_per_game == 7_920
    assert reference_solver.analyzer_timeout == 900

    hybrid = HarnessConfig.local(seed=2)
    hybrid.apply_environment()
    hybrid_solver = make_solver(hybrid)
    assert type(hybrid_solver) is HybridHarnessSolver
    assert hybrid_solver.max_runtime_s_per_game == 360


def test_reference_prompt_and_tool_contract_exclude_hybrid_features(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OURO3_HARNESS_MODE", "duck-reference")
    agent = ToolAgent(
        model="test",
        base_url="http://127.0.0.1:1/v1",
        provider="openai",
    )
    prompt = agent._build_user_prompt(0, valid_actions=["RIGHT"])
    description = python_tool_description()
    combined = f"{prompt}\n{description}"

    for hybrid_only in (
        "hypothesis_ledger",
        "changed_regions",
        "object_tracks",
        "prediction mismatch",
        "`expect`",
    ):
        assert hybrid_only not in combined
    assert "Only letter-coded board views" in prompt
    assert "only `.ascii`, `.segmentation`, `.step`, and `.level`" in description
