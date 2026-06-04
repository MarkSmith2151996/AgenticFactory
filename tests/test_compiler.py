from __future__ import annotations

import pytest

from agentic_factory.compiler import compile_prompt
from agentic_factory.loader import get_agent


def test_compile_prompt_includes_task_tools_and_schema() -> None:
    spec = get_agent("brand-amazon-screener")

    compiled = compile_prompt(spec, {"brand": "Corelle"})

    assert "Navigate to amazon.com using browser_gather" in compiled["system"]
    assert "Search for Corelle" in compiled["system"]
    assert "browser_gather" in compiled["system"]
    assert "[Tool Calling Format]" in compiled["system"]
    assert '"tool_call"' in compiled["system"]
    assert '"final_answer"' in compiled["system"]
    assert "typical_price_band" in compiled["system"]
    assert "e.g. '$10-$30'" in compiled["system"]


def test_compile_prompt_uses_box_tool_defs_when_provided() -> None:
    spec = get_agent("brand-amazon-screener")

    compiled = compile_prompt(
        spec,
        {"brand": "Corelle"},
        tool_defs=[
            {
                "name": "box_browser",
                "description": "Box-provided browser tool",
                "params": {"query": {"type": "string"}},
            }
        ],
    )

    assert "- box_browser:" in compiled["system"]
    assert "- browser_gather:" not in compiled["system"]


def test_compile_prompt_substitutes_input_in_user_message() -> None:
    spec = get_agent("brand-amazon-screener")

    compiled = compile_prompt(spec, {"brand": "Corelle"})

    assert "Search for Corelle" in compiled["user"]
    assert '"brand": "Corelle"' in compiled["user"]


def test_system_rules_appear_at_top() -> None:
    spec = get_agent("brand-amazon-screener").model_copy(update={"mode": "development"})

    compiled = compile_prompt(spec, {"brand": "Corelle"})

    assert compiled["system"].startswith("# System Rules")


def test_operational_mode_omits_system_rules() -> None:
    spec = get_agent("brand-amazon-screener")

    compiled = compile_prompt(spec, {"brand": "Corelle"})

    assert spec.mode == "operational"
    assert "# System Rules" not in compiled["system"]


def test_guidance_appears_at_end_when_present() -> None:
    spec = get_agent("brand-amazon-screener").model_copy(
        update={"mode": "development", "guidance": "Prefer conservative claims."}
    )

    compiled = compile_prompt(spec, {"brand": "Corelle"})

    assert compiled["system"].endswith("[Guidance]\nPrefer conservative claims.")


def test_missing_schema_name_raises_value_error() -> None:
    spec = get_agent("brand-amazon-screener").model_copy(update={"output": "MissingSchema"})

    with pytest.raises(ValueError, match="Unknown schema: MissingSchema"):
        compile_prompt(spec, {"brand": "Corelle"})


def test_missing_required_input_placeholder_raises_clear_error() -> None:
    spec = get_agent("brand-amazon-screener")

    with pytest.raises(ValueError, match="Missing required input for prompt placeholder: brand"):
        compile_prompt(spec, {})
