from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_factory.loader import get_agent, load_agents
from agentic_factory.spec import LlmAgentSpec, ServiceAgentSpec, validate_agent_spec


def write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_default_yaml_specs_load() -> None:
    specs = load_agents()

    assert {"brand-amazon-screener", "keepa-downloader", "data-runner"}.issubset(specs)
    assert isinstance(specs["brand-amazon-screener"], LlmAgentSpec)
    assert isinstance(specs["keepa-downloader"], ServiceAgentSpec)
    assert isinstance(specs["data-runner"], LlmAgentSpec)


def test_get_agent_returns_concrete_spec_types() -> None:
    assert isinstance(get_agent("brand-amazon-screener"), LlmAgentSpec)
    assert isinstance(get_agent("keepa-downloader"), ServiceAgentSpec)


def test_missing_required_fields_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        validate_agent_spec({"name": "missing-fields", "type": "llm_agent"})


def test_llm_spec_runtime_block_loads() -> None:
    spec = validate_agent_spec(
        {
            "name": "runtime-agent",
            "type": "llm_agent",
            "project": "fba",
            "task": "Do work",
            "tools": ["browser_gather"],
            "output": "BrandScreenerResult",
            "model": "openai:gpt-5.4",
            "runtime": {"base_url": "http://127.0.0.1:4096/v1"},
        }
    )

    assert isinstance(spec, LlmAgentSpec)
    assert spec.runtime is not None
    assert spec.runtime.base_url == "http://127.0.0.1:4096/v1"
    assert spec.runtime.api_key is None


def test_llm_spec_runtime_api_key_is_optional() -> None:
    spec = validate_agent_spec(
        {
            "name": "runtime-agent",
            "type": "llm_agent",
            "project": "fba",
            "task": "Do work",
            "tools": ["browser_gather"],
            "output": "BrandScreenerResult",
            "model": "openai:gpt-5.4",
            "runtime": {"base_url": "http://127.0.0.1:4096/v1", "api_key": "test-key"},
        }
    )

    assert isinstance(spec, LlmAgentSpec)
    assert spec.runtime is not None
    assert spec.runtime.api_key == "test-key"


def test_llm_spec_without_runtime_defaults_to_none() -> None:
    spec = validate_agent_spec(
        {
            "name": "plain-agent",
            "type": "llm_agent",
            "project": "fba",
            "task": "Do work",
            "tools": ["browser_gather"],
            "output": "BrandScreenerResult",
            "model": "openai:gpt-5.4",
        }
    )

    assert isinstance(spec, LlmAgentSpec)
    assert spec.runtime is None


def test_llm_spec_defaults_to_operational_mode() -> None:
    spec = validate_agent_spec(
        {
            "name": "plain-agent",
            "type": "llm_agent",
            "project": "fba",
            "task": "Do work",
            "tools": ["browser_gather"],
            "output": "BrandScreenerResult",
            "model": "openai:gpt-5.4",
        }
    )

    assert isinstance(spec, LlmAgentSpec)
    assert spec.mode == "operational"


def test_llm_spec_accepts_development_mode() -> None:
    spec = validate_agent_spec(
        {
            "name": "dev-agent",
            "type": "llm_agent",
            "mode": "development",
            "project": "fba",
            "task": "Do work",
            "tools": ["browser_gather"],
            "output": "BrandScreenerResult",
            "model": "openai:gpt-5.4",
        }
    )

    assert isinstance(spec, LlmAgentSpec)
    assert spec.mode == "development"


def test_llm_spec_allows_toolbox_only() -> None:
    spec = validate_agent_spec(
        {
            "name": "toolbox-agent",
            "type": "llm_agent",
            "project": "fba",
            "task": "Do work",
            "tools": [],
            "toolbox": ["db_read"],
            "output": "DataRunnerResult",
            "model": "openai:gpt-5.4",
        }
    )

    assert isinstance(spec, LlmAgentSpec)
    assert spec.tools == []
    assert spec.toolbox == ["db_read"]


def test_llm_spec_requires_tools_or_toolbox() -> None:
    with pytest.raises(ValidationError, match="At least one of 'tools' or 'toolbox' must be non-empty"):
        validate_agent_spec(
            {
                "name": "empty-tools-agent",
                "type": "llm_agent",
                "project": "fba",
                "task": "Do work",
                "tools": [],
                "toolbox": [],
                "output": "DataRunnerResult",
                "model": "openai:gpt-5.4",
            }
        )


def test_unknown_type_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        validate_agent_spec({"name": "unknown-agent", "type": "unknown"})


def test_duplicate_agent_names_raise_clear_error(tmp_path: Path) -> None:
    content = """
name: duplicate-agent
type: service_agent
project: fba
service:
  url_default: "http://example.test"
  endpoint: /download
  timeout: 1
input:
  required: [urls]
output: KeepaDownloadResult
""".strip()
    write_yaml(tmp_path / "one.yaml", content)
    write_yaml(tmp_path / "two.yaml", content)

    with pytest.raises(ValueError, match="Duplicate agent name 'duplicate-agent'"):
        load_agents(tmp_path)
