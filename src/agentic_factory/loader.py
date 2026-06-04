"""YAML agent spec loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_factory.spec import AgentSpec, validate_agent_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agents"


def load_agents(agents_dir: Path | None = None) -> dict[str, AgentSpec]:
    """Load and validate every YAML agent spec keyed by agent name."""
    directory = agents_dir or AGENTS_DIR
    specs: dict[str, AgentSpec] = {}
    for path in sorted(directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        spec = validate_agent_spec(data)
        existing = specs.get(spec.name)
        if existing is not None:
            raise ValueError(f"Duplicate agent name {spec.name!r} in {path}")
        specs[spec.name] = spec
    return specs


def get_agent(name: str) -> AgentSpec:
    """Return one loaded agent spec by name."""
    specs = load_agents()
    try:
        return specs[name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent: {name}") from exc


def load_agent(name: str) -> AgentSpec:
    """Return one loaded agent spec by name."""
    return get_agent(name)
