"""Declarative agent spec models."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class ServiceConfig(BaseModel):
    url_env: str | None = None
    url_default: str
    endpoint: str
    timeout: int = Field(gt=0)


class AgentInputSpec(BaseModel):
    required: list[str] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    base_url: str
    api_key: str | None = None


class LlmAgentSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    type: Literal["llm_agent"]
    mode: Literal["operational", "development"] = "operational"
    project: str
    task: str
    tools: list[str] = Field(default_factory=list)
    toolbox: list[str] = Field(default_factory=list)
    output: str
    model: str
    guidance: str | None = None
    runtime: RuntimeConfig | None = None

    @model_validator(mode="after")
    def validate_tool_sources(self) -> LlmAgentSpec:
        if not self.tools and not self.toolbox:
            raise ValueError("At least one of 'tools' or 'toolbox' must be non-empty")
        return self


class ServiceAgentSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    type: Literal["service_agent"]
    project: str
    service: ServiceConfig
    input: AgentInputSpec
    output: str


AgentSpec: TypeAlias = Annotated[
    LlmAgentSpec | ServiceAgentSpec,
    Field(discriminator="type"),
]

AGENT_SPEC_ADAPTER: TypeAdapter[AgentSpec] = TypeAdapter(AgentSpec)


def validate_agent_spec(data: object) -> AgentSpec:
    """Validate raw YAML data into a concrete agent spec."""
    return AGENT_SPEC_ADAPTER.validate_python(data)
