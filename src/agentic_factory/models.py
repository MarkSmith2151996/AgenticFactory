"""HTTP request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class InvokeAgentRequest(BaseModel):
    name: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)


class InvokeAgentResponse(BaseModel):
    invocation_id: str
    status: Literal["running", "success", "error", "timeout"]
    output: dict[str, Any] | None = None
    error: str | None = None
