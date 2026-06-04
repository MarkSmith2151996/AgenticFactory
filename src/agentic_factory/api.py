"""FastAPI entrypoint for the AgenticFactory runtime."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from agentic_factory.agents import invoke_agent
from agentic_factory.db import complete_invocation, create_invocation, fail_invocation
from agentic_factory.models import InvokeAgentRequest, InvokeAgentResponse


DEFAULT_MODEL = "openai/gpt-5.4"

app = FastAPI(title="AgenticFactory Runtime", version="0.1.0")


@app.post("/invoke_agent", response_model=InvokeAgentResponse)
async def invoke_agent_endpoint(request: InvokeAgentRequest) -> InvokeAgentResponse:
    invocation_id = create_invocation(
        agent_name=request.name,
        input_payload=request.input,
        model_used=DEFAULT_MODEL,
    )
    try:
        result = await invoke_agent(request.name, request.input)
    except Exception as exc:
        error = str(exc)
        fail_invocation(invocation_id, error)
        return InvokeAgentResponse(
            invocation_id=invocation_id,
            status="error",
            output=None,
            error=error,
        )

    complete_invocation(
        invocation_id,
        result.output,
        tokens_input=result.tokens_input,
        tokens_output=result.tokens_output,
        cost_usd=result.cost_usd,
    )
    return InvokeAgentResponse(
        invocation_id=invocation_id,
        status="success",
        output=result.output,
        error=None,
    )


def main() -> None:
    uvicorn.run("agentic_factory.api:app", host="127.0.0.1", port=8080)


if __name__ == "__main__":
    main()
