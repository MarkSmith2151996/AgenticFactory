# AgenticFactory Architecture

## V1 Scope

AgenticFactory v1 is intentionally small:

- one manual HTTP invocation endpoint: `POST /invoke_agent`
- one loopback listener: `127.0.0.1:8080`
- one AOODB SQLite database: `data/aoodb.db`
- one manual FBA screening agent: `brand-amazon-screener`
- one Custodian MCP bridge tool: `invoke_agent`

Everything else is deferred: scheduling, triggers, pipelines, organizer/cache, broader browser automation, per-project MCP servers, and eval infrastructure.

II-006 adds the first project-scoped MCP server as an exception to the original CP1 boundary: FBA lives at `mcp/fba/` and invokes the runtime HTTP API. This proves the per-project MCP pattern before moving that server into a dedicated FBA repo.

## Runtime Flow

```text
Claude/OpenCode MCP client
  -> FBA MCP server invoke_agent(name, input)
  -> HTTP POST http://127.0.0.1:8080/invoke_agent
  -> AgenticFactory FastAPI endpoint
  -> AOODB row status=running
  -> agent dispatcher
  -> AOODB row status=success/error
  -> JSON response returned through the MCP bridge
```

## AOODB

The AOODB schema lives inside `agentic_factory.db` and is applied idempotently whenever the runtime opens a connection. Invocation ids use `AI-NNNNN` with zero padding.

For degraded local-browser runs, token and cost fields remain `NULL`. Successful structured analysis runs populate token fields for the `openai/gpt-5.4` phase.

## Dependencies

`pyproject.toml` pins `pydantic-ai==1.97.0`, which was the latest available version from `pip index versions pydantic-ai` during CP1.

## FBA MCP Server

- Location: `mcp/fba/`
- Launcher: `bin/fba-mcp-serve`
- Local bind: `127.0.0.1:8090`
- Public hostname target: `https://fba.lamannalogistics.com/mcp`
- OAuth DB: `data/fba_mcp_oauth.db` (gitignored)
- Initial tool: `invoke_agent(name: str, input: dict)`

The FBA MCP server uses the official MCP Python SDK (`mcp.server.Server` plus `StreamableHTTPSessionManager`) and Starlette, matching the Custodian HTTP MCP pattern. FastMCP was not used because the existing Workbench pattern already has explicit OAuth route wiring, protected-resource metadata, and a `/tool` verification route that are useful for this proof of concept.

Port `8090` was chosen because the runtime API already owns `8080` and Custodian HTTP MCP owns `8223`.
