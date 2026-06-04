# AgenticFactory

AgenticFactory is a local loopback runtime for declarative agents. The v1 slice hosts one manual invocation endpoint, records each run in its own SQLite AOODB, and exposes project-scoped MCP servers that can invoke runtime agents.

This repo also temporarily hosts experimental project MCP servers under `mcp/`. That is tactical: once the pattern is proven, project MCP servers can move into their owning repos, such as `fba-command-center` for FBA-specific tools.

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
bin/runtime-serve
```

The server binds to `127.0.0.1:8080` and exposes `POST /invoke_agent` with a JSON body like:

```json
{"name": "brand-amazon-screener", "input": {"brand": "Corelle"}}
```

The response includes the AOODB invocation id and the agent output. Invocation rows are stored in `data/aoodb.db`, which is intentionally gitignored.

## FBA MCP Server

Start the FBA project MCP server on `127.0.0.1:8090`:

```bash
. .venv/bin/activate
bin/fba-mcp-serve
```

The server exposes `/mcp` for streamable HTTP MCP clients and `/tool` for simple authenticated JSON verification. OAuth clients are registered separately from Custodian:

```bash
bin/register-fba-oauth-client --name claude-desktop --redirect-uri https://claude.ai/api/mcp/auth_callback
```
