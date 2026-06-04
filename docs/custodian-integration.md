# Custodian Integration

## Project Registration

Custodian stores registered projects in `nai-workbench/custodian/custodian.db`, table `projects`. The Admin TUI's registration helper is `custodian/admin.py::register_project(name, path, stack="")`.

The helper executes this idempotent SQLite pattern:

```sql
INSERT INTO projects (name, path, stack)
VALUES (?, ?, ?)
ON CONFLICT(name) DO UPDATE SET
    path = excluded.path,
    stack = excluded.stack,
    status = 'active';
```

For this repo, CP1 used the direct equivalent below because the helper is only exposed inside the NAI Workbench codebase, not as an MCP tool:

```bash
sqlite3 /home/dev/projects/nai-workbench/custodian/custodian.db "INSERT INTO projects (name, path, stack) VALUES ('agentic-factory', '/home/dev/projects/AgenticFactory', 'Python + FastAPI + Pydantic-AI + SQLite') ON CONFLICT(name) DO UPDATE SET path = excluded.path, stack = excluded.stack, status = 'active';"
```

Verification query:

```bash
sqlite3 /home/dev/projects/nai-workbench/custodian/custodian.db "SELECT name, path, stack, status FROM projects WHERE name = 'agentic-factory';"
```

Expected result:

```text
agentic-factory|/home/dev/projects/AgenticFactory|Python + FastAPI + Pydantic-AI + SQLite|active
```

## MCP Bridge History

II-005 CP1 temporarily added a Custodian MCP tool named `invoke_agent`. It posted the MCP payload to the AgenticFactory runtime at `http://127.0.0.1:8080/invoke_agent` and returned the runtime response unchanged.

II-006 replaces that proxy with a project-scoped FBA MCP server in AgenticFactory. Custodian should not own the runtime bridge long-term.

## FBA MCP OAuth Setup

Custodian's OAuth setup uses `custodian/register_oauth_client.py`, backed by SQLite tables and `custodian/oauth_provider.py`. The FBA MCP server follows the same local bootstrap pattern but stores credentials separately in `data/fba_mcp_oauth.db` and uses the `fba.full` scope.

Register or rotate an FBA client:

```bash
bin/register-fba-oauth-client --name claude-desktop --redirect-uri https://claude.ai/api/mcp/auth_callback
bin/register-fba-oauth-client --name claude-desktop --rotate-secret --redirect-uri https://claude.ai/api/mcp/auth_callback
```

The client secret is printed only when created or rotated.

## Cloudflare Route

The intended public route is:

```text
https://fba.lamannalogistics.com/mcp -> http://127.0.0.1:8090/mcp
```

This follows the existing Cloudflare tunnel pattern in `/mnt/c/Users/Big A/.cloudflared/config.yml`, where `custodian.lamannalogistics.com` routes to `127.0.0.1:8223`.
