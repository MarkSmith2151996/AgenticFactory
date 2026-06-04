"""AOODB persistence for AgenticFactory invocations."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "aoodb.db"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agent_invocations (
  id TEXT PRIMARY KEY,
  agent_name TEXT NOT NULL,
  agent_version TEXT NOT NULL,
  input_json TEXT NOT NULL,
  output_json TEXT,
  status TEXT NOT NULL,
  error_message TEXT,
  model_used TEXT NOT NULL,
  tokens_input INTEGER,
  tokens_output INTEGER,
  cost_usd REAL,
  created_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_invocations_agent
  ON agent_invocations(agent_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_invocations_status
  ON agent_invocations(status, created_at DESC);
"""


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_agent_version() -> str:
    """Return the current git short SHA, or dev when the repo has no commits yet."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "dev"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "dev"


def connect() -> sqlite3.Connection:
    """Open the AOODB connection and ensure the schema exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    """Context-managed AOODB connection."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def next_invocation_id(conn: sqlite3.Connection) -> str:
    """Generate the next AI-NNNNN invocation id."""
    row = conn.execute(
        "SELECT id FROM agent_invocations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return "AI-00001"
    try:
        numeric = int(str(row["id"]).split("-", 1)[1])
    except (IndexError, ValueError):
        numeric = 0
    return f"AI-{numeric + 1:05d}"


def create_invocation(
    *,
    agent_name: str,
    input_payload: dict[str, Any],
    model_used: str,
) -> str:
    """Create a running invocation row and return its id."""
    with db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        invocation_id = next_invocation_id(conn)
        conn.execute(
            """
            INSERT INTO agent_invocations (
              id, agent_name, agent_version, input_json, status, model_used, created_at
            ) VALUES (?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                invocation_id,
                agent_name,
                get_agent_version(),
                json.dumps(input_payload, sort_keys=True),
                model_used,
                utc_now(),
            ),
        )
        conn.commit()
        return invocation_id


def complete_invocation(
    invocation_id: str,
    output_payload: dict[str, Any],
    *,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
    cost_usd: float | None = None,
) -> None:
    """Mark an invocation as successful."""
    with db_connection() as conn:
        conn.execute(
            """
            UPDATE agent_invocations
            SET status = 'success', output_json = ?, tokens_input = ?, tokens_output = ?,
                cost_usd = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(output_payload, sort_keys=True),
                tokens_input,
                tokens_output,
                cost_usd,
                utc_now(),
                invocation_id,
            ),
        )
        conn.commit()


def fail_invocation(invocation_id: str, error_message: str) -> None:
    """Mark an invocation as errored."""
    with db_connection() as conn:
        conn.execute(
            """
            UPDATE agent_invocations
            SET status = 'error', error_message = ?, completed_at = ?
            WHERE id = ?
            """,
            (error_message, utc_now(), invocation_id),
        )
        conn.commit()
