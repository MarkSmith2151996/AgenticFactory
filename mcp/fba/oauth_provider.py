"""Persistent OAuth provider for the FBA MCP server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import bcrypt
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "fba_mcp_oauth.db"
ACCESS_TOKEN_PREFIX = "fba_at_"
REFRESH_TOKEN_PREFIX = "fba_rt_"
AUTH_CODE_PREFIX = "fba_code_"
DEFAULT_SCOPES = ["fba.full"]
ACCESS_TOKEN_TTL = 3600
AUTH_CODE_TTL = 600


class FBAAccessToken(AccessToken):
    token_id: str
    family_id: str | None = None


class FBARefreshToken(RefreshToken):
    token_id: str
    family_id: str | None = None
    resource: str | None = None


class FBAAuthorizationCode(AuthorizationCode):
    code_id: str
    code_hash: str


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_oauth_schema() -> None:
    """Create OAuth tables if they do not already exist."""
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS oauth_clients (
                id TEXT PRIMARY KEY,
                name TEXT,
                secret TEXT,
                redirect_uris TEXT NOT NULL,
                token_endpoint_auth_method TEXT NOT NULL DEFAULT 'none',
                grant_types TEXT NOT NULL,
                response_types TEXT NOT NULL,
                scope TEXT,
                client_uri TEXT,
                logo_uri TEXT,
                contacts TEXT,
                tos_uri TEXT,
                policy_uri TEXT,
                jwks_uri TEXT,
                jwks TEXT,
                software_id TEXT,
                software_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                client_id_issued_at INTEGER,
                secret_expires_at INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS oauth_tokens (
                id TEXT PRIMARY KEY,
                family_id TEXT,
                token_hash TEXT NOT NULL,
                token_type TEXT NOT NULL,
                client_id TEXT NOT NULL REFERENCES oauth_clients(id),
                scopes TEXT NOT NULL,
                resource TEXT,
                expires_at INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                revoked INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS oauth_auth_codes (
                code_hash TEXT PRIMARY KEY,
                code_id TEXT NOT NULL UNIQUE,
                client_id TEXT NOT NULL REFERENCES oauth_clients(id),
                redirect_uri TEXT NOT NULL,
                redirect_uri_provided_explicitly INTEGER DEFAULT 1,
                code_challenge TEXT NOT NULL,
                code_challenge_method TEXT NOT NULL,
                scopes TEXT NOT NULL,
                resource TEXT,
                expires_at INTEGER NOT NULL,
                used INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_oauth_clients_name ON oauth_clients(name);
            CREATE INDEX IF NOT EXISTS idx_oauth_tokens_hash ON oauth_tokens(token_hash);
            CREATE INDEX IF NOT EXISTS idx_oauth_tokens_client_type ON oauth_tokens(client_id, token_type);
            CREATE INDEX IF NOT EXISTS idx_oauth_tokens_family ON oauth_tokens(family_id);
            CREATE INDEX IF NOT EXISTS idx_oauth_auth_codes_expires ON oauth_auth_codes(expires_at);
            """
        )
        conn.commit()


def _json_default(obj: Any) -> str:
    # Pydantic URL types stringify cleanly but are not stdlib JSON serializable.
    if hasattr(obj, "__str__"):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=_json_default)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    return json.loads(value)


def _normalize_scopes(scopes: list[str] | None, fallback: list[str] | None = None) -> list[str]:
    if scopes:
        return list(scopes)
    if fallback:
        return list(fallback)
    return list(DEFAULT_SCOPES)


def _hash_value(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return bcrypt.hashpw(digest, bcrypt.gensalt()).decode("utf-8")


def _verify_hash(value: str, stored_hash: str) -> bool:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return bcrypt.checkpw(digest, stored_hash.encode("utf-8"))


def _build_secret_value(prefix: str) -> tuple[str, str, str]:
    record_id = uuid4().hex
    secret = secrets.token_hex(32)
    plain = f"{prefix}{record_id}.{secret}"
    return record_id, secret, plain


def _extract_token_id(prefix: str, value: str) -> str | None:
    if not value.startswith(prefix):
        return None
    token_id, sep, _secret = value[len(prefix) :].partition(".")
    if not sep or not token_id:
        return None
    return token_id


def _row_to_client(row: sqlite3.Row) -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=row["id"],
        client_secret=row["secret"],
        client_id_issued_at=row["client_id_issued_at"],
        client_secret_expires_at=row["secret_expires_at"],
        redirect_uris=_json_loads(row["redirect_uris"], []),
        token_endpoint_auth_method=row["token_endpoint_auth_method"],
        grant_types=_json_loads(row["grant_types"], ["authorization_code", "refresh_token"]),
        response_types=_json_loads(row["response_types"], ["code"]),
        scope=row["scope"],
        client_name=row["name"],
        client_uri=row["client_uri"],
        logo_uri=row["logo_uri"],
        contacts=_json_loads(row["contacts"], None),
        tos_uri=row["tos_uri"],
        policy_uri=row["policy_uri"],
        jwks_uri=row["jwks_uri"],
        jwks=_json_loads(row["jwks"], None),
        software_id=row["software_id"],
        software_version=row["software_version"],
    )


class FBAOAuthProvider(
    OAuthAuthorizationServerProvider[
        FBAAuthorizationCode,
        FBARefreshToken,
        FBAAccessToken,
    ]
):
    def __init__(self, resource_server_url: str, default_scopes: list[str] | None = None):
        self.resource_server_url = resource_server_url
        self.default_scopes = default_scopes or list(DEFAULT_SCOPES)

    async def ensure_schema(self) -> None:
        await asyncio.to_thread(ensure_oauth_schema)

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return await asyncio.to_thread(self._get_client_sync, client_id)

    def _get_client_sync(self, client_id: str) -> OAuthClientInformationFull | None:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM oauth_clients WHERE id = ?", (client_id,)).fetchone()
            if row is None:
                return None
            return _row_to_client(row)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        await asyncio.to_thread(self._register_client_sync, client_info)

    def _register_client_sync(self, client_info: OAuthClientInformationFull) -> None:
        now = int(time.time())
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO oauth_clients (
                    id, name, secret, redirect_uris, token_endpoint_auth_method,
                    grant_types, response_types, scope, client_uri, logo_uri,
                    contacts, tos_uri, policy_uri, jwks_uri, jwks,
                    software_id, software_version, client_id_issued_at, secret_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    secret=excluded.secret,
                    redirect_uris=excluded.redirect_uris,
                    token_endpoint_auth_method=excluded.token_endpoint_auth_method,
                    grant_types=excluded.grant_types,
                    response_types=excluded.response_types,
                    scope=excluded.scope,
                    client_uri=excluded.client_uri,
                    logo_uri=excluded.logo_uri,
                    contacts=excluded.contacts,
                    tos_uri=excluded.tos_uri,
                    policy_uri=excluded.policy_uri,
                    jwks_uri=excluded.jwks_uri,
                    jwks=excluded.jwks,
                    software_id=excluded.software_id,
                    software_version=excluded.software_version,
                    secret_expires_at=excluded.secret_expires_at
                """,
                (
                    client_info.client_id,
                    client_info.client_name,
                    client_info.client_secret,
                    _json_dumps(client_info.redirect_uris or []),
                    client_info.token_endpoint_auth_method or "none",
                    _json_dumps(client_info.grant_types or ["authorization_code", "refresh_token"]),
                    _json_dumps(client_info.response_types or ["code"]),
                    client_info.scope,
                    client_info.client_uri,
                    client_info.logo_uri,
                    _json_dumps(client_info.contacts) if client_info.contacts is not None else None,
                    client_info.tos_uri,
                    client_info.policy_uri,
                    client_info.jwks_uri,
                    _json_dumps(client_info.jwks) if client_info.jwks is not None else None,
                    client_info.software_id,
                    client_info.software_version,
                    client_info.client_id_issued_at or now,
                    client_info.client_secret_expires_at or 0,
                ),
            )
            conn.commit()

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        return await asyncio.to_thread(self._authorize_sync, client, params)

    def _authorize_sync(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        code_id, code_secret, plain_code = _build_secret_value(AUTH_CODE_PREFIX)
        scopes = _normalize_scopes(params.scopes, self.default_scopes)
        now = int(time.time())
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO oauth_auth_codes (
                    code_hash, code_id, client_id, redirect_uri,
                    redirect_uri_provided_explicitly, code_challenge,
                    code_challenge_method, scopes, resource, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _hash_value(code_secret),
                    code_id,
                    client.client_id,
                    str(params.redirect_uri),
                    1,
                    params.code_challenge,
                    "S256",
                    _json_dumps(scopes),
                    str(params.resource) if params.resource else self.resource_server_url,
                    now + AUTH_CODE_TTL,
                ),
            )
            conn.commit()
        return construct_redirect_uri(str(params.redirect_uri), code=plain_code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> FBAAuthorizationCode | None:
        return await asyncio.to_thread(self._load_authorization_code_sync, client, authorization_code)

    def _load_authorization_code_sync(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> FBAAuthorizationCode | None:
        code_id = _extract_token_id(AUTH_CODE_PREFIX, authorization_code)
        if not code_id:
            return None
        _prefix, _sep, secret = authorization_code.partition(".")
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM oauth_auth_codes WHERE code_id = ? AND client_id = ? AND used = 0",
                (code_id, client.client_id),
            ).fetchone()
            if row is None or row["expires_at"] < int(time.time()):
                return None
            if not _verify_hash(secret, row["code_hash"]):
                return None
            return FBAAuthorizationCode(
                code=authorization_code,
                code_id=row["code_id"],
                code_hash=row["code_hash"],
                scopes=_json_loads(row["scopes"], self.default_scopes),
                expires_at=row["expires_at"],
                client_id=row["client_id"],
                redirect_uri=row["redirect_uri"],
                redirect_uri_provided_explicitly=bool(row["redirect_uri_provided_explicitly"]),
                code_challenge=row["code_challenge"],
                resource=row["resource"],
            )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: FBAAuthorizationCode
    ) -> OAuthToken:
        return await asyncio.to_thread(self._exchange_authorization_code_sync, client, authorization_code)

    def _exchange_authorization_code_sync(
        self, client: OAuthClientInformationFull, authorization_code: FBAAuthorizationCode
    ) -> OAuthToken:
        with get_db() as conn:
            cursor = conn.execute(
                "UPDATE oauth_auth_codes SET used = 1 WHERE code_hash = ? AND used = 0",
                (authorization_code.code_hash,),
            )
            if cursor.rowcount == 0:
                raise TokenError("invalid_grant", "authorization code already used")
            token = self._issue_token_pair_sync(
                conn,
                client_id=client.client_id or "",
                scopes=authorization_code.scopes,
                resource=authorization_code.resource,
            )
            conn.commit()
            return token

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> OAuthToken:
        return await asyncio.to_thread(self._exchange_refresh_token_sync, client, refresh_token)

    def _exchange_refresh_token_sync(self, client: OAuthClientInformationFull, refresh_token: str) -> OAuthToken:
        token_id = _extract_token_id(REFRESH_TOKEN_PREFIX, refresh_token)
        if not token_id:
            raise TokenError("invalid_grant", "invalid refresh token")
        _prefix, _sep, secret = refresh_token.partition(".")
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT * FROM oauth_tokens
                WHERE id = ? AND token_type = 'refresh' AND revoked = 0 AND client_id = ?
                """,
                (token_id, client.client_id),
            ).fetchone()
            if row is None or not _verify_hash(secret, row["token_hash"]):
                raise TokenError("invalid_grant", "invalid refresh token")
            conn.execute("UPDATE oauth_tokens SET revoked = 1 WHERE family_id = ?", (row["family_id"],))
            token = self._issue_token_pair_sync(
                conn,
                client_id=row["client_id"],
                scopes=_json_loads(row["scopes"], self.default_scopes),
                resource=row["resource"],
            )
            conn.commit()
            return token

    async def load_access_token(self, token: str) -> FBAAccessToken | None:
        return await asyncio.to_thread(self._load_access_token_sync, token)

    def _load_access_token_sync(self, token: str) -> FBAAccessToken | None:
        token_id = _extract_token_id(ACCESS_TOKEN_PREFIX, token)
        if not token_id:
            return None
        _prefix, _sep, secret = token.partition(".")
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM oauth_tokens WHERE id = ? AND token_type = 'access' AND revoked = 0",
                (token_id,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] and row["expires_at"] < int(time.time()):
                return None
            if not _verify_hash(secret, row["token_hash"]):
                return None
            return FBAAccessToken(
                token=token,
                token_id=row["id"],
                family_id=row["family_id"],
                client_id=row["client_id"],
                scopes=_json_loads(row["scopes"], self.default_scopes),
                expires_at=row["expires_at"],
                resource=row["resource"],
            )

    async def revoke_token(self, token: str, token_type_hint: str | None = None) -> None:
        await asyncio.to_thread(self._revoke_token_sync, token, token_type_hint)

    def _revoke_token_sync(self, token: str, token_type_hint: str | None = None) -> None:
        prefix = REFRESH_TOKEN_PREFIX if token_type_hint == "refresh_token" else ACCESS_TOKEN_PREFIX
        token_id = _extract_token_id(prefix, token)
        if not token_id and token_type_hint != "refresh_token":
            token_id = _extract_token_id(REFRESH_TOKEN_PREFIX, token)
        if not token_id:
            return
        with get_db() as conn:
            row = conn.execute("SELECT family_id FROM oauth_tokens WHERE id = ?", (token_id,)).fetchone()
            if row and row["family_id"]:
                conn.execute("UPDATE oauth_tokens SET revoked = 1 WHERE family_id = ?", (row["family_id"],))
            else:
                conn.execute("UPDATE oauth_tokens SET revoked = 1 WHERE id = ?", (token_id,))
            conn.commit()

    def _issue_token_pair_sync(
        self,
        conn: sqlite3.Connection,
        *,
        client_id: str,
        scopes: list[str],
        resource: str | None,
    ) -> OAuthToken:
        family_id = uuid4().hex
        access_id, access_secret, access_plain = _build_secret_value(ACCESS_TOKEN_PREFIX)
        refresh_id, refresh_secret, refresh_plain = _build_secret_value(REFRESH_TOKEN_PREFIX)
        expires_at = int(time.time()) + ACCESS_TOKEN_TTL
        conn.execute(
            """
            INSERT INTO oauth_tokens (
                id, family_id, token_hash, token_type, client_id,
                scopes, resource, expires_at
            ) VALUES (?, ?, ?, 'access', ?, ?, ?, ?)
            """,
            (access_id, family_id, _hash_value(access_secret), client_id, _json_dumps(scopes), resource, expires_at),
        )
        conn.execute(
            """
            INSERT INTO oauth_tokens (
                id, family_id, token_hash, token_type, client_id,
                scopes, resource, expires_at
            ) VALUES (?, ?, ?, 'refresh', ?, ?, ?, NULL)
            """,
            (refresh_id, family_id, _hash_value(refresh_secret), client_id, _json_dumps(scopes), resource),
        )
        return OAuthToken(
            access_token=access_plain,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL,
            refresh_token=refresh_plain,
            scope=" ".join(scopes),
        )


async def create_or_rotate_named_client(
    name: str,
    *,
    rotate_secret: bool = False,
    redirect_uris: list[str] | None = None,
) -> tuple[str, str | None, bool, str]:
    return await asyncio.to_thread(
        _create_or_rotate_named_client_sync,
        name,
        rotate_secret,
        redirect_uris,
    )


def _create_or_rotate_named_client_sync(
    name: str,
    rotate_secret: bool,
    redirect_uris: list[str] | None,
) -> tuple[str, str | None, bool, str]:
    ensure_oauth_schema()
    now = int(time.time())
    normalized_redirects = list(redirect_uris or [])
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, created_at FROM oauth_clients WHERE name = ? ORDER BY created_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        if row is None:
            client_id = uuid4().hex
            client_secret = secrets.token_hex(32)
            conn.execute(
                """
                INSERT INTO oauth_clients (
                    id, name, secret, redirect_uris, token_endpoint_auth_method,
                    grant_types, response_types, scope, client_id_issued_at, secret_expires_at
                ) VALUES (?, ?, ?, ?, 'client_secret_post', ?, ?, ?, ?, 0)
                """,
                (
                    client_id,
                    name,
                    client_secret,
                    _json_dumps(normalized_redirects),
                    _json_dumps(["authorization_code", "refresh_token"]),
                    _json_dumps(["code"]),
                    " ".join(DEFAULT_SCOPES),
                    now,
                ),
            )
            conn.commit()
            return client_id, client_secret, True, "created"

        client_id = row["id"]
        created_at = row["created_at"]
        if not rotate_secret:
            return client_id, None, False, created_at

        client_secret = secrets.token_hex(32)
        if redirect_uris is None:
            conn.execute(
                "UPDATE oauth_clients SET secret = ?, secret_expires_at = 0 WHERE id = ?",
                (client_secret, client_id),
            )
        else:
            conn.execute(
                "UPDATE oauth_clients SET secret = ?, redirect_uris = ?, secret_expires_at = 0 WHERE id = ?",
                (client_secret, _json_dumps(normalized_redirects), client_id),
            )
        conn.commit()
        return client_id, client_secret, False, "rotated"
