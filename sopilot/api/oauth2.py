"""OAuth2 JWT authentication for SOPilot.

Provides:
- ``POST /auth/token`` — issue a JWT from client_id + client_secret
- ``decode_token(token)`` — validate and decode a JWT
- ``require_role(*roles)`` — FastAPI dependency for role-based access control

Client configuration is static (env var ``SOPILOT_OAUTH_CLIENTS``), keeping
the on-prem deployment simple with no DB table for clients.

Example clients JSON::

    [
      {"client_id": "admin-ui", "client_secret_hash": "<bcrypt>", "role": "admin"},
      {"client_id": "camera-01", "client_secret_hash": "<bcrypt>", "role": "viewer"}
    ]

Generate a bcrypt hash::

    python -c "from passlib.hash import bcrypt; print(bcrypt.hash('mysecret'))"

Auth resolution in middleware (additive — existing X-API-Key still works):
    1. X-API-Key header → ApiKeyMiddleware (unchanged)
    2. Authorization: Bearer <jwt> → decoded here; attaches auth_client_id + auth_role to request.state
    3. If neither and route requires role → 401
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Token schemas ──────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ── JWT helpers ────────────────────────────────────────────────────────────

def _encode_token(payload: dict[str, Any], secret: str) -> str:
    try:
        from jose import jwt  # type: ignore[import-untyped]
        return jwt.encode(payload, secret, algorithm="HS256")
    except ImportError as exc:
        raise ImportError(
            "python-jose[cryptography] is required for JWT support. "
            "Install it with: pip install 'python-jose[cryptography]'"
        ) from exc


def decode_token(token: str, secret: str) -> dict[str, Any]:
    """Decode and validate a JWT.  Raises ``HTTPException`` on any failure."""
    try:
        from jose import JWTError, jwt  # type: ignore[import-untyped]
        payload: dict[str, Any] = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT library not installed",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _verify_secret(plain: str, hashed: str) -> bool:
    """Verify a plain-text secret against a bcrypt hash."""
    try:
        from passlib.hash import bcrypt  # type: ignore[import-untyped]
        return bcrypt.verify(plain, hashed)
    except ImportError:
        # Fallback: plain equality (development only — always log a warning)
        logger.warning(
            "passlib[bcrypt] not installed — using plain-text secret comparison (insecure)"
        )
        return plain == hashed
    except Exception:
        return False


def _load_clients(clients_json: str) -> list[dict[str, Any]]:
    try:
        clients = json.loads(clients_json)
        if not isinstance(clients, list):
            return []
        return clients
    except Exception:
        return []


# ── FastAPI dependency for role-based access control ───────────────────────

def require_role(*roles: str):  # type: ignore[return]
    """FastAPI dependency that enforces role membership.

    Reads the JWT from the ``Authorization: Bearer`` header and verifies the
    embedded role is in ``roles``.  Returns the decoded payload on success.

    Example::

        @router.get("/admin/audit-log")
        async def audit_log(request: Request, _=Depends(require_role("admin"))):
            ...
    """
    async def _check(request: Request) -> dict[str, Any]:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth_header.removeprefix("Bearer ").strip()
        settings = request.app.state.settings
        payload = decode_token(token, settings.jwt_secret)

        token_role = payload.get("role", "")
        if token_role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{token_role}' not authorised; required: {roles}",
            )
        # Attach identity to request state for AuditMiddleware
        request.state.auth_client_id = payload.get("sub", "unknown")
        request.state.auth_role = token_role
        return payload

    return Depends(_check)


# ── Router factory ─────────────────────────────────────────────────────────

def build_oauth2_router(settings: Any) -> APIRouter:
    router = APIRouter(tags=["Auth"])

    @router.post(
        "/auth/token",
        response_model=TokenResponse,
        summary="Issue JWT access token",
        description="Exchange client_id + client_secret for a short-lived JWT. "
                    "Use the returned ``access_token`` in ``Authorization: Bearer <token>``.",
    )
    async def issue_token(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
        clients = _load_clients(settings.oauth_clients_json)
        matched = None
        for client in clients:
            if client.get("client_id") == form.username:
                if _verify_secret(form.password, client.get("client_secret_hash", "")):
                    matched = client
                    break

        if matched is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid client credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        expire_minutes = settings.jwt_expire_minutes
        now = int(time.time())
        payload = {
            "sub": matched["client_id"],
            "role": matched.get("role", "viewer"),
            "iat": now,
            "exp": now + expire_minutes * 60,
        }
        token = _encode_token(payload, settings.jwt_secret)
        return TokenResponse(access_token=token, expires_in=expire_minutes * 60)

    return router
