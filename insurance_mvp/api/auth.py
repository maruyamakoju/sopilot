"""Authentication and Authorization

Simple API key authentication for MVP.
Production systems should use OAuth2/JWT.
"""

import hashlib
import secrets
from datetime import datetime
from typing import Any

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# API Key Configuration


class APIKey(BaseModel):
    """API Key metadata"""

    key_hash: str
    name: str
    created_at: datetime
    permissions: dict[str, bool]
    active: bool = True


class APIKeyStore:
    """
    Simple in-memory API key store.

    Production: Replace with database storage + Redis caching.
    """

    def __init__(self):
        self._keys: dict[str, APIKey] = {}

    def generate_key(self, name: str, permissions: dict[str, bool] | None = None) -> str:
        """
        Generate new API key.

        Args:
            name: Human-readable key name (e.g., "frontend-app", "reviewer-dashboard")
            permissions: Permission dict (e.g., {"read": True, "write": True})

        Returns:
            Plain-text API key (only shown once!)
        """
        # Generate cryptographically secure random key
        api_key = f"ins_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(api_key)

        # Default permissions: read-only
        if permissions is None:
            permissions = {"read": True, "write": False, "admin": False}

        # Store hashed key
        self._keys[key_hash] = APIKey(
            key_hash=key_hash,
            name=name,
            created_at=datetime.utcnow(),
            permissions=permissions,
            active=True,
        )

        return api_key

    def validate_key(self, api_key: str) -> APIKey | None:
        """
        Validate API key and return metadata.

        Args:
            api_key: Plain-text API key

        Returns:
            APIKey metadata if valid, None otherwise
        """
        key_hash = self._hash_key(api_key)
        key_meta = self._keys.get(key_hash)

        if key_meta and key_meta.active:
            return key_meta
        return None

    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke API key.

        Returns:
            True if revoked, False if not found
        """
        key_hash = self._hash_key(api_key)
        key_meta = self._keys.get(key_hash)

        if key_meta:
            key_meta.active = False
            return True
        return False

    def _hash_key(self, api_key: str) -> str:
        """Hash API key using SHA256"""
        return hashlib.sha256(api_key.encode()).hexdigest()


# Global key store (singleton)
api_key_store = APIKeyStore()


# Initialize with default keys for development
def initialize_dev_keys():
    """
    Initialize development API keys.

    WARNING: These keys are for development only!
    Production keys should be managed via environment variables or secret management.
    """
    # Full access key (for testing)
    dev_key = api_key_store.generate_key(
        name="dev-full-access", permissions={"read": True, "write": True, "admin": True}
    )
    print(f"[DEV] Full access API key: {dev_key}")

    # Read-only key
    readonly_key = api_key_store.generate_key(
        name="dev-readonly", permissions={"read": True, "write": False, "admin": False}
    )
    print(f"[DEV] Read-only API key: {readonly_key}")

    return {"full_access": dev_key, "readonly": readonly_key}


# FastAPI Security Dependencies

# API Key Header: X-API-Key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Bearer Token (alternative)
bearer_scheme = HTTPBearer(auto_error=False)


async def get_api_key(
    api_key_header: str | None = Security(api_key_header),
    bearer_token: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> APIKey:
    """
    Validate API key from header or bearer token.

    Usage in route:
        @app.get("/protected")
        async def protected_route(api_key: APIKey = Depends(get_api_key)):
            return {"message": f"Authenticated as {api_key.name}"}
    """
    # Try header first
    api_key_str = api_key_header

    # Fall back to bearer token
    if not api_key_str and bearer_token:
        api_key_str = bearer_token.credentials

    if not api_key_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate key
    key_meta = api_key_store.validate_key(api_key_str)

    if not key_meta:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return key_meta


async def get_api_key_optional(
    api_key_header: str | None = Security(api_key_header),
    bearer_token: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> APIKey | None:
    """
    Optional authentication (for public endpoints with optional enhanced access).

    Returns None if no valid API key provided.
    """
    try:
        return await get_api_key(api_key_header, bearer_token)
    except HTTPException:
        return None


# Permission Checks


def require_permission(permission: str):
    """
    Dependency factory for permission checks.

    Usage:
        @app.post("/admin/reset", dependencies=[Depends(require_permission("admin"))])
        async def admin_reset():
            return {"message": "Database reset"}
    """

    async def check_permission(api_key: APIKey = Security(get_api_key)):
        if not api_key.permissions.get(permission, False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}",
            )
        return api_key

    return check_permission


# Rate Limiting (Simple Token Bucket)


class RateLimiter:
    """
    Simple in-memory rate limiter using token bucket algorithm.

    Production: Replace with Redis-based rate limiting.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.buckets: dict[str, dict[str, Any]] = {}

    def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if request is within rate limit.

        Args:
            identifier: Unique identifier (e.g., API key, IP address)

        Returns:
            True if allowed, False if rate limited
        """
        now = datetime.utcnow()

        if identifier not in self.buckets:
            self.buckets[identifier] = {
                "tokens": self.requests_per_minute,
                "last_update": now,
            }

        bucket = self.buckets[identifier]

        # Refill tokens based on time elapsed
        time_elapsed = (now - bucket["last_update"]).total_seconds() / 60.0
        bucket["tokens"] = min(self.requests_per_minute, bucket["tokens"] + time_elapsed * self.requests_per_minute)
        bucket["last_update"] = now

        # Check if token available
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False


# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=60)


async def check_rate_limit(api_key: APIKey = Security(get_api_key)):
    """
    Rate limiting dependency.

    Usage:
        @app.post("/claims/upload", dependencies=[Depends(check_rate_limit)])
        async def upload_claim(...):
            ...
    """
    if not rate_limiter.check_rate_limit(api_key.key_hash):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 60 requests per minute.",
            headers={"Retry-After": "60"},
        )


# Reviewer Authentication (for human review endpoints)


class ReviewerStore:
    """
    Simple reviewer authentication.

    Production: Integrate with corporate SSO (SAML, LDAP, OAuth2).
    """

    def __init__(self):
        self._reviewers: dict[str, dict[str, Any]] = {}

    def create_reviewer(self, reviewer_id: str, name: str, password: str) -> bool:
        """Create reviewer account (hashed password)"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        self._reviewers[reviewer_id] = {
            "name": name,
            "password_hash": password_hash,
            "active": True,
        }
        return True

    def authenticate(self, reviewer_id: str, password: str) -> bool:
        """Authenticate reviewer"""
        reviewer = self._reviewers.get(reviewer_id)
        if not reviewer or not reviewer["active"]:
            return False

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == reviewer["password_hash"]


# Global reviewer store
reviewer_store = ReviewerStore()


def initialize_dev_reviewers():
    """Initialize development reviewers"""
    reviewer_store.create_reviewer("reviewer_alice", "Alice Smith", "dev_password_123")
    reviewer_store.create_reviewer("reviewer_bob", "Bob Johnson", "dev_password_456")
    print("[DEV] Reviewers initialized: reviewer_alice, reviewer_bob")


# Audit Logging Helper


def get_actor_info(api_key: APIKey | None = None) -> tuple[str, str]:
    """
    Extract actor info for audit logging.

    Returns:
        (actor_type, actor_id)
    """
    if api_key:
        return ("API", api_key.name)
    return ("SYSTEM", "internal")
