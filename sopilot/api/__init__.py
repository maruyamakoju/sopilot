"""FastAPI routers."""

# Paths exempt from authentication and rate-limiting.
PUBLIC_PATHS: frozenset[str] = frozenset({
    "/",
    "/health",
    "/readiness",
    "/metrics",
    "/status",
    "/docs",
    "/redoc",
    "/openapi.json",
})
