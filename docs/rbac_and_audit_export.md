# RBAC and Signed Audit Export

## Roles

- `viewer`: read-only APIs (`/videos`, `/score/{id}`, `/search`, `/audit/trail`, etc.)
- `operator`: viewer + ingest and score creation, queue metrics
- `admin`: operator + destructive/admin operations (`/videos/{id}` delete, `/train/nightly`, signed audit export)

## Environment Variables

- `SOPILOT_API_TOKEN`
- `SOPILOT_AUTH_REQUIRED=true|false`
- `SOPILOT_API_TOKEN_ROLE=admin|operator|viewer`
- `SOPILOT_API_ROLE_TOKENS=admin:tokenA,operator:tokenB,viewer:tokenC`
- `SOPILOT_BASIC_USER`
- `SOPILOT_BASIC_PASSWORD`
- `SOPILOT_BASIC_ROLE=admin|operator|viewer`

When `SOPILOT_AUTH_REQUIRED=true` and no credentials are configured, non-public APIs return `503`.
For local development-only bypass, set `SOPILOT_AUTH_REQUIRED=false` and use `SOPILOT_AUTH_DEFAULT_ROLE`.

## Signed Audit Export

1. Configure signing key:
   - `SOPILOT_AUDIT_SIGNING_KEY=<secret>`
   - `SOPILOT_AUDIT_SIGNING_KEY_ID=<key-id>`
2. Call:
   - `POST /audit/export?limit=500`
3. Download:
   - `GET /audit/export/{export_id}/file`

The export contains HMAC-SHA256 signature metadata:

- `algorithm`
- `key_id`
- `payload_sha256`
- `signature_hex`

To verify, recompute canonical JSON for payload excluding `signature`, then compare both digest and HMAC.
