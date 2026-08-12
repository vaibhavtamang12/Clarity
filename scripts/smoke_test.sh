#!/usr/bin/env bash
# Smoke test: verify a running stack answers liveness + readiness probes.
set -euo pipefail

BASE="${BASE_URL:-http://localhost:8000}"

echo "==> liveness  (${BASE}/healthz)"
curl -fsS "${BASE}/healthz"
echo

echo "==> readiness (${BASE}/api/v1/health)"
curl -fsS "${BASE}/api/v1/health"
echo

echo "Smoke test passed."