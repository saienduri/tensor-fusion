#!/usr/bin/env bash
set -euo pipefail

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: missing required command: $1" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd jq

TF_ENDPOINT="${TF_ENDPOINT:-http://66.42.113.99:30080}"
TF_API_KEY="${TF_API_KEY:-}"
TF_NAMESPACE="${TF_NAMESPACE:-tensor-fusion}"
TF_POOL_NAME="${TF_POOL_NAME:-amd-remote-cluster-amd-remote-pool}"
TF_TTL_SECONDS="${TF_TTL_SECONDS:-21600}"
TF_CLIENT_ID="${TF_CLIENT_ID:-$(hostname 2>/dev/null || echo external-client)}"

TF_VRAM_REQUEST="${TF_VRAM_REQUEST:-4Gi}"
TF_VRAM_LIMIT="${TF_VRAM_LIMIT:-4Gi}"

TF_STUB_URL="${TF_STUB_URL:-https://github.com/saienduri/tensor-fusion/releases/download/external-client-v0.1.0/libhip_client_stub.so}"
TF_STUB_PATH="${TF_STUB_PATH:-./libhip_client_stub.so}"
TF_DEBUG="${TF_DEBUG:-0}"

if [[ -z "$TF_API_KEY" ]]; then
  echo "error: TF_API_KEY is required" >&2
  exit 1
fi

create_payload="$(
  jq -nc \
    --arg poolName "$TF_POOL_NAME" \
    --arg namespace "$TF_NAMESPACE" \
    --arg clientId "$TF_CLIENT_ID" \
    --argjson ttlSeconds "$TF_TTL_SECONDS" \
    --arg vramRequest "$TF_VRAM_REQUEST" \
    --arg vramLimit "$TF_VRAM_LIMIT" \
    '{
      poolName: $poolName,
      namespace: $namespace,
      clientId: $clientId,
      ttlSeconds: $ttlSeconds,
      resources: {
        vramRequest: $vramRequest,
        vramLimit: $vramLimit
      }
    }'
)"

resp="$(
  curl -sS -X POST "${TF_ENDPOINT}/api/v1/external-connections" \
    -H "X-API-Key: ${TF_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$create_payload"
)"

TF_CONN_NAME="$(echo "$resp" | jq -r .name)"
if [[ -z "$TF_CONN_NAME" || "$TF_CONN_NAME" == "null" ]]; then
  echo "error: failed to create connection; response:" >&2
  echo "$resp" | jq . >&2 || echo "$resp" >&2
  exit 1
fi

echo "Created connection: ${TF_NAMESPACE}/${TF_CONN_NAME}" >&2
echo "" >&2
echo "CONNECTION_NAME=${TF_CONN_NAME}" >&2

TF_CONNECTION_URL=""
while :; do
  get="$(
    curl -sS "${TF_ENDPOINT}/api/v1/external-connections/${TF_NAMESPACE}/${TF_CONN_NAME}" \
      -H "X-API-Key: ${TF_API_KEY}"
  )"
  TF_CONNECTION_URL="$(echo "$get" | jq -r .connectionURL)"
  st="$(echo "$get" | jq -r .status)"
  if [[ -n "$TF_CONNECTION_URL" && "$TF_CONNECTION_URL" != "null" ]]; then
    break
  fi
  echo "Waiting for connectionURL... status=${st}" >&2
  sleep 1
done

TF_WORKER_HOST="$(echo "$TF_CONNECTION_URL" | cut -d+ -f2)"
TF_WORKER_PORT="$(echo "$TF_CONNECTION_URL" | cut -d+ -f3)"

if [[ -z "$TF_WORKER_HOST" || -z "$TF_WORKER_PORT" ]]; then
  echo "error: unexpected connectionURL format: $TF_CONNECTION_URL" >&2
  exit 1
fi

if [[ ! -f "$TF_STUB_PATH" ]]; then
  echo "Downloading client stub to: $TF_STUB_PATH" >&2
  curl -fsSL -o "$TF_STUB_PATH" "$TF_STUB_URL"
  chmod +x "$TF_STUB_PATH" || true
fi

export TF_CONNECTION_URL
export TF_WORKER_HOST
export TF_WORKER_PORT
export LD_PRELOAD="$TF_STUB_PATH"
export TF_DEBUG

echo "TF_CONNECTION_URL=$TF_CONNECTION_URL" >&2
echo "TF_WORKER_HOST=$TF_WORKER_HOST" >&2
echo "TF_WORKER_PORT=$TF_WORKER_PORT" >&2
echo "LD_PRELOAD=$LD_PRELOAD" >&2
echo "TF_DEBUG=$TF_DEBUG" >&2

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

cat <<EOF

Connection ready. TTL is $TF_TTL_SECONDS seconds.

Set your environment variables and start hacking away:
  export LD_PRELOAD="$LD_PRELOAD"
  export TF_WORKER_HOST="$TF_WORKER_HOST"
  export TF_WORKER_PORT="$TF_WORKER_PORT"

"Disconnect command: ./disconnect-remote-gpu.sh "$TF_CONN_NAME"
EOF
