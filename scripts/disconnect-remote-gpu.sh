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

# Connection name can be provided as:
#  - first arg (preferred): ./disconnect-remote-gpu.sh <conn-name>
#  - env var: TF_CONN_NAME
TF_CONN_NAME="${TF_CONN_NAME:-${1:-}}"

if [[ -z "$TF_API_KEY" ]]; then
  echo "error: TF_API_KEY is required" >&2
  exit 1
fi
if [[ -z "$TF_CONN_NAME" ]]; then
  echo "error: connection name is required" >&2
  echo "usage: TF_API_KEY=... $0 <connection-name>" >&2
  echo "  or:  TF_API_KEY=... TF_CONN_NAME=<connection-name> $0" >&2
  exit 1
fi

echo "Disconnecting: ${TF_NAMESPACE}/${TF_CONN_NAME}" >&2

resp="$(
  curl -sS -X DELETE "${TF_ENDPOINT}/api/v1/external-connections/${TF_NAMESPACE}/${TF_CONN_NAME}" \
    -H "X-API-Key: ${TF_API_KEY}"
)"

echo "$resp" | jq .

echo "Disconnected: ${TF_NAMESPACE}/${TF_CONN_NAME}" >&2
