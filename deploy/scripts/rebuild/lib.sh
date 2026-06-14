#!/usr/bin/env bash
# Shared helpers for rebuild scripts. Source this from each 0N-*.sh script.
# Usage: `source "$(dirname "$0")/lib.sh"`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.env"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "error: $CONFIG_FILE not found. Copy config.env.example to config.env and edit."
  exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

if [[ -z "${ES_PASSWORD:-}" ]]; then
  echo "error: ES_PASSWORD is empty in $CONFIG_FILE"
  exit 1
fi

: "${ES_URL:?ES_URL not set}"
: "${KB_URL:?KB_URL not set}"
: "${ES_USER:=elastic}"
: "${CURL_FLAGS:=-sk}"

# Colored output helpers
info()  { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; exit 1; }

# curl wrappers — call ES / Kibana with auth and xsrf as needed
es_get()    { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" "$ES_URL/$1"; }
es_put()    { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -X PUT    -H "Content-Type: application/json" "$ES_URL/$1" --data "$2"; }
es_post()   { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -X POST   -H "Content-Type: application/json" "$ES_URL/$1" --data "$2"; }
es_put_f()  { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -X PUT    -H "Content-Type: application/json" "$ES_URL/$1" --data-binary @"$2"; }
es_del()    { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -X DELETE "$ES_URL/$1"; }

kb_get()    { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -H "kbn-xsrf: true" "$KB_URL/$1"; }
kb_post()   { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -H "kbn-xsrf: true" -X POST -H "Content-Type: application/json" "$KB_URL/$1" --data "$2"; }
kb_put()    { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -H "kbn-xsrf: true" -X PUT  -H "Content-Type: application/json" "$KB_URL/$1" --data "$2"; }
kb_post_f() { curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -H "kbn-xsrf: true" -X POST -F "file=@$2" "$KB_URL/$1"; }

# Check HTTP status — pass the URL and expected codes; prints result
check_http() {
  local url="$1"; shift
  local expected="${1:-200}"
  local code
  code=$(curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -o /dev/null -w '%{http_code}' "$url")
  if [[ "$code" == "$expected" ]]; then
    ok "$url -> $code"
    return 0
  else
    warn "$url -> $code (expected $expected)"
    return 1
  fi
}
