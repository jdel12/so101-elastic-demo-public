#!/usr/bin/env bash
# 01-start-trial.sh — start or restart a 30-day trial license.
#
# This only works if a trial has NOT been used on this cluster yet.
# If the trial has already been used, this will fail with a specific error —
# in that case, you need to either:
#   (a) wipe the cluster entirely (destroying the cluster UUID) and rebuild
#       to get a fresh trial, or
#   (b) upload a Platinum/Enterprise license via POST _license with a JSON file.

source "$(dirname "$0")/lib.sh"

info "Current license before:"
es_get "_license" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  {d[\"license\"][\"type\"]} / {d[\"license\"][\"status\"]}')"

info "Attempting to start trial..."
resp=$(es_post "_license/start_trial?acknowledge=true" "")
echo "$resp" | python3 -m json.tool

if echo "$resp" | grep -q '"trial_was_started":true'; then
  ok "Trial started successfully. Re-run 00-preflight.sh to confirm Agent Builder is now available."
else
  warn "Trial could not be started."
  warn "Likely cause: a trial has already been used on this cluster UUID."
  warn "Options:"
  warn "  1. Wipe the cluster (see scripts/rebuild/teardown/wipe-elasticsearch.sh)"
  warn "     and start over — the new cluster gets a fresh trial."
  warn "  2. Upload a Platinum/Enterprise license file with:"
  warn "       curl -sk -u elastic:\$PW -X PUT -H 'Content-Type: application/json' \\"
  warn "         $ES_URL/_license --data @license.json"
  exit 1
fi
