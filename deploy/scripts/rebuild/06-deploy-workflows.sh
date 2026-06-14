#!/usr/bin/env bash
# 06-deploy-workflows.sh — deploy the 3 Elastic workflows to Kibana.
#
# Reads YAML files from deploy/infrastructure/workflows/ and pushes each one
# via PUT /api/workflows/workflow/<id> with the YAML as the request body.
#
# Workflow IDs are derived from the filenames:
#   flywheel-curator.yaml  -> flywheel-curator
#   flywheel-stats.yaml    -> flywheel-stats
#   model-comparison.yaml  -> model-comparison
#
# Idempotent — PUT overwrites any existing workflow with the same ID.
#
# Prerequisites:
# - Trial license or Platinum/Enterprise (workflows require a paid-tier license)
# - Kibana reachable at KB_URL

source "$(dirname "$0")/lib.sh"

WORKFLOWS_DIR="$REPO_ROOT/deploy/infrastructure/workflows"

WORKFLOW_FILES=(
  flywheel-curator.yaml
  flywheel-stats.yaml
  model-comparison.yaml
)

# ---------------------------------------------------------------------------
# Preflight — verify the YAML files exist
# ---------------------------------------------------------------------------
info "Checking workflow source files..."
for f in "${WORKFLOW_FILES[@]}"; do
  if [[ ! -f "$WORKFLOWS_DIR/$f" ]]; then
    fail "Not found: $WORKFLOWS_DIR/$f"
  fi
done
ok "All ${#WORKFLOW_FILES[@]} workflow YAML files present"

# ---------------------------------------------------------------------------
# Deploy each workflow
# ---------------------------------------------------------------------------
deploy_failures=0

for f in "${WORKFLOW_FILES[@]}"; do
  workflow_id="${f%.yaml}"
  yaml_path="$WORKFLOWS_DIR/$f"

  info "Deploying workflow '$workflow_id' from $f..."

  # Read YAML content and JSON-encode it for the request body
  yaml_content=$(cat "$yaml_path")
  json_body=$(python3 -c 'import json,sys; print(json.dumps({"yaml": sys.stdin.read()}))' <<< "$yaml_content")

  resp=$(kb_put "api/workflows/workflow/$workflow_id" "$json_body")

  # Check for success indicators in the response
  if echo "$resp" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    # Success if response contains the workflow id or name, or no error field
    if d.get("id") or d.get("name") or "error" not in d:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
' 2>/dev/null; then
    ok "  $workflow_id deployed"
  else
    warn "  $workflow_id — unexpected response: $resp"
    deploy_failures=$((deploy_failures + 1))
  fi
done

echo
if [[ "$deploy_failures" -gt 0 ]]; then
  warn "$deploy_failures workflow(s) may not have deployed correctly — review output above"
else
  ok "All ${#WORKFLOW_FILES[@]} workflows deployed"
fi

# ---------------------------------------------------------------------------
# Verify — GET each workflow back from Kibana
# ---------------------------------------------------------------------------
echo
info "Verifying deployed workflows..."
verify_failures=0

for f in "${WORKFLOW_FILES[@]}"; do
  workflow_id="${f%.yaml}"

  resp=$(kb_get "api/workflows/workflow/$workflow_id")

  if echo "$resp" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    if "error" in d or d.get("statusCode", 200) >= 400:
        sys.exit(1)
    sys.exit(0)
except Exception:
    sys.exit(1)
' 2>/dev/null; then
    ok "  $workflow_id verified"
  else
    warn "  $workflow_id — could not verify: $resp"
    verify_failures=$((verify_failures + 1))
  fi
done

echo
if [[ "$verify_failures" -eq 0 ]]; then
  ok "ALL ${#WORKFLOW_FILES[@]} WORKFLOWS DEPLOYED AND VERIFIED"
else
  warn "$verify_failures workflow(s) failed verification"
  exit 1
fi
