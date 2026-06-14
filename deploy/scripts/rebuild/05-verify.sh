#!/usr/bin/env bash
# 05-verify.sh — end-to-end health check after a rebuild.
#
# Exits 0 if everything looks good, 1 otherwise. Non-destructive.

source "$(dirname "$0")/lib.sh"

errors=0
check() {
  if "$@"; then return 0; else errors=$((errors+1)); return 1; fi
}

info "=== ES cluster ==="
check check_http "$ES_URL/_cluster/health" 200

info "=== License ==="
lic=$(es_get "_license" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["license"]["type"])')
if [[ "$lic" == "trial" || "$lic" == "platinum" || "$lic" == "enterprise" ]]; then
  ok "license: $lic"
else
  warn "license: $lic  (Agent Builder / Workflows will not work)"
  errors=$((errors+1))
fi

info "=== Index templates ==="
count=$(es_get "_index_template/robot-*" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d.get("index_templates",[])))')
if [[ "$count" -eq 8 ]]; then
  ok "8 robot index templates present"
else
  warn "expected 8 robot index templates, found $count"
  errors=$((errors+1))
fi

info "=== Kibana dashboards ==="
dcount=$(kb_get "api/saved_objects/_find?type=dashboard&per_page=1000" | python3 -c '
import json, sys
d = json.load(sys.stdin)
n = sum(1 for o in d.get("saved_objects", []) if any(p in o.get("attributes",{}).get("title","") for p in ["Live Performance","Flywheel","Model Comparison","Training Observability"]))
print(n)')
if [[ "$dcount" -eq 4 ]]; then
  ok "4 robot dashboards imported"
else
  warn "expected 4 robot dashboards, found $dcount"
  errors=$((errors+1))
fi

info "=== Action connectors ==="
for name in "Ollama Local" "Ollama Vision" "robot-arm"; do
  found=$(kb_get "api/actions/connectors" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(any(c.get('name')=='$name' for c in d if isinstance(d, list)))
")
  if [[ "$found" == "True" ]]; then
    ok "connector: $name"
  else
    warn "connector missing: $name"
    errors=$((errors+1))
  fi
done

info "=== Agent Builder agent ==="
if check_http "$KB_URL/api/agent_builder/agents" 200 2>/dev/null; then
  agent_count=$(kb_get "api/agent_builder/agents" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    if isinstance(d, dict) and "results" in d:
        print(len(d["results"]))
    elif isinstance(d, list):
        print(len(d))
    else:
        print(0)
except Exception:
    print(0)
')
  if [[ "$agent_count" -gt 0 ]]; then
    ok "$agent_count agent(s) defined"
  else
    warn "no agents defined yet (run 04-create-agent-builder-agent.sh)"
    errors=$((errors+1))
  fi
else
  warn "Agent Builder not available (check license)"
  errors=$((errors+1))
fi

info "=== MCP server reachability ==="
mcp_code=$(curl -sk -m 3 -X POST "$MCP_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"verify","version":"0.1"}}}' \
  -o /dev/null -w '%{http_code}')
if [[ "$mcp_code" == "200" ]]; then
  ok "MCP server responds at $MCP_URL"
else
  warn "MCP server at $MCP_URL returned $mcp_code"
  errors=$((errors+1))
fi

echo
if [[ "$errors" -eq 0 ]]; then
  ok "ALL CHECKS PASSED"
  exit 0
else
  warn "$errors check(s) failed"
  exit 1
fi
