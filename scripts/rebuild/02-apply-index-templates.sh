#!/usr/bin/env bash
# 02-apply-index-templates.sh — PUT the 7 robot index templates against the cluster.
#
# Reads from infra/elastic/index-templates/*.json. Each file is a single
# PUT-ready template body (index_patterns + template). The filename (minus .json)
# is the template name.
#
# Idempotent — safe to re-run.

source "$(dirname "$0")/lib.sh"

TEMPLATES_DIR="$REPO_ROOT/infra/elastic/index-templates"

if [[ ! -d "$TEMPLATES_DIR" ]]; then
  fail "Templates directory not found: $TEMPLATES_DIR"
fi

count=0
for f in "$TEMPLATES_DIR"/*.json; do
  name=$(basename "$f" .json)
  info "PUT _index_template/$name"
  resp=$(es_put_f "_index_template/$name" "$f")
  if echo "$resp" | grep -q '"acknowledged":true'; then
    ok "  $name"
    count=$((count + 1))
  else
    warn "  $name -> $resp"
  fi
done

echo
ok "Applied $count index template(s)"

echo
info "Verification:"
es_get "_index_template/robot-*" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for t in d.get('index_templates', []):
    patterns = t['index_template'].get('index_patterns', [])
    print(f'  {t[\"name\"]}: patterns={patterns}')
"
