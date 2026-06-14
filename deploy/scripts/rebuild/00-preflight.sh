#!/usr/bin/env bash
# 00-preflight.sh — verify the cluster is reachable and report current state
# before any destructive action. Safe to run anytime.
#
# Exit 0 if ready, 1 if anything fails.

source "$(dirname "$0")/lib.sh"

info "ES URL: $ES_URL"
info "KB URL: $KB_URL"
echo

info "Checking ES reachability..."
check_http "$ES_URL" 200 || fail "Elasticsearch not reachable"

info "Checking Kibana reachability..."
check_http "$KB_URL/api/status" 200 || fail "Kibana not reachable"

echo
info "Cluster info:"
es_get "" | python3 -m json.tool | sed -n '1,20p'

echo
info "License:"
es_get "_license" | python3 -c "
import json, sys
d = json.load(sys.stdin)
lic = d.get('license', {})
print(f'  type:     {lic.get(\"type\")}')
print(f'  status:   {lic.get(\"status\")}')
print(f'  issued:   {lic.get(\"issue_date\")}')
print(f'  expires:  {lic.get(\"expiry_date\", \"n/a\")}')
"

echo
info "Robot indices (live):"
es_get "_cat/indices/robot-*?v&s=index" || warn "no robot indices yet"

echo
info "Robot index templates installed:"
es_get "_index_template/robot-*" | python3 -c "
import json, sys
d = json.load(sys.stdin)
tpls = d.get('index_templates', [])
print(f'  count: {len(tpls)}')
for t in tpls:
    print(f'    - {t[\"name\"]}')
"

echo
info "Kibana Agent Builder feature:"
resp=$(kb_get "api/features")
echo "$resp" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ids = [f.get('id','') for f in d if isinstance(d, list)]
ab = [i for i in ids if 'agent' in i.lower() or 'workflow' in i.lower()]
if ab:
    print(f'  found: {ab}')
else:
    print('  NOT REGISTERED — license likely does not permit Agent Builder/Workflows')
"

echo
ok "Preflight complete"
