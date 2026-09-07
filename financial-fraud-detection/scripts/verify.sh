#!/usr/bin/env bash
# End-to-end smoke test: is the demo actually scoring and explaining real data?
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PORT=$(grep -E '^DEMO_PORT=' .env | cut -d= -f2); URL="http://localhost:$PORT"
fail=0
chk(){ printf '  %-34s' "$1"; shift; if out=$("$@" 2>&1); then echo "OK   $out"; else echo "FAIL $out"; fail=1; fi; }

echo "verifying $URL"
chk "triton health" bash -c 'curl -fsS -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready'
chk "demo api"      bash -c "curl -fsS -o /dev/null -w '%{http_code}' $URL/api/stats"
chk "scoring"       bash -c "curl -fsS $URL/api/stats | python3 -c \"import sys,json;d=json.load(sys.stdin);assert d['n']>0 and d['f1']>0.9;print(f\\\"{d['n']:,} txns, F1={d['f1']}, {d['throughput']:,}/s\\\")\""
chk "feed has data" bash -c "curl -fsS '$URL/api/feed?offset=0&limit=3' | python3 -c \"import sys,json;r=json.load(sys.stdin);assert len(r)==3 and r[0]['amount'] is not None;print(f\\\"\\\${r[0]['amount']} {r[0]['mcc_name']} in {r[0]['city']}\\\")\""
chk "live explanation" bash -c "curl -fsS $URL/api/explain/1 | python3 -c \"import sys,json;d=json.load(sys.stdin);a=d['attributions'][0];assert len(d['attributions'])>=8;print(f\\\"top driver '{a['feature']}' {a['value']:+.2f} in {d['latency']}s\\\")\""
echo
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED — demo is booth-ready." || echo "SOME CHECKS FAILED — see above."
exit $fail
