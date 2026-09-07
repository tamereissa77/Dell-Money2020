#!/usr/bin/env bash
# Smoke test: is the portfolio demo booth-ready?
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PORT=$(grep -E '^APP_PORT=' .env | cut -d= -f2); URL="http://localhost:$PORT"
fail=0
chk(){ printf '  %-30s' "$1"; shift; if out=$("$@" 2>&1); then echo "OK   $out"; else echo "FAIL $out"; fail=1; fi; }
echo "verifying $URL"
chk "streamlit health" bash -c "curl -fsS -o /dev/null -w '%{http_code}' $URL/_stcore/health"
chk "dataset present"  bash -c "[ -f data/stock_data/sp500.csv ] && echo \"\$(wc -l < data/stock_data/sp500.csv) rows\""
chk "cuOpt on GPU"     bash -c "docker compose exec -T portfolio python -c \"
import cuopt,cuml,torch
assert torch.cuda.is_available()
print(f'cuopt {cuopt.__version__} cuml {cuml.__version__} on {torch.cuda.get_device_name(0)}')\""
echo
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED — demo is booth-ready." || echo "SOME CHECKS FAILED."
exit $fail
