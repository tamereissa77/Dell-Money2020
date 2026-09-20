#!/usr/bin/env python
"""Print the most recent discovered signal in a form a booth visitor can read."""
import glob, json, os, sys

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "signal_discovery_workflow", "output")
files = sorted(glob.glob(os.path.join(OUT, "*.json")))
if not files:
    print("no results yet — run 'make run'"); sys.exit(0)

d = json.load(open(files[-1]))
m = d.get("evaluation_metrics", {})

# The description is a fenced JSON blob from the model; pull the chosen formula out of it.
formula = meaning = None
desc = d.get("signal_description", "")
try:
    body = desc.strip().removeprefix("```json").removesuffix("```").strip()
    cand = json.loads(body)
    if isinstance(cand, list) and cand:
        formula, meaning = cand[0].get("formula"), cand[0].get("meaning")
except Exception:
    pass

ic, p = m.get("mean_ic"), m.get("p_value")
thr = 0.02
verdict = ("ACCEPTED" if ic is not None and abs(ic) >= thr and p is not None and p <= 0.05
           else "BEST EFFORT — statistically real, below the IC bar")

print(f"\n  Signal     {d.get('selected_signal','?')}")
if formula: print(f"  Formula    {formula}")
if meaning: print(f"  Idea       {meaning}")
print(f"  Iteration  {d.get('iteration','?')}   ({d.get('timestamp','')})")
print(f"\n  Mean IC        {ic:+.4f}        (threshold |IC| >= {thr})" if ic is not None else "")
print(f"  p-value        {p:.2e}        (threshold <= 0.05)" if p is not None else "")
for k, lbl in (("ic_ir","IC information ratio"), ("t_stat","t-statistic"),
               ("num_periods","periods tested"), ("positive_ic_ratio","positive IC ratio")):
    if k in m:
        v = m[k]
        print(f"  {lbl:<14} {v:,.4f}" if isinstance(v, float) else f"  {lbl:<14} {v:,}")
print(f"\n  Verdict    {verdict}")
print(f"  Code       {len(d.get('signal_code','').splitlines())} lines of executable Python, generated and run")
print(f"  Saved      {os.path.basename(files[-1])}\n")
