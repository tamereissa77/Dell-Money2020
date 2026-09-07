"""GB10 feasibility proof: cuOpt + cuML on sm_121, Mean-CVaR GPU vs CPU."""
import time, sys, numpy as np, pandas as pd, cvxpy as cp

N_ASSETS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
N_SCEN   = int(sys.argv[2]) if len(sys.argv) > 2 else 50000

import torch, cuml, cuopt
print("=== environment ===")
print(f"  gpu {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
print(f"  cuml {cuml.__version__}   cuopt {cuopt.__version__}")

from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.cvar_optimizer import CVaR
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.settings import (
    ReturnsComputeSettings, ScenarioGenerationSettings, KDESettings, ApiSettings)
from portfolio_optimization.utils import calculate_returns

rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", periods=1260, freq="B")
tickers = [f"A{i:04d}" for i in range(N_ASSETS)]
shocks = rng.standard_normal((len(dates), N_ASSETS)) * rng.uniform(.008,.025,N_ASSETS) \
         + rng.normal(.0004,.0002,N_ASSETS)
prices = pd.DataFrame(100*np.exp(np.cumsum(shocks,axis=0)), index=dates, columns=tickers)
print(f"\n=== problem ===\n  {N_ASSETS} assets x {N_SCEN:,} scenarios  ({len(dates)} days history)")

rd = calculate_returns(prices, regime_dict=None,
      returns_compute_settings=ReturnsComputeSettings(return_type="LOG", freq=1))

print("\n=== 1. scenario generation (KDE) ===")
times = {}
for label, dev in (("GPU (cuML)","GPU"), ("CPU (sklearn)","CPU")):
    try:
        t=time.time()
        d = generate_cvar_data(dict(rd), ScenarioGenerationSettings(
            num_scen=N_SCEN, fit_type="kde",
            kde_settings=KDESettings(bandwidth=0.005, kernel="gaussian", device=dev)))
        dt=time.time()-t; times[label]=dt
        print(f"  {label:<14}: {dt:8.3f}s")
        if dev=="GPU": data_gpu=d
        else: data_cpu=d
    except Exception as e:
        print(f"  {label:<14}: FAILED {type(e).__name__}: {str(e)[:110]}")
if len(times)==2:
    print(f"  {'speedup':<14}: {times['CPU (sklearn)']/times['GPU (cuML)']:8.1f}x")

rd = data_gpu if 'data_gpu' in dir() else data_cpu   # already the updated returns_dict
params = CvarParameters(w_min=0.0, w_max=0.2, c_min=0.0, c_max=1.0,
                        risk_aversion=1.0, confidence=0.95, L_tar=1.6)

print("\n=== 2. Mean-CVaR solve ===")
solve = {}
for label, api, ss in (("GPU (cuOpt)","cuopt_python",{"time_limit":120}),
                       ("CPU (CVXPY)","cvxpy",{"solver":cp.CLARABEL,"verbose":False})):
    try:
        opt = CVaR(returns_dict=rd, cvar_params=params, api_settings=ApiSettings(api=api))
        t=time.time()
        row, pf = opt.solve_optimization_problem(ss, print_results=False)
        dt=time.time()-t; solve[label]=dt
        print(f"  {label:<12}: {dt:8.3f}s   CVaR={row.get('CVaR',float('nan')):.6f}  return={row.get('return',float('nan')):.6f}")
    except Exception as e:
        print(f"  {label:<12}: FAILED {type(e).__name__}: {str(e)[:150]}")
if len(solve)==2:
    print(f"  {'speedup':<12}: {solve['CPU (CVXPY)']/solve['GPU (cuOpt)']:8.1f}x")
print("\nGB10_BENCH_OK")
