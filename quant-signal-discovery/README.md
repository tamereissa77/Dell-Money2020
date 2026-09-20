# Quantitative signal discovery

Three agents in a closed loop invent an alpha signal, write it as executable Python, and
backtest it on fourteen years of S&P 500 prices — then read their own results and try again.

Based on NVIDIA's [Quantitative Signal Discovery Agent](https://github.com/NVIDIA-AI-Blueprints/quantitative-signal-discovery-agent)
(Apache 2.0), adapted here to run **entirely on the GB10**. Upstream ships pointed at
NVIDIA's hosted inference endpoint; this version serves the model locally, so after
`make data` nothing touches the network.

---

## What the audience sees

`make run` and a trace viewer on the second screen. In roughly a minute the loop runs three
times and prints what it found:

```
  Signal     signal_momentum_adjusted_volume
  Formula    Mul(Rank(TS_Return(Close, 20)), Rank(Decay_Linear(Volume, 20)))
  Idea       Combines price momentum with weighted volume activity, scaling
             momentum by relative trading intensity

  Mean IC        -0.0122        (threshold |IC| >= 0.02)
  p-value        1.74e-07       (threshold <= 0.05)
  t-statistic    -5.2353
  periods tested 3,494

  Verdict    BEST EFFORT — statistically real, below the IC bar
  Code       43 lines of executable Python, generated and run
```

Nobody wrote that formula. The signal agent composed it from an operator catalogue, the code
agent turned it into a runnable module with the operators inlined, and the eval agent
backtested it and reported back.

**The interesting result is the verdict.** That signal is real — a t-statistic of −5.2 over
3,494 periods is not noise — but its information coefficient is 0.012 against a 0.02 bar, so
the agent refused to accept it and went round the loop again. A system that reports
"statistically significant but too weak to trade" is behaving like a quant, not like a demo.

The negative sign is not a failure either: a consistently negative IC is a predictive signal
with its sign inverted. Worth saying out loud when someone asks.

---

## Measured on this machine

| | |
|---|---|
| Full run, 3 iterations | **49–74 s** |
| Per iteration | 11–36 s |
| Model generation | **14.7 tok/s** (30B MoE, ~3B active, FP8) |
| Model load, cold | ~2 min (31 GB of weights) |
| Backtest universe | 3,519 trading days × 380 tickers |
| Network at run time | none |

The model is NVIDIA Nemotron 3 Nano 30B A3B FP8 on vLLM, the same deployment validated for
DGX Spark (Grace ARM64, sm_121). 14.7 tok/s is the GB10's memory-bandwidth ceiling for this
class of model — a smaller model does not go meaningfully faster, so the lever for pacing
the demo is the token budget in `configs/config-gb10-local.yml`, not the model choice.

---

## Running it

```bash
make up        # model server + Phoenix trace viewer (~2 min cold)
make prewarm   # force the weight load before anyone is watching
make verify    # seven checks; all must pass
make run       # discover momentum signals
make results   # reprint the last finding
make down
```

Other prompts:

```bash
make run-vol                              # volatility signals
make run-rev                              # mean reversion
make run-volume                           # volume-price divergence
INPUT="signals that work in illiquid names" make run
```

Any phrasing works — the input is a natural-language brief, not a menu.

**The booth screen is Phoenix at `http://localhost:6006`.** Open the
`signal-discovery-workflow` project and every agent call is there: the prompt, the model's
output, the latency. It is the most convincing part of the demo, because it shows three
distinct agents rather than one model being asked three questions.

### First time on a fresh machine

```bash
make install   # venv from the lockfile (uv)
make data      # S&P 500 history from Yahoo Finance — the only step needing internet
```

---

## How it was adapted

| Upstream | Here |
|---|---|
| `base_url: https://integrate.api.nvidia.com/v1/` | `http://localhost:8000/v1/` |
| `model_name: nvidia/nemotron-3.5-lightning-30b-a3b` | `nemotron-nano` (local Nemotron 3 Nano 30B A3B FP8) |
| Requires `NVIDIA_API_KEY` | No key, no egress |
| Data fetched at run time | Pre-fetched to `src/signal_discovery_workflow/data/sp500/` |
| No container story | `docker-compose.yml`: vLLM + Phoenix |

`configs/config-gb10-local.yml` is generated from upstream's `config-optimization.yml` by
those two substitutions. Upstream's own configs are left untouched, so pulling a newer
version of the blueprint is a re-run of the two `sed` lines in the git history.

Pacing levers, all in `configs/config-gb10-local.yml`:

- `max_iterations` (3) — the dominant cost
- `num_signals` (2) — ideas generated per iteration
- `max_tokens` per agent (4000 / 3000 / 2000) — rarely reached in practice
- `ic_threshold` (0.02) — lower it and runs end sooner with `accepted`

---

## Honest notes

- **This is a research loop, not a trading system.** It searches for statistical association
  between a formula and forward returns. No transaction costs, no slippage, no capacity
  analysis, no live execution. Say so before anyone asks.
- **Backtest, not out-of-sample validation.** The signal is fitted and evaluated on the same
  fourteen-year window. A real process holds out a period the search never saw.
- **Results vary between runs.** The signal agent runs at temperature 0.8 by design — that is
  the exploration. Two runs of `make run` will not produce the same formula, which is worth
  showing rather than hiding.
- **`best_effort` is the common outcome** at the default 0.02 threshold, and that is the
  honest result for a one-minute search over a standard universe. Signals that clear a 0.02
  IC bar are not usually found in sixty seconds.
- Yahoo Finance dropped ~12 tickers as delisted when the dataset was built; the universe is
  380 names, not the full 500. It does not affect the method.

## Attribution

Upstream: [NVIDIA-AI-Blueprints/quantitative-signal-discovery-agent](https://github.com/NVIDIA-AI-Blueprints/quantitative-signal-discovery-agent),
Apache 2.0 — see `LICENSE.txt` and `README-upstream.md`. Market data from Yahoo Finance via
`yfinance`; dataset licensing is the user's responsibility, as upstream notes.
