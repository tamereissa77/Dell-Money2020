"""txn-generator - replays the TabFormer test set as a live transaction stream.

Publishes transactions to `txn.raw` and their labels to `txn.truth`, on separate
topics, so the scoring path is structurally incapable of seeing ground truth.
Compose reinforces this: the label CSV is mounted into this service only.

Control API on :8091 lets the presenter steer the stream without touching a
terminal - rate, fraud-rate mode, scenario injection, pause/resume.
"""
import os, sys, json, time, threading, random, itertools
import numpy as np, pandas as pd
from flask import Flask, jsonify, request

sys.path.insert(0, "/svc")
from common import pipeline as P

B = os.environ.get("GNN_TEST_DIR", "/work/data/TabFormer/gnn_np/test_gnn")
SEED = int(os.environ.get("SEED", "20260920"))
PORT = int(os.environ.get("PORT", "8091"))

# --- state the control API mutates -----------------------------------------
STATE = {
    "rate": float(os.environ.get("RATE", "400")),   # target transactions/sec
    "mode": os.environ.get("MODE", "demo"),         # "demo" | "realistic"
    "paused": False,
    "emitted": 0,
    "frauds_emitted": 0,
    "injected": 0,
    "started": time.time(),
    "last_scenario": None,
}
LOCK = threading.Lock()

# Fraud rate per mode. `realistic` is the true TabFormer base rate; `demo`
# is elevated so something happens on screen inside thirty seconds. The UI
# displays which is active - this is the build's main honesty control.
FRAUD_RATE = {"realistic": 0.00122, "demo": 0.04}
TRUE_BASE_RATE = 0.00122
TEST_SET_RATE = None            # measured at load; the set is rebalanced

print("[gen] loading test set ...", flush=True)
L = lambda p: pd.read_csv(p)
disp = L(f"{B}/nodes/transaction_display.csv")
y = L(f"{B}/nodes/transaction_label.csv").values.ravel().astype(int)
e_ut = L(f"{B}/edges/user_to_transaction.csv").values.T.astype(np.int64)
e_tm = L(f"{B}/edges/transaction_to_merchant.csv").values.T.astype(np.int64)
assert len(disp) == len(y), "display/label row mismatch"
TEST_SET_RATE = float(y.mean())
print(f"[gen] {len(y)} rows, test-set fraud rate {100*TEST_SET_RATE:.3f}%", flush=True)

MCC_NAMES = {5411: "Grocery", 5812: "Restaurant", 5541: "Fuel", 4121: "Rideshare",
             5912: "Pharmacy", 7995: "Gambling", 5999: "Retail", 6011: "ATM"}


def _mcc_name(m):
    if m in MCC_NAMES: return MCC_NAMES[m]
    if 3000 <= m <= 3299: return "Airline"
    if 3300 <= m <= 3499: return "Car rental"
    if 3500 <= m <= 3999: return "Hotel"
    return f"MCC {m}"


# Replay in temporal order, never shuffled: the GNN's card/merchant/time
# relationships are the whole point, and a shuffle destroys them.
_order_cols = [c for c in ("Year", "Month", "Day") if c in disp.columns]
ORDER = (disp[_order_cols].reset_index().sort_values(_order_cols + ["index"])["index"].values
         if _order_cols else np.arange(len(disp)))
FRAUD_ROWS = ORDER[y[ORDER] == 1]
CLEAN_ROWS = ORDER[y[ORDER] == 0]
print(f"[gen] temporal order over {len(ORDER)} rows "
      f"({len(FRAUD_ROWS)} fraud / {len(CLEAN_ROWS)} clean)", flush=True)


_SEQ = itertools.count(1)


def row_payload(i, scenario=None):
    """The transaction as it goes on the wire. Carries no label, by design.

    `row` is the index into the preprocessed test set. scoring-svc uses it to
    look up feature vectors, exactly as a real system would hit a feature store
    rather than recompute features inline.
    """
    d = disp.iloc[i]
    mcc = int(d["MCC"]) if not pd.isna(d["MCC"]) else 0
    err = str(d["Errors"]).strip()
    return {
        # Unique per emission, not per row. A replayed row is a NEW transaction
        # event; deriving the id from the row index made repeats collide and
        # silently overwrite each other downstream.
        "txn_id": f"t{next(_SEQ):09d}",
        "row": int(i),
        "produced_ms": P.now_ms(),
        "amount": float(d["Amount"]),
        "mcc": mcc, "mcc_name": _mcc_name(mcc),
        "city": str(d["City"]).strip(), "state": str(d["State"]).strip(),
        "merchant": str(d["Merchant"])[-6:],
        "chip": str(d["Chip"]).replace(" Transaction", ""),
        "errors": "" if err in ("XX", "nan") else err,
        "when": f'{int(d["Year"])}-{int(d["Month"]):02d}-{int(d["Day"]):02d}',
        "user_id": int(e_ut[0][i]), "merchant_id": int(e_tm[1][i]),
        "scenario": scenario,
    }


# --- scenarios --------------------------------------------------------------
# Every typology is a curated sequence of REAL rows from the test set, not a
# fabricated transaction: scoring-svc looks up features by row index, so a
# synthetic row would have no features and could not be scored. See
# scenarios.py for how each is selected.
from scenarios import Catalogue

CAT = Catalogue(disp, y, e_ut, e_tm, seed=SEED)
SCENARIOS = ["merchant-city-anomaly", "geo-velocity", "cnp-burst",
             "account-takeover", "mule-fanin-fanout"]
for _n in SCENARIOS:
    print(f"[gen] scenario {_n}: {CAT.describe(_n)}", flush=True)

PENDING = []        # rows queued by an inject call, emitted next


def pick_row(rng):
    """Choose the next row so the emitted stream hits the configured fraud rate.

    The test set is rebalanced (~8% fraud), so neither mode can just replay it
    in order - both modes resample against their target rate.
    """
    target = FRAUD_RATE.get(STATE["mode"], 0.04)
    pool = FRAUD_ROWS if rng.random() < target else CLEAN_ROWS
    return int(pool[rng.randrange(len(pool))])


def run():
    P.wait_for_broker()
    prod = P.producer("txn-generator")
    rng = random.Random(SEED)       # deterministic: a rehearsed demo repeats exactly
    print(f"[gen] producing to {P.T_RAW} / {P.T_TRUTH} at {STATE['rate']} TPS", flush=True)
    while True:
        if STATE["paused"]:
            time.sleep(0.05)
            continue
        rate = max(1.0, STATE["rate"])
        # Emit in slices so a high TPS does not become one huge burst per second.
        batch = max(1, int(rate / 50))
        t0 = time.perf_counter()
        for _ in range(batch):
            with LOCK:
                scen = PENDING.pop(0) if PENDING else None
            if scen:
                i, name = scen
                STATE["injected"] += 1
            else:
                i, name = pick_row(rng), None
            msg = row_payload(i, name)
            # Key by card/account so per-entity ordering holds within a partition.
            P.send(prod, P.T_RAW, msg["user_id"], msg)
            P.send(prod, P.T_TRUTH, msg["txn_id"],
                   {"txn_id": msg["txn_id"], "row": int(i), "label": int(y[i])})
            STATE["emitted"] += 1
            STATE["frauds_emitted"] += int(y[i])
        prod.poll(0)
        spent = time.perf_counter() - t0
        time.sleep(max(0.0, batch / rate - spent))


# --- control API ------------------------------------------------------------
app = Flask(__name__)


@app.get("/status")
def status():
    up = max(1e-9, time.time() - STATE["started"])
    return jsonify({
        **{k: STATE[k] for k in ("rate", "mode", "paused", "emitted",
                                 "frauds_emitted", "injected", "last_scenario")},
        "actual_tps": round(STATE["emitted"] / up, 1),
        "emitted_fraud_rate": round(STATE["frauds_emitted"] / max(STATE["emitted"], 1), 5),
        "configured_fraud_rate": FRAUD_RATE.get(STATE["mode"]),
        "true_base_rate": TRUE_BASE_RATE,
        "test_set_rate": round(TEST_SET_RATE, 5),
        "scenarios": {n: CAT.describe(n)["status"] for n in SCENARIOS},
        "seed": SEED,
    })


@app.post("/rate")
def set_rate():
    v = float(request.json.get("tps", 400))
    STATE["rate"] = max(1.0, min(v, 50000.0))
    return jsonify({"rate": STATE["rate"]})


@app.post("/mode")
def set_mode():
    m = str(request.json.get("mode", "demo"))
    if m not in FRAUD_RATE:
        return jsonify({"error": f"mode must be one of {list(FRAUD_RATE)}"}), 400
    STATE["mode"] = m
    return jsonify({"mode": m, "fraud_rate": FRAUD_RATE[m]})


@app.post("/inject/<scenario>")
def inject(scenario):
    if scenario not in SCENARIOS:
        return jsonify({"error": "unknown scenario", "known": SCENARIOS}), 404
    rows = CAT.built.get(scenario) or []
    if not rows:
        return jsonify({"error": "no rows in this dataset match the typology",
                        "scenario": scenario}), 503
    with LOCK:
        # The whole sequence goes next, in order and back-to-back: the pattern
        # only reads as a pattern if the transactions arrive together.
        for i in rows:
            PENDING.append((int(i), scenario))
    STATE["last_scenario"] = scenario
    return jsonify({"injected": scenario, "rows": len(rows),
                    "detail": CAT.describe(scenario), "queued": len(PENDING)})


@app.get("/scenarios")
def scenarios():
    return jsonify({n: CAT.describe(n) for n in SCENARIOS})


@app.post("/pause")
def pause():
    STATE["paused"] = True
    return jsonify({"paused": True})


@app.post("/resume")
def resume():
    STATE["paused"] = False
    return jsonify({"paused": False})


@app.get("/health")
def health():
    return jsonify({"ok": True, "emitted": STATE["emitted"]})


def _shutdown(signum, frame):
    print("[gen] SIGTERM - exiting", flush=True)
    os._exit(0)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
