"""screening-stub - stands in for the bank's existing certified screening engine.

Deliberately simple and deliberately noisy. Real sanctions/AML screening engines
are threshold-and-watchlist rule systems that fire on a large fraction of
traffic and run at a 90-98% false-positive rate; that is the operational reality
this platform is meant to help with, not a strawman.

Three things about this service are commercial, not technical:

  * It consumes txn.raw directly. It does NOT see model scores, and nothing
    upstream can disable it. The model never writes to screening.alerts.
  * Its output is preserved intact. alert-svc re-ranks it; it never overrules,
    filters or replaces it.
  * Its false-positive ratio is measured and published, so the "we reduce the
    queue you already work" claim is arithmetic rather than assertion.

Never present this as the product. It is the incumbent.
"""
import os, sys, time, threading, collections
from flask import Flask, jsonify

sys.path.insert(0, "/svc")
from common import pipeline as P

PORT = int(os.environ.get("PORT", "8093"))

# Tuned to land in the band real engines live in. Raising these makes the
# incumbent quieter and the augmentation argument weaker - which is exactly the
# trade a bank is making, so it is configurable and displayed, never hidden.
AMOUNT_THRESHOLD = float(os.environ.get("RULE_AMOUNT", "300"))
VELOCITY_WINDOW_S = float(os.environ.get("RULE_VELOCITY_WINDOW", "60"))
VELOCITY_COUNT = int(os.environ.get("RULE_VELOCITY_COUNT", "3"))
ROUND_AMOUNTS = {100.0, 200.0, 250.0, 500.0, 1000.0}

# Stands in for a sanctions/high-risk-geography list. Real lists are licensed
# and certified; this is a plausible shape, not a real list, and is labelled as
# such wherever it surfaces.
WATCHLIST_CITIES = {c.strip() for c in os.environ.get(
    "RULE_WATCHLIST",
    "Rome,Naples,Moscow,Lagos,Caracas,Tehran,Karachi,Odessa").split(",") if c.strip()}

RULES = {
    "R001": "amount above reporting threshold",
    "R002": "merchant city on high-risk geography list",
    "R003": "card velocity: repeated use inside window",
    "R004": "round-value amount consistent with structuring",
    "R005": "card-not-present with terminal error",
}

_recent = collections.defaultdict(collections.deque)   # user_id -> deque[ts]
# Lifetime counters blend every regime the demo has run through. Switching the
# generator from demo to the true base rate changes the incumbent's
# false-positive rate from ~65% to ~99%, and a lifetime average takes tens of
# minutes to follow - long enough that a presenter quoting the real figure is
# contradicted by the screen. WINDOW holds the most recent judged alerts so the
# published rate reflects the regime that is actually running.
WINDOW = collections.deque(maxlen=int(os.environ.get("FP_WINDOW", "5000")))
STATE = {"seen": 0, "alerted": 0, "by_rule": collections.Counter(),
         "true_pos": 0, "false_pos": 0}
LOCK = threading.Lock()


def evaluate(txn):
    """Return the rules this transaction trips. Order is significant: the first
    is reported as the primary reason, as an incumbent's case file would."""
    hits = []
    amt = float(txn.get("amount", 0.0))
    city = str(txn.get("city", "")).strip()
    chip = str(txn.get("chip", "")).strip().lower()
    err = str(txn.get("errors", "")).strip()

    if amt >= AMOUNT_THRESHOLD:
        hits.append("R001")
    if city in WATCHLIST_CITIES:
        hits.append("R002")

    now = time.time()
    dq = _recent[txn["user_id"]]
    dq.append(now)
    while dq and now - dq[0] > VELOCITY_WINDOW_S:
        dq.popleft()
    if len(dq) >= VELOCITY_COUNT:
        hits.append("R003")

    if amt in ROUND_AMOUNTS:
        hits.append("R004")
    if ("online" in chip or "not present" in chip) and err:
        hits.append("R005")
    return hits


def run():
    P.wait_for_broker()
    # Two consumers: txn.raw drives screening; txn.truth is read ONLY to publish
    # the incumbent's own false-positive rate for the comparison screen. The
    # label never influences whether an alert is raised - see evaluate(), which
    # takes only the transaction.
    cons = P.consumer("screening-stub", [P.T_RAW, P.T_TRUTH], offset="latest")
    prod = P.producer("screening-stub")
    truth = {}
    print(f"[screen] rules active: {sorted(RULES)} "
          f"(amount>={AMOUNT_THRESHOLD}, watchlist={len(WATCHLIST_CITIES)} cities)", flush=True)
    while True:
        msg = cons.poll(0.05)
        if msg is None or msg.error():
            continue
        rec = P.decode(msg)
        if msg.topic() == P.T_TRUTH:
            truth[rec["txn_id"]] = rec["label"]
            if len(truth) > 200000:
                for k in list(truth)[:50000]:
                    truth.pop(k, None)
            continue

        STATE["seen"] += 1
        hits = evaluate(rec)
        if not hits:
            continue

        alert = {
            "alert_id": f"S{rec['txn_id'][1:]}",
            "txn_id": rec["txn_id"], "row": rec["row"],
            "raised_ms": P.now_ms(),
            "source": "incumbent-screening",
            "rule_id": hits[0],
            "rule_reason": RULES[hits[0]],
            "all_rules": hits,
            "amount": rec.get("amount"), "city": rec.get("city"),
            "user_id": rec["user_id"], "merchant_id": rec["merchant_id"],
            "scenario": rec.get("scenario"),
        }
        P.send(prod, P.T_SCREENING, alert["alert_id"], alert)
        with LOCK:
            STATE["alerted"] += 1
            for h in hits:
                STATE["by_rule"][h] += 1
            lbl = truth.get(rec["txn_id"])
            if lbl is not None:
                WINDOW.append(int(lbl))
                if lbl == 1:
                    STATE["true_pos"] += 1
                else:
                    STATE["false_pos"] += 1
        prod.poll(0)


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"ok": True, "seen": STATE["seen"], "alerted": STATE["alerted"]})


@app.get("/metrics")
def metrics():
    with LOCK:
        tp, fp = STATE["true_pos"], STATE["false_pos"]
        judged = tp + fp
        return jsonify({
            "seen": STATE["seen"],
            "alerted": STATE["alerted"],
            "alert_rate": round(STATE["alerted"] / max(STATE["seen"], 1), 5),
            "true_positives": tp, "false_positives": fp,
            "lifetime_false_positive_rate": round(fp / max(judged, 1), 4),
            # The headline for the comparison screen: recent judgements only, so
            # it follows a mode switch within a minute instead of dragging the
            # previous regime along for half an hour.
            "false_positive_rate": round(
                sum(1 for l in WINDOW if l == 0) / max(len(WINDOW), 1), 4),
            "window": len(WINDOW),
            "judged": judged,
            "by_rule": dict(STATE["by_rule"]),
            "rules": RULES,
            "config": {"amount_threshold": AMOUNT_THRESHOLD,
                       "velocity_window_s": VELOCITY_WINDOW_S,
                       "velocity_count": VELOCITY_COUNT,
                       "watchlist_cities": sorted(WATCHLIST_CITIES)},
            "disclaimer": "Stand-in for a certified screening engine. "
                          "Watchlist is illustrative, not a real sanctions list.",
        })


def _shutdown(signum, frame):
    print("[screen] SIGTERM - exiting", flush=True)
    os._exit(0)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
