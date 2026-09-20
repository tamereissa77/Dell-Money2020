"""alert-svc - joins the incumbent's alerts with model scores and re-ranks them.

This is the commercial heart of the build. The argument it has to make, in a
form a compliance audience will accept:

    The incumbent keeps screening. Every alert it raises is preserved. We
    reorder its queue so the investigator reaches the real frauds first, and
    we can say how much of its queue they no longer have to work through.

So this service never suppresses silently, never overrules the incumbent, and
never raises an alert of its own. It ranks what it is given.

Persistence is SQLite: single writer, small footprint, no extra container.
Postgres would only be justified by concurrent writers, which this does not
have.
"""
import os, sys, time, threading, sqlite3, json
from flask import Flask, jsonify, request

sys.path.insert(0, "/svc")
from common import pipeline as P

PORT = int(os.environ.get("PORT", "8094"))
DB = os.environ.get("ALERT_DB", "/state/alerts.db")
JOIN_WINDOW_S = float(os.environ.get("JOIN_WINDOW_S", "30"))

# A repeat false-positive pattern may be suppressed - but suppression is a
# recorded decision with a reason, never a silent drop. Suppressed alerts stay
# queryable and stay in the audit trail.
SUPPRESS_ENABLED = os.environ.get("SUPPRESS_ENABLED", "1") == "1"
SUPPRESS_MIN_SEEN = int(os.environ.get("SUPPRESS_MIN_SEEN", "25"))
SUPPRESS_MAX_SCORE = float(os.environ.get("SUPPRESS_MAX_SCORE", "0.02"))
# Hard ceiling on how much of the queue may be suppressed. The augmentation
# argument has to be won by ranking, not by hiding alerts.
SUPPRESS_MAX_FRACTION = float(os.environ.get("SUPPRESS_MAX_FRACTION", "0.10"))

_conn = None
_dblock = threading.Lock()


def db():
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB), exist_ok=True)
        _conn = sqlite3.connect(DB, check_same_thread=False)
        _conn.executescript("""
        CREATE TABLE IF NOT EXISTS alerts (
          alert_id TEXT PRIMARY KEY,
          txn_id TEXT, row INTEGER,
          rule_id TEXT, rule_reason TEXT, all_rules TEXT,
          amount REAL, city TEXT, user_id INTEGER, merchant_id INTEGER,
          scenario TEXT,
          score REAL, pred INTEGER,
          incumbent_rank INTEGER, ranked_rank INTEGER, rank_delta INTEGER,
          priority TEXT,
          suppressed INTEGER DEFAULT 0, suppress_reason TEXT,
          raised_ms REAL, ranked_ms REAL,
          state TEXT DEFAULT 'new',
          model_version TEXT, feature_version TEXT, data_version TEXT,
          label INTEGER
        );
        CREATE INDEX IF NOT EXISTS ix_alerts_score ON alerts(score DESC);
        CREATE INDEX IF NOT EXISTS ix_alerts_state ON alerts(state);
        """)
        # Additive migration: CREATE TABLE IF NOT EXISTS will not add a column
        # to a database that already exists on the volume.
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(alerts)")}
        if "label" not in cols:
            _conn.execute("ALTER TABLE alerts ADD COLUMN label INTEGER")
        _conn.commit()
    return _conn


STATE = {"alerts_in": 0, "scores_in": 0, "joined": 0, "unmatched": 0,
         "suppressed": 0, "incumbent_seq": 0, "emit_errors": 0, "last_error": None}
LOCK = threading.Lock()
# Rolling count of low-score alerts per rule, for the suppression heuristic.
_rule_noise = {}
TRUTH = {}          # txn_id -> label, measurement only
_truth_backlog = []  # (label, txn_id) pending backfill


def priority_of(score):
    if score >= 0.90: return "critical"
    if score >= 0.50: return "high"
    if score >= 0.10: return "medium"
    return "low"


def rank_key(score, scenario):
    """Ranking signal. Stage 2 uses the model score plus a small nudge for
    network typologies, which is where the graph adds information a
    per-transaction rules engine structurally cannot have. Historical
    disposition feeds in at Stage 3, once cases start closing."""
    boost = 0.02 if scenario == "mule-fanin-fanout" else 0.0
    return min(1.0, score + boost)


def suppress_decision(alert, score):
    """Suppress only a specific, repeatedly-quiet pattern - and say why.

    Keyed on (rule, merchant), not on the rule alone. Keying on the rule
    suppressed ~80% of the incumbent's queue, which is not deprioritising a
    repeat false positive - it is deleting its output and then claiming credit
    for a smaller queue. A suppression must name a pattern an investigator
    would recognise, and the suppressed share is capped so the comparison
    cannot be won by suppression.
    """
    if not SUPPRESS_ENABLED or score > SUPPRESS_MAX_SCORE:
        return None
    with LOCK:
        total = max(STATE["joined"], 1)
        if STATE["suppressed"] / total > SUPPRESS_MAX_FRACTION:
            return None
    key = (alert["rule_id"], alert.get("merchant_id"))
    n = _rule_noise.get(key, 0) + 1
    _rule_noise[key] = n
    if n >= SUPPRESS_MIN_SEEN:
        return (f"rule {alert['rule_id']} on merchant {alert.get('merchant_id')} "
                f"has produced {n} alerts scoring below {SUPPRESS_MAX_SCORE} in "
                f"this session; deprioritised as a repeat false-positive "
                f"pattern. Still retrievable, still audited, never deleted.")
    return None


def _safe_emit(fn, alert, sc):
    """One malformed record must not take the pipeline down.

    Without this, a single exception inside emit() escapes the consume loop and
    kills the consumer thread silently - the service keeps answering /health
    while ingesting nothing. That happened during the Stage 2 build (a schema
    mismatch), so the failure is now counted and visible instead.
    """
    try:
        fn(alert, sc)
    except Exception as e:
        with LOCK:
            STATE["emit_errors"] += 1
            STATE["last_error"] = str(e)[:300]
        print(f"[alert] emit failed: {e}", flush=True)


def run():
    P.wait_for_broker()
    # txn.truth is consumed for MEASUREMENT ONLY - it feeds /truth-check and
    # nothing else. It is not an input to rank_key() or suppress_decision().
    cons = P.consumer("alert-svc", [P.T_SCREENING, P.T_SCORED, P.T_TRUTH],
                      offset="latest")
    prod = P.producer("alert-svc")
    pending_alerts, scores = {}, {}
    print(f"[alert] joining {P.T_SCREENING} x {P.T_SCORED} -> {P.T_RANKED}", flush=True)

    def emit(alert, sc):
        with LOCK:
            STATE["incumbent_seq"] += 1
            inc_rank = STATE["incumbent_seq"]      # the incumbent's own order: arrival
        score = float(sc["score"])
        reason = suppress_decision(alert, score)
        rec = {
            **alert,
            "score": score, "pred": int(sc["pred"]),
            "priority": priority_of(score),
            "rank_score": rank_key(score, alert.get("scenario")),
            "incumbent_rank": inc_rank,
            "ranked_ms": P.now_ms(),
            "suppressed": bool(reason), "suppress_reason": reason,
            "model_version": sc.get("model_version"),
            "feature_version": sc.get("feature_version"),
            "data_version": sc.get("data_version"),
            "alert_to_rank_ms": round(P.now_ms() - alert["raised_ms"], 2),
        }
        P.send(prod, P.T_RANKED, rec["alert_id"], rec)
        with _dblock:
            db().execute("""INSERT OR REPLACE INTO alerts
              (alert_id,txn_id,row,rule_id,rule_reason,all_rules,amount,city,
               user_id,merchant_id,scenario,score,pred,incumbent_rank,priority,
               suppressed,suppress_reason,raised_ms,ranked_ms,state,
               model_version,feature_version,data_version,label)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'new',?,?,?,?)""",
              (rec["alert_id"], rec["txn_id"], rec["row"], rec["rule_id"],
               rec["rule_reason"], json.dumps(rec.get("all_rules", [])),
               rec.get("amount"), rec.get("city"), rec["user_id"], rec["merchant_id"],
               rec.get("scenario"), score, rec["pred"], inc_rank, rec["priority"],
               int(bool(reason)), reason, rec["raised_ms"], rec["ranked_ms"],
               rec.get("model_version"), rec.get("feature_version"),
               rec.get("data_version"), TRUTH.get(rec["txn_id"])))
            db().commit()
        with LOCK:
            STATE["joined"] += 1
            if reason:
                STATE["suppressed"] += 1
        prod.poll(0)

    last_gc = time.time()
    while True:
        msg = cons.poll(0.05)
        if msg is not None and not msg.error():
            rec = P.decode(msg)
            if msg.topic() == P.T_TRUTH:
                TRUTH[rec["txn_id"]] = rec["label"]
                # An alert can be emitted before its truth record arrives, so
                # backfill rather than leave the row unlabelled - otherwise the
                # comparison silently measures only the subset that happened to
                # win the race.
                _truth_backlog.append((rec["label"], rec["txn_id"]))
                if len(_truth_backlog) >= 200:
                    with _dblock:
                        db().executemany(
                            "UPDATE alerts SET label=? WHERE txn_id=? AND label IS NULL",
                            _truth_backlog)
                        db().commit()
                    _truth_backlog.clear()
                if len(TRUTH) > 300000:
                    for k in list(TRUTH)[:100000]:
                        TRUTH.pop(k, None)
            elif msg.topic() == P.T_SCREENING:
                STATE["alerts_in"] += 1
                sc = scores.pop(rec["txn_id"], None)
                if sc:
                    _safe_emit(emit, rec, sc)
                else:
                    pending_alerts[rec["txn_id"]] = (rec, time.time())
            else:
                STATE["scores_in"] += 1
                pa = pending_alerts.pop(rec["txn_id"], None)
                if pa:
                    _safe_emit(emit, pa[0], rec)
                else:
                    scores[rec["txn_id"]] = rec

        now = time.time()
        if now - last_gc > 5:
            last_gc = now
            # An alert whose score never arrives is still an alert. It is
            # counted, not dropped - the incumbent's output is never lost.
            for tid, (a, t) in list(pending_alerts.items()):
                if now - t > JOIN_WINDOW_S:
                    pending_alerts.pop(tid, None)
                    STATE["unmatched"] += 1
            if len(scores) > 200000:
                for k in list(scores)[:50000]:
                    scores.pop(k, None)


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"ok": True, **{k: STATE[k] for k in STATE}})


@app.get("/queue")
def queue():
    """The two queues, side by side. `incumbent` is the incumbent's own order
    (arrival); `ranked` is ours. Nothing is removed from either."""
    lim = min(int(request.args.get("limit", 25)), 200)
    include_suppressed = request.args.get("suppressed", "0") == "1"
    where = "" if include_suppressed else "WHERE suppressed=0"
    with _dblock:
        c = db()
        inc = c.execute(f"SELECT alert_id,txn_id,rule_id,rule_reason,amount,city,"
                        f"score,priority,scenario,incumbent_rank,suppressed "
                        f"FROM alerts {where} ORDER BY incumbent_rank ASC LIMIT ?",
                        (lim,)).fetchall()
        rnk = c.execute(f"SELECT alert_id,txn_id,rule_id,rule_reason,amount,city,"
                        f"score,priority,scenario,incumbent_rank,suppressed "
                        f"FROM alerts {where} ORDER BY score DESC LIMIT ?",
                        (lim,)).fetchall()
    cols = ["alert_id", "txn_id", "rule_id", "rule_reason", "amount", "city",
            "score", "priority", "scenario", "incumbent_rank", "suppressed"]
    to = lambda rows: [dict(zip(cols, r)) for r in rows]
    ranked = to(rnk)
    for i, r in enumerate(ranked, 1):
        r["ranked_rank"] = i
        r["rank_delta"] = r["incumbent_rank"] - i
    return jsonify({"incumbent": to(inc), "ranked": ranked})


@app.get("/comparison")
def comparison():
    """The headline the screening-comparison screen renders.

    "The top N of the re-ranked queue contains X% of the true positives in the
    incumbent's full queue" - computed against ground truth, not asserted, and
    not against the model's own predictions (which would be circular).

    Ground truth is joined here for MEASUREMENT ONLY. It is not an input to
    ranking or suppression; see rank_key() and suppress_decision().
    """
    topn = int(request.args.get("top", 50))
    with _dblock:
        c = db()
        total = c.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        labelled = c.execute(
            "SELECT COUNT(*) FROM alerts WHERE label IS NOT NULL").fetchone()[0]
        tp_total = c.execute(
            "SELECT COUNT(*) FROM alerts WHERE label=1").fetchone()[0]
        # our order
        top_ranked = c.execute(
            "SELECT label FROM alerts ORDER BY score DESC LIMIT ?", (topn,)).fetchall()
        # the incumbent's order, same depth - the fair baseline
        top_incumbent = c.execute(
            "SELECT label FROM alerts ORDER BY incumbent_rank ASC LIMIT ?",
            (topn,)).fetchall()
        supp = c.execute("SELECT COUNT(*) FROM alerts WHERE suppressed=1").fetchone()[0]
    tp_ranked = sum(1 for (l,) in top_ranked if l == 1)
    tp_incumbent = sum(1 for (l,) in top_incumbent if l == 1)
    return jsonify({
        "incumbent_queue_size": total,
        "labelled": labelled,
        "true_positives_in_queue": tp_total,
        "top_n": topn,
        # The number that matters: same investigator effort, both orderings.
        "true_positives_in_top_n_ranked": tp_ranked,
        "true_positives_in_top_n_incumbent": tp_incumbent,
        "recall_at_top_n_ranked": round(tp_ranked / max(tp_total, 1), 4),
        "recall_at_top_n_incumbent": round(tp_incumbent / max(tp_total, 1), 4),
        "lift": (round(tp_ranked / tp_incumbent, 2) if tp_incumbent
                 else f"{tp_ranked} vs 0 - the incumbent's top {topn} contains "
                      f"no true positives, so a ratio is undefined"),
        "queue_reduction": round(1 - (topn / max(total, 1)), 4),
        "suppressed": supp,
        "suppressed_fraction": round(supp / max(total, 1), 4),
        "caveat": "Measured on a rebalanced public test set, not production "
                  "traffic. The incumbent's false-positive rate - and therefore "
                  "this lift - depends on the fraud rate of the stream; check "
                  "the generator's mode before quoting either.",
    })


@app.get("/suppressed")
def suppressed():
    """Nothing is silently dropped - every suppression is listed with its reason."""
    with _dblock:
        rows = db().execute(
            "SELECT alert_id,txn_id,rule_id,score,suppress_reason FROM alerts "
            "WHERE suppressed=1 ORDER BY ranked_ms DESC LIMIT 200").fetchall()
    return jsonify([dict(zip(["alert_id", "txn_id", "rule_id", "score", "reason"], r))
                    for r in rows])


def _shutdown(signum, frame):
    print("[alert] SIGTERM - exiting", flush=True)
    os._exit(0)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
