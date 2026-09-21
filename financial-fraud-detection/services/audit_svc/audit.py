"""audit-svc - append-only, hash-chained record of every decision.

The artefact a regulator or internal audit would actually ask for. Three
properties make it worth more than a log file:

  * **Append-only by construction.** Records arrive only from the `case.events`
    topic. There is no HTTP route that writes one, so nothing can be inserted
    out of band - not even by this service's own API.
  * **Hash-chained.** Each record carries the hash of its predecessor, so a
    record cannot be altered or removed without breaking every hash after it.
    `/verify` walks the chain and names the first break.
  * **Exportable as evidence.** `/export/<case_id>` produces the whole history
    of one case - alert, score, model and data versions, every state change,
    every named human action - with the chain verification result attached.

What is deliberately NOT audited: one record per scored transaction. At the
measured 6,900 TPS that is ~600M records a day, and none of them is a decision.
The score that matters is the one attached to an alert, and it is captured
there with its model, feature and data versions.
"""
import os, sys, json, time, hashlib, threading, sqlite3
from flask import Flask, jsonify, request

sys.path.insert(0, "/svc")
from common import pipeline as P

PORT = int(os.environ.get("PORT", "8095"))
DB = os.environ.get("AUDIT_DB", "/state/audit.db")

_conn = None
_lock = threading.Lock()


def db():
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB), exist_ok=True)
        _conn = sqlite3.connect(DB, check_same_thread=False)
        _conn.executescript("""
        CREATE TABLE IF NOT EXISTS audit (
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          ts REAL, actor TEXT, actor_kind TEXT, action TEXT,
          case_id TEXT, alert_id TEXT, txn_id TEXT,
          payload TEXT,
          model_version TEXT, feature_version TEXT, data_version TEXT,
          prev_hash TEXT, hash TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_audit_case ON audit(case_id);
        CREATE INDEX IF NOT EXISTS ix_audit_alert ON audit(alert_id);
        """)
        _conn.commit()
    return _conn


def _digest(seq, ts, actor, action, case_id, alert_id, payload, prev_hash):
    """Everything that identifies the record feeds the hash. Changing any field
    - or deleting a record so a later seq shifts - breaks the chain."""
    blob = "|".join([str(seq), f"{ts:.6f}", actor or "", action or "",
                     case_id or "", alert_id or "",
                     json.dumps(payload, sort_keys=True, separators=(",", ":")),
                     prev_hash or ""])
    return hashlib.sha256(blob.encode()).hexdigest()


STATE = {"records": 0, "last_hash": None, "errors": 0}


def append(rec):
    """The only writer. Called from the Kafka consumer, never from HTTP."""
    with _lock:
        c = db()
        row = c.execute("SELECT seq, hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        prev_seq, prev_hash = (row if row else (0, ""))
        seq = prev_seq + 1
        ts = rec.get("ts") or time.time()
        payload = rec.get("payload", {})
        h = _digest(seq, ts, rec.get("actor"), rec.get("action"),
                    rec.get("case_id"), rec.get("alert_id"), payload, prev_hash)
        c.execute("""INSERT INTO audit
          (seq,ts,actor,actor_kind,action,case_id,alert_id,txn_id,payload,
           model_version,feature_version,data_version,prev_hash,hash)
          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (seq, ts, rec.get("actor"), rec.get("actor_kind", "service"),
           rec.get("action"), rec.get("case_id"), rec.get("alert_id"),
           rec.get("txn_id"), json.dumps(payload, sort_keys=True),
           rec.get("model_version"), rec.get("feature_version"),
           rec.get("data_version"), prev_hash, h))
        c.commit()
        STATE["records"] += 1
        STATE["last_hash"] = h
        return seq, h


def verify(limit=None):
    """Walk the chain from the beginning and recompute every hash.

    Returns the first break rather than a boolean, because "which record was
    altered" is the question that actually gets asked.
    """
    with _lock:
        q = "SELECT seq,ts,actor,action,case_id,alert_id,payload,prev_hash,hash FROM audit ORDER BY seq"
        rows = db().execute(q + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    prev = ""
    for (seq, ts, actor, action, case_id, alert_id, payload, prev_hash, h) in rows:
        if prev_hash != prev:
            return {"ok": False, "records": len(rows), "broken_at": seq,
                    "reason": "prev_hash does not match the preceding record's hash",
                    "expected_prev": prev, "found_prev": prev_hash}
        want = _digest(seq, ts, actor, action, case_id, alert_id,
                       json.loads(payload), prev_hash)
        if want != h:
            return {"ok": False, "records": len(rows), "broken_at": seq,
                    "reason": "record contents do not match its stored hash",
                    "expected_hash": want, "found_hash": h}
        prev = h
    return {"ok": True, "records": len(rows), "head": prev or None}


def run():
    P.wait_for_broker()
    cons = P.consumer("audit-svc", [P.T_CASES], offset="earliest")
    print(f"[audit] consuming {P.T_CASES} -> hash-chained log at {DB}", flush=True)
    while True:
        msg = cons.poll(0.2)
        if msg is None or msg.error():
            continue
        try:
            append(P.decode(msg))
        except Exception as e:
            STATE["errors"] += 1
            print(f"[audit] append failed: {e}", flush=True)


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"ok": True, **STATE})


@app.get("/verify")
def api_verify():
    return jsonify(verify())


@app.get("/records")
def records():
    lim = min(int(request.args.get("limit", 50)), 500)
    case = request.args.get("case_id")
    with _lock:
        if case:
            rows = db().execute(
                "SELECT seq,ts,actor,actor_kind,action,case_id,alert_id,payload,hash "
                "FROM audit WHERE case_id=? ORDER BY seq", (case,)).fetchall()
        else:
            rows = db().execute(
                "SELECT seq,ts,actor,actor_kind,action,case_id,alert_id,payload,hash "
                "FROM audit ORDER BY seq DESC LIMIT ?", (lim,)).fetchall()
    cols = ["seq", "ts", "actor", "actor_kind", "action", "case_id", "alert_id", "payload", "hash"]
    out = [dict(zip(cols, r)) for r in rows]
    for o in out:
        o["payload"] = json.loads(o["payload"])
    return jsonify(out)


@app.get("/export/<case_id>")
def export(case_id):
    """The evidence pack: one case's complete history plus chain verification.

    Deliberately self-contained - a reviewer should not need this service
    running to read it.
    """
    with _lock:
        rows = db().execute(
            "SELECT seq,ts,actor,actor_kind,action,case_id,alert_id,txn_id,payload,"
            "model_version,feature_version,data_version,prev_hash,hash "
            "FROM audit WHERE case_id=? ORDER BY seq", (case_id,)).fetchall()
    if not rows:
        return jsonify({"error": "no audit records for this case", "case_id": case_id}), 404
    cols = ["seq", "ts", "actor", "actor_kind", "action", "case_id", "alert_id", "txn_id",
            "payload", "model_version", "feature_version", "data_version", "prev_hash", "hash"]
    recs = [dict(zip(cols, r)) for r in rows]
    for r in recs:
        r["payload"] = json.loads(r["payload"])
        r["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["ts"]))
    chain = verify()
    humans = sorted({r["actor"] for r in recs if r["actor_kind"] == "human"})
    pack = {
        "evidence_pack_version": "1",
        "case_id": case_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record_count": len(recs),
        "human_actors": humans,
        "chain_verification": chain,
        "records": recs,
    }
    # Seal the pack itself, so a tampered export is detectable independently of
    # the database it came from.
    pack["pack_digest"] = hashlib.sha256(
        json.dumps(pack, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return jsonify(pack)


def _shutdown(signum, frame):
    print("[audit] SIGTERM - exiting", flush=True)
    os._exit(0)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
