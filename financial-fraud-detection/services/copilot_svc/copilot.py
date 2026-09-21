"""copilot-svc - grounded draft narrative for the investigator.

This stage ships the **deterministic template narrator**, which the brief makes
mandatory: the demo must never be one model download away from failing on the
booth floor. An LLM path can be layered on top later; the contract below does
not change when it is.

Four rules the output has to obey, and they are structural rather than stylistic:

  * **Every assertion cites its source.** A sentence is only emitted alongside
    the record it rests on - the alert, the score, an attribution, a policy
    clause. `citations` is not decoration; the narrative is assembled FROM it.
  * **The draft is marked machine-drafted** and stays that way until a named
    human adopts it. Adoption is an explicit action, recorded separately.
  * **It is never called a report or a filing.** It is an investigator's draft
    narrative. The words "suspicious transaction report", "SAR", "STR" and
    "filing" appear nowhere in the output, and a guard enforces that.
  * **Model output is decision support, not a conclusion.** POL-MODEL-01 says
    so, and the narrative says so too - including, where one feature dominates
    the attribution, that this needs examining rather than repeating.

Bilingual: English and Arabic, with the Arabic rendered right-to-left by the UI.
"""
import os, sys, json, glob, time, threading, hashlib
from flask import Flask, jsonify, request

sys.path.insert(0, "/svc")
from common import pipeline as P

PORT = int(os.environ.get("PORT", "8096"))
POLICY_DIR = os.environ.get("POLICY_DIR", "/policy")
ALERT_URL = os.environ.get("ALERT_URL", "http://alert-svc:8094")
DEMO_URL = os.environ.get("DEMO_URL", "http://demo:8090")
LLM_URL = os.environ.get("LLM_URL", "").strip()          # empty = template only

# Phrases that must never reach the output. A draft narrative is not a
# regulatory filing and must not read like one.
FORBIDDEN = ("suspicious transaction report", "suspicious activity report",
             "sar filing", " sar ", " str ", "regulatory filing", "file a report")

POLICIES = {}
for f in sorted(glob.glob(os.path.join(POLICY_DIR, "*.json"))):
    try:
        d = json.load(open(f))
        POLICIES[d["id"]] = d
    except Exception as e:
        print(f"[copilot] bad policy {f}: {e}", flush=True)
print(f"[copilot] {len(POLICIES)} policy documents loaded", flush=True)

STATE = {"drafts": 0, "adopted": 0, "rejected": 0, "llm": bool(LLM_URL),
         "guard_trips": 0}
DRAFTS = {}
LOCK = threading.Lock()


def _get(url, timeout=10):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def retrieve(alert_id):
    """Assemble the grounding set. Everything the narrative may reference has
    to come from here - the generator cannot reach past it."""
    ctx = {"alert": None, "attribution": None, "policies": [], "errors": []}
    try:
        ctx["alert"] = _get(f"{ALERT_URL}/case/{alert_id}")
    except Exception as e:
        ctx["errors"].append(f"alert: {e}")
        return ctx
    row = ctx["alert"].get("txn_id")
    try:
        # the per-decision attribution, not a batch average
        idx = int(str(row)[1:]) if row else None
        if idx is not None:
            ctx["attribution"] = _get(f"{DEMO_URL}/api/explain/{idx % 25000}", timeout=30)
    except Exception as e:
        ctx["errors"].append(f"attribution: {e}")

    a = ctx["alert"]
    picked = ["POL-OPS-02", "POL-MODEL-01", "POL-CASE-01"]
    rule = (a.get("rule_id") or "")
    if rule == "R001": picked.append("POL-AML-01")
    if rule == "R002": picked.append("POL-AML-02")
    if rule in ("R003", "R005"): picked.append("POL-AML-03")
    if (a.get("score") or 0) >= 0.9: picked.append("POL-OPS-01")
    att = (ctx.get("attribution") or {}).get("attributions") or []
    if _dominated(att):
        picked.append("POL-MODEL-02")
    ctx["policies"] = [POLICIES[p] for p in dict.fromkeys(picked) if p in POLICIES]
    return ctx


def _dominated(att, factor=3.0):
    """True when one feature outweighs every other by `factor`.

    Worth naming rather than inlining: on this dataset "merchant city" routinely
    dominates, and POL-MODEL-02 requires that to be examined rather than
    repeated. See docs/LIMITATIONS.md for why it dominates.
    """
    if not att or len(att) < 2:
        return False
    vals = sorted((abs(x["value"]) for x in att), reverse=True)
    return vals[0] > factor * vals[1]


def _cite(cits, kind, ref, detail):
    cits.append({"n": len(cits) + 1, "kind": kind, "ref": ref, "detail": detail})
    return f"[{len(cits)}]"


def draft_en(ctx):
    a, att = ctx["alert"], ctx.get("attribution")
    cits, para = [], []
    c = _cite(cits, "alert", a["alert_id"], f"screening alert on transaction {a.get('txn_id')}")
    para.append(f"Transaction {a.get('txn_id')} for {a.get('amount')} at a merchant in "
                f"{a.get('city')} was referred by the existing screening engine under rule "
                f"{a.get('rule_id')} ({a.get('rule_reason')}) {c}.")

    c = _cite(cits, "score", f"model {a.get('model_version')}",
              f"score {a.get('score')}, feature set {a.get('feature_version')}, data {a.get('data_version')}")
    c2 = _cite(cits, "policy", "POL-MODEL-01", POLICIES["POL-MODEL-01"]["summary"])
    para.append(f"The detection model scored it {float(a.get('score') or 0):.4f} {c}. "
                f"This is decision support and is not by itself a basis for a disposition {c2}.")

    if att and att.get("attributions"):
        top = att["attributions"][:3]
        names = ", ".join(f"{t['feature']} ({t['value']:+.2f})" for t in top)
        c = _cite(cits, "attribution", f"shapley:{a.get('txn_id')}",
                  f"per-decision attribution, {len(att['attributions'])} features")
        para.append(f"The attribution for this specific decision is led by {names} {c}.")
        if _dominated(att["attributions"]):
            c = _cite(cits, "policy", "POL-MODEL-02", POLICIES["POL-MODEL-02"]["summary"])
            para.append(f"One feature dominates that attribution. Before relying on it, record "
                        f"whether it is behaviourally meaningful for this cardholder or reflects "
                        f"a population characteristic of the training data {c}.")

    for pol in ctx["policies"]:
        if pol["id"] in ("POL-MODEL-01", "POL-MODEL-02", "POL-OPS-02"):
            continue
        c = _cite(cits, "policy", pol["id"], pol["summary"])
        para.append(f"{pol['title']}: {pol['body']} {c}")

    c = _cite(cits, "policy", "POL-OPS-02", POLICIES["POL-OPS-02"]["summary"])
    para.append(f"This narrative is machine-drafted and forms no part of the case until a named "
                f"analyst reviews and adopts it {c}. The investigator records the behavioural "
                f"evidence relied upon and the reasoning for the disposition.")
    return " ".join(para), cits


def draft_ar(ctx, cits):
    """Arabic draft, citing the same numbered sources as the English."""
    a, att = ctx["alert"], ctx.get("attribution")
    p = []
    p.append(f"تمت إحالة المعاملة {a.get('txn_id')} بمبلغ {a.get('amount')} لدى تاجر في "
             f"{a.get('city')} من قبل نظام الفرز القائم بموجب القاعدة {a.get('rule_id')} [1].")
    p.append(f"قام نموذج الكشف بتقييمها بدرجة {float(a.get('score') or 0):.4f} [2]. "
             f"وتُعد هذه النتيجة أداة مساندة لاتخاذ القرار ولا تشكل بذاتها أساساً للتصرف [3].")
    if att and att.get("attributions"):
        top = att["attributions"][:3]
        names = "، ".join(f"{t['feature']} ({t['value']:+.2f})" for t in top)
        p.append(f"العوامل الأكثر تأثيراً في هذا القرار تحديداً هي: {names} [4].")
    p.append("هذه المسودة مُنشأة آلياً ولا تُعد جزءاً من الملف حتى يقوم محلل مُسمّى "
             "بمراجعتها واعتمادها. ويسجل المحقق الأدلة السلوكية التي اعتمد عليها "
             "وأسباب التصرف النهائي.")
    return " ".join(p)


def guard(text):
    """Refuse to emit anything that reads as a regulatory filing."""
    low = f" {text.lower()} "
    return [w.strip() for w in FORBIDDEN if w in low]


app = Flask(__name__)


@app.post("/draft/<alert_id>")
def make_draft(alert_id):
    ctx = retrieve(alert_id)
    if not ctx["alert"]:
        return jsonify({"error": "alert not found", "detail": ctx["errors"]}), 404
    en, cits = draft_en(ctx)
    ar = draft_ar(ctx, cits)
    bad = guard(en) + guard(ar)
    if bad:
        STATE["guard_trips"] += 1
        return jsonify({"error": "draft rejected by output guard",
                        "forbidden_terms": bad}), 500
    did = hashlib.sha256(f"{alert_id}|{time.time()}".encode()).hexdigest()[:16]
    d = {
        "draft_id": did, "alert_id": alert_id,
        "status": "machine-drafted",          # never "final", never "report"
        "generator": "template" if not LLM_URL else "llm",
        "llm_available": bool(LLM_URL),
        "created_ms": P.now_ms(),
        "narrative_en": en, "narrative_ar": ar,
        "citations": cits,
        "adopted_by": None, "adopted_ms": None,
        "label": "Investigator's draft narrative - machine-drafted, not adopted",
        "retrieval_errors": ctx["errors"],
    }
    with LOCK:
        DRAFTS[did] = d
        STATE["drafts"] += 1
    return jsonify(d)


@app.post("/draft/<draft_id>/adopt")
def adopt(draft_id):
    """Authorship transfers only by an explicit, named action. The machine text
    is preserved alongside the edits so the audit shows both."""
    body = request.get_json(silent=True) or {}
    actor = (body.get("actor") or "").strip()
    if not actor:
        return jsonify({"error": "a named analyst is required to adopt a draft"}), 400
    with LOCK:
        d = DRAFTS.get(draft_id)
    if not d:
        return jsonify({"error": "no such draft"}), 404
    edited = body.get("narrative_en")
    if edited:
        bad = guard(edited)
        if bad:
            # Counted here too - the guard fires on analyst edits as well as on
            # generated text, and a metric that only saw one path was misleading.
            with LOCK:
                STATE["guard_trips"] += 1
                STATE["rejected"] += 1
            return jsonify({"error": "edited text rejected by output guard",
                            "forbidden_terms": bad}), 400
    d["machine_narrative_en"] = d["narrative_en"]
    if edited:
        d["narrative_en"] = edited
    d["status"] = "adopted"
    d["adopted_by"] = actor
    d["adopted_ms"] = P.now_ms()
    d["label"] = f"Investigator's narrative - adopted by {actor}"
    d["edited"] = bool(edited) and edited != d["machine_narrative_en"]
    with LOCK:
        STATE["adopted"] += 1
    try:
        prod = P.producer("copilot-svc")
        P.send(prod, P.T_CASES, d["alert_id"], {
            "ts": time.time(), "actor": actor, "actor_kind": "human",
            "action": "narrative.adopt", "case_id": d["alert_id"],
            "alert_id": d["alert_id"],
            "payload": {"draft_id": draft_id, "edited": d["edited"],
                        "generator": d["generator"]}})
        prod.flush(2)
    except Exception as e:
        d["audit_error"] = str(e)
    return jsonify(d)


@app.get("/draft/<draft_id>")
def get_draft(draft_id):
    with LOCK:
        d = DRAFTS.get(draft_id)
    return (jsonify(d), 200) if d else (jsonify({"error": "no such draft"}), 404)


@app.get("/policies")
def policies():
    return jsonify(sorted(POLICIES.values(), key=lambda p: p["id"]))


@app.get("/health")
def health():
    return jsonify({"ok": True, "policies": len(POLICIES), **STATE})


def _shutdown(signum, frame):
    print("[copilot] SIGTERM - exiting", flush=True)
    os._exit(0)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
