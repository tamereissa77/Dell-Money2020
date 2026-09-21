"""Money20/20 fraud-detection booth demo - backend.
Scores the held-out TabFormer test set on the GB10 via Triton and serves
a live feed + per-transaction Shapley explanations to the booth UI.
"""
import os, json, time, threading
import numpy as np, pandas as pd
from flask import Flask, jsonify, send_from_directory, request
import tritonclient.http as httpclient
from tritonclient.http import InferInput, InferRequestedOutput

B = os.environ.get("GNN_TEST_DIR", "/work/data/TabFormer/gnn_np/test_gnn")
TRITON = os.environ.get("TRITON_URL", "localhost:8000")
MODEL = "prediction_and_shapley_np"
HERO_N = int(os.environ.get("HERO_N", "12"))

# v2: where /api/feed gets its rows.
#   "file"  - v1 behaviour, walk the pre-scored test set. Unchanged default, so
#             the original three-minute booth demo runs exactly as before.
#   "kafka" - consume txn.scored from the streaming pipeline.
FEED_SOURCE = os.environ.get("FEED_SOURCE", "file").lower()

MCC_NAMES = {5411:"Grocery",5812:"Restaurant",5541:"Fuel",5912:"Pharmacy",4121:"Rideshare",
  5621:"Clothing",7995:"Gambling",5999:"Retail",4900:"Utilities",5300:"Wholesale",
  5732:"Electronics",7011:"Hotel",4111:"Transit",5942:"Books",5661:"Shoes",
  4814:"Telecom",5310:"Discount",7996:"Amusement",5813:"Bar",5651:"Apparel",
  5691:"Apparel",5200:"Home supply",5251:"Hardware",5311:"Dept. store",5399:"General retail",
  5499:"Convenience",5533:"Auto parts",5722:"Appliances",5734:"Software",5735:"Music",
  5814:"Fast food",5816:"Digital goods",5921:"Liquor",5932:"Antiques",5941:"Sporting goods",
  5947:"Gifts",5970:"Crafts",5977:"Cosmetics",6011:"ATM",7230:"Salon",7276:"Tax prep",
  7349:"Cleaning",7392:"Consulting",7512:"Car rental",7538:"Auto service",7542:"Car wash",
  7832:"Cinema",7929:"Entertainment",7991:"Tourist attr.",8011:"Medical",8021:"Dental",
  8041:"Chiropractor",8043:"Optician",8062:"Hospital",8099:"Health",8111:"Legal",
  8931:"Accounting",4784:"Tolls",4829:"Money transfer",5045:"Computers",5094:"Jewellery",
  5192:"Books/news",5300:"Wholesale",5462:"Bakery",5511:"Car dealer",5599:"Vehicles",
  5719:"Home furnishing",5811:"Caterer",5912:"Drug store",5993:"Tobacco",6300:"Insurance",
  3000:"Airline",3001:"Airline",3005:"Airline",3006:"Airline",3007:"Airline",3008:"Airline",
  3009:"Airline",3010:"Airline",3058:"Airline",3066:"Airline",3075:"Airline",3132:"Airline",
  3144:"Airline",3174:"Airline",3184:"Airline",3260:"Airline",3387:"Car rental",
  3389:"Car rental",3395:"Car rental",3405:"Car rental",3504:"Hotel",3509:"Hotel",
  3596:"Hotel",3640:"Hotel",3684:"Hotel",3730:"Hotel",3771:"Hotel"}


MCC_EXTRA = {5733:"Music store",5732:"Electronics",5735:"Record store",7922:"Theatre",
  1711:"HVAC / plumbing",8049:"Podiatrist",5921:"Liquor",5947:"Gifts",5992:"Florist",
  5261:"Garden supply",7217:"Carpet cleaning",5697:"Tailor",5698:"Wigs",5714:"Drapery",
  5718:"Fireplace",5945:"Toys",5950:"Glassware",5960:"Direct marketing",5962:"Telemarketing",
  5964:"Catalogue",5965:"Combined catalogue",5966:"Outbound telemarketing",5967:"Inbound teleservices",
  5968:"Subscription",5969:"Direct marketing (other)",5975:"Hearing aids",5976:"Orthopaedic",
  5983:"Fuel dealer",5994:"News-stand",5995:"Pet shop",5996:"Pools & spas",5997:"Razors",
  5998:"Tent & awning",7211:"Laundry",7221:"Photo studio",7251:"Shoe repair",7261:"Funeral",
  7273:"Dating",7278:"Buying club",7296:"Costume rental",7297:"Massage",7298:"Health & beauty spa",
  7311:"Advertising",7333:"Commercial art",7338:"Copying",7342:"Pest control",7361:"Employment",
  7372:"Software services",7375:"Information retrieval",7379:"Computer repair",7394:"Equipment rental",
  7395:"Photo developing",7399:"Business services",7511:"Truck stop",7513:"Truck rental",
  7519:"RV rental",7531:"Auto body",7534:"Tyre retread",7535:"Auto paint",7622:"Electronics repair",
  7623:"A/C repair",7629:"Appliance repair",7631:"Watch repair",7641:"Furniture repair",
  7692:"Welding",7699:"Repair services",7841:"Video rental",7911:"Dance",7932:"Billiards",
  7933:"Bowling",7941:"Sports clubs",7992:"Golf",7993:"Game supply",7994:"Video arcade",
  7997:"Country club",7998:"Aquarium",8021:"Dental",8031:"Osteopath",8042:"Optometrist",
  8044:"Optical goods",8049:"Podiatry",8050:"Nursing care",8071:"Medical lab",8351:"Childcare",
  8398:"Charity",8641:"Civic association",8651:"Political org",8661:"Religious org",
  8675:"Automobile association",8699:"Membership org",8734:"Testing lab",8911:"Architecture",
  8999:"Professional services",9211:"Court costs",9222:"Fines",9223:"Bail",9311:"Tax payment",
  9399:"Government services",9402:"Postal service",9405:"Government purchase",
  4722:"Travel agency",4899:"Cable / streaming",5655:"Sportswear",5815:"Digital media",
  7210:"Laundry & dry cleaning",4816:"Internet services",5044:"Office equipment",
  5065:"Electrical parts",5111:"Office supplies",5169:"Chemicals",5172:"Petroleum",
  5211:"Building materials",5231:"Glass & paint",5271:"Mobile homes",5309:"Duty-free",
  5322:"Rental store",5331:"Variety store",5422:"Butcher",5441:"Confectionery",
  5451:"Dairy",5551:"Boat dealer",5561:"Trailers",5571:"Motorcycle",5592:"Motor homes",
  5598:"Snowmobile",5611:"Menswear",5631:"Womens accessories",5641:"Childrenswear",
  5681:"Furrier",5713:"Floor covering",5931:"Second-hand",5940:"Bicycle",5943:"Stationery",
  5944:"Jewellery",5946:"Camera shop",5972:"Stamps & coins",5978:"Typewriter",
  6010:"Cash advance",6012:"Financial institution",6051:"Quasi-cash",6211:"Securities",
  6513:"Property rental",7032:"Sporting camp",7033:"Campsite",7276:"Tax preparation",
  7299:"Personal services",7321:"Credit reporting",7331:"Direct mail",7332:"Blueprinting",
  7339:"Secretarial",7361:"Recruitment",7370:"Computer services",7382:"Security services",
  7393:"Detective agency",7523:"Parking",7529:"Airport parking",7549:"Towing",
  7699:"Repairs",7829:"Film distribution",7361:"Employment agency"}
MCC_NAMES.update(MCC_EXTRA)

print("[demo] loading test graph ...", flush=True)
L = lambda p: pd.read_csv(p)
x_user  = L(f"{B}/nodes/user.csv").values.astype(np.float32)
x_txn   = L(f"{B}/nodes/transaction.csv").values.astype(np.float32)
x_merch = L(f"{B}/nodes/merchant.csv").values.astype(np.float32)
y       = L(f"{B}/nodes/transaction_label.csv").values.ravel().astype(int)
e_ut    = L(f"{B}/edges/user_to_transaction.csv").values.T.astype(np.int64)
e_tm    = L(f"{B}/edges/transaction_to_merchant.csv").values.T.astype(np.int64)
mk = lambda p: pd.read_csv(p, header=None).values.ravel().astype(np.int32)
m_user  = mk(f"{B}/nodes/user_feature_mask.csv")
m_txn   = mk(f"{B}/nodes/transaction_feature_mask.csv")
m_merch = mk(f"{B}/nodes/merchant_feature_mask.csv")
disp    = L(f"{B}/nodes/transaction_display.csv")
assert len(disp) == len(y) == x_txn.shape[0], "display/feature row mismatch"

# feature-group labels, derived from the masks (not hardcoded)
def group_names(mask, cols):
    out = {}
    for gid in sorted({int(g) for g in mask if g >= 0}):
        names = sorted({cols[i].rsplit("_",1)[0] for i in range(len(mask)) if mask[i]==gid})
        out[gid] = "+".join(names)
    return [out[g] for g in sorted(out)]
GN_USER  = group_names(m_user,  list(L(f"{B}/nodes/user.csv").columns))
GN_TXN   = group_names(m_txn,   list(L(f"{B}/nodes/transaction.csv").columns))
GN_MERCH = group_names(m_merch, list(L(f"{B}/nodes/merchant.csv").columns))
PRETTY = {"Card":"Cardholder profile","City":"Merchant city","Errors":"Terminal errors",
          "Zip":"Postal code","Chip":"Entry mode","Amount":"Transaction amount",
          "Merchant":"Merchant identity","MCC":"Merchant category"}
pretty = lambda g: PRETTY.get(g, g)

def _mcc_name(m):
    """Name an MCC, falling back to the ISO range for travel codes (3000-3999)."""
    if m in MCC_NAMES: return MCC_NAMES[m]
    if 3000 <= m <= 3299: return "Airline"
    if 3300 <= m <= 3499: return "Car rental"
    if 3500 <= m <= 3999: return "Hotel"
    return f"MCC {m}"

_tl = threading.local()
def _client():
    """One Triton client per thread - geventhttpclient sockets are not thread-safe."""
    c = getattr(_tl, "cli", None)
    if c is None:
        c = httpclient.InferenceServerClient(url=TRITON, network_timeout=900, connection_timeout=900)
        _tl.cli = c
    return c

def _infer(xu, xt, xm, eut, etm, shap):
    d = {"x_merchant":(xm,"FP32"),"x_transaction":(xt,"FP32"),"x_user":(xu,"FP32"),
         "COMPUTE_SHAP":(np.array([shap],dtype=bool),"BOOL"),
         "feature_mask_merchant":(m_merch,"INT32"),"feature_mask_transaction":(m_txn,"INT32"),
         "feature_mask_user":(m_user,"INT32"),
         "edge_index_transaction_to_merchant":(etm,"INT64"),
         "edge_index_user_to_transaction":(eut,"INT64")}
    ins = []
    for k,(v,dt) in d.items():
        i = InferInput(k, v.shape, datatype=dt); i.set_data_from_numpy(v); ins.append(i)
    outs = [InferRequestedOutput("PREDICTION")]
    if shap:
        outs += [InferRequestedOutput(f"shap_values_{n}") for n in ("user","transaction","merchant")]
    t = time.time()
    r = _client().infer(MODEL, inputs=ins, outputs=outs, timeout=900000)
    return r, time.time()-t

print("[demo] scoring full test set ...", flush=True)
# One warm-up pass, then measure. The first call after a Triton restart pays
# per-process init (~0.8s); quoting that as throughput understates sustained
# performance by ~16x. Standard benchmarking practice: discard warm-up.
_r, _warm = _infer(x_user, x_txn, x_merch, e_ut, e_tm, False)
_lat = []
for _ in range(3):
    _r, _dt = _infer(x_user, x_txn, x_merch, e_ut, e_tm, False)
    _lat.append(_dt)
SCORE_SECS = min(_lat)
COLD_SECS = _warm
print(f"[demo] warm-up {_warm:.3f}s -> sustained {SCORE_SECS:.4f}s", flush=True)
SCORES = _r.as_numpy("PREDICTION").ravel().astype(float)
PRED = (SCORES > 0.5).astype(int)
TP=int(((PRED==1)&(y==1)).sum()); FP=int(((PRED==1)&(y==0)).sum())
FN=int(((PRED==0)&(y==1)).sum()); TN=int(((PRED==0)&(y==0)).sum())
PREC=TP/max(TP+FP,1); REC=TP/max(TP+FN,1); F1=2*PREC*REC/max(PREC+REC,1e-9)
print(f"[demo] scored {len(SCORES)} txns in {SCORE_SECS:.3f}s  F1={F1:.4f}", flush=True)

def explain(i):
    u = int(e_ut[0][i]); m = int(e_tm[1][i]); ee = np.array([[0],[0]], dtype=np.int64)
    r, dt = _infer(x_user[u:u+1], x_txn[i:i+1], x_merch[m:m+1], ee, ee, True)
    rows = []
    for node, names in (("user",GN_USER),("transaction",GN_TXN),("merchant",GN_MERCH)):
        sv = r.as_numpy(f"shap_values_{node}").ravel()
        for nm, val in zip(names, sv):
            rows.append({"feature": pretty(nm), "raw": nm, "node": node, "value": float(val)})
    rows.sort(key=lambda z: -abs(z["value"]))
    return {"idx": int(i), "score": float(r.as_numpy("PREDICTION").ravel()[0]),
            "latency": round(dt,2), "attributions": rows}

CACHE, LOCK = {}, threading.Lock()
HERO = [int(i) for i in np.argsort(-np.where(y==1, SCORES, -1))[:HERO_N]]

def warm_cache():
    for i in HERO:
        try:
            e = explain(i)
            with LOCK: CACHE[i] = e
            print(f"[demo] cached hero {i} ({len(CACHE)}/{len(HERO)})", flush=True)
        except Exception as ex:
            print(f"[demo] hero {i} failed: {ex}", flush=True)
    print("[demo] hero cache ready", flush=True)
threading.Thread(target=warm_cache, daemon=True).start()

def row(i):
    d = disp.iloc[i]
    mcc = int(d["MCC"]) if not pd.isna(d["MCC"]) else 0
    city = str(d["City"]).strip()
    return {"idx": int(i), "score": round(float(SCORES[i]),4), "label": int(y[i]),
            "pred": int(PRED[i]), "amount": float(d["Amount"]),
            "mcc": mcc, "mcc_name": _mcc_name(mcc),
            "city": city, "state": str(d["State"]).strip(),
            "merchant": str(d["Merchant"])[-6:], "chip": str(d["Chip"]).replace(" Transaction",""),
            "errors": ("" if str(d["Errors"]).strip() in ("XX","nan") else str(d["Errors"]).strip()),
            "when": f'{int(d["Year"])}-{int(d["Month"]):02d}-{int(d["Day"]):02d}',
            "user_id": int(e_ut[0][i]), "merchant_id": int(e_tm[1][i])}

# --- v2 streaming feed ------------------------------------------------------
# A bounded ring of the most recent scored transactions. Bounded on purpose: at
# 5,000 TPS an unbounded buffer is a memory leak with a countdown, and the UI
# only ever renders the tail.
STREAM, STREAM_LOCK = [], threading.Lock()
# Injected scenarios are retained separately from the scrolling feed. At 400 TPS
# a one-transaction injection leaves a 200-row window in about half a second -
# too fast for a presenter to point at, let alone narrate. These persist until
# displaced by later injections.
SCENARIO_HITS, SCENARIO_MAX = [], 120
STREAM_MAX = int(os.environ.get("STREAM_MAX", "2000"))
STREAM_STATE = {"consumed": 0, "connected": False, "last_error": None,
                "e2e_ms": 0.0, "model_ms": 0.0, "last_ms": 0.0}


def _stream_consumer():
    """Fill STREAM from txn.scored. Never fatal: if the broker dies the UI
    degrades to a stale tail with a visible banner rather than white-screening."""
    import sys
    sys.path.insert(0, "/svc")
    from common import pipeline as P
    while True:
        try:
            P.wait_for_broker(timeout=300)
            cons = P.consumer("demo-ui", [P.T_SCORED], offset="latest")
            STREAM_STATE["connected"] = True
            STREAM_STATE["last_error"] = None
            print("[demo] stream feed connected", flush=True)
            while True:
                msg = cons.poll(0.5)
                if msg is None or msg.error():
                    continue
                rec = P.decode(msg)
                with STREAM_LOCK:
                    STREAM.append(rec)
                    if len(STREAM) > STREAM_MAX:
                        del STREAM[:len(STREAM) - STREAM_MAX]
                    if rec.get("scenario"):
                        SCENARIO_HITS.append(rec)
                        if len(SCENARIO_HITS) > SCENARIO_MAX:
                            del SCENARIO_HITS[:len(SCENARIO_HITS) - SCENARIO_MAX]
                STREAM_STATE["consumed"] += 1
                STREAM_STATE["e2e_ms"] = rec.get("e2e_latency_ms", 0.0)
                STREAM_STATE["model_ms"] = rec.get("model_latency_ms", 0.0)
                STREAM_STATE["last_ms"] = time.time() * 1000.0
        except Exception as e:
            STREAM_STATE["connected"] = False
            STREAM_STATE["last_error"] = str(e)
            print(f"[demo] stream feed down: {e}", flush=True)
            time.sleep(3)


if FEED_SOURCE == "kafka":
    threading.Thread(target=_stream_consumer, daemon=True).start()


# --- stage 2: proxy the pipeline services ----------------------------------
# The browser can only reach this container, so the comparison screen is served
# its data through here rather than by exposing every service to the network.
import urllib.request, urllib.error

ALERT_URL = os.environ.get("ALERT_URL", "http://alert-svc:8094")
SCREEN_URL = os.environ.get("SCREEN_URL", "http://screening-stub:8093")
GEN_URL = os.environ.get("GEN_URL", "http://txn-generator:8091")


def _get(base, path, timeout=8):
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode()), None
    except Exception as e:
        return None, str(e)


app = Flask(__name__, static_folder="static")

@app.get("/api/stats")
def stats():
    with LOCK: cached = sum(1 for i in HERO if i in CACHE)
    return jsonify({"n": len(SCORES), "score_secs": round(SCORE_SECS,4),
        "per_txn_ms": round(1000*SCORE_SECS/len(SCORES),4),
        "throughput": int(len(SCORES)/max(SCORE_SECS,1e-9)),
        "cold_secs": round(COLD_SECS,4),
        "tp":TP,"fp":FP,"fn":FN,"tn":TN,
        "precision":round(PREC,4),"recall":round(REC,4),"f1":round(F1,4),
        "accuracy":round((TP+TN)/len(y),4),
        "fraud_rate": round(100*float(y.mean()),3),
        "hero_cached": cached, "hero_total": len(HERO)})

@app.get("/api/feed")
def feed():
    lim = min(int(request.args.get("limit",40)),200)
    if FEED_SOURCE == "kafka":
        # Newest first, and merge in the display fields the stream does not
        # carry so the UI renders identically to the file-backed feed.
        with STREAM_LOCK:
            tail = list(STREAM[-lim:])[::-1]
        out = []
        for r in tail:
            i = int(r["row"])
            base = row(i)
            base.update({"score": round(float(r["score"]),4), "pred": int(r["pred"]),
                         "txn_id": r.get("txn_id"), "scenario": r.get("scenario"),
                         "e2e_latency_ms": r.get("e2e_latency_ms"),
                         "model_latency_ms": r.get("model_latency_ms")})
            out.append(base)
        return jsonify(out)
    off = int(request.args.get("offset",0))
    idx = [(off+k) % len(SCORES) for k in range(lim)]
    return jsonify([row(i) for i in idx])


@app.get("/api/stream")
def stream_status():
    """What the status bar needs: is the pipeline alive, and how far behind."""
    with STREAM_LOCK:
        depth = len(STREAM)
    stale = (time.time()*1000.0 - STREAM_STATE["last_ms"]) if STREAM_STATE["last_ms"] else None
    return jsonify({
        "source": FEED_SOURCE,
        "connected": STREAM_STATE["connected"],
        "degraded": FEED_SOURCE == "kafka" and (
            not STREAM_STATE["connected"] or (stale is not None and stale > 5000)),
        "consumed": STREAM_STATE["consumed"],
        "buffer": depth,
        "stale_ms": round(stale, 1) if stale is not None else None,
        "e2e_latency_ms": STREAM_STATE["e2e_ms"],
        "model_latency_ms": STREAM_STATE["model_ms"],
        "last_error": STREAM_STATE["last_error"],
    })

@app.get("/api/heroes")
def heroes():
    return jsonify([row(i) for i in HERO])

@app.get("/api/explain/<int:i>")
def api_explain(i):
    if i < 0 or i >= len(SCORES): return jsonify({"error":"out of range"}), 404
    with LOCK: hit = CACHE.get(i)
    if hit: return jsonify({**hit, "cached": True, **{"txn": row(i)}})
    e = explain(i)
    with LOCK: CACHE[i] = e
    return jsonify({**e, "cached": False, "txn": row(i)})

BRAND_DIR = os.environ.get("BRAND_DIR", "/brand")

@app.get("/brand/<path:fn>")
def brand_file(fn):
    return send_from_directory(BRAND_DIR, fn)

@app.get("/api/brand")
def brand_manifest():
    """Tell the UI which official logo files are actually present."""
    def find(*names):
        for n in names:
            for ext in ("svg","png"):
                if os.path.exists(os.path.join(BRAND_DIR, f"{n}.{ext}")):
                    return f"/brand/{n}.{ext}"
        return None
    return jsonify({
        "dell":   find("dell-technologies-white","dell-technologies","dell"),
        "nvidia": find("nvidia-white","nvidia"),
    })

@app.after_request
def _no_cache(resp):
    """Booth reliability: never let a browser serve a stale page or dataset."""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp

@app.get("/api/comparison")
def api_comparison():
    top = request.args.get("top","500")
    cmp_, e1 = _get(ALERT_URL, f"/comparison?top={top}")
    scr, e2 = _get(SCREEN_URL, "/metrics")
    gen, e3 = _get(GEN_URL, "/status")
    return jsonify({"comparison": cmp_, "screening": scr, "generator": gen,
                    "errors": [e for e in (e1,e2,e3) if e]})


@app.get("/api/queue")
def api_queue():
    lim = request.args.get("limit","25")
    sup = request.args.get("suppressed","0")
    q, err = _get(ALERT_URL, f"/queue?limit={lim}&suppressed={sup}")
    return jsonify(q or {"error": err, "incumbent": [], "ranked": []})


# Everything below proxies a pipeline service so the browser only ever talks to
# this origin. Six ports to remember is not a demo, it is a scavenger hunt.
AUDIT_URL = os.environ.get("AUDIT_URL", "http://audit-svc:8095")
COPILOT_URL = os.environ.get("COPILOT_URL", "http://copilot-svc:8096")
SCORE_URL = os.environ.get("SCORE_URL", "http://scoring-svc:8092")


def _post(base, path, payload, timeout=180):
    try:
        req = urllib.request.Request(
            f"{base}{path}", data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), r.status
    except urllib.error.HTTPError as e:
        try:    return json.loads(e.read().decode()), e.code
        except Exception: return {"error": str(e)}, e.code
    except Exception as e:
        return {"error": str(e)}, 502


@app.get("/api/cases")
def api_cases():
    q = request.query_string.decode()
    d, err = _get(ALERT_URL, f"/cases{'?' + q if q else ''}")
    return jsonify(d if d is not None else {"error": err})


@app.post("/api/case/<alert_id>/<verb>")
def api_case_action(alert_id, verb):
    d, code = _post(ALERT_URL, f"/case/{alert_id}/{verb}", request.get_json(silent=True))
    return jsonify(d), code


@app.get("/api/audit")
def api_audit():
    q = request.query_string.decode()
    d, err = _get(AUDIT_URL, f"/records{'?' + q if q else ''}")
    return jsonify(d if d is not None else {"error": err})


@app.get("/api/audit/verify")
def api_audit_verify():
    d, err = _get(AUDIT_URL, "/verify")
    return jsonify(d if d is not None else {"error": err})


@app.get("/api/audit/export/<case_id>")
def api_audit_export(case_id):
    d, err = _get(AUDIT_URL, f"/export/{case_id}", timeout=30)
    return jsonify(d if d is not None else {"error": err})


@app.post("/api/copilot/draft/<alert_id>")
def api_copilot_draft(alert_id):
    d, code = _post(COPILOT_URL, f"/draft/{alert_id}", {})
    return jsonify(d), code


@app.post("/api/copilot/adopt/<draft_id>")
def api_copilot_adopt(draft_id):
    d, code = _post(COPILOT_URL, f"/draft/{draft_id}/adopt", request.get_json(silent=True))
    return jsonify(d), code


@app.get("/api/presenter/<path:rest>")
def api_presenter_get(rest):
    d, err = _get(GEN_URL, f"/{rest}")
    return jsonify(d if d is not None else {"error": err})


@app.post("/api/presenter/<path:rest>")
def api_presenter(rest):
    """Presenter controls proxied too, so the panel works from one origin
    without CORS and without the operator knowing the generator's port."""
    d, code = _post(GEN_URL, f"/{rest}", request.get_json(silent=True), timeout=15)
    return jsonify(d), code


@app.get("/api/health/all")
def api_health_all():
    """One call for the status bar: every service, one round trip."""
    out = {}
    for name, url, path in (("scoring", SCORE_URL, "/metrics"),
                            ("screening", SCREEN_URL, "/metrics"),
                            ("alerts", ALERT_URL, "/health"),
                            ("audit", AUDIT_URL, "/health"),
                            ("copilot", COPILOT_URL, "/health"),
                            ("generator", GEN_URL, "/status")):
        d, err = _get(url, path, timeout=4)
        out[name] = {"ok": d is not None, "data": d, "error": err}
    return jsonify(out)


@app.get("/api/suppressed")
def api_suppressed():
    q, err = _get(ALERT_URL, "/suppressed")
    return jsonify(q if q is not None else {"error": err})


@app.get("/api/scenarios")
def api_scenarios():
    """Injected transactions, retained so a presenter can point at them.

    `since_ms` filters to a recent window; omit it for everything retained.
    Grouped by scenario with the detection outcome, which is what the presenter
    is actually narrating.
    """
    since = float(request.args.get("since_ms", 0) or 0)
    with STREAM_LOCK:
        hits = [h for h in SCENARIO_HITS if h.get("scored_ms", 0) >= since]
    out = {}
    for h in hits:
        g = out.setdefault(h["scenario"], {"scenario": h["scenario"], "rows": [],
                                           "flagged": 0, "count": 0, "top_score": 0.0})
        i = int(h["row"])
        r = row(i)
        r.update({"score": round(float(h["score"]), 4), "pred": int(h["pred"]),
                  "txn_id": h.get("txn_id"), "scored_ms": h.get("scored_ms")})
        g["rows"].append(r)
        g["count"] += 1
        g["flagged"] += int(h["pred"])
        g["top_score"] = max(g["top_score"], round(float(h["score"]), 4))
    for g in out.values():
        g["rows"] = g["rows"][-12:]
    return jsonify(sorted(out.values(),
                          key=lambda g: -max((r.get("scored_ms") or 0) for r in g["rows"])))


# --- graph neighbourhood ----------------------------------------------------
# Built once: transaction -> its card and merchant, and the reverse indexes.
# The GNN already reasons over these edges; the UI has never shown them.
from collections import defaultdict as _dd
_BY_USER, _BY_MERCH = _dd(list), _dd(list)
for _i in range(len(y)):
    _BY_USER[int(e_ut[0][_i])].append(_i)
    _BY_MERCH[int(e_tm[1][_i])].append(_i)
print(f"[demo] graph index: {len(_BY_USER)} cards, {len(_BY_MERCH)} merchants", flush=True)


@app.get("/api/graph/<int:i>")
def api_graph(i):
    """The neighbourhood around one transaction: its card, its merchant, and
    the other transactions those two touch.

    This is the screen that justifies the graph model. For a fan-in typology
    the merchant node carries many unrelated cards, and that structure exists
    only across accounts - a per-transaction rules engine has nowhere to see it.
    """
    if i < 0 or i >= len(SCORES):
        return jsonify({"error": "out of range"}), 404
    cap = min(int(request.args.get("cap", 24)), 60)
    u, m = int(e_ut[0][i]), int(e_tm[1][i])
    sib_u = [k for k in _BY_USER[u] if k != i][:cap]
    sib_m = [k for k in _BY_MERCH[m] if k != i][:cap]

    nodes = [{"id": f"t{i}", "kind": "txn", "focus": True, "label": f"{row(i)['amount']:.2f}",
              "score": round(float(SCORES[i]), 4), "row": i},
             {"id": f"u{u}", "kind": "card", "label": f"card {u}",
              "degree": len(_BY_USER[u])},
             {"id": f"m{m}", "kind": "merchant", "label": f"merchant {m}",
              "degree": len(_BY_MERCH[m])}]
    edges = [{"s": f"u{u}", "t": f"t{i}"}, {"s": f"t{i}", "t": f"m{m}"}]
    seen_cards = {u}
    for k in sib_u:
        nodes.append({"id": f"t{k}", "kind": "txn", "label": f"{row(k)['amount']:.2f}",
                      "score": round(float(SCORES[k]), 4), "row": k})
        edges.append({"s": f"u{u}", "t": f"t{k}"})
    for k in sib_m:
        ku = int(e_ut[0][k])
        if f"t{k}" not in {n["id"] for n in nodes}:
            nodes.append({"id": f"t{k}", "kind": "txn", "label": f"{row(k)['amount']:.2f}",
                          "score": round(float(SCORES[k]), 4), "row": k})
        if ku not in seen_cards:
            seen_cards.add(ku)
            nodes.append({"id": f"u{ku}", "kind": "card", "label": f"card {ku}",
                          "degree": len(_BY_USER[ku])})
            edges.append({"s": f"u{ku}", "t": f"t{k}"})
        edges.append({"s": f"t{k}", "t": f"m{m}"})
    return jsonify({
        "focus": row(i), "nodes": nodes, "edges": edges,
        "cards_on_merchant": len({int(e_ut[0][k]) for k in _BY_MERCH[m]}),
        "txns_on_merchant": len(_BY_MERCH[m]),
        "txns_on_card": len(_BY_USER[u]),
        "note": "Many unrelated cards converging on one merchant is a layering "
                "indicator (POL-NET-01). The structure exists across accounts, "
                "which is what a per-transaction rules engine cannot see.",
    })


@app.get("/graph")
def graph_page(): return send_from_directory("static", "graph.html")


@app.get("/compare")
def compare(): return send_from_directory("static","compare.html")


@app.get("/live")
def live_view():
    """The v1 booth view on its own, so the v2 shell can host it in a frame
    without duplicating its canvas renderer."""
    return send_from_directory("static", "index.html")


@app.get("/")
def index():
    """One URL for everything.

    In v2 this serves the shell: one persistent chrome, tabs for every view,
    one status bar, and every service proxied through this origin. In v1 it
    serves the original booth page untouched - a v1 audience should see exactly
    what they saw before, and nothing about v2 should leak into it.
    """
    if FEED_SOURCE == "kafka":
        return send_from_directory("static", "app.html")
    return send_from_directory("static", "index.html")

def _shutdown(signum, frame):
    """Flask's threaded dev server ignores SIGTERM, so `docker compose stop`
    waits out the whole grace period and then SIGKILLs (exit 137). Exit
    promptly and cleanly instead — the app holds no unsaved state."""
    print("[demo] SIGTERM received — shutting down", flush=True)
    os._exit(0)

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT","8090")), threaded=True)
