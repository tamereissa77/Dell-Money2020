import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure the directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
from config import LOAN_CONFIGS, LOAN_TYPE_DISPLAY, DOC_DISPLAY_NAMES
from graph import build_graph

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=template_dir)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# --- INITIAL STATE ---
print("[FLASK] 🏗️ Initializing...", flush=True)
graph = build_graph(use_checkpointer=False)
print("[FLASK] ✅ Ready.", flush=True)

@app.route("/")
def index():
    print("[FLASK] 🏠 Home page accessed!", flush=True)
    return render_template("index.html", 
                         loan_types=LOAN_TYPE_DISPLAY,
                         configs=LOAN_CONFIGS)

@app.route("/api/loan-types")
def get_loan_types():
    print("[FLASK] 🔍 Fetching loan types...", flush=True)
    types = []
    for key, display in LOAN_TYPE_DISPLAY.items():
        config = LOAN_CONFIGS.get(key, {})
        docs_info = []
        for d_key in config.get("required_docs", []):
            docs_info.append({
                "key": d_key,
                "display": DOC_DISPLAY_NAMES.get(d_key, d_key),
                "condition": config.get("conditions", {}).get(d_key, {}).get("description", "")
            })
        types.append({"key": key, "display": display, "docs": docs_info})
    return jsonify(types)

@app.route("/api/sample-docs")
def get_sample_docs():
    import base64
    print("[FLASK] 🖼️ Serving sample demo document images...", flush=True)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    doc_paths = {
        "national_id_front": os.path.join(root_dir, "shared_data", "test_id.jpg"),
        "national_id_back": os.path.join(root_dir, "shared_data", "test_input", "back", "Aya-Hassan-ID-Back_jpg.rf.3f74eb65498cba2d4c06ed33ceb90996.jpg"),
        "hr_letter": os.path.join(root_dir, "shared_data", "hr_doc_0.png"),
        "tax_card": os.path.join(root_dir, "data", "types", "بطاقة ضريبية.jfif"),
        "clinic_license": os.path.join(root_dir, "data", "types", "ترخيص عيادة.jfif"),
        "pension_transfer_pledge": os.path.join(root_dir, "data", "types", "تعهد.jfif"),
        "salary_transfer_pledge": os.path.join(root_dir, "data", "types", "تعهد.jfif"),
        "commercial_register": os.path.join(root_dir, "data", "types", "سجل تجاري.jfif"),
        "ownership_contract": os.path.join(root_dir, "data", "types", "عقد ايجار.jfif"),
        "utility_bill": os.path.join(root_dir, "data", "types", "فاتورة تليفون.jfif"),
        "electricity_bill": os.path.join(root_dir, "data", "types", "فاتورة كهرباء.jfif"),
        "syndicate_card": os.path.join(root_dir, "data", "types", "كارنيه نقابة.jfif"),
        "bank_statement": os.path.join(root_dir, "data", "types", "كشف حساب.jfif"),
        "practice_license": os.path.join(root_dir, "data", "types", "مزاولة مهنة.jfif")
    }
    
    samples = {}
    for key, path in doc_paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = "png" if path.endswith(".png") else "jpeg"
                samples[key] = f"data:image/{ext};base64,{b64}"
                
    return jsonify(samples)

@app.route("/api/process", methods=["POST"])
def process_loan():
    print("\n" + "="*70, flush=True)
    print("🚀 [AGENT THINKING & AUDIT] NEW LOAN REQUEST RECEIVED!", flush=True)
    data = request.json
    loan_type = data.get("loan_type")
    form_data = data.get("form_data", {})
    uploaded_docs = data.get("uploaded_docs", {})

    print(f"📌 Selected Loan Type: {loan_type}", flush=True)
    print(f"📝 HTML FORM DATA (Submitted by User):", flush=True)
    for k, v in form_data.items():
        print(f"   • {k:<22}: {v}", flush=True)
    print("-" * 70, flush=True)

    initial_state = {
        "loan_type": loan_type,
        "form_data": form_data,
        "uploaded_docs": uploaded_docs,
    }
    try:
        result = graph.invoke(initial_state)
        
        print("\n" + "="*70, flush=True)
        print("🧠 [AGENT DECISION & EXPLANATION]", flush=True)
        print(f"🎯 Final Decision : {result.get('final_decision')}", flush=True)
        print(f"📊 Overall Score  : {result.get('satisfaction_score', 0)*100:.1f}%", flush=True)
        print("❓ Reasons / Explanations:", flush=True)
        for r in result.get("reasons", []):
            print(f"   👉 {r}", flush=True)
            
        print("\n📄 Document Validation Details (How the Agent Scored Each Doc):", flush=True)
        val_results = result.get("validation_results", {})
        for doc_k, doc_info in val_results.items():
            doc_score = doc_info.get("overall_score", 0.0) * 100
            print(f"   ► [{doc_k}] -> Score: {doc_score:.1f}%", flush=True)
            for reason in doc_info.get("reasons", []):
                print(f"       • {reason}", flush=True)
        print("="*70 + "\n", flush=True)

        return jsonify({
            "status": "success",
            "final_decision": result.get("final_decision"),
            "satisfaction_score": result.get("satisfaction_score"),
            "validation_results": val_results,
            "missing_docs": result.get("missing_docs"),
            "reasons": result.get("reasons")
        })
    except Exception as e:
        print(f"[FLASK] ❌ ERROR: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[FLASK] 🚀 Server starting on http://{host}:{port}", flush=True)
    app.run(debug=False, port=port, host=host)
