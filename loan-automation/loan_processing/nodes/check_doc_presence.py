
import json
try:
    from ..state import GraphState
    from ..config import LOAN_CONFIGS, DOC_DISPLAY_NAMES
except ImportError:
    from state import GraphState
    from config import LOAN_CONFIGS, DOC_DISPLAY_NAMES


def check_doc_presence(state: GraphState) -> GraphState:
    """
    Verify completeness of the uploaded document set using a direct key-based check.
    A document is considered PRESENT if:
      - It was uploaded (key exists in uploaded_docs with non-empty value), AND
      - Its extracted data does NOT contain only an 'error' key (i.e. extraction succeeded or
        returned partial data).
    """
    loan_type = state["loan_type"]
    config    = LOAN_CONFIGS[loan_type]
    req_docs  = state.get("required_docs", config["required_docs"])
    uploaded  = state.get("uploaded_docs", {})
    extracted = state.get("extracted_docs", {})

    present_docs    = []
    missing_docs    = []
    unreadable_docs = []

    for doc_key in req_docs:
        # 1. Was it uploaded at all?
        doc_value = uploaded.get(doc_key)
        if not doc_value:
            missing_docs.append(doc_key)
            print(f"[check_doc_presence] MISSING  : {doc_key} (not uploaded)", flush=True)
            continue

        # 2. Did extraction succeed?
        ext_data = extracted.get(doc_key)

        # Hard extraction error -> {"error": "..."}
        if isinstance(ext_data, dict) and list(ext_data.keys()) == ["error"]:
            unreadable_docs.append(doc_key)
            print(f"[check_doc_presence] UNREADABLE: {doc_key} -> {ext_data.get('error','?')}", flush=True)
            continue

        # Extraction_failed flag with no other useful keys
        if isinstance(ext_data, dict) and ext_data.get("extraction_failed") and len(ext_data) <= 2:
            unreadable_docs.append(doc_key)
            print(f"[check_doc_presence] UNREADABLE: {doc_key} -> extraction_failed", flush=True)
            continue

        # Everything else = present (including partial parse, real data, etc.)
        present_docs.append(doc_key)
        keys = list(ext_data.keys()) if isinstance(ext_data, dict) else "n/a"
        print(f"[check_doc_presence] PRESENT  : {doc_key} -> fields={keys}", flush=True)

    all_present = len(missing_docs) == 0 and len(unreadable_docs) == 0
    combined_missing = missing_docs + unreadable_docs

    summary = (
        "All documents present and readable ({}/{})".format(len(present_docs), len(req_docs))
        if all_present
        else "Missing documents: {}".format(combined_missing)
    )

    print(f"[check_doc_presence] all_present={all_present}  missing={combined_missing}", flush=True)
    print(f"[check_doc_presence] {summary}", flush=True)

    return {
        **state,
        "missing_docs":    combined_missing,
        "presence_passed": all_present,
    }
