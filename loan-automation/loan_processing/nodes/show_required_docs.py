"""
nodes/show_required_docs.py
Displays the required document checklist for the chosen loan type.
"""
try:
    from ..state import GraphState
    from ..config import LOAN_CONFIGS, DOC_DISPLAY_NAMES
except ImportError:
    from state import GraphState
    from config import LOAN_CONFIGS, DOC_DISPLAY_NAMES



def show_required_docs(state: GraphState) -> GraphState:
    """
    Renders a human-readable checklist of required documents.
    In a web app you would return this list to the UI layer.
    """
    loan_type  = state["loan_type"]
    req_docs   = state.get("required_docs", [])
    config     = LOAN_CONFIGS[loan_type]

    print("\n" + "═" * 55)
    print(f"  المستندات المطلوبة لـ: {config['display']}")
    print("═" * 55)
    for i, doc_key in enumerate(req_docs, 1):
        display = DOC_DISPLAY_NAMES.get(doc_key, doc_key)
        rule_desc = config["conditions"].get(doc_key, {}).get("description", "")
        print(f"  {i}. {display}")
        if rule_desc:
            print(f"       ↳ الشرط: {rule_desc}")
    print("═" * 55 + "\n")

    # Nothing changes in state – this is a display-only node
    return state
