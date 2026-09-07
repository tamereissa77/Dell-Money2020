"""
edges/routers.py
Conditional edge functions for the LangGraph StateGraph.
"""
from state import GraphState
from config import THRESHOLDS


def route_doc_check(state: GraphState) -> str:
    """
    After check_doc_presence:
      - "all_present"  → proceed to validate_conditions
      - "missing_docs" → loop back to upload_docs_and_form
    """
    if state.get("error"):
        return "missing_docs"        # also catches bad loan_type

    presence_passed = state.get("presence_passed", False)

    if presence_passed:
        print("[route_doc_check] → all_present")
        return "all_present"
    else:
        missing = state.get("missing_docs", [])
        print(f"[route_doc_check] → missing_docs: {missing}")
        return "missing_docs"


def route_final_decision(state: GraphState) -> str:
    """
    After score_threshold:
      - "approve"       if score >= THRESHOLDS["approve"]
      - "manual_review" if score >= THRESHOLDS["manual_review"]
      - "reject"        otherwise
    """
    score = state.get("satisfaction_score", 0.0)

    if score >= THRESHOLDS["approve"]:
        print(f"[route_final_decision] → approve  ({score:.1%})")
        return "approve"
    elif score >= THRESHOLDS["manual_review"]:
        print(f"[route_final_decision] → manual_review  ({score:.1%})")
        return "manual_review"
    else:
        print(f"[route_final_decision] → reject  ({score:.1%})")
        return "reject"
