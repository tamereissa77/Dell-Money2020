"""
Human node: receives loan_type from the user and populates required_docs
"""
from state import GraphState
from config import LOAN_CONFIGS, LOAN_TYPE_DISPLAY


def select_loan_type(state: GraphState) -> GraphState:

    loan_type = state.get("loan_type", "")

    if loan_type not in LOAN_CONFIGS:
        valid = list(LOAN_CONFIGS.keys())
        return {
            **state,
            "error": (
                f"نوع القرض '{loan_type}' غير معروف. "
                f"الأنواع المتاحة: {valid}"
            ),
        }

    required_docs = LOAN_CONFIGS[loan_type]["required_docs"]
    print(f"[select_loan_type] نوع القرض: {LOAN_TYPE_DISPLAY[loan_type]}")
    print(f"[select_loan_type] المستندات المطلوبة: {required_docs}")

    return {
        **state,
        "required_docs": required_docs,
        "error": None,
    }
