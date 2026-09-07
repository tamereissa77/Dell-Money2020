"""
nodes/final_nodes.py
Terminal nodes: approve_loan, reject_loan, manual_review.
"""
from state import GraphState
from config import LOAN_CONFIGS, DOC_DISPLAY_NAMES


# ─────────────────────────────────────────────────────────────────────────────
def approve_loan(state: GraphState) -> GraphState:
    score     = state.get("satisfaction_score", 0.0)
    loan_type = state.get("loan_type", "")
    display   = LOAN_CONFIGS.get(loan_type, {}).get("display", loan_type)
    form      = state.get("form_data", {})
    name      = form.get("full_name", "العميل")
    amount    = form.get("requested_amount", "—")

    message = (
        f"✅ تم قبول طلب القرض\n"
        f"   العميل        : {name}\n"
        f"   نوع القرض     : {display}\n"
        f"   المبلغ المطلوب: {amount} جنيه\n"
        f"   درجة الاستيفاء: {score:.1%}\n"
        f"   الحالة        : موافق عليه – سيتم التواصل معك خلال 2-3 أيام عمل."
    )
    print(message)
    return {
        **state,
        "final_decision":   "approve",
        "approval_message": message,
    }


# ─────────────────────────────────────────────────────────────────────────────
def reject_loan(state: GraphState) -> GraphState:
    score     = state.get("satisfaction_score", 0.0)
    loan_type = state.get("loan_type", "")
    display   = LOAN_CONFIGS.get(loan_type, {}).get("display", loan_type)
    results   = state.get("validation_results", {})
    form      = state.get("form_data", {})
    name      = form.get("full_name", "العميل")

    reasons = list(state.get("reasons", []))
    if not reasons:
        for doc_key, result in results.items():
            doc_display = DOC_DISPLAY_NAMES.get(doc_key, doc_key)
            for r_item in result.get("reasons", []):
                if any(k in r_item for k in ["اختلاف", "منتهي", "العمر", "مرفوض", "ناقصة", "خطأ", "انتهت"]):
                    reasons.append(f"{doc_display}: {r_item}")
    if not reasons:
        reasons.append("درجة استيفاء الشروط أقل من الحد الأدنى المطلوب للموافقة")

    message = (
        f"❌ تم رفض طلب القرض\n"
        f"   العميل        : {name}\n"
        f"   نوع القرض     : {display}\n"
        f"   درجة الاستيفاء: {score:.1%}\n"
        f"   أسباب الرفض:\n" + "\n".join([f"  • {r}" for r in reasons])
    )
    print(message)
    return {
        **state,
        "final_decision":   "reject",
        "rejection_reason": message,
        "reasons": list(dict.fromkeys(reasons))
    }


# ─────────────────────────────────────────────────────────────────────────────
def manual_review(state: GraphState) -> GraphState:
    score     = state.get("satisfaction_score", 0.0)
    loan_type = state.get("loan_type", "")
    display   = LOAN_CONFIGS.get(loan_type, {}).get("display", loan_type)
    results   = state.get("validation_results", {})
    form      = state.get("form_data", {})
    name      = form.get("full_name", "العميل")

    # Highlight borderline conditions
    borderline = []
    for doc_key, result in results.items():
        doc_display = DOC_DISPLAY_NAMES.get(doc_key, doc_key)
        doc_score   = result.get("overall_score", result.get("score", 1.0))
        if 0.0 < doc_score < 1.0:
            borderline.append(
                f"  • {doc_display} ({doc_score:.0%}): {result.get('summary','')}"
            )

    notes_text = "\n".join(borderline) if borderline else "  • يرجى مراجعة النتائج الجزئية"

    notes = (
        f"⚠️  إحالة للمراجعة اليدوية\n"
        f"   العميل        : {name}\n"
        f"   نوع القرض     : {display}\n"
        f"   درجة الاستيفاء: {score:.1%}\n"
        f"   بنود تحتاج مراجعة:\n{notes_text}"
    )
    print(notes)
    return {
        **state,
        "final_decision": "manual_review",
        "review_notes":   notes,
    }
