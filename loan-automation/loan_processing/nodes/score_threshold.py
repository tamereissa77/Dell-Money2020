"""

Reads the satisfaction_score and marks it in state for the edge router.
Also prints a human-readable validation report.
"""
import json
try:
    from ..state import GraphState
except ImportError:
    from state import GraphState

from config import LOAN_CONFIGS, DOC_DISPLAY_NAMES, THRESHOLDS


def score_threshold(state: GraphState) -> GraphState:
    """Print the validation report and pass state to the routing edge."""
    score   = state.get("satisfaction_score", 0.0)
    results = state.get("validation_results", {})
    loan_type = state.get("loan_type", "")
    config  = LOAN_CONFIGS.get(loan_type, {})

    print("\n" + "═" * 60)
    print(f"  تقرير التحقق – {config.get('display', loan_type)}")
    print("═" * 60)
    for doc_key, result in results.items():
        doc_display = DOC_DISPLAY_NAMES.get(doc_key, doc_key)
        doc_score   = result.get("overall_score", result.get("score", 0.0))
        summary     = result.get("summary", "")
        bar_len     = int(doc_score * 20)
        bar         = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {doc_display}")
        print(f"    [{bar}] {doc_score:.0%}")
        print(f"    {summary}")
        # Extract and print checks from 3-phase structure
        all_checks = []
        if "phase_1_doc_conditions" in result or "phase_2_cross_doc" in result:
            p1 = result.get("phase_1_doc_conditions", {})
            for c in p1.get("checks", []):
                all_checks.append({
                    "passed": c.get("passed"),
                    "text": f"الشرط ({c.get('condition', '')}): {c.get('detail', '')}"
                })
            p2 = result.get("phase_2_cross_doc", {})
            for c in p2.get("checks", []):
                all_checks.append({
                    "passed": c.get("passed"),
                    "text": f"تطابق المستندات (مقارنة {c.get('field', '')} مع {c.get('compared_with', '')}): {c.get('detail', '')}"
                })
            p3 = result.get("phase_3_form_match", {})
            for c in p3.get("checks", []):
                all_checks.append({
                    "passed": c.get("passed"),
                    "text": f"مطابقة النموذج (حقل {c.get('field', '')}): {c.get('detail', '')}"
                })
        else:
            # Legacy format fallback
            for check in result.get("checks", []):
                all_checks.append({
                    "passed": check.get("passed"),
                    "text": f"{check.get('condition','')} – {check.get('detail','')}"
                })

        for check in all_checks:
            icon = "✓" if check.get("passed") else "✗"
            print(f"      {icon} {check.get('text','')}")
    print("─" * 60)
    print(f"  النتيجة الإجمالية : {score:.1%}")

    if score >= THRESHOLDS["approve"]:
        decision_text = "✅  يُرشَّح للموافقة"
    elif score >= THRESHOLDS["manual_review"]:
        decision_text = "⚠️  يُحال للمراجعة اليدوية"
    else:
        decision_text = "❌  يُرشَّح للرفض"

    print(f"  القرار المقترح   : {decision_text}")
    print("═" * 60 + "\n")

    return state