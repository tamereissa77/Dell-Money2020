
#Human interrupt node

from state import GraphState
from config import LOAN_CONFIGS, DOC_DISPLAY_NAMES


REQUIRED_FORM_FIELDS = [
    "full_name",            # الأسم بالكامل
    "national_id",          # الرقم القومي
    "gender",               # النوع
    "marital_status",       # الحالة الأجتماعية
    "address",              # العنوان بالكامل
    "phone",                # رقم الهاتف
    "job_title",            # الوظيفة
    "monthly_income",       # الدخل الشهري
    "bank_account_number",  # رقم الحساب البنكي
    "requested_amount",     # مقدار القرض المطلوب
]


def upload_docs_and_form(state: GraphState) -> GraphState:
    """
    Validates that the human has provided all expected inputs.
    Missing fields are flagged as errors but do not block the graph –
    downstream nodes will surface detailed messages.
    """
    uploaded  = state.get("uploaded_docs", {})
    form_data = state.get("form_data", {})
    loan_type = state.get("loan_type", "")
    req_docs  = LOAN_CONFIGS.get(loan_type, {}).get("required_docs", [])

    # Check which docs were actually uploaded
    present_docs   = [d for d in req_docs if d in uploaded and uploaded[d]]
    missing_upload = [d for d in req_docs if d not in uploaded or not uploaded[d]]

    # Check form completeness (warn only)
    missing_fields = [f for f in REQUIRED_FORM_FIELDS if not form_data.get(f)]

    if missing_upload:
        names = [DOC_DISPLAY_NAMES.get(d, d) for d in missing_upload]
        print(f"[upload_docs_and_form] ⚠ مستندات ناقصة: {names}")
    if missing_fields:
        print(f"[upload_docs_and_form] ⚠ حقول النموذج الناقصة: {missing_fields}")

    print(f"[upload_docs_and_form] ✓ تم رفع {len(present_docs)} من {len(req_docs)} مستند")
    print(f"[upload_docs_and_form] ✓ بيانات النموذج: {list(form_data.keys())}")

    return {
        **state,
        "uploaded_docs": uploaded,
        "form_data": form_data,
    }
