"""
main.py
CLI entry point and test runner for the loan processing graph.

Usage:
    python -m loan_processing.main                    # runs default demo
    python -m loan_processing.main --loan personal_doctors
    python -m loan_processing.main --loan car
    python -m loan_processing.main --list             # list all loan types

Supported --loan values:
    car | personal_default_income | personal_doctors |
    personal_pension | personal_freelance |
    personal_salary_secured | personal_salary_unsecured
"""

import argparse
from graph import build_graph
from config import LOAN_TYPE_DISPLAY


# ─────────────────────────────────────────────────────────────────────────────
#  Mock document content – simulates what an OCR/upload would provide.
#  In production these come from actual file uploads.
# ─────────────────────────────────────────────────────────────────────────────
MOCK_DOCS = {
    "national_id": """
        بطاقة الرقم القومي
        الاسم: أحمد محمد علي
        رقم البطاقة: 29901011234567
        تاريخ الميلاد: 1999-01-01
        العمر: 26 سنة
        تاريخ الإصدار: 2022-05-10
        تاريخ الانتهاء: 2027-05-10
        الحالة: سارية
    """,
    "practice_license": """
        رخصة مزاولة المهنة
        الاسم: د. أحمد محمد علي
        التخصص: طب الأسنان
        رقم الرخصة: DEN-2019-45678
        تاريخ الإصدار: 2019-03-15
        تاريخ الانتهاء: 2026-03-15
        الحالة: سارية
    """,
    "syndicate_card": """
        كارنيه نقابة الأطباء
        الاسم: د. أحمد محمد علي
        رقم العضوية: 78923
        تاريخ الإصدار: 2023-01-01
        تاريخ الانتهاء: 2025-12-31
        الحالة: سارية
    """,
    "clinic_license": """
        ترخيص عيادة
        اسم العيادة: عيادة الدكتور أحمد لطب الأسنان
        العنوان: 15 شارع الجمهورية، المعادي، القاهرة
        رقم الترخيص: CLN-2020-1122
        تاريخ الإصدار: 2020-06-01
        تاريخ الانتهاء: 2026-06-01
        الحالة: سارية
    """,
    "utility_bill": """
        إيصال مرافق - شركة الغاز
        اسم المشترك: أحمد محمد علي
        تاريخ الإصدار: 2025-03-15
        المبلغ المستحق: 450 جنيه
        عنوان الاستهلاك: 15 شارع الجمهورية، المعادي
    """,
    "ownership_contract": """
        عقد إيجار
        المؤجر: شركة العقارات المتحدة
        المستأجر: أحمد محمد علي
        مقر النشاط: 22 شارع التحرير، وسط البلد، القاهرة
        تاريخ العقد: 2023-01-01
        مدة الإيجار: 3 سنوات
        الغرض: نشاط تجاري
    """,
    "tax_card": """
        البطاقة الضريبية
        الاسم: أحمد محمد علي
        الرقم الضريبي: 123-456-789
        نوع النشاط: استشارات تجارية
        تاريخ بدء النشاط: 2020-04-15
        تاريخ الإصدار: 2023-04-15
        تاريخ الانتهاء: 2026-04-15
        الحالة: سارية
    """,
    "commercial_register": """
        السجل التجاري
        اسم المنشأة: شركة أحمد للاستشارات
        رقم السجل: 78901234
        تاريخ الإصدار: 2025-02-10
        الحالة: ساري
    """,
    "bank_statement": """
        كشف حساب بنكي
        البنك: بنك مصر
        اسم العميل: أحمد محمد علي
        رقم الحساب: 1234567890
        من تاريخ: 2024-03-01
        إلى تاريخ:  2025-03-01
        فترة الكشف: 13 شهراً
        متوسط الرصيد: 85,000 جنيه
    """,
    "hr_letter": """
        خطاب من جهة العمل (HR Letter)
        الشركة: مجموعة الفجر للمقاولات
        الموظف: أحمد محمد علي
        المسمى الوظيفي: مهندس أول
        الدخل الشهري: 18,500 جنيه مصري
        تاريخ التعيين: 2018-09-01
        تاريخ الخطاب: 2025-04-01
    """,
    "electricity_bill": """
        فاتورة كهرباء
        اسم المشترك: أحمد محمد علي
        عنوان العداد: 5 شارع النيل، الدقي، الجيزة
        تاريخ الفاتورة: 2025-03-20
        قيمة الفاتورة: 32,500 جنيه مصري
        رقم العداد: 987654321
    """,
    "salary_transfer_pledge": """
        تعهد تحويل الراتب
        الشركة: مجموعة الفجر للمقاولات
        الموظف: أحمد محمد علي
        نتعهد بتحويل راتب الموظف المذكور إلى البنك
        طوال فترة سداد القرض حتى انتهائها.
        تاريخ التعهد: 2025-04-05
        توقيع المفوض: محمد سالم (مدير الموارد البشرية)
    """,
    "pension_transfer_pledge": """
        تعهد بتحويل المعاش
        الاسم: محمود سعيد إبراهيم
        رقم المعاش: PEN-2019-55432
        نتعهد بتحويل المعاش الشهري إلى البنك طوال فترة السداد.
        تاريخ التعهد: 2025-04-01
        توقيع: محمود سعيد إبراهيم
    """,
}

MOCK_FORM = {
    "full_name":           "أحمد محمد علي",
    "national_id":         "29901011234567",
    "gender":              "ذكر",
    "marital_status":      "أعزب / آنسة",
    "address":             "القاهرة، مصر الجديدة، شارع الثورة ١٢٣",
    "phone":               "01012345678",
    "job_title":           "مهندس برمجيات",
    "monthly_income":      "18500",
    "bank_account_number": "12345678901234",
    "requested_amount":    "200000",
}


# ─────────────────────────────────────────────────────────────────────────────
def run_loan(loan_type: str):
    from config import LOAN_CONFIGS
    if loan_type not in LOAN_CONFIGS:
        print(f"❌ نوع قرض غير معروف: '{loan_type}'")
        print(f"   الأنواع المتاحة: {list(LOAN_CONFIGS.keys())}")
        return

    config   = LOAN_CONFIGS[loan_type]
    req_docs = config["required_docs"]

    # Supply only the docs required for this loan type
    uploaded = {k: MOCK_DOCS[k] for k in req_docs if k in MOCK_DOCS}

    initial_state = {
        "loan_type":     loan_type,
        "uploaded_docs": uploaded,
        "form_data":     MOCK_FORM,
    }

    print(f"\n{'═'*60}")
    print(f"  بدء معالجة طلب قرض: {config['display']}")
    print(f"{'═'*60}\n")

    # Use graph without interrupt (single-pass for CLI testing)
    graph  = build_graph(use_checkpointer=False)
    result = graph.invoke(initial_state)

    print(f"\n{'═'*60}")
    print(f"  القرار النهائي: {result.get('final_decision', '—').upper()}")
    print(f"  درجة الاستيفاء: {result.get('satisfaction_score', 0):.1%}")
    print(f"{'═'*60}\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Loan Processing CLI")
    parser.add_argument(
        "--loan", default="personal_doctors",
        help="Loan type key (default: personal_doctors)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available loan types"
    )
    args = parser.parse_args()

    if args.list:
        print("\nأنواع القروض المتاحة:")
        for key, display in LOAN_TYPE_DISPLAY.items():
            print(f"  {key:35s} → {display}")
        return

    run_loan(args.loan)


if __name__ == "__main__":
    main()
