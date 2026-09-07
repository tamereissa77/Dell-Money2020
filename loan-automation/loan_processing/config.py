"""All loan types, required documents, and validation rules.
Conditions are evaluated by validate_conditions.py using the LLM.
"""


#  Document display names (Arabic → key mapping)

DOC_DISPLAY_NAMES = {
    "national_id":           "بطاقة الرقم القومي",
    "practice_license":      "رخصة مزاولة المهنة",
    "syndicate_card":        "كارنيه النقابة",
    "clinic_license":        "ترخيص العيادة أو المركز الطبي",
    "utility_bill":          "إيصال مرافق (كهرباء / غاز)",
    "ownership_contract":    "عقد ملكية أو إيجار",
    "tax_card":              "البطاقة الضريبية",
    "commercial_register":   "السجل التجاري",
    "bank_statement":        "كشف حساب بنكي",
    "hr_letter":             "HR Letter (خطاب من جهة العمل)",
    "electricity_bill":      "فاتورة كهرباء",
    "salary_transfer_pledge":"تعهد بتحويل الراتب",
    "pension_transfer_pledge":"تعهد بتحويل المعاش",
}


#  Loan type display names

LOAN_TYPE_DISPLAY = {
    "car":                     "قرض السيارة",
    "personal_default_income": "قرض شخصي – برنامج الدخل الافتراضي",
    "personal_doctors":        "قرض شخصي – للأطباء وأصحاب العيادات",
    "personal_pension":        "قرض شخصي – أصحاب المعاشات",
    "personal_freelance":      "قرض شخصي – أصحاب الأعمال والمهن الحرة",
    "personal_salary_secured": "قرض شخصي – بضمان تحويل الراتب",
    "personal_salary_unsecured":"قرض شخصي – بدون ضمان تحويل الراتب",
}

# ─────────────────────────────────────────────
#  Main config: required docs + validation rules
#  Rule fields (all optional, LLM checks what is provided):
#    present          : bool  – doc must exist and be readable
#    valid            : bool  – doc must not be expired
#    age_min          : int   – minimum applicant age (years)
#    age_max          : int   – maximum applicant age at loan END
#    max_months_old   : int   – document issue date not older than N months
#    min_history_months: int  – document must cover at least N months
#    min_amount_egp   : float – monetary value must be >= this (EGP)
#    contains_fields  : list  – LLM checks these fields exist in the doc
#    description      : str   – human-readable rule summary shown in report
# ─────────────────────────────────────────────

LOAN_CONFIGS = {

    # قرض السيارة  
    "car": {
        "display": LOAN_TYPE_DISPLAY["car"],
        "required_docs": ["national_id", "utility_bill"],
        "conditions": {
            "national_id": {
                "present": True,
                "valid": True,
                "description": "البطاقة سارية المفعول",
            },
            "utility_bill": {
                "present": True,
                "max_months_old": 3,
                "description": "لم يمض على تاريخ إصدار الإيصال أكثر من 3 شهور",
            },
        },
    },

    #  قرض شخصي: برنامج الدخل الافتراضي  
    "personal_default_income": {
        "display": LOAN_TYPE_DISPLAY["personal_default_income"],
        "required_docs": ["national_id"],
        "conditions": {
            "national_id": {
                "present": True,
                "valid": True,
                "age_min": 21,
                "age_max": 65,
                "description": (
                    "البطاقة سارية. العمر > 21 ولا يزيد عن 65 سنة "
                    "لموظفي الأعمال والمهن الحرة"
                ),
            },
        },
    },

    #  قرض شخصي: أطباء  
    "personal_doctors": {
        "display": LOAN_TYPE_DISPLAY["personal_doctors"],
        "required_docs": [
            "national_id",
            "practice_license",
            "syndicate_card",
            "clinic_license",
        ],
        "conditions": {
            "national_id": {
                "present": True,
                "valid": True,
                "age_min": 25,
                "age_max": 65,
                "description": "البطاقة سارية. العمر من 25 إلى 65 سنة",
            },
            "practice_license": {
                "present": True,
                "valid": True,
                "description": "رخصة مزاولة المهنة سارية",
            },
            "syndicate_card": {
                "present": True,
                "valid": True,
                "description": "كارنيه النقابة ساري",
            },
            "clinic_license": {
                "present": True,
                "valid": True,
                "description": "ترخيص العيادة أو المركز الطبي ساري",
            },
        },
    },

    #  قرض شخصي: أصحاب المعاشات  
    "personal_pension": {
        "display": LOAN_TYPE_DISPLAY["personal_pension"],
        "required_docs": ["national_id", "pension_transfer_pledge"],
        "conditions": {
            "national_id": {
                "present": True,
                "valid": True,
                "age_max": 65,   # age must be <= 65 at END of financing period
                "description": "البطاقة سارية. العمر لا يتجاوز 65 سنة في نهاية فترة التمويل",
            },
            "pension_transfer_pledge": {
                "present": True,
                "description": (
                    "تعهد بتحويل المعاش على البنك طوال فترة السداد"
                ),
            },
        },
    },

    #  قرض شخصي: أصحاب الأعمال والمهن الحرة  
    "personal_freelance": {
        "display": LOAN_TYPE_DISPLAY["personal_freelance"],
        "required_docs": [
            "national_id",
            "utility_bill",
            "ownership_contract",
            "tax_card",
            "commercial_register",
            "bank_statement",
        ],
        "conditions": {
            "national_id": {
                "present": True,
                "valid": True,
                "age_min": 21,
                "age_max": 65,
                "description": "البطاقة سارية. العمر من 21 إلى 65 سنة",
            },
            "utility_bill": {
                "present": True,
                "max_months_old": 3,
                "description": "إيصال مرافق: لم يمض على تاريخه أكثر من 3 شهور",
            },
            "ownership_contract": {
                "present": True,
                "description": "عقد ملكية أو إيجار لمقر النشاط",
            },
            "tax_card": {
                "present": True,
                "valid": True,
                "contains_fields": ["تاريخ بدء النشاط"],
                "description": "البطاقة الضريبية سارية وتتضمن تاريخ بدء النشاط",
            },
            "commercial_register": {
                "present": True,
                "max_months_old": 3,
                "description": "السجل التجاري: لم يمض عليه أكثر من 3 شهور",
            },
            "bank_statement": {
                "present": True,
                "min_history_months": 12,
                "description": (
                    "كشف حساب بنكي شخصي أو للنشاط لا يقل تاريخه عن سنة كاملة"
                ),
            },
        },
    },

    #   قرض شخصي: بضمان تحويل الراتب  
    "personal_salary_secured": {
        "display": LOAN_TYPE_DISPLAY["personal_salary_secured"],
        "required_docs": [
            "hr_letter",
            "national_id",
        ],
        "conditions": {
            "hr_letter": {
                "present": True,
                "contains_fields": ["الدخل الشهري", "تاريخ التعيين"],
                "match_identity": True,
                "description": (
                    "HR Letter يتضمن: الدخل الشهري وتاريخ التعيين"
                ),
            },
            "national_id": {
                "present": True,
                "valid": True,
                "age_min": 21,
                "age_max": 65,
                "match_identity": True,
                "check_front_back": True,
                "description": "البطاقة سارية. العمر من 21 إلى 60 سنة",
            },
        },
    },

    #   قرض شخصي: بدون ضمان تحويل الراتب  
    "personal_salary_unsecured": {
        "display": LOAN_TYPE_DISPLAY["personal_salary_unsecured"],
        "required_docs": [
            "hr_letter",
            "national_id",
            "electricity_bill",
        ],
        "conditions": {
            "hr_letter": {
                "present": True,
                "contains_fields": ["الدخل الشهري", "تاريخ التعيين"],
                "description": (
                    "HR Letter يتضمن: الدخل الشهري وتاريخ التعيين"
                ),
            },
            "national_id": {
                "present": True,
                "valid": True,
                "age_min": 21,
                "age_max": 65,
                "description": "البطاقة سارية. العمر من 21 إلى 65 سنة",
            },
            "electricity_bill": {
                "present": True,
                "max_months_old": 3,
                "min_amount_egp": 25,
                "description": (
                    "فاتورة كهرباء: لم يمض على تاريخها 3 شهور، "
                    "والقيمة أكبر من أو تساوي 25 جنيه مصري"
                ),
            },
        },
    },
}


#  Decision thresholds

THRESHOLDS = {
    "approve":       0.85,   # score >= 85% → approve
    "manual_review": 0.65,   # 65% ≤ score < 85% → manual review
    # below 65% → reject
}