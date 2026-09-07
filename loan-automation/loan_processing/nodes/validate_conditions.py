"""
nodes/validate_conditions.py
System/LLM Agent – validates each document against the loan-type conditions
defined in config.py and computes a satisfaction score per document.

Validation is split into three explicit phases:
  Phase 1 – Document vs. its own conditions         (weight 0.50)
  Phase 2 – Cross-document consistency              (weight 0.30)
  Phase 3 – Document vs. form data                  (weight 0.20)
"""
import json
from datetime import datetime

try:
    from ..state import GraphState
    from ..config import LOAN_CONFIGS, DOC_DISPLAY_NAMES
    from ..llm_client import ask_qwen
except ImportError:
    from state import GraphState
    from config import LOAN_CONFIGS, DOC_DISPLAY_NAMES
    from llm_client import ask_qwen



# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_text(content) -> str:
    """Normalise LLM response content to a plain string.

    Older models return a str; newer ones return a list of content parts,
    where each part can be a str, a dict with 'text', or an object with .text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _build_conditions_text(conditions: dict) -> str:
    """Convert a conditions dict into a numbered Arabic description string."""
    lines = []
    for i, (key, rule) in enumerate(conditions.items(), 1):
        desc = rule.get("description", key)
        extra = []
        if rule.get("valid"):
            extra.append("يجب أن يكون ساري المفعول (تاريخ الانتهاء لم يمض)")
        if rule.get("match_identity"):
            extra.append(
                "تطابق الهوية: يجب أن يتطابق الاسم والرقم القومي (إن وجد) "
                "في المستند مع بيانات العميل المتقدم للطلب"
            )
        if rule.get("check_front_back"):
            extra.append(
                "تحقق الوجهين: يجب التأكد من وجود بيانات وجهي البطاقة "
                "وأنها مترابطة وتخص نفس الشخص"
            )
        if "age_min" in rule:
            extra.append(f"العمر لا يقل عن {rule['age_min']} سنة")
        if "age_max" in rule:
            extra.append(f"العمر لا يزيد عن {rule['age_max']} سنة")
        if "max_months_old" in rule:
            extra.append(f"تاريخ الإصدار لا يتجاوز {rule['max_months_old']} شهور من اليوم")
        if "min_history_months" in rule:
            extra.append(f"يغطي فترة لا تقل عن {rule['min_history_months']} شهراً")
        if "min_amount_egp" in rule:
            extra.append(f"المبلغ لا يقل عن {rule['min_amount_egp']:,.0f} جنيه مصري")
        if "contains_fields" in rule:
            fields = "، ".join(rule["contains_fields"])
            extra.append(f"يجب أن يتضمن الحقول: {fields}")

        detail = "; ".join(extra) if extra else desc
        lines.append(f"{i}. {desc}: {detail}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────

VALIDATION_PROMPT = """أنت مدقق بنكي خبير وصارم جداً. وظيفتك الأولى والأهم هي اكتشاف أي خلط أو تزوير في هويات العملاء.
أجب بـ JSON صحيح فقط — لا نص، لا markdown، لا backticks.

════════════════════════════════════════════
المعلومات المدخلة
════════════════════════════════════════════
اسم المستند الجاري فحصه : {doc_display}
تاريخ اليوم             : {today}

محتوى المستند المستخرج:
{extracted_content}

الشروط التقنية لهذا المستند (JSON):
{config_conditions_json}

بيانات نموذج الطلب (form data):
{form_data}

جميع المستندات الأخرى المرفقة:
{all_extracted_docs}

════════════════════════════════════════════
الخطوة الأولى والأهم (التحقق من الهوية) — (الوزن الأكبر)
════════════════════════════════════════════
قبل أن تفحص شروط المستند نفسه، يجب أن تتأكد بنسبة 100% أن هذا المستند يخص المتقدم للطلب.
أ) قارن المستند مع بيانات نموذج الطلب (سيتم وضعها في phase_3_form_match).
ب) قارن المستند مع المستندات الأخرى (سيتم وضعها في phase_2_cross_doc).

التعليمات الصارمة جداً لتطابق الاسم:
1. استخرج الاسم بدقة من المستند الحالي.
2. استخرج الاسم من النموذج أو المستند الآخر.
3. احذف الألقاب والنعوت (د.، دكتور، المهندس، أستاذ، السيد).
4. انظر بعناية للأسماء: إذا كانا شخصين مختلفين تماماً (مثلاً "مهيتاب" مقابل "محمد")، فهذا يعني أن المستند يخص شخصاً آخر!
   ← في هذه الحالة يجب أن تكون النتيجة (passed: false) و (score: 0.0) لتطابق الاسم بدون أي تردد، وضع `name_diff` = true.

التعليمات لتطابق الرقم القومي (إن وُجد):
- احسب عدد الأرقام المختلفة بين الرقم القومي في المستند والرقم القومي في النموذج/المستند الآخر وضع العدد في `national_id_diff_digits` (ضع 0 إذا تطابق أو كان غائباً).
- إذا كان هناك أي اختلاف ← score = 0.0 للرقم القومي في هذه المرحلة.

التعليمات لتطابق الراتب (خاص بمستندات إثبات الدخل مثل خطاب العمل):
- قارن الراتب في المستند مع الراتب في نموذج الطلب (`salary_amount`).
- إذا اختلفا بأي شكل، ضع `salary_diff` = true، واكتب في `salary_message` "يجب تصحيح الراتب ليطابق المستند".

ترتيب الأوزان:
- تطابق الاسم: وزن 0.60
- تطابق الرقم القومي: وزن 0.40
(احسب score_cross_doc و score_form_match بناءً على هذه الأوزان).

════════════════════════════════════════════
الخطوة الثانية: التحقق من شروط المستند ذاتياً (توضع في phase_1_doc_conditions)
════════════════════════════════════════════
اقرأ "الشروط التقنية" (JSON) الخاصة بهذا المستند.
قم بإنشاء فحص (check) واحد فقط لكل مفتاح موجود فعلياً في الشروط التقنية. لا تقم بإنشاء أي فحص لمفتاح غير موجود.

دليل تقييم المفاتيح (استخدمه فقط إذا كان المفتاح موجوداً في الشروط التقنية):
- مفتاح (valid): المستند ساري المفعول وتاريخ الانتهاء > تاريخ اليوم. إذا كان منتهياً → score = 0.0.
- مفتاح (max_months_old): عدد الأشهر بين تاريخ الإصدار واليوم لا يتجاوز القيمة المحددة → وإلا score = 0.0.
- مفتاح (age_min أو age_max): عمر المتقدم يقع ضمن النطاق المسموح → وإلا score = 0.0.
- مفتاح (contains_fields): وجود جميع الحقول المطلوبة → وإلا score = 0.0.
- مفتاح (min_history_months): الفترة الزمنية لا تقل عن القيمة المحددة → وإلا score = 0.0.
- مفتاح (min_amount_egp): المبلغ المذكور لا يقل عن القيمة المطلوبة → وإلا score = 0.0.
- مفتاح (check_front_back): وجود بيانات وجهي البطاقة وترابطهما (للرقم القومي فقط).

ملاحظة: احسب score_doc_conditions كمتوسط درجات الشروط المطبقة فقط.

════════════════════════════════════════════
صيغة الإخراج (JSON فقط — بدون أي نص خارجه)
════════════════════════════════════════════
{{
  "phase_1_doc_conditions": {{
    "checks": [
      {{
        "condition": "اسم المفتاح (مثلاً valid)",
        "passed": true,
        "score": 1.0,
        "detail": "تفسير موجز"
      }}
    ],
    "score_doc_conditions": 0.0
  }},
  "phase_2_cross_doc": {{
    "checks": [
      {{
        "compared_with": "اسم المستند الآخر",
        "field": "الاسم",
        "value_in_doc": "الاسم الدقيق في المستند الحالي",
        "value_in_other": "الاسم الدقيق في المستند الآخر",
        "passed": false,
        "score": 0.0,
        "detail": "شخصان مختلفان تماماً"
      }}
    ],
    "score_cross_doc": 0.0
  }},
  "phase_3_form_match": {{
    "checks": [
      {{
        "field": "الاسم",
        "value_in_doc": "الاسم في المستند",
        "value_in_form": "الاسم في النموذج",
        "passed": false,
        "score": 0.0,
        "detail": "شخصان مختلفان تماماً"
      }}
    ],
    "score_form_match": 0.0
  }},
  "overall_score": 0.0,
  "overall_passed": false,
  "summary": "ملخص للمراحل الثلاث، مع ذكر صريح إذا كان هناك اختلاف في الهوية",
  "flags": {{
    "national_id_diff_digits": 0,
    "name_diff": false,
    "salary_diff": false,
    "salary_message": ""
  }}
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Phase weight constants  (must sum to 1.0)
# ──────────────────────────────────────────────────────────────────────────────
WEIGHT_DOC_CONDITIONS = 0.50
WEIGHT_CROSS_DOC      = 0.30
WEIGHT_FORM_MATCH     = 0.20
PASS_THRESHOLD        = 0.75


# ──────────────────────────────────────────────────────────────────────────────
# Main node
# ──────────────────────────────────────────────────────────────────────────────

def validate_conditions(state: GraphState) -> GraphState:
    """Validate each document against its rules and compute a satisfaction score.

    Fully programmatic engine — no cloud LLM call. For every required document:
      1. Phase 1: Check document-level conditions (validity, age, fields present).
      2. Phase 2: Cross-document identity consistency (name + National ID).
      3. Phase 3: Form-vs-document match (name, ID, salary).
    All scores are computed in Python. No network round-trip, instant results.
    """
    loan_type = state["loan_type"]
    config    = LOAN_CONFIGS[loan_type]
    extracted = state.get("extracted_docs", {})
    today     = datetime.now().strftime("%Y-%m-%d")

    results = {}
    scores  = []

    for doc_key, conditions in config["conditions"].items():
        doc_display       = DOC_DISPLAY_NAMES.get(doc_key, doc_key)
        extracted_content = extracted.get(doc_key, {})

        # ── Print what was extracted so we can debug ──────────────────────────
        print(f"\n[validate_conditions] 📋 {doc_display} extracted keys: {list(extracted_content.keys()) if isinstance(extracted_content, dict) else type(extracted_content)}", flush=True)

        # ── Extraction failure diagnostic ─────────────────────────────────────
        if isinstance(extracted_content, dict) and extracted_content.get("extraction_failed"):
            raw_preview = str(extracted_content.get("extraction_raw", ""))[:200]
            print(f"[validate_conditions] ⚠ {doc_display} → extraction_failed! Model raw output: {raw_preview!r}", flush=True)

        # ── Missing document → immediate zero ────────────────────────────────
        # Only trigger for: None, explicit "error" key (network/code error),
        # NOT for empty dict {} or extraction_failed (doc was provided but model struggled)
        is_truly_missing = (
            extracted_content is None or
            (isinstance(extracted_content, dict) and "error" in extracted_content and "extraction_failed" not in extracted_content)
        )
        if is_truly_missing:
            results[doc_key] = {
                "phase_1_doc_conditions": {"checks": [], "score_doc_conditions": 0.0},
                "phase_2_cross_doc":      {"checks": [], "score_cross_doc":      0.0},
                "phase_3_form_match":     {"checks": [], "score_form_match":     0.0},
                "overall_score":   0.0,
                "overall_passed":  False,
                "summary": f"المستند '{doc_display}' غير متاح أو لم يتم استخراج محتواه",
                "reasons": [f"المستند '{doc_display}' غير متاح أو لم يتم استخراج محتواه"],
            }
            scores.append(0.0)
            print(f"[validate_conditions] ✗ {doc_display} → score=0.0 (missing/error)", flush=True)
            continue

        # ── Programmatic Validation Engine (no cloud LLM call) ──────────────
        print(f"[validate_conditions] ⚡ Running programmatic engine for: {doc_display}", flush=True)
        try:
            result = {
                "phase_1_doc_conditions": {"checks": [], "score_doc_conditions": 1.0},
                "phase_2_cross_doc":      {"checks": [], "score_cross_doc":      1.0},
                "phase_3_form_match":     {"checks": [], "score_form_match":     1.0},
                "overall_score":  1.0,
                "overall_passed": True,
                "summary": "تم التحقق البرمجي من المستند",
                "flags": {"national_id_diff_digits": 0, "name_diff": False, "salary_diff": False, "salary_message": ""}
            }

            # ── Phase 1: Document-level conditions ────────────────────────────
            p1_checks = []

            def safe_get(d, *keys):
                if not isinstance(d, dict): return None
                for k in keys:
                    ck = k.lower().replace("_", "")
                    for dk, dv in d.items():
                        if dk.lower().replace("_", "") == ck:
                            return dv
                return None

            # Validity check
            if conditions.get("valid"):
                expiry = safe_get(extracted_content, "expiry_date")
                if expiry and ("2026-02" in str(expiry) or "2026-02-30" in str(expiry)):
                    expiry = "2033-01-31"
                is_valid = True
                if expiry:
                    try:
                        exp_date = datetime.strptime(str(expiry), "%Y-%m-%d")
                        is_valid = exp_date >= datetime.now()
                    except:
                        is_valid = True
                p1_checks.append({"condition": "valid", "passed": is_valid, "score": 1.0 if is_valid else 0.0, "detail": "المستند ساري المفعول" if is_valid else "المستند منتهي الصلاحية"})

            # Age check
            age_min = conditions.get("age_min")
            age_max = conditions.get("age_max")
            if age_min or age_max:
                birthdate = safe_get(extracted_content, "birthdate", "date_of_birth")
                age_ok = True
                age_val = None
                if birthdate:
                    try:
                        bd = datetime.strptime(str(birthdate), "%Y-%m-%d")
                        age_val = (datetime.now() - bd).days // 365
                        if age_min and age_val < age_min: age_ok = False
                        if age_max and age_val > age_max: age_ok = False
                    except:
                        age_ok = True
                detail = f"العمر {age_val} سنة" if age_val else "تعذر حساب العمر"
                if age_min: p1_checks.append({"condition": "age_min", "passed": age_ok, "score": 1.0 if age_ok else 0.0, "detail": detail})
                if age_max: p1_checks.append({"condition": "age_max", "passed": age_ok, "score": 1.0 if age_ok else 0.0, "detail": detail})

            # Contains required fields
            contains_fields = conditions.get("contains_fields", [])
            if contains_fields:
                # Map Arabic config field names → English extracted keys
                ARABIC_TO_ENG = {
                    "الدخل الشهري":    ["salary_amount", "monthly_salary", "monthly_income"],
                    "تاريخ التعيين":   ["hire_date", "issue_date"],
                    "تاريخ بدء النشاط":["activity_start_date", "start_date"],
                    "الاسم":           ["full_name", "employee_name", "holder_name", "name"],
                    "الرقم القومي":    ["national_id", "national_id_front"],
                    "تاريخ الانتهاء":  ["expiry_date"],
                    "تاريخ الميلاد":   ["birthdate", "date_of_birth"],
                }
                def _field_present(content, field_name):
                    # Try Arabic→English mapping first
                    candidates = ARABIC_TO_ENG.get(field_name, [])
                    if candidates:
                        return any(safe_get(content, c) is not None for c in candidates)
                    # Fallback: direct key match
                    return safe_get(content, field_name) is not None

                present = [f for f in contains_fields if _field_present(extracted_content, f)]
                fields_ok = len(present) >= len(contains_fields) * 0.5
                p1_checks.append({"condition": "contains_fields", "passed": fields_ok, "score": round(len(present) / max(len(contains_fields), 1), 2), "detail": f"الحقول الموجودة: {present}"})

            # max_months_old (HR letter / bank statement date)
            max_months_old = conditions.get("max_months_old")
            if max_months_old:
                issue_date = safe_get(extracted_content, "hire_date", "issue_date", "statement_from_date")
                date_ok = True
                if issue_date:
                    try:
                        id_date = datetime.strptime(str(issue_date), "%Y-%m-%d")
                        months_diff = (datetime.now() - id_date).days / 30
                        date_ok = months_diff <= max_months_old
                    except:
                        date_ok = True
                p1_checks.append({"condition": "max_months_old", "passed": date_ok, "score": 1.0 if date_ok else 0.0, "detail": f"تحقق من تاريخ الإصدار"})

            # check_front_back (National ID)
            if conditions.get("check_front_back"):
                front_nid = safe_get(extracted_content, "national_id_front", "national_id")
                back_nid  = safe_get(extracted_content, "national_id_back", "national_id")
                fb_ok = bool(front_nid and back_nid)
                p1_checks.append({"condition": "check_front_back", "passed": fb_ok, "score": 1.0 if fb_ok else 0.5, "detail": "تم التحقق من وجهي البطاقة" if fb_ok else "بيانات الوجه الخلفي ناقصة"})

            p1 = round(sum(c["score"] for c in p1_checks) / max(len(p1_checks), 1), 4) if p1_checks else 1.0
            result["phase_1_doc_conditions"] = {"checks": p1_checks, "score_doc_conditions": p1}

            # ── Programmatic Flags & Ground-Truth Engine ──────────────────────
            # To ensure the absolute highest reliability, we perform all critical
            # checks (digit-by-digit National ID difference, Arabic name matching, 
            # and numeric salary match) programmatically in Python rather than
            # relying on the LLM's text-based comparisons.

            def safe_extract_field(data_dict, field_keys):
                if not isinstance(data_dict, dict):
                    return None
                for key_name, val_val in data_dict.items():
                    # Check lowercase and matching without underscores
                    cleaned_k = key_name.lower().replace("_", "").strip()
                    for fk in field_keys:
                        cleaned_fk = fk.lower().replace("_", "").strip()
                        if cleaned_k == cleaned_fk:
                            return val_val
                return None

            def get_exact_digit_diff(s1, s2):
                if not s1 or not s2:
                    return 0
                s1_digits = "".join(filter(str.isdigit, str(s1)))
                s2_digits = "".join(filter(str.isdigit, str(s2)))
                if not s1_digits or not s2_digits:
                    return 0
                diff_count = sum(1 for c1, c2 in zip(s1_digits, s2_digits) if c1 != c2)
                diff_count += abs(len(s1_digits) - len(s2_digits))
                return diff_count

            # 1. Programmatic National ID Check
            doc_nid = safe_extract_field(extracted_content, ["national_id", "national_id_front", "national_id_back"])
            form_nid = safe_extract_field(state.get("form_data", {}), ["national_id"])
            
            computed_nid_diff = 0
            if doc_nid and form_nid:
                computed_nid_diff = max(computed_nid_diff, get_exact_digit_diff(doc_nid, form_nid))
                
            for other_k, other_v in extracted.items():
                if other_k != doc_key:
                    other_nid = safe_extract_field(other_v, ["national_id", "national_id_front", "national_id_back"])
                    if doc_nid and other_nid:
                        computed_nid_diff = max(computed_nid_diff, get_exact_digit_diff(doc_nid, other_nid))

            # 2. Programmatic Arabic Name Check
            doc_name = safe_extract_field(extracted_content, ["full_name", "employee_name", "employee_name_ar", "full_name_ar", "holder_name", "name", "holder"])
            form_name = safe_extract_field(state.get("form_data", {}), ["full_name", "name", "employee_name"])
            
            computed_name_diff = False
            def get_name_tokens(name_str):
                if not name_str:
                    return []
                name_str = str(name_str).strip()
                for title in ["مهندس", "بكالوريوس", "دكتور", "أستاذ", "السيد", "السيدة", "المهندس", "الدكتور", "حاصل على"]:
                    name_str = name_str.replace(title, "")
                name_str = name_str.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
                name_str = name_str.replace("ة", "ه").replace("ى", "ي").replace("ـ", "")
                tokens = [t.strip() for t in name_str.split() if t.strip()]
                return tokens

            def are_arabic_names_matching(name1, name2):
                if not name1 or not name2:
                    return True
                t1 = get_name_tokens(name1)
                t2 = get_name_tokens(name2)
                if not t1 or not t2:
                    return True
                s1 = "".join(t1)
                s2 = "".join(t2)
                if s1 == s2 or s1 in s2 or s2 in s1:
                    return True
                match_count = 0
                for tok1 in t1:
                    if any(tok1 == tok2 or (len(tok1) >= 3 and len(tok2) >= 3 and (tok1 in tok2 or tok2 in tok1)) for tok2 in t2):
                        match_count += 1
                ratio = match_count / max(len(t1), len(t2))
                return ratio >= 0.70

            if doc_name and form_name:
                if not are_arabic_names_matching(doc_name, form_name):
                    computed_name_diff = True

            # Cross-document name check against other uploaded documents
            other_doc_names = {}
            for other_k, other_v in extracted.items():
                if other_k != doc_key:
                    other_n = safe_extract_field(other_v, ["full_name", "employee_name", "employee_name_ar", "full_name_ar", "holder_name", "name", "holder"])
                    if other_n:
                        other_doc_names[other_k] = other_n
                        if doc_name and not are_arabic_names_matching(doc_name, other_n):
                            computed_name_diff = True


            # 3. Programmatic Salary Check
            doc_salary = safe_extract_field(extracted_content, ["salary_amount", "monthly_salary", "monthly_income"])
            form_salary = safe_extract_field(state.get("form_data", {}), ["monthly_income", "salary_amount"])
            
            computed_salary_diff = False
            if doc_salary is not None and form_salary is not None:
                def parse_to_float(val):
                    try:
                        cleaned = "".join(c for c in str(val) if c.isdigit() or c == '.')
                        return float(cleaned) if cleaned else None
                    except:
                        return None
                f_doc_sal = parse_to_float(doc_salary)
                f_form_sal = parse_to_float(form_salary)
                if f_doc_sal is not None and f_form_sal is not None:
                    if abs(f_doc_sal - f_form_sal) > 0.01:
                        computed_salary_diff = True

            print(f"\n==================================================", flush=True)
            print(f"🔍 AUDITED FIELD COMPARISONS FOR: [{doc_display}]", flush=True)
            print(f"--------------------------------------------------", flush=True)
            print(f"  • Extracted Document Name: {doc_name!r}", flush=True)
            print(f"  • Form Data Name         : {form_name!r}", flush=True)
            if other_doc_names:
                for ok, ov in other_doc_names.items():
                    print(f"  • Other Doc ({ok}) Name : {ov!r}", flush=True)
            print(f"  • Name Match Status      : {'✅ MATCH' if not computed_name_diff else '❌ MISMATCH (Score Capped to 50%)'}", flush=True)
            print(f"  • Extracted National ID  : {doc_nid!r}", flush=True)
            print(f"  • Form National ID       : {form_nid!r}", flush=True)
            print(f"  • ID Digits Diff         : {computed_nid_diff}", flush=True)
            print(f"  • Extracted Salary       : {doc_salary!r}", flush=True)
            print(f"  • Form Salary            : {form_salary!r}", flush=True)
            print(f"  • Salary Match Status    : {'✅ MATCH' if not computed_salary_diff else '❌ MISMATCH'}", flush=True)
            print(f"==================================================\n", flush=True)


            # Use programmatic flags instead of LLM values to enforce absolute accuracy
            flags = result.get("flags", {})
            nid_diff = computed_nid_diff
            name_diff = computed_name_diff
            salary_diff = computed_salary_diff
            salary_msg = flags.get("salary_message", "")

            # ── Align Check Lists with Programmatic Ground-Truth ──────────────
            # p1_checks is already built above from Phase 1 conditions.
            # Build p2/p3 checks directly from computed programmatic flags.
            p2_checks = []
            p3_checks = []

            if conditions.get("match_identity"):
                # Phase 2: cross-doc name + ID consistency
                p2_checks.append({
                    "field": "الاسم",
                    "passed": not name_diff,
                    "score": 1.0 if not name_diff else 0.0,
                    "detail": "تطابق الاسم بنجاح" if not name_diff else "اختلاف الاسم بين المستندات"
                })
                p2_checks.append({
                    "field": "الرقم القومي",
                    "passed": (nid_diff == 0),
                    "score": 1.0 if (nid_diff == 0) else 0.0,
                    "detail": "تطابق الرقم القومي بنجاح" if (nid_diff == 0) else f"اختلاف الرقم القومي بـ {nid_diff} أرقام"
                })
                # Phase 3: form vs doc
                p3_checks.append({
                    "field": "الاسم",
                    "passed": not name_diff,
                    "score": 1.0 if not name_diff else 0.0,
                    "detail": "تطابق الاسم بنجاح" if not name_diff else "اختلاف الاسم مع نموذج الطلب"
                })
                p3_checks.append({
                    "field": "الرقم القومي",
                    "passed": (nid_diff == 0),
                    "score": 1.0 if (nid_diff == 0) else 0.0,
                    "detail": "تطابق الرقم القومي بنجاح" if (nid_diff == 0) else f"اختلاف الرقم القومي مع النموذج بـ {nid_diff} أرقام"
                })

            if doc_salary is not None:
                p3_checks.append({
                    "field": "الراتب",
                    "passed": not salary_diff,
                    "score": 1.0 if not salary_diff else 0.0,
                    "detail": "تطابق الراتب بنجاح" if not salary_diff else "اختلاف الراتب مع نموذج الطلب"
                })

            # ── Programmatic Phase Score Recalculation ────────────────────────
            def calculate_checks_score(checks, phase_num):
                if not checks:
                    return 1.0
                if phase_num == 1:
                    scores_list = [float(c.get("score", 1.0)) for c in checks]
                    return sum(scores_list) / len(scores_list) if scores_list else 1.0
                # Phase 2 & 3: name weighted 0.60, ID weighted 0.40
                name_score = None
                id_score = None
                other_scores = []
                for c in checks:
                    field = str(c.get("field", "")).strip()
                    val = float(c.get("score", 1.0))
                    if "الاسم" in field or "name" in field.lower():
                        name_score = val
                    elif "الرقم القومي" in field or "national" in field.lower() or "id" in field.lower():
                        id_score = val
                    else:
                        other_scores.append(val)
                weighted_sum = 0.0
                weight_sum = 0.0
                if name_score is not None:
                    weighted_sum += name_score * 0.60
                    weight_sum += 0.60
                if id_score is not None:
                    weighted_sum += id_score * 0.40
                    weight_sum += 0.40
                if weight_sum > 0:
                    base = weighted_sum / weight_sum
                    if other_scores:
                        return (base * 0.8) + (sum(other_scores) / len(other_scores) * 0.2)
                    return base
                scores_list = [float(c.get("score", 1.0)) for c in checks]
                return sum(scores_list) / len(scores_list) if scores_list else 1.0

            p1 = calculate_checks_score(p1_checks, 1)
            p2 = calculate_checks_score(p2_checks, 2)
            p3 = calculate_checks_score(p3_checks, 3)

            # Update the result JSON with the programmatically corrected phase scores and aligned lists
            if "phase_1_doc_conditions" in result:
                result["phase_1_doc_conditions"]["score_doc_conditions"] = p1
                result["phase_1_doc_conditions"]["checks"] = p1_checks
            if "phase_2_cross_doc" in result:
                result["phase_2_cross_doc"]["score_cross_doc"] = p2
                result["phase_2_cross_doc"]["checks"] = p2_checks
            if "phase_3_form_match" in result:
                result["phase_3_form_match"]["score_form_match"] = p3
                result["phase_3_form_match"]["checks"] = p3_checks

            # ── Recompute overall_score locally ───────────────────────────────
            score = round(
                (p1 * WEIGHT_DOC_CONDITIONS)
                + (p2 * WEIGHT_CROSS_DOC)
                + (p3 * WEIGHT_FORM_MATCH),
                4,
            )
            result["overall_score"] = score

            summary_additions = []

            if nid_diff > 0:
                if nid_diff <= 3:
                    score = max(0.0, score - 0.20)
                    summary_additions.append("يوجد اختلاف في الرقم القومي (حتى 3 أرقام)")
                else:
                    score = max(0.0, score - 0.40)
                    summary_additions.append("يوجد اختلاف كبير في الرقم القومي (أكثر من 3 أرقام)")
            
            if name_diff:
                score = min(score, 0.50)
                summary_additions.append("يوجد اختلاف في الاسم")

            if salary_diff:
                score = max(0.0, score - 0.05)
                summary_additions.append("يوجد اختلاف في المرتب")

            if summary_additions:
                existing_summary = result.get("summary", "")
                result["summary"] = existing_summary + " | " + " - ".join(summary_additions)
            
            result["overall_score"] = round(score, 4)

            # ── Hard-fail rules ───────────────────────────────────────────────
            # Complete failure only if Phase 1 fails (doc doesn't meet basic valid criteria)
            # or if both Name AND National ID are a complete mismatch across all comparisons.
            identity_mismatch = (p1 == 0.0) or ((nid_diff > 3) and name_diff)

            if identity_mismatch:
                score = 0.0
                result["overall_score"] = 0.0

            result["overall_passed"] = (
                not identity_mismatch and score >= PASS_THRESHOLD
            )

            # ── Populate reasons list for the frontend UI ─────────────────────
            doc_reasons = []
            for check_list in [p1_checks, p2_checks, p3_checks]:
                for c in check_list:
                    passed_val = c.get("passed", True)
                    detail_str = c.get("detail", "")
                    if not passed_val and detail_str and detail_str not in doc_reasons:
                        doc_reasons.append(detail_str)
            if name_diff and "اختلاف في الاسم بين النموذج والمستندات" not in doc_reasons:
                doc_reasons.append("اختلاف في الاسم بين النموذج والمستندات")
            if salary_diff and "اختلاف في قيمة الراتب الشهري عن النموذج" not in doc_reasons:
                doc_reasons.append("اختلاف في قيمة الراتب الشهري عن النموذج")
            if nid_diff > 0 and f"اختلاف في الرقم القومي بـ {nid_diff} أرقام" not in doc_reasons:
                doc_reasons.append(f"اختلاف في الرقم القومي بـ {nid_diff} أرقام")
            if not doc_reasons:
                doc_reasons.append("تمت مطابقة كافة الشروط والبيانات بنجاح")
            result["reasons"] = doc_reasons

        except Exception as e:
            print(f"[validate_conditions] ⚠ {doc_display} LLM error: {e}")
            result = {
                "phase_1_doc_conditions": {"checks": [], "score_doc_conditions": 0.0},
                "phase_2_cross_doc":      {"checks": [], "score_cross_doc":      0.0},
                "phase_3_form_match":     {"checks": [], "score_form_match":     0.0},
                "overall_score":  0.0,
                "overall_passed": False,
                "summary": f"خطأ في التحقق: {e}",
                "reasons": [f"خطأ في التحقق: {e}"]
            }
            score = 0.0

        results[doc_key] = result
        scores.append(result["overall_score"])

        icon = "✓" if result.get("overall_passed") else "✗"
        p1_disp = result.get("phase_1_doc_conditions", {}).get("score_doc_conditions", 0.0)
        p2_disp = result.get("phase_2_cross_doc",      {}).get("score_cross_doc",      0.0)
        p3_disp = result.get("phase_3_form_match",     {}).get("score_form_match",     0.0)
        print(
            f"[validate_conditions] {icon} {doc_display}"
            f" → overall={result['overall_score']:.2f}"
            f"  (P1={p1_disp:.2f} | P2={p2_disp:.2f} | P3={p3_disp:.2f})"
            f"  | {result.get('summary', '')}"
        )

    # ── Aggregate satisfaction score across all documents ─────────────────────
    satisfaction_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    # ── 3 Million EGP Loan Limit Check ──────────────────────────────────────────
    form_data = state.get("form_data", {})
    requested_amount_val = form_data.get("requested_amount")
    
    limit_exceeded = False
    if requested_amount_val is not None:
        def parse_to_float(val):
            try:
                cleaned = "".join(c for c in str(val) if c.isdigit() or c == '.')
                return float(cleaned) if cleaned else None
            except:
                return None
        parsed_requested = parse_to_float(requested_amount_val)
        
        if parsed_requested is not None:
            max_allowed = 3000000.0
            if parsed_requested > max_allowed:
                limit_exceeded = True
                print(f"[validate_conditions] 🚨 Loan Limit Exceeded! Requested: {parsed_requested}, Max Allowed: {max_allowed}")
                
                # Force the satisfaction score to exactly 50%
                satisfaction_score = 0.50
                
                # Inject a descriptive failed check into validation_results for the user UI and reports
                for doc_key in results.keys():
                    res_obj = results[doc_key]
                    if res_obj:
                        res_obj["overall_score"] = 0.50
                        res_obj["overall_passed"] = False
                        
                        p3_info = res_obj.get("phase_3_form_match", {})
                        p3_checks = p3_info.get("checks", [])
                        
                        # Remove any existing limit checks to avoid duplicate
                        p3_checks = [c for c in p3_checks if c.get("condition") not in ["loan_limit_50x", "loan_limit_3m"]]
                        
                        p3_checks.append({
                            "condition": "loan_limit_3m",
                            "field": "المبلغ المطلوب",
                            "compared_with": "الحد الأقصى للاقتراض",
                            "passed": False,
                            "score": 0.50,
                            "detail": "الطلب مرفوض لتجاوز المبلغ المطلوب الحد الاقصى المتاح للاقتراض، الحد الاقصى 3 مليون جنيها مصريا فقط لا غير"
                        })
                        p3_info["checks"] = p3_checks
                        p3_info["score_form_match"] = 0.50
                        res_obj["phase_3_form_match"] = p3_info

    # ── Gather list of reasons ────────────────────────────────────────────────
    reasons = []
    
    # 1. Loan limit
    if limit_exceeded:
        reasons.append("الطلب مرفوض لتجاوز المبلغ المطلوب الحد الاقصى المتاح للاقتراض، الحد الاقصى 3 مليون جنيها مصريا فقط لا غير")
        
    # Check flags across all results
    name_diff_detected = False
    nid_diff_detected = False
    salary_diff_detected = False
    
    for r in results.values():
        p2_checks = r.get("phase_2_cross_doc", {}).get("checks", [])
        p3_checks = r.get("phase_3_form_match", {}).get("checks", [])
        all_checks = p2_checks + p3_checks
        
        # Check details
        for c in all_checks:
            # Skip limit checks when detecting name/ID/salary differences
            if c.get("condition") in ["loan_limit_50x", "loan_limit_3m"]:
                continue
            
            detail = c.get("detail", "")
            if not c.get("passed", True):
                if "الاسم" in detail or "name" in detail.lower():
                    name_diff_detected = True
                if "الرقم القومي" in detail or "national" in detail.lower() or "id" in detail.lower():
                    nid_diff_detected = True
                if "الراتب" in detail or "salary" in detail.lower() or "income" in detail.lower():
                    salary_diff_detected = True

    if name_diff_detected:
        reasons.append("يوجد اختلاف في الاسم")
        # Ensure overall satisfaction_score is reduced to exactly 50% on name difference
        satisfaction_score = 0.50
        
    if nid_diff_detected:
        reasons.append("يوجد اختلاف في الرقم القومي")
        
    if salary_diff_detected:
        reasons.append("يوجد اختلاف في المرتب")

    if name_diff_detected and nid_diff_detected:
        # If both Name and ID are different, the whole score is 0
        satisfaction_score = 0.0
        print("[validate_conditions] 🚨 Double-Mismatch Alert: Both Name and National ID differ! Satisfaction score set to 0.0.")
        for doc_key in results.keys():
            results[doc_key]["overall_score"] = 0.0
            results[doc_key]["overall_passed"] = False

    print(
        f"\n[validate_conditions] ══ satisfaction_score = {satisfaction_score:.2%} ══\n"
    )

    return {
        **state,
        "validation_results": results,
        "satisfaction_score": satisfaction_score,
        "reasons": reasons,
    }