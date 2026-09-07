"""Frozen prompts. Identical text is used at training, eval, and inference time.

The eval harness asserts that the prompt hash in a JSONL record equals the
hash of the prompt this module currently produces. If you intentionally change
a prompt, bump PROMPT_VERSION and rebuild the dataset.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

DocType = Literal["id_front", "id_back", "hr_letter"]

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are an information-extraction assistant for Egyptian identity documents "
    "and HR letters. Look at the image and emit a single JSON object that follows "
    "the schema specified by the user. Each field must be an object with two keys: "
    '"value" (the extracted text, or null if not present / unreadable) and '
    '"bbox_2d" ([x1, y1, x2, y2] in absolute pixel coordinates of the image you see, '
    "or null if value is null). "
    "Normalize Arabic-Indic digits to Western digits (0-9). Format dates as YYYY-MM-DD. "
    "Return strictly valid JSON. No prose, no markdown fences."
)


_USER_PROMPT_ID_FRONT = (
    "Extract the following fields from this Egyptian National ID — FRONT side:\n"
    "  - name (الاسم)\n"
    "  - address (العنوان)\n"
    "  - national_id (الرقم القومي, 14 digits)\n"
    "  - date_of_birth (تاريخ الميلاد, YYYY-MM-DD)\n"
    "Return JSON with these exact keys, in this order. Use null for any field that is missing or unreadable."
)

_USER_PROMPT_ID_BACK = (
    "Extract the following fields from this Egyptian National ID — BACK side:\n"
    "  - national_id (الرقم القومي, 14 digits)\n"
    "  - expiry_date (تاريخ انتهاء الصلاحية; the date after \"هذه البطاقة سارية حتى\"; YYYY-MM-DD)\n"
    "  - profession (المهنة / الوظيفة)\n"
    "  - gender (النوع; emit \"M\" for ذكر, \"F\" for أنثى)\n"
    "  - marital_status (الحالة الاجتماعية; one of: أعزب, متزوج, مطلق, أرمل — store in masculine form)\n"
    "Return JSON with these exact keys, in this order. Use null for any field that is missing or unreadable."
)

_USER_PROMPT_HR_LETTER = (
    "Extract the following fields from this Arabic HR / salary letter:\n"
    "  - name (اسم الموظف)\n"
    "  - national_id (الرقم القومي للموظف, 14 digits)\n"
    "  - company_name (اسم الشركة المُصدِرة)\n"
    "  - hire_date (تاريخ التعيين, YYYY-MM-DD)\n"
    "  - issue_date (تاريخ إصدار الخطاب, YYYY-MM-DD)\n"
    "  - salary (المرتب; digits only, no commas, no currency)\n"
    "  - job_title (المسمى الوظيفي)\n"
    "  - signatured (boolean: true if an authorized signature is present, false otherwise; "
    "the bbox_2d for this field is the signature region itself)\n"
    "Return JSON with these exact keys, in this order. Use null for any unreadable string field, "
    "and false for signatured if absent."
)

_USER_PROMPTS = {
    "id_front": _USER_PROMPT_ID_FRONT,
    "id_back": _USER_PROMPT_ID_BACK,
    "hr_letter": _USER_PROMPT_HR_LETTER,
}


def user_prompt(doc_type: DocType) -> str:
    return _USER_PROMPTS[doc_type]


def prompt_hash() -> str:
    """Stable hash over (version, system, all user prompts). Used to detect drift."""
    payload = json.dumps(
        {"v": PROMPT_VERSION, "system": SYSTEM_PROMPT, "users": _USER_PROMPTS},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
