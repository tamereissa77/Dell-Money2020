"""Canonical extraction schema and normalization rules.

This module is the single source of truth for:
  - field names per doc_type
  - value types per field
  - normalization rules applied during labeling AND eval

The model is trained on values that have ALREADY been normalized; the
normalization functions here are reused at eval time to make ground-truth
comparison apples-to-apples.

If you change anything here, you change the training data format — rerun
build_vl_dataset.py and retrain.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Literal, Optional

DocType = Literal["id_front", "id_back", "hr_letter"]
DOC_TYPES: List[DocType] = ["id_front", "id_back", "hr_letter"]

# ---------------------------------------------------------------------------
# Schema: ordered list of (field_name, value_type) per doc type.
# Order matters: it's the order in which the model is asked to emit fields,
# and the order the eval harness uses for stable reporting.
# ---------------------------------------------------------------------------

FIELDS: Dict[DocType, List[tuple[str, str]]] = {
    "id_front": [
        ("name", "str"),
        ("address", "str"),
        ("national_id", "digits14"),
        ("date_of_birth", "iso_date"),
    ],
    "id_back": [
        ("national_id", "digits14"),
        ("expiry_date", "iso_date"),
        ("profession", "str"),
        ("gender", "gender"),
        ("marital_status", "marital_status"),
    ],
    "hr_letter": [
        ("name", "str"),
        ("national_id", "digits14"),
        ("company_name", "str"),
        ("hire_date", "iso_date"),
        ("issue_date", "iso_date"),
        ("salary", "money"),
        ("job_title", "str"),
        ("signatured", "bool"),
    ],
}


def field_names(doc_type: DocType) -> List[str]:
    return [f for f, _ in FIELDS[doc_type]]


# ---------------------------------------------------------------------------
# Normalization rules
# ---------------------------------------------------------------------------

ARABIC_INDIC_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_INDIC_TO_WESTERN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def _strip_diacritics(s: str) -> str:
    """Drop Arabic harakat (tashkeel)."""
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def normalize_text(s: Optional[str]) -> Optional[str]:
    """Generic Arabic-aware text normalization for non-numeric string fields."""
    if s is None:
        return None
    s = unicodedata.normalize("NFC", str(s))
    s = s.translate(ARABIC_INDIC_TO_WESTERN).translate(PERSIAN_INDIC_TO_WESTERN)
    s = _strip_diacritics(s)
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_digits14(s: Optional[str]) -> Optional[str]:
    """14-digit Egyptian national ID. Strip everything but digits, validate length."""
    if s is None:
        return None
    s = unicodedata.normalize("NFC", str(s))
    s = s.translate(ARABIC_INDIC_TO_WESTERN).translate(PERSIAN_INDIC_TO_WESTERN)
    digits = re.sub(r"\D", "", s)
    return digits if len(digits) == 14 else (digits or None)


_DATE_PATTERNS = [
    re.compile(r"^(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})$"),
    re.compile(r"^(?P<d>\d{1,2})[-/](?P<m>\d{1,2})[-/](?P<y>\d{4})$"),  # DD/MM/YYYY (Egyptian convention)
]


def normalize_iso_date(s: Optional[str]) -> Optional[str]:
    """Return ISO YYYY-MM-DD, or None if unparseable."""
    if s is None:
        return None
    s = unicodedata.normalize("NFC", str(s))
    s = s.translate(ARABIC_INDIC_TO_WESTERN).translate(PERSIAN_INDIC_TO_WESTERN)
    s = s.strip()
    for pat in _DATE_PATTERNS:
        m = pat.match(s)
        if m:
            y, mo, d = int(m["y"]), int(m["m"]), int(m["d"])
            # Sanity: month in [1,12], day in [1,31]. Don't validate calendar — Feb 30 is the labeller's problem.
            if 1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2099:
                return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


_GENDER_MAP = {
    "ذكر": "M", "ذكور": "M", "m": "M", "male": "M",
    "أنثى": "F", "انثى": "F", "اناث": "F", "f": "F", "female": "F",
}


def normalize_gender(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = normalize_text(s)
    if s is None:
        return None
    return _GENDER_MAP.get(s.lower(), None)


# Closed vocabulary, stored in masculine singular form.
_MARITAL_MAP = {
    "اعزب": "أعزب", "أعزب": "أعزب", "انسة": "أعزب", "آنسة": "أعزب",
    "متزوج": "متزوج", "متزوجة": "متزوج",
    "مطلق": "مطلق", "مطلقة": "مطلق",
    "ارمل": "أرمل", "أرمل": "أرمل", "ارملة": "أرمل", "أرملة": "أرمل",
}


def normalize_marital(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = normalize_text(s)
    if s is None:
        return None
    return _MARITAL_MAP.get(s, None)


def normalize_money(s: Optional[str]) -> Optional[str]:
    """Salary as digits-only string. '7,500.00 EGP' -> '7500'."""
    if s is None:
        return None
    s = unicodedata.normalize("NFC", str(s))
    s = s.translate(ARABIC_INDIC_TO_WESTERN).translate(PERSIAN_INDIC_TO_WESTERN)
    # Drop currency words.
    s = re.sub(r"\b(EGP|ج\.?م\.?|جنيه|pound|pounds)\b", "", s, flags=re.IGNORECASE)
    # Take the integer part before any decimal.
    s = re.sub(r"[,،\s]", "", s).strip()
    m = re.match(r"^(-?\d+)(?:\.\d+)?$", s)
    return m.group(1) if m else None


def normalize_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y", "نعم", "موقع", "مختوم"}


NORMALIZERS = {
    "str": normalize_text,
    "digits14": normalize_digits14,
    "iso_date": normalize_iso_date,
    "gender": normalize_gender,
    "marital_status": normalize_marital,
    "money": normalize_money,
    "bool": normalize_bool,
}


def normalize_record(doc_type: DocType, record: Dict) -> Dict:
    """Apply per-field normalization to a raw label record.

    Input shape (per field):  {"value": <raw>, "bbox_2d": [x1,y1,x2,y2]}  or  {"value": null, "bbox_2d": null}
    Output shape: same, but value is canonicalized.

    bbox_2d is passed through unchanged — it lives in pixel space and is
    independent of the value normalization.
    """
    out = {}
    for fname, ftype in FIELDS[doc_type]:
        entry = record.get(fname) or {"value": None, "bbox_2d": None}
        raw_val = entry.get("value") if isinstance(entry, dict) else entry
        bbox = entry.get("bbox_2d") if isinstance(entry, dict) else None
        out[fname] = {
            "value": NORMALIZERS[ftype](raw_val),
            "bbox_2d": bbox,
        }
    return out


def empty_record(doc_type: DocType) -> Dict:
    """Skeleton record with all fields null. Use as the starting point for new labels."""
    return {f: {"value": (False if t == "bool" else None), "bbox_2d": None}
            for f, t in FIELDS[doc_type]}
