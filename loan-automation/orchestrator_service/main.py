"""
Orchestrator Microservice
Merges outputs from all 3 pipelines (National ID, HR Letter, Loan Form)
and performs cross-validation + LLM-based risk assessment.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import requests
import json
import re
from datetime import datetime

app = FastAPI(title="NBE Loan Orchestrator Service")

# ─── Service URLs (internal Docker network or localhost) ─────────────
NATIONAL_ID_URL = os.environ.get("NATIONAL_ID_SERVICE_URL", "http://national_id_service:8010")
HR_LETTER_URL = os.environ.get("HR_LETTER_SERVICE_URL", "http://hr_letter_service:8011")
FORM_URL = os.environ.get("FORM_SERVICE_URL", "http://form_service:8012")

# ─── NVIDIA API for LLM risk assessment ────────────────────────────
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "meta/llama-3.1-70b-instruct")
VISION_KEY = os.environ.get("VISION_API_KEY", "")

# ─── LLM System Prompt for Risk Assessment ─────────────────────────
SYSTEM_PROMPT = """You are the Lead Loan Underwriting AI at the National Bank of Egypt (NBE). Your objective is to validate a loan application by cross-referencing three distinct data sources:

National ID (Image/Data): Primary identity verification.
Loan Application Form (Digital Input): Customer's requested parameters.
HR Salary Letter (Image/Data): Employment and financial capability proof.

Verification & Validation Logic (The Core Task):
Execute the following checks with 100% strictness:
Identity Linkage: Does the "Name" and "National ID" in the Form match the ID Card and HR Letter? (Allow for minor Arabic phonetic variations).
Financial Eligibility: Is the "Requested Loan Amount" in the Form sustainable based on the "Net Salary" in the HR Letter? (Calculate DTI - Debt to Income ratio if applicable).
Employment Consistency: Does the "Employer Name" and "Job Title" in the Form match the Official HR Letter?
Document Integrity: Confirm that the HR Letter has both a Stamp and a Signature as reported by the Vision model.

Final Output Schema (JSON):
{
  "application_status": "VALIDATED | REJECTED | MANUAL_REVIEW",
  "reconciliation_report": {
    "identity_match": "boolean",
    "financial_check": "boolean",
    "employment_check": "boolean",
    "integrity_check": "boolean"
  },
  "mismatch_details": [
    "List any discrepancies found between the 3 sources"
  ],
  "risk_score": "0.0 (Low) - 1.0 (High)",
  "decision_summary": "Short Arabic summary of why the application was accepted or flagged."
}

IMPORTANT: Output strictly the final Output Schema (JSON) with no extra conversational text or markdown blocks outside the JSON format."""


# ─── Request Schemas ────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    """Full pipeline: provide document paths and form will be fetched."""
    national_id_image: str              # Path to ID image in shared volume
    hr_letter_file: str                 # Path to HR letter (PDF or image) in shared volume
    form_national_id: Optional[str] = None  # National ID to look up in form service
    shared_dir: str = "/shared_data"


class MergeRequest(BaseModel):
    """Merge pre-extracted data from all 3 pipelines."""
    id_data: Dict[str, Any]
    hr_data: Dict[str, Any]
    form_data: Dict[str, Any]
    signature_data: Optional[Dict[str, Any]] = None


class ValidateRequest(BaseModel):
    """Legacy: LLM-based validation with pre-extracted data."""
    form_data: Dict[str, Any]
    id_extraction: Dict[str, Any]
    hr_extraction: Dict[str, Any]


# ─── Helper Functions ───────────────────────────────────────────────
def extract_json_from_text(content: str) -> Optional[dict]:
    """Extract JSON from LLM output with various fallback strategies."""
    # Try direct parse
    try:
        return json.loads(content.strip())
    except:
        pass

    # Try regex extraction
    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass

    # Strip markdown
    cleaned = content.strip()
    if cleaned.startswith('```json'):
        cleaned = cleaned.replace('```json', '').replace('```', '')
    elif cleaned.startswith('```'):
        cleaned = cleaned.replace('```', '')
    try:
        return json.loads(cleaned.strip())
    except:
        pass

    # Brace-balancing
    for match in re.finditer(r'\{', content):
        start = match.start()
        balance = 0
        for i in range(start, len(content)):
            if content[i] == '{':
                balance += 1
            elif content[i] == '}':
                balance -= 1
                if balance == 0:
                    try:
                        return json.loads(content[start:i+1])
                    except:
                        break
    return None


def safe_get(d: dict, *keys, default=None):
    """Safely navigate nested dict keys."""
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current


def normalize_arabic_name(name: str) -> str:
    """Normalize Arabic name for comparison (remove diacritics, normalize hamza, etc.)."""
    if not name:
        return ""
    # Remove common diacritics
    import unicodedata
    normalized = unicodedata.normalize('NFKD', name)
    # Remove tatweel, diacritics
    result = ""
    for c in normalized:
        cat = unicodedata.category(c)
        if cat.startswith('M'):  # Mark category (diacritics)
            continue
        result += c
    # Normalize alef variants
    result = result.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    # Normalize taa marbuta
    result = result.replace('ة', 'ه')
    return result.strip()


def name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names (simple token-based)."""
    if not name1 or not name2:
        return 0.0
    
    n1 = normalize_arabic_name(name1)
    n2 = normalize_arabic_name(name2)
    
    if n1 == n2:
        return 1.0

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


# ─── Cross-Validation Logic ────────────────────────────────────────
def cross_validate(id_data: dict, hr_data: dict, form_data: dict, sig_data: dict) -> dict:
    """
    Deterministic cross-validation between the 3 data sources.
    Returns validation results with explanations.
    """
    checks = {}
    issues = []

    # --- 1. Name Match: ID vs HR Letter ---
    id_name = (
        safe_get(id_data, "full_name_ar") or
        safe_get(id_data, "extracted_data", "primary_entities", "full_name") or ""
    )
    hr_name_ar = (
        safe_get(hr_data, "employee_name_ar") or
        safe_get(hr_data, "extracted_data", "primary_entities", "full_name") or ""
    )
    hr_name = safe_get(hr_data, "employee_name") or hr_name_ar

    name_sim = name_similarity(id_name, hr_name_ar) if hr_name_ar else name_similarity(id_name, hr_name)
    checks["name_match_id_vs_hr"] = name_sim >= 0.5
    checks["name_similarity_score"] = round(name_sim, 2)
    if name_sim < 0.5:
        issues.append(f"Name mismatch: ID='{id_name}' vs HR='{hr_name_ar or hr_name}' (similarity: {name_sim:.0%})")

    # --- 2. National ID Consistency: ID card vs Form ---
    id_nid = (
        safe_get(id_data, "national_id") or
        safe_get(id_data, "extracted_data", "primary_entities", "national_id") or ""
    )
    form_nid = safe_get(form_data, "national_id") or ""
    checks["national_id_consistent"] = (
        id_nid.strip() == form_nid.strip() if id_nid and form_nid else False
    )
    if not checks["national_id_consistent"] and id_nid and form_nid:
        issues.append(f"National ID mismatch: ID card='{id_nid}' vs Form='{form_nid}'")

    # --- 3. ID Expiry Check ---
    expiry = (
        safe_get(id_data, "expiry_date") or
        safe_get(id_data, "extracted_data", "primary_entities", "expiry_date")
    )
    now_str = datetime.utcnow().strftime("%Y-%m-%d")
    if expiry and isinstance(expiry, str) and len(expiry) >= 10:
        try:
            checks["id_not_expired"] = expiry >= now_str
            if not checks["id_not_expired"]:
                issues.append(f"National ID expired: {expiry}")
        except:
            checks["id_not_expired"] = None
            issues.append(f"Could not parse expiry date: {expiry}")
    else:
        checks["id_not_expired"] = None
        issues.append("Expiry date not available from ID extraction")

    # --- 4. HR Letter Signature & Stamp ---
    has_sig = (
        safe_get(sig_data, "has_handwritten_signature") or
        safe_get(hr_data, "is_signed") or
        safe_get(hr_data, "extracted_data", "integrity_check", "signature_detected") or
        False
    )
    has_stamp = (
        safe_get(sig_data, "has_official_stamp") or
        safe_get(hr_data, "extracted_data", "integrity_check", "stamp_detected") or
        False
    )
    sig_confidence = safe_get(sig_data, "confidence") or "unknown"

    checks["hr_letter_signed"] = bool(has_sig)
    checks["hr_letter_stamped"] = bool(has_stamp)
    checks["signature_confidence"] = sig_confidence

    if not has_sig:
        issues.append("HR letter signature not detected")
    if not has_stamp:
        issues.append("HR letter official stamp not detected")

    # --- 5. Financial Eligibility (DTI) ---
    salary = (
        safe_get(hr_data, "monthly_salary") or
        safe_get(hr_data, "extracted_data", "primary_entities", "salary")
    )
    requested_amount = safe_get(form_data, "requested_amount")
    repayment_months = safe_get(form_data, "repayment_period_months")

    try:
        salary_num = float(re.sub(r'[^\d.]', '', str(salary))) if salary else None
        amount_num = float(re.sub(r'[^\d.]', '', str(requested_amount))) if requested_amount else None
        months_num = float(repayment_months) if repayment_months else 60  # Default 5 years

        if salary_num and amount_num and salary_num > 0:
            monthly_payment = amount_num / months_num
            dti = monthly_payment / salary_num
            checks["dti_ratio"] = round(dti, 3)
            checks["financial_eligible"] = dti < 0.4  # Max 40% DTI
            if dti >= 0.4:
                issues.append(f"DTI ratio {dti:.1%} exceeds 40% threshold (salary={salary_num}, monthly payment={monthly_payment:.0f})")
        else:
            checks["dti_ratio"] = None
            checks["financial_eligible"] = None
            issues.append("Cannot calculate DTI: missing salary or loan amount")
    except Exception as e:
        checks["dti_ratio"] = None
        checks["financial_eligible"] = None
        issues.append(f"DTI calculation error: {str(e)}")

    # --- 6. Employment Consistency: Form vs HR Letter ---
    form_job = safe_get(form_data, "job_title") or ""
    hr_job = (
        safe_get(hr_data, "job_title") or
        safe_get(hr_data, "extracted_data", "primary_entities", "job_title") or ""
    )
    if form_job and hr_job:
        job_sim = name_similarity(form_job, hr_job)
        checks["employment_consistent"] = job_sim >= 0.3
        if job_sim < 0.3:
            issues.append(f"Job title mismatch: Form='{form_job}' vs HR='{hr_job}'")
    else:
        checks["employment_consistent"] = None

    return {
        "checks": checks,
        "issues": issues,
        "all_passed": len(issues) == 0
    }


# ─── Merge Function ────────────────────────────────────────────────
def merge_extraction_outputs(
    id_data: dict, hr_data: dict, form_data: dict, sig_data: dict
) -> dict:
    """
    Merge outputs from all 3 pipelines into a unified application record
    with cross-validation results.
    """
    validation = cross_validate(id_data, hr_data, form_data, sig_data)

    return {
        "metadata": {
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "pipeline_version": "2.0"
        },
        "applicant": {
            "full_name": (
                safe_get(hr_data, "employee_name") or
                safe_get(hr_data, "extracted_data", "primary_entities", "full_name")
            ),
            "full_name_ar": (
                safe_get(id_data, "full_name_ar") or
                safe_get(id_data, "extracted_data", "primary_entities", "full_name")
            ),
            "national_id": (
                safe_get(id_data, "national_id") or
                safe_get(id_data, "extracted_data", "primary_entities", "national_id")
            ),
            "id_expiry_date": (
                safe_get(id_data, "expiry_date") or
                safe_get(id_data, "extracted_data", "primary_entities", "expiry_date")
            ),
            "gender": (
                safe_get(id_data, "gender") or
                safe_get(form_data, "gender")
            ),
            "birth_date": safe_get(id_data, "birth_date"),
            "governorate": safe_get(id_data, "governorate"),
            "phone": safe_get(form_data, "phone"),
            "address": safe_get(form_data, "address"),
            "marital_status": safe_get(form_data, "marital_status"),
        },
        "employment": {
            "company_name": (
                safe_get(hr_data, "company_name") or
                safe_get(form_data, "company_name")
            ),
            "job_title": (
                safe_get(hr_data, "job_title") or
                safe_get(hr_data, "extracted_data", "primary_entities", "job_title")
            ),
            "monthly_salary_egp": (
                safe_get(hr_data, "monthly_salary") or
                safe_get(hr_data, "extracted_data", "primary_entities", "salary")
            ),
            "hr_letter_date": (
                safe_get(hr_data, "issue_date") or
                safe_get(hr_data, "extracted_data", "primary_entities", "date")
            ),
        },
        "loan_request": {
            "loan_type": safe_get(form_data, "loan_type"),
            "personal_loan_type": safe_get(form_data, "personal_loan_type"),
            "requested_amount_egp": safe_get(form_data, "requested_amount"),
            "bank_account": safe_get(form_data, "bank_account"),
            "repayment_period_months": safe_get(form_data, "repayment_period_months"),
        },
        "validation": {
            "hr_letter_signed": validation["checks"].get("hr_letter_signed"),
            "hr_letter_stamped": validation["checks"].get("hr_letter_stamped"),
            "signature_confidence": validation["checks"].get("signature_confidence"),
            "name_match_id_vs_hr": validation["checks"].get("name_match_id_vs_hr"),
            "name_similarity_score": validation["checks"].get("name_similarity_score"),
            "national_id_consistent": validation["checks"].get("national_id_consistent"),
            "id_not_expired": validation["checks"].get("id_not_expired"),
            "dti_ratio": validation["checks"].get("dti_ratio"),
            "financial_eligible": validation["checks"].get("financial_eligible"),
            "employment_consistent": validation["checks"].get("employment_consistent"),
            "issues": validation["issues"],
            "all_checks_passed": validation["all_passed"],
        },
        "id_parsed": safe_get(id_data, "id_parsed"),
    }


# ─── Full Pipeline Endpoint ────────────────────────────────────────
@app.post("/process-application")
def process_application(req: ProcessRequest):
    """
    Full end-to-end pipeline:
    1. Call national_id_service to extract ID data
    2. Call hr_letter_service to extract HR data + verify signature
    3. Call form_service to fetch form data
    4. Merge + cross-validate all outputs
    5. Optionally run LLM risk assessment
    """
    errors = []

    # ── Step 1: National ID extraction ───────────────────────────
    id_data = {}
    try:
        resp = requests.post(
            f"{NATIONAL_ID_URL}/extract",
            json={"image_filename": req.national_id_image, "shared_dir": req.shared_dir},
            timeout=120
        )
        resp.raise_for_status()
        id_result = resp.json()
        if id_result.get("status") == "Success":
            id_data = id_result.get("data", {})
        else:
            errors.append(f"ID extraction: {id_result.get('detail', 'Unknown error')}")
    except Exception as e:
        errors.append(f"ID service error: {str(e)}")

    # ── Step 2a: HR Letter extraction ────────────────────────────
    hr_data = {}
    try:
        resp = requests.post(
            f"{HR_LETTER_URL}/extract",
            json={"image_filename": req.hr_letter_file, "shared_dir": req.shared_dir},
            timeout=120
        )
        resp.raise_for_status()
        hr_result = resp.json()
        if hr_result.get("status") == "Success":
            hr_data = hr_result.get("data", {})
        else:
            errors.append(f"HR extraction: {hr_result.get('detail', 'Unknown error')}")
    except Exception as e:
        errors.append(f"HR service error: {str(e)}")

    # ── Step 2b: Signature verification ──────────────────────────
    sig_data = {}
    try:
        resp = requests.post(
            f"{HR_LETTER_URL}/verify-signature",
            json={"image_filename": req.hr_letter_file, "shared_dir": req.shared_dir},
            timeout=60
        )
        resp.raise_for_status()
        sig_result = resp.json()
        if sig_result.get("status") == "Success":
            sig_data = sig_result.get("data", {})
    except Exception as e:
        errors.append(f"Signature verification error: {str(e)}")

    # ── Step 3: Form data ────────────────────────────────────────
    form_data = {}
    form_nid = req.form_national_id or id_data.get("national_id")
    if form_nid:
        try:
            resp = requests.post(
                f"{FORM_URL}/fetch-by-id",
                json={"national_id": form_nid},
                timeout=30
            )
            if resp.status_code == 200:
                form_result = resp.json()
                form_data = form_result.get("data", {})
            else:
                # Try latest
                resp = requests.get(f"{FORM_URL}/fetch-latest", timeout=30)
                if resp.status_code == 200:
                    form_data = resp.json().get("data", {})
                else:
                    errors.append("No form data found")
        except Exception as e:
            errors.append(f"Form service error: {str(e)}")
    else:
        try:
            resp = requests.get(f"{FORM_URL}/fetch-latest", timeout=30)
            if resp.status_code == 200:
                form_data = resp.json().get("data", {})
        except Exception as e:
            errors.append(f"Form service error: {str(e)}")

    # ── Step 4: Merge + Cross-validate ───────────────────────────
    merged = merge_extraction_outputs(id_data, hr_data, form_data, sig_data)
    merged["pipeline_errors"] = errors

    # ── Step 5: LLM Risk Assessment (optional, if API key available)
    if VISION_KEY and id_data and hr_data:
        try:
            llm_result = _run_llm_assessment(form_data, id_data, hr_data)
            merged["llm_assessment"] = llm_result
        except Exception as e:
            merged["llm_assessment"] = {"error": str(e)}

    return {"status": "Success", "data": merged}


# ─── Merge Endpoint (pre-extracted data) ────────────────────────────
@app.post("/merge")
def merge_data(req: MergeRequest):
    """Merge pre-extracted data from all 3 pipelines."""
    sig_data = req.signature_data or {}
    merged = merge_extraction_outputs(req.id_data, req.hr_data, req.form_data, sig_data)
    return {"status": "Success", "data": merged}


# ─── Legacy Validate Endpoint ──────────────────────────────────────
@app.post("/validate")
def validate_application(req: ValidateRequest):
    """Legacy: LLM-based validation with pre-extracted data."""
    if not VISION_KEY:
        raise HTTPException(status_code=500, detail="VISION_API_KEY not configured")

    result = _run_llm_assessment(req.form_data, req.id_extraction, req.hr_extraction)
    return {"status": "Success", "data": result}


def _run_llm_assessment(form_data: dict, id_data: dict, hr_data: dict) -> dict:
    """Run the LLM-based risk assessment."""
    user_content = (
        f"Form_Data: {json.dumps(form_data, ensure_ascii=False)}\n\n"
        f"ID_Extraction: {json.dumps(id_data, ensure_ascii=False)}\n\n"
        f"HR_Extraction: {json.dumps(hr_data, ensure_ascii=False)}"
    )

    headers = {
        "Authorization": f"Bearer {VISION_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": ORCHESTRATOR_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }

    resp = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()['choices'][0]['message']['content']

    result = extract_json_from_text(content)
    if result is None:
        raise ValueError(f"Could not parse LLM JSON output: {content[:500]}")

    return result


@app.get("/health")
def health():
    return {"status": "healthy", "service": "orchestrator_service"}
