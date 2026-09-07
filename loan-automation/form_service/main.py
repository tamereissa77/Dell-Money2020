"""
Loan Application Form Microservice
Pipeline 3: Google Sheets API integration for loan form data
Supports both Google Sheets API mode and manual POST mode.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import json
from datetime import datetime

app = FastAPI(title="NBE Loan Form Service")

# ─── Configuration ───────────────────────────────────────────────────
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")
GOOGLE_CREDENTIALS_PATH = os.environ.get("GOOGLE_CREDENTIALS_PATH", "service_account.json")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Arabic → English field mapping (from the Google Form)
FIELD_MAP = {
    "الأسم بالكامل": "full_name",
    "الاسم بالكامل": "full_name",
    "الرقم القومي": "national_id",
    "النوع": "gender",
    "الحالة الأجتماعية": "marital_status",
    "الحالة الاجتماعية": "marital_status",
    "العنوان بالكامل": "address",
    "العنوان": "address",
    "رقم الهاتف": "phone",
    "الوظيفة": "job_title",
    "الدخل الشهري": "monthly_income",
    "رقم الحساب البنكي": "bank_account",
    "نوع القرض": "loan_type",
    "نوع القرض الشخصي": "personal_loan_type",
    "مقدار القرض المطلوب": "requested_amount",
    "مدة السداد": "repayment_period_months",
    "Timestamp": "timestamp",
}

# ─── In-memory store for manual mode ────────────────────────────────
_manual_store: List[Dict[str, Any]] = []


# ─── Google Sheets Functions ────────────────────────────────────────
def _get_sheets_service():
    """Initialize and return Google Sheets API service."""
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
            raise FileNotFoundError(f"Service account file not found: {GOOGLE_CREDENTIALS_PATH}")

        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_PATH, scopes=SCOPES
        )
        return build("sheets", "v4", credentials=creds)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Google API libraries not installed. Use manual mode via POST /submit"
        )


def _translate_row(headers: list, row: list) -> dict:
    """Map Arabic headers to English keys."""
    raw = dict(zip(headers, row))
    return {FIELD_MAP.get(k, k): v for k, v in raw.items()}


def _fetch_all_rows() -> tuple:
    """Fetch all rows from the Google Sheet. Returns (headers, rows)."""
    service = _get_sheets_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=GOOGLE_SHEET_ID,
        range="Form Responses 1!A:Z"
    ).execute()
    rows = result.get("values", [])
    if len(rows) < 2:
        raise HTTPException(status_code=404, detail="No form responses found in sheet")
    return rows[0], rows[1:]


# ─── Request / Response Schemas ─────────────────────────────────────
class FormSubmission(BaseModel):
    """Manual form submission schema."""
    full_name: str
    national_id: str
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    job_title: Optional[str] = None
    monthly_income: Optional[str] = None
    bank_account: Optional[str] = None
    loan_type: Optional[str] = None
    personal_loan_type: Optional[str] = None
    requested_amount: Optional[str] = None
    repayment_period_months: Optional[str] = None


class FetchByIdRequest(BaseModel):
    national_id: str


# ─── Endpoints: Google Sheets Mode ──────────────────────────────────
@app.get("/fetch-latest")
def fetch_latest():
    """Fetch the most recent form response from Google Sheets."""
    if not GOOGLE_SHEET_ID:
        # Fallback to manual store
        if _manual_store:
            return {"status": "Success", "source": "manual", "data": _manual_store[-1]}
        raise HTTPException(status_code=404, detail="No GOOGLE_SHEET_ID configured and no manual submissions")

    try:
        headers, data_rows = _fetch_all_rows()
        latest = data_rows[-1]
        translated = _translate_row(headers, latest)
        return {
            "status": "Success",
            "source": "google_sheets",
            "data": translated,
            "total_responses": len(data_rows)
        }
    except Exception as e:
        if _manual_store:
            return {"status": "Success", "source": "manual_fallback", "data": _manual_store[-1]}
        raise HTTPException(status_code=502, detail=f"Google Sheets API error: {str(e)}")


@app.post("/fetch-by-id")
def fetch_by_id(req: FetchByIdRequest):
    """Fetch a specific form response by national ID lookup."""
    # Check manual store first
    for entry in reversed(_manual_store):
        if entry.get("national_id") == req.national_id:
            return {"status": "Success", "source": "manual", "data": entry}

    if not GOOGLE_SHEET_ID:
        raise HTTPException(status_code=404, detail="No matching entry found")

    try:
        headers, data_rows = _fetch_all_rows()
        # Find the national_id column index
        nid_header_candidates = ["الرقم القومي", "national_id"]
        nid_col = None
        for i, h in enumerate(headers):
            if h in nid_header_candidates or FIELD_MAP.get(h) == "national_id":
                nid_col = i
                break

        if nid_col is None:
            raise HTTPException(status_code=500, detail="Could not find national_id column in sheet")

        for row in reversed(data_rows):
            if len(row) > nid_col and row[nid_col].strip() == req.national_id.strip():
                translated = _translate_row(headers, row)
                return {"status": "Success", "source": "google_sheets", "data": translated}

        raise HTTPException(status_code=404, detail=f"No form response found for national ID: {req.national_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Google Sheets API error: {str(e)}")


@app.get("/fetch-all")
def fetch_all():
    """Fetch all form responses."""
    results = []

    # Manual store entries
    for entry in _manual_store:
        results.append({"source": "manual", "data": entry})

    if GOOGLE_SHEET_ID:
        try:
            headers, data_rows = _fetch_all_rows()
            for row in data_rows:
                translated = _translate_row(headers, row)
                results.append({"source": "google_sheets", "data": translated})
        except Exception as e:
            pass  # Include whatever we have

    return {
        "status": "Success",
        "total": len(results),
        "responses": results
    }


# ─── Endpoints: Manual Mode ─────────────────────────────────────────
@app.post("/submit")
def submit_form(form: FormSubmission):
    """
    Manual form submission endpoint.
    Use this when Google Sheets is not configured.
    """
    entry = form.dict()
    entry["timestamp"] = datetime.utcnow().isoformat()
    entry["source"] = "manual_api"
    _manual_store.append(entry)

    return {
        "status": "Success",
        "message": "Form submitted successfully",
        "data": entry,
        "total_stored": len(_manual_store)
    }


@app.post("/submit-raw")
def submit_raw(data: Dict[str, Any]):
    """
    Submit raw form data (Arabic or English keys).
    Automatically translates Arabic keys to English.
    """
    translated = {FIELD_MAP.get(k, k): v for k, v in data.items()}
    translated["timestamp"] = datetime.utcnow().isoformat()
    translated["source"] = "manual_raw_api"
    _manual_store.append(translated)

    return {
        "status": "Success",
        "message": "Raw form data submitted",
        "data": translated,
        "total_stored": len(_manual_store)
    }


@app.get("/health")
def health():
    sheets_configured = bool(GOOGLE_SHEET_ID)
    creds_exist = os.path.exists(GOOGLE_CREDENTIALS_PATH) if GOOGLE_CREDENTIALS_PATH else False
    return {
        "status": "healthy",
        "service": "form_service",
        "google_sheets_configured": sheets_configured,
        "credentials_found": creds_exist,
        "manual_entries": len(_manual_store)
    }
