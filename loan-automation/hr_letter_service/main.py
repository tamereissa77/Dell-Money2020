"""
HR Letter Extraction Microservice
Pipeline 2: PDF/Image → NeMo OCR + VLM extraction + Signature Verification
Extracts structured data from Egyptian HR salary letters.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import base64
import requests
import json
import re
from datetime import datetime

app = FastAPI(title="NBE HR Letter Extraction Service")

# ─── API Configuration ───────────────────────────────────────────────
OCR_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"
VISION_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

OCR_KEY = os.environ.get("NVIDIA_API_KEY", "")
VISION_KEY = os.environ.get("VISION_API_KEY", "")

VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
TEXT_MODEL = "meta/llama-3.1-70b-instruct"


# ─── Utility Functions ──────────────────────────────────────────────
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json(content: str) -> Optional[dict]:
    """Brace-balancing algorithm to extract nested JSON from LLM output."""
    for match in re.finditer(r'\{', content):
        start = match.start()
        balance = 0
        for i in range(start, len(content)):
            if content[i] == '{':
                balance += 1
            elif content[i] == '}':
                balance -= 1
                if balance == 0:
                    candidate = content[start:i+1]
                    try:
                        cleaned = candidate.strip()
                        if cleaned.startswith('```json'):
                            cleaned = cleaned.replace('```json', '').replace('```', '')
                        return json.loads(cleaned)
                    except:
                        break
    return None


def pdf_to_base64_images(pdf_path: str) -> list:
    """Convert PDF pages to base64 JPEG images using PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("jpeg")
        images.append(base64.b64encode(img_bytes).decode("utf-8"))
    doc.close()
    return images


def pdf_crop_bottom(pdf_path: str, ratio: float = 0.3) -> str:
    """Crop the bottom portion of the first page of a PDF, return as base64."""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[0]
    h = page.rect.height
    clip = fitz.Rect(0, h * (1 - ratio), page.rect.width, h)
    pix = page.get_pixmap(dpi=200, clip=clip)
    b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
    doc.close()
    return b64


def image_crop_bottom(image_path: str, ratio: float = 0.3) -> str:
    """Crop the bottom portion of an image file, return as base64."""
    from PIL import Image
    import io
    img = Image.open(image_path)
    w, h = img.size
    crop_box = (0, int(h * (1 - ratio)), w, h)
    cropped = img.crop(crop_box)
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Request Schema ─────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    file_path: Optional[str] = None        # Path on shared volume (PDF or image)
    image_base64: Optional[str] = None     # Or direct base64
    image_filename: Optional[str] = None   # Legacy: filename in shared_dir
    shared_dir: str = "/shared_data"


class SignatureRequest(BaseModel):
    file_path: Optional[str] = None
    image_base64: Optional[str] = None
    image_filename: Optional[str] = None
    shared_dir: str = "/shared_data"
    crop_ratio: float = 0.3               # Bottom % to crop for signature check


# ─── Prompts ─────────────────────────────────────────────────────────
CLASSIFICATION_PROMPT = """Look at this document image. This is an HR salary letter from an Egyptian company.
Text may be in Arabic, English, or mixed. Read ALL text carefully.
List every piece of text you can read: employee name, salary, company name, job title, dates, stamps, signatures.
Be as detailed and accurate as possible."""

RECONSTRUCTION_PROMPT = """You are an expert Egyptian HR salary letter data extractor. You have:

1. Document type: HR_LETTER
2. Raw OCR data (noisy, with scrambled Arabic RTL text): 
{ocr_text}
3. Visual analysis from image:
{visual_hints}

The OCR data has these known issues:
- Arabic words are in REVERSED order (RTL was read as LTR)
- Character-level errors common in Arabic OCR
- Arabic numerals need conversion to Western digits

YOUR TASK: Cross-reference the OCR text with the visual analysis to reconstruct the correct data.

Extract these fields:
- employee_name: The employee's name as written (English if available)
- employee_name_ar: Arabic name if present, else null
- monthly_salary: Numeric value only (EGP)
- issue_date: Date in YYYY-MM-DD format
- company_name: Employer name
- job_title: Position/role
- is_signed: true or false (did you see a signature?)

Return ONLY this JSON:
{{
  "employee_name": "name as written",
  "employee_name_ar": "Arabic name or null",
  "monthly_salary": "numeric value only",
  "issue_date": "YYYY-MM-DD",
  "company_name": "employer name",
  "job_title": "position",
  "is_signed": true
}}

Return ONLY the JSON object. No markdown. No explanation."""

SIGNATURE_PROMPT = """This is the bottom section of an official HR letter.
Look carefully for: handwritten signature, official stamp/seal, or both.
Return ONLY valid JSON:
{{
  "has_handwritten_signature": true,
  "has_official_stamp": true,
  "confidence": "high",
  "notes": "brief explanation in English"
}}"""


def resolve_image(req, allow_pdf=True):
    """Resolve the image source from request parameters. Returns (base64, file_path, is_pdf)."""
    if req.image_base64:
        return req.image_base64, None, False

    file_path = None
    if req.file_path:
        file_path = req.file_path
    elif hasattr(req, 'image_filename') and req.image_filename:
        file_path = os.path.join(req.shared_dir, req.image_filename)

    if file_path:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        is_pdf = file_path.lower().endswith('.pdf')
        
        if is_pdf and allow_pdf:
            pages = pdf_to_base64_images(file_path)
            if not pages:
                raise HTTPException(status_code=400, detail="PDF has no pages")
            return pages[0], file_path, True
        else:
            return encode_image(file_path), file_path, False

    raise HTTPException(status_code=400, detail="Provide file_path, image_base64, or image_filename")


# ─── HR Letter Extraction Endpoint ──────────────────────────────────
@app.post("/extract")
def extract_hr_letter(req: ExtractRequest):
    """
    3-stage HR Letter extraction pipeline:
    Stage 1: NeMo Retriever OCR
    Stage 2: Llama 3.2 11B Vision (visual reading)
    Stage 3: Llama 3.1 70B (intelligent reconstruction)
    """
    base64_image, file_path, is_pdf = resolve_image(req)

    # ── Stage 1: NeMo OCR ────────────────────────────────────────
    ocr_headers = {"Authorization": f"Bearer {OCR_KEY}", "Content-Type": "application/json"}
    ocr_payload = {"input": [{"url": f"data:image/jpeg;base64,{base64_image}"}]}

    ocr_texts = []
    raw_ocr = {}
    try:
        ocr_resp = requests.post(OCR_URL, headers=ocr_headers, json=ocr_payload)
        ocr_resp.raise_for_status()
        raw_ocr = ocr_resp.json()
        if "data" in raw_ocr:
            for item in raw_ocr["data"]:
                for det in item.get("text_detections", []):
                    text = det.get("text_prediction", {}).get("text", "")
                    conf = det.get("text_prediction", {}).get("confidence", 0)
                    if text:
                        ocr_texts.append(f'- "{text}" (confidence: {conf:.2f})')
    except Exception as e:
        raw_ocr = {"error": str(e)}

    ocr_text_summary = "\n".join(ocr_texts) if ocr_texts else "OCR returned no text"

    # ── Stage 2: Vision Reading (11B) ────────────────────────────
    vision_headers = {"Authorization": f"Bearer {VISION_KEY}", "Content-Type": "application/json"}
    vision_payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }

    visual_hints = ""
    try:
        vision_resp = requests.post(VISION_URL, headers=vision_headers, json=vision_payload)
        vision_resp.raise_for_status()
        visual_hints = vision_resp.json()['choices'][0]['message']['content']
    except Exception as e:
        visual_hints = f"Vision analysis failed: {str(e)}"

    # ── Stage 3: Intelligent Reconstruction (70B) ────────────────
    reconstruction_input = RECONSTRUCTION_PROMPT.format(
        ocr_text=ocr_text_summary,
        visual_hints=visual_hints
    )

    text_payload = {
        "model": TEXT_MODEL,
        "messages": [{"role": "user", "content": reconstruction_input}],
        "temperature": 0.1,
        "max_tokens": 1024
    }

    try:
        text_resp = requests.post(VISION_URL, headers=vision_headers, json=text_payload)
        text_resp.raise_for_status()
        text_content = text_resp.json()['choices'][0]['message']['content']

        extracted = extract_json(text_content)
        if extracted is None:
            raise ValueError(f"No valid JSON in LLM output: {text_content[:500]}")

    except Exception as e:
        return {
            "status": "Error",
            "detail": str(e),
            "ocr_fallback": raw_ocr,
            "visual_hints": visual_hints
        }

    # ── Build Response ───────────────────────────────────────────
    response = {
        "status": "Success",
        "pipeline": "hr_letter_v1",
        "data": extracted,
        "processing": {
            "ocr_lines_detected": len(ocr_texts),
            "source_type": "pdf" if is_pdf else "image",
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    # Save output
    try:
        if req.image_filename:
            out_name = os.path.splitext(req.image_filename)[0] + "_hr_result.json"
        elif file_path:
            out_name = os.path.splitext(os.path.basename(file_path))[0] + "_hr_result.json"
        else:
            out_name = None

        if out_name:
            out_path = os.path.join(req.shared_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return response


# ─── Signature Verification Endpoint ────────────────────────────────
@app.post("/verify-signature")
def verify_signature(req: SignatureRequest):
    """
    Dedicated signature + stamp verification.
    Crops the bottom portion of the document and analyzes it with VLM.
    """
    # Get the cropped bottom section
    file_path = req.file_path
    if not file_path and req.image_filename:
        file_path = os.path.join(req.shared_dir, req.image_filename)

    if req.image_base64:
        # Can't crop from base64 directly without decoding, use full image
        b64_cropped = req.image_base64
    elif file_path:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if file_path.lower().endswith('.pdf'):
            b64_cropped = pdf_crop_bottom(file_path, req.crop_ratio)
        else:
            b64_cropped = image_crop_bottom(file_path, req.crop_ratio)
    else:
        raise HTTPException(status_code=400, detail="Provide file_path, image_base64, or image_filename")

    # ── VLM Signature Check ──────────────────────────────────────
    vision_headers = {"Authorization": f"Bearer {VISION_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_cropped}"}},
                    {"type": "text", "text": SIGNATURE_PROMPT}
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1
    }

    try:
        resp = requests.post(VISION_URL, headers=vision_headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        result = extract_json(raw)
        if result is None:
            result = {
                "has_handwritten_signature": False,
                "has_official_stamp": False,
                "confidence": "low",
                "notes": f"Could not parse VLM response: {raw[:200]}"
            }
    except Exception as e:
        result = {
            "has_handwritten_signature": False,
            "has_official_stamp": False,
            "confidence": "low",
            "notes": f"Verification failed: {str(e)}"
        }

    return {
        "status": "Success",
        "pipeline": "signature_verification_v1",
        "data": result,
        "processing": {
            "crop_ratio": req.crop_ratio,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "hr_letter_service"}
