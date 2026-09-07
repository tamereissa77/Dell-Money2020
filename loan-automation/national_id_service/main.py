"""
National ID Extraction Microservice
Pipeline 1: NeMo Retriever OCR + VLM (Llama 3.2 11B Vision) + Llama 70B Reconstruction
Extracts structured data from Egyptian National ID card images.
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

app = FastAPI(title="NBE National ID Extraction Service")

# ─── API Configuration ───────────────────────────────────────────────
OCR_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"
VISION_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

OCR_KEY = os.environ.get("NVIDIA_API_KEY", "")
VISION_KEY = os.environ.get("VISION_API_KEY", "")

VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
TEXT_MODEL = "meta/llama-3.1-70b-instruct"

# ─── Egyptian Governorate Codes ──────────────────────────────────────
GOVERNORATE_CODES = {
    "01": "القاهرة",        # Cairo
    "02": "الإسكندرية",     # Alexandria
    "03": "بورسعيد",        # Port Said
    "04": "السويس",         # Suez
    "11": "دمياط",          # Damietta
    "12": "الدقهلية",       # Dakahlia
    "13": "الشرقية",        # Sharqia
    "14": "القليوبية",      # Qalyubia
    "15": "كفر الشيخ",      # Kafr El Sheikh
    "16": "الغربية",        # Gharbia
    "17": "المنوفية",       # Menoufia
    "18": "البحيرة",        # Beheira
    "19": "الإسماعيلية",    # Ismailia
    "21": "الجيزة",         # Giza
    "22": "بني سويف",       # Beni Suef
    "23": "الفيوم",         # Fayoum
    "24": "المنيا",         # Minya
    "25": "أسيوط",          # Asyut
    "26": "سوهاج",          # Sohag
    "27": "قنا",            # Qena
    "28": "أسوان",          # Aswan
    "29": "الأقصر",         # Luxor
    "31": "البحر الأحمر",    # Red Sea
    "32": "الوادي الجديد",   # New Valley
    "33": "مطروح",          # Matrouh
    "34": "شمال سيناء",     # North Sinai
    "35": "جنوب سيناء",     # South Sinai
    "88": "خارج الجمهورية",  # Outside Egypt
}


# ─── National ID Parser ─────────────────────────────────────────────
def parse_national_id(nid: str) -> dict:
    """
    Parse the 14-digit Egyptian National ID number.
    Format: CYYMMDGGSSSC
      C  = Century indicator (2=1900s, 3=2000s)
      YY = Year of birth
      MM = Month of birth
      DD = Day of birth  
      GG = Governorate code
      SSS = Sequence number (odd=male, even=female)
      C  = Check digit
    """
    if not nid or len(nid) != 14 or not nid.isdigit():
        return {"valid": False, "error": f"Invalid national ID format: '{nid}'"}

    century_code = nid[0]
    year = nid[1:3]
    month = nid[3:5]
    day = nid[5:7]
    gov_code = nid[7:9]
    sequence = nid[9:12]
    check_digit = nid[13]

    # Determine century
    if century_code == "2":
        full_year = f"19{year}"
    elif century_code == "3":
        full_year = f"20{year}"
    else:
        full_year = f"?{year}"

    # Gender from sequence (odd = male, even = female)
    gender = "ذكر" if int(sequence) % 2 == 1 else "أنثى"

    # Governorate
    governorate = GOVERNORATE_CODES.get(gov_code, f"Unknown ({gov_code})")

    return {
        "valid": True,
        "birth_date": f"{full_year}-{month}-{day}",
        "gender": gender,
        "governorate": governorate,
        "governorate_code": gov_code,
        "century": full_year[:2] + "00s",
        "sequence": sequence,
        "check_digit": check_digit,
    }


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


# ─── Request Schema ─────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    image_path: Optional[str] = None       # Path on shared volume
    image_base64: Optional[str] = None     # Or direct base64
    image_filename: Optional[str] = None   # Legacy: filename in shared_dir
    shared_dir: str = "/shared_data"


# ─── OCR Prompts ─────────────────────────────────────────────────────
CLASSIFICATION_PROMPT = """Look at this document image. This is an Egyptian National ID card (بطاقة الرقم القومي).
Read ALL text visible on the card carefully — names, numbers, dates, addresses.
The text is primarily in Arabic. Be as detailed and accurate as possible.
List every piece of text you can read from the image."""

RECONSTRUCTION_PROMPT = """You are an expert Egyptian National ID card data extractor. You have:

1. Document type: ID_CARD (Egyptian National ID / بطاقة الرقم القومي)
2. Raw OCR data (noisy, with scrambled Arabic RTL text): 
{ocr_text}
3. Visual analysis from image:
{visual_hints}

The OCR data has these known issues:
- Arabic words are in REVERSED order (RTL was read as LTR)
- Character-level errors common in Arabic OCR
- Arabic numerals (٠١٢٣٤٥٦٧٨٩) need conversion to Western digits (0123456789)

YOUR TASK: Cross-reference the OCR text with the visual analysis to reconstruct the correct data.

Extract these fields:
- full_name_ar: The person's full Arabic name (correct RTL order, fix spelling)
- national_id: The 14-digit national ID number (digits only, convert Arabic numerals)
- expiry_date: The expiration date in YYYY-MM-DD format
- address: Full address if visible

Return ONLY this JSON:
{{
  "full_name_ar": "الاسم بالكامل",
  "national_id": "14 digits",
  "expiry_date": "YYYY-MM-DD or null",
  "address": "address or null"
}}

Return ONLY the JSON object. No markdown. No explanation. Start with {{ end with }}."""


# ─── Main Extraction Endpoint ───────────────────────────────────────
@app.post("/extract")
def extract_national_id(req: ExtractRequest):
    """
    3-stage National ID extraction pipeline:
    Stage 1: NeMo Retriever OCR (spatial text detection)
    Stage 2: Llama 3.2 11B Vision (visual reading + classification)
    Stage 3: Llama 3.1 70B (intelligent OCR reconstruction)
    + National ID number parsing for birth date, gender, governorate
    """
    # Resolve image
    if req.image_base64:
        base64_image = req.image_base64
    elif req.image_path:
        if not os.path.exists(req.image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {req.image_path}")
        base64_image = encode_image(req.image_path)
    elif req.image_filename:
        full_path = os.path.join(req.shared_dir, req.image_filename)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {full_path}")
        base64_image = encode_image(full_path)
    else:
        raise HTTPException(status_code=400, detail="Provide image_path, image_base64, or image_filename")

    # ── Stage 1: NeMo OCR ────────────────────────────────────────
    ocr_headers = {"Authorization": f"Bearer {OCR_KEY}", "Content-Type": "application/json"}
    ocr_payload = {"input": [{"url": f"data:image/png;base64,{base64_image}"}]}

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

    # ── Stage 2: Vision Classification + Reading (11B) ───────────
    vision_headers = {"Authorization": f"Bearer {VISION_KEY}", "Content-Type": "application/json"}
    vision_payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
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

    # ── Stage 3: Intelligent OCR Reconstruction (70B) ────────────
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

    # ── Parse National ID Number ─────────────────────────────────
    nid = extracted.get("national_id", "")
    # Clean: remove any non-digit characters
    nid_clean = re.sub(r'[^\d]', '', nid)
    parsed = parse_national_id(nid_clean)

    # Enrich extracted data with parsed info
    if parsed.get("valid"):
        extracted["gender"] = parsed["gender"]
        extracted["governorate"] = parsed["governorate"]
        extracted["birth_date"] = parsed["birth_date"]
        extracted["national_id"] = nid_clean  # Ensure clean digits
    
    extracted["id_parsed"] = parsed

    # ── Build Final Response ─────────────────────────────────────
    response = {
        "status": "Success",
        "pipeline": "national_id_v1",
        "data": extracted,
        "processing": {
            "ocr_lines_detected": len(ocr_texts),
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    # Save output if path available
    try:
        if req.image_filename:
            out_name = os.path.splitext(req.image_filename)[0] + "_id_result.json"
            out_path = os.path.join(req.shared_dir, out_name)
        elif req.image_path:
            out_name = os.path.splitext(os.path.basename(req.image_path))[0] + "_id_result.json"
            out_path = os.path.join(req.shared_dir, out_name)
        else:
            out_path = None

        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Don't fail the request if saving fails

    return response


@app.get("/health")
def health():
    return {"status": "healthy", "service": "national_id_service"}
