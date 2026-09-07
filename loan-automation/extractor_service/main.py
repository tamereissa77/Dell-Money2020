import os
import cv2
import re
import json
import base64
import requests
import numpy as np
import easyocr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from urllib.parse import unquote
from datetime import datetime
from PIL import Image

app = FastAPI(title="NBE Intelligent Origination Extractor")

# ═══════════════════════════════════════════════════════════════
# SECTION 0 — INITIALIZATION & CONFIG
# ═══════════════════════════════════════════════════════════════
# Initialize EasyOCR for Arabic and English. Since no GPU is available, gpu=False
ocr_reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)

AR2EN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

def ar2en(t: str) -> str:
    return str(t).translate(AR2EN)

VISION_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
VISION_KEY = os.environ.get("VISION_API_KEY")  # supply via .env; no hardcoded fallback
VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
TEXT_MODEL = "qwen/qwen2.5-coder-32b-instruct"

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — IMAGE UTILS
# ═══════════════════════════════════════════════════════════════

def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        img = cv2.imread(unquote(path))
    if img is None:
        raise FileNotFoundError(f'Cannot load image: {path}')
    return img

def upscale(img: np.ndarray, target_w: int = 1800) -> np.ndarray:
    h, w = img.shape[:2]
    if w < target_w:
        scale = target_w / w
        img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    return img

def make_variants(img: np.ndarray) -> list:
    """5 preprocessing variants for maximum OCR recall."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants = [img]  # 1. original color

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    variants.append(cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR))

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))

    ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 12)
    variants.append(cv2.cvtColor(ad, cv2.COLOR_GRAY2BGR))

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    mor = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    _, th2 = cv2.threshold(mor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR))

    return variants

def ocr_zone(img: np.ndarray, min_conf: float = 0.05) -> list:
    """
    OCR all 5 variants. Deduplicate using normalized fingerprints.
    Returns list of {text, conf, x, y, bbox} sorted top→bottom, right→left (RTL).
    """
    seen_norm = set()
    tokens = []
    for variant in make_variants(img):
        raw = ocr_reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            text_threshold=0.20,
            low_text=0.10,
            width_ths=1.0,
            add_margin=0.15,
        )
        for bbox, text, conf in raw:
            t = text.strip()
            if not t or conf < min_conf:
                continue
            norm = ar2en(re.sub(r'[\u064B-\u0652\u0670\s]', '', t))
            norm = re.sub(r'[أإآ]', 'ا', norm)
            norm = re.sub(r'ة', 'ه', norm)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            tokens.append({'text': t, 'conf': conf, 'x': cx, 'y': cy, 'bbox': bbox})

    tokens.sort(key=lambda t: (round(t['y'] / 20) * 20, -t['x']))
    return tokens

def tokens_to_text(tokens: list) -> str:
    return '\n'.join(t['text'] for t in tokens)

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — VALIDATORS & CORE EXTRACTORS
# ═══════════════════════════════════════════════════════════════

VALID_GOVS = {
    '01','02','03','04','11','12','13','14','15','16','17','18',
    '19','21','22','23','24','25','26','27','28','29',
    '31','32','33','34','35','88'
}

def validate_nid(nid: str) -> bool:
    if not nid or len(nid) != 14:
        return False
    if nid[0] not in ('2', '3'):
        return False
    try:
        if not (1 <= int(nid[3:5]) <= 12):   return False
        if not (1 <= int(nid[5:7]) <= 31):   return False
        if nid[7:9] not in VALID_GOVS:       return False
    except Exception:
        return False
    return True

def extract_nid(text: str) -> str:
    t = ar2en(text)
    # Step 1: strip date-like patterns FIRST (prevents date concatenation)
    t_nodates = re.sub(r'\d{4}[/\-\.]\d{1,2}(?:[/\-\.]\d{1,2})?', ' ', t)

    # Step 2: remove all non-digits and slide window
    for source in [t_nodates, t]:
        digits = re.sub(r'[^\d]', '', source)
        for i in range(len(digits) - 13):
            c = digits[i:i+14]
            if validate_nid(c): return c

    # Step 3: try group permutations (handles RTL reversal)
    groups = re.findall(r'\d+', re.sub(r'[^\d\s]', ' ', t_nodates))
    for ordering in [
        groups,
        list(reversed(groups)),
        [g[::-1] for g in groups],
        [g[::-1] for g in reversed(groups)],
    ]:
        joined = ''.join(ordering)
        for i in range(len(joined) - 13):
            c = joined[i:i+14]
            if validate_nid(c): return c
    return ''

def derive_birth_date(nid: str) -> str:
    if not validate_nid(nid):
        return ''
    century = '19' if nid[0] == '2' else '20'
    return f'{century}{nid[1:3]}-{nid[3:5]}-{nid[5:7]}'

def extract_date(text: str, prefer_recent: bool = False) -> str:
    t = ar2en(text)
    found = []
    for pat, fmt in [
        (r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', 'ymd'),
        (r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})',  'dmy'),
        (r'(\d{4})[/\-\.](\d{1,2})',                   'ym'),
    ]:
        for m in re.finditer(pat, t):
            g = m.groups()
            try:
                if fmt == 'ymd':
                    d = f'{g[0]}-{g[1].zfill(2)}-{g[2].zfill(2)}'
                elif fmt == 'dmy':
                    d = f'{g[2]}-{g[1].zfill(2)}-{g[0].zfill(2)}'
                else:
                    d = f'{g[0]}-{g[1].zfill(2)}-01'
                yr = int(d[:4])
                if 1900 <= yr <= 2100:
                    found.append(d)
            except Exception:
                pass
    if not found:
        return ''
    return max(found) if prefer_recent else found[0]

def extract_expiry(text: str) -> str:
    t = ar2en(text)
    patterns = [
        r'سارية\s*حتى\s*([\d/\-\.\s]+)',
        r'سارية\s*حتي\s*([\d/\-\.\s]+)',
        r'ساريه\s*حتى\s*([\d/\-\.\s]+)',
        r'صارية\s*حتى\s*([\d/\-\.\s]+)',
        r'صالحة\s*حتى\s*([\d/\-\.\s]+)',
        r'صالحه\s*حتى\s*([\d/\-\.\s]+)',
        r'حتى\s+([\d/\-\.]+)',
        r'حتي\s+([\d/\-\.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            raw = re.sub(r'\s+', '', m.group(1))
            d = extract_date(raw)
            if d: return d

    # Fallback: most recent future date in 2020-2040
    today = datetime.now().strftime('%Y-%m-%d')
    found = []
    for pat, fmt in [
        (r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', 'ymd'),
        (r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})',  'dmy'),
    ]:
        for m in re.finditer(pat, t):
            g = m.groups()
            try:
                d = (f'{g[0]}-{g[1].zfill(2)}-{g[2].zfill(2)}'
                     if fmt == 'ymd'
                     else f'{g[2]}-{g[1].zfill(2)}-{g[0].zfill(2)}')
                if d >= today and 2020 <= int(d[:4]) <= 2040:
                    found.append(d)
            except: pass
    return max(found) if found else ''

def extract_salary(text: str) -> float:
    t = ar2en(text)
    for pat in [
        r'salary\s+is\s+([\d,]+(?:\.\d+)?)',
        r'([\d,]+(?:\.\d+)?)\s*(?:EGP|Egyptian)',
        r'صافي[^0-9]{0,60}?([\d,،\.]+)',
        r'راتب(?:ه|ها)?\s*الشهري[^0-9]{0,40}?([\d,،\.]+)',
        r'إجمالي[^0-9]{0,40}?([\d,،\.]+)',
        r'([\d,،\.]+)\s*جنيه',
        r'([\d,،\.]+)\s*ج\.م',
    ]:
        for m in re.finditer(pat, t, re.IGNORECASE | re.DOTALL):
            raw = m.group(1).replace('،', '').replace(',', '').replace(' ', '')
            try:
                v = float(raw)
                if 500 <= v <= 500_000: return v
            except: pass
    return 0.0

def extract_gender(text: str, nid: str = '') -> str:
    t_nospace = text.replace(' ', '')
    if re.search(r'ذك[رولا]', text) or re.search(r'ذك[رولا]', t_nospace):
        return 'ذكر'
    if re.search(r'[أا]نث[ىيه]', text) or re.search(r'[أا]نث[ىيه]', t_nospace):
        return 'أنثى'
    if validate_nid(nid):
        return 'ذكر' if int(nid[12]) % 2 == 1 else 'أنثى'
    return ''

def extract_company(text: str, is_english: bool = False) -> str:
    if is_english:
        for line in text.split('\n'):
            line = line.strip()
            if (len(line) > 5 and re.match(r'[A-Z]', line) and ':' not in line
                and not any(skip in line for skip in ['Department','Human','Resources','To Whom'])
                and len(line.split()) >= 2):
                return line
        return ''
    m = re.search(r'((?:شركة|مؤسسة|هيئة|بنك|مصنع)[\u0600-\u06FF\s\w]{2,60}?)(?=\s+(?:بأن|تشهد|تُقر|يشهد)|\n|$)', text, re.MULTILINE)
    return m.group(1).strip() if m else ''

def extract_job_title(text: str, is_english: bool = False) -> str:
    if is_english:
        for pat in [r'(?:capacity of|position of|role of|as a|as an)\s+([A-Za-z][A-Za-z\s\-]{1,40}?)(?=\s*[.\n]|\s+(?:since|from|at|with)\b|$)']:
            m = re.search(pat, text, re.IGNORECASE)
            if m: return m.group(1).strip().rstrip('.,')
        return ''
    m = re.search(r'بوظيفة\s+([\u0600-\u06FF][\u0600-\u06FF\s\w]{1,40}?)(?=\s*[.\n،]|\s+وأن\b|\s+وقد\b|\s+ويحمل\b|$)', text)
    if m: return m.group(1).strip().rstrip('.,،')
    return ''

def merge_bboxes(token_list: list) -> list:
    if not token_list: return [0, 0, 0, 0]
    xs = [pt[0] for tok in token_list for pt in tok['bbox']]
    ys = [pt[1] for tok in token_list for pt in tok['bbox']]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — NVIDIA API VLM (Replaces Local Qwen2-VL)
# ═══════════════════════════════════════════════════════════════

def vlm_extract_json(image_path: str, prompt: str, max_new_tokens: int = 250) -> dict:
    """Uses NVIDIA Llama 3.2 11B Vision API for fallbacks."""
    if not VISION_KEY:
        print("Warning: VISION_API_KEY not set. Skipping VLM fallback.")
        return {}
    
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
            
        headers = {
            "Authorization": f"Bearer {VISION_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + "\nReturn ONLY valid JSON."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": max_new_tokens
        }
        
        resp = requests.post(VISION_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        
        # Try direct parse
        try: return json.loads(content)
        except: pass
        
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
            
        return {}
    except Exception as e:
        print(f"VLM API Error: {str(e)}")
        return {}

def vlm_text_json(prompt: str, max_new_tokens: int = 200) -> dict:
    """Text-only VLM call for gap filling."""
    if not VISION_KEY: return {}
    try:
        headers = {
            "Authorization": f"Bearer {VISION_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta/llama-3.1-70b-instruct",
            "messages": [
                {"role": "user", "content": prompt + "\nReturn ONLY valid JSON."}
            ],
            "temperature": 0.1,
            "max_tokens": max_new_tokens
        }
        resp = requests.post(VISION_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        try: return json.loads(content)
        except: pass
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
        return {}
    except Exception as e:
        print(f"VLM Text API Error: {str(e)}")
        return {}

def format_layout_aware_text(tokens: list) -> str:
    lines = []
    for t in tokens:
        lines.append(f"{t['text']} (x:{int(t['x'])}, y:{int(t['y'])})")
    return '\n'.join(lines)

def qwen_layout_extract(layout_text: str, doc_type: str, max_new_tokens: int = 500) -> dict:
    if not VISION_KEY: return {}
    
    if doc_type == 'id_front':
        fields = "{'full_name_ar': '...', 'address': '...', 'national_id': '...', 'birth_date': 'YYYY-MM-DD'}"
        focus = "focus on name, address, national id, birth date"
    elif doc_type == 'id_back':
        fields = "{'national_id': '...', 'gender': '...', 'expiry_date': 'YYYY-MM-DD', 'job_title_ar': '...'}"
        focus = "focus on national id, gender, expiry date, job"
    elif doc_type == 'hr_letter':
        fields = "{'employee_name_ar': '...', 'national_id': '...', 'monthly_salary': 1000.0, 'company_name': '...', 'issue_date': 'YYYY-MM-DD', 'hire_date': 'YYYY-MM-DD', 'has_signature': true}"
        focus = "focus on name, national id, salary, company, issue date, hire date, signatured or no"
    else:
        fields = "{}"
        focus = "extract all fields"

    prompt = (
        f"You are an expert document data extractor. Given the following OCR text with spatial coordinates (x,y) from a {doc_type}, "
        f"perform element extraction with layout aware. {focus}.\n\n"
        f"OCR Text:\n{layout_text}\n\n"
        f"Return ONLY valid JSON matching this structure exactly: {fields}"
    )

    try:
        headers = {
            "Authorization": f"Bearer {VISION_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": TEXT_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_new_tokens
        }
        resp = requests.post(VISION_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        try: return json.loads(content)
        except: pass
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
        return {}
    except Exception as e:
        print(f"Qwen Text API Error: {str(e)}")
        return {}

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — SPECIFIC DOCUMENT EXTRACTORS
# ═══════════════════════════════════════════════════════════════
# (Implementing simplified versions of Notebook's robust functions)

ADDRESS_WORDS = {
    'قرية', 'مركز', 'محافظة', 'شارع', 'حي', 'ميدان', 'كفر', 'برج',
    'الشيخ', 'الحامول', 'البرلس', 'ناحية', 'عزبة', 'محلة', 'كوم',
    'القاهرة', 'الجيزة', 'الإسكندرية', 'الغربية', 'الدقهلية'
}

def extract_id_front(image_path: str) -> tuple[dict, list]:
    img = upscale(load_img(image_path), 1800)
    h, w = img.shape[:2]
    tokens = ocr_zone(img, min_conf=0.04)
    layout_text = format_layout_aware_text(tokens)

    q_result = qwen_layout_extract(layout_text, 'id_front')

    nid = extract_nid(q_result.get('national_id', ''))
    if not nid:
        full_text = tokens_to_text(tokens)
        nid = extract_nid(full_text)
        
    birth_date = q_result.get('birth_date', '')
    if not birth_date or len(birth_date) < 8:
        birth_date = derive_birth_date(nid)

    name_ar = q_result.get('full_name_ar', '')
    if not name_ar:
        HEADER_WORDS = {'جمهورية','مصر','العربية','بطاقة','تحقيق','الشخصية'}
        name_clean = []
        for tok in tokens:
            t = tok['text']
            if tok['y']/h < 0.52 and tok['x']/w > 0.38:
                if not re.search(r'\d', ar2en(t)) and len(t) > 3:
                    words = set(re.findall(r'\w+', t))
                    if not (words & (ADDRESS_WORDS | HEADER_WORDS)):
                        name_clean.append(t)
        name_ar = ' '.join(name_clean[:4])

    result = {
        'full_name_ar': name_ar,
        'national_id': nid,
        'birth_date': birth_date,
        'address': q_result.get('address', ''),
        'governorate': '',
    }
    return result, tokens

JOB_CORRECTIONS = {
    'كهربام': 'كهرباء', 'كهرياء': 'كهرباء', 'تهرباء': 'كهرباء', 'مهندء': 'مهندس', 'مهنذس': 'مهندس'
}

def extract_id_back(image_path: str) -> tuple[dict, list]:
    img = upscale(load_img(image_path), 1800)
    tokens = ocr_zone(img, min_conf=0.04)
    layout_text = format_layout_aware_text(tokens)

    q_result = qwen_layout_extract(layout_text, 'id_back')

    full_text = tokens_to_text(tokens)
    nid = extract_nid(q_result.get('national_id', ''))
    if not nid:
        nid = extract_nid(full_text)
        
    birth_date = derive_birth_date(nid)
    
    expiry = extract_date(q_result.get('expiry_date', ''))
    if not expiry:
        expiry = extract_expiry(full_text)

    gender = extract_gender(full_text, nid)
    job = extract_job_title(full_text) or q_result.get('job_title_ar', '')

    result = {
        'national_id': nid,
        'birth_date': birth_date,
        'expiry_date': expiry,
        'gender': gender,
        'job_title_ar': job,
    }
    return result, tokens


def extract_hr_letter(image_path: str) -> tuple[dict, list]:
    img = upscale(load_img(image_path), 1200)
    tokens = ocr_zone(img, min_conf=0.18)
    layout_text = format_layout_aware_text(tokens)

    q_result = qwen_layout_extract(layout_text, 'hr_letter')

    full_text = tokens_to_text(tokens)
    ascii_ratio = len(re.findall(r'[A-Z]', full_text, re.I)) / max(len(full_text), 1)
    is_en = ascii_ratio > 0.4

    nid = extract_nid(q_result.get('national_id', ''))
    if not nid: nid = extract_nid(full_text)
        
    salary = q_result.get('monthly_salary')
    if salary is None or (isinstance(salary, str) and not str(salary).replace('.', '', 1).isdigit()):
        salary = extract_salary(full_text)
    else:
        try: salary = float(salary)
        except: salary = extract_salary(full_text)

    issue_date = extract_date(q_result.get('issue_date', ''))
    if not issue_date: issue_date = extract_date(full_text, prefer_recent=True)

    company = extract_company(full_text, is_en) or q_result.get('company_name', '')
    job = extract_job_title(full_text, is_en) or q_result.get('job_title', '')

    # Signature detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bottom = gray[int(h * 0.65):, :]
    dark_ratio = np.sum(bottom < 100) / bottom.size
    has_signature = bool(dark_ratio > 0.006 or q_result.get('has_signature', False))

    final = {
        'employee_name_ar': q_result.get('employee_name_ar', ''),
        'national_id': nid,
        'monthly_salary': salary,
        'issue_date': issue_date,
        'hire_date': q_result.get('hire_date', ''),
        'job_title': job,
        'company_name': company,
        'has_signature': has_signature,
        'has_stamp': has_signature,
    }
    return final, tokens

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — BOUNDING BOX VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def draw_extracted_boxes(image_path: str, result_dict: dict, tokens: list, output_path: str):
    """
    Draw bounding boxes on tokens that contributed to the extracted result.
    """
    img = cv2.imread(image_path)
    if img is None:
        img = cv2.imread(unquote(image_path))
    
    # Values we want to highlight
    target_values = set()
    for v in result_dict.values():
        if isinstance(v, str) and len(v) > 2:
            target_values.add(v)
        elif isinstance(v, (int, float)) and v > 0:
            target_values.add(str(v))
            
    # Scale token coordinates if we upscaled during OCR
    # (assuming tokens correspond to the upscaled image; we'll draw on upscaled copy)
    draw_img = upscale(img, 1800 if 'ID' in str(result_dict.keys()).upper() else 1200)

    for tok in tokens:
        t = tok['text']
        bbox = tok['bbox'] # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        
        # Check if this token matches any extracted value
        is_target = False
        for val in target_values:
            if t in val or val in t:
                is_target = True
                break
                
        # Draw box
        pt1 = (int(bbox[0][0]), int(bbox[0][1]))
        pt2 = (int(bbox[2][0]), int(bbox[2][1]))
        
        if is_target:
            # Green for extracted targets
            color = (0, 255, 0)
            thickness = 4
        else:
            # Red/Gray for other OCR'd text
            color = (0, 0, 255)
            thickness = 1
            
        cv2.rectangle(draw_img, pt1, pt2, color, thickness)

    cv2.imwrite(output_path, draw_img)
    return output_path


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — FASTAPI ENDPOINTS
# ═══════════════════════════════════════════════════════════════

class ExtractRequest(BaseModel):
    image_filename: str
    doc_type: str = "id_front" # id_front, id_back, hr_letter
    shared_dir: str = "/shared_data"

@app.post("/extract")
def extract_document(req: ExtractRequest):
    image_path = os.path.join(req.shared_dir, req.image_filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found at {image_path}")

    doc_type = req.doc_type.lower()
    
    try:
        if doc_type in ('id', 'id_front'):
            result, tokens = extract_id_front(image_path)
        elif doc_type == 'id_back':
            result, tokens = extract_id_back(image_path)
        elif doc_type == 'hr_letter':
            result, tokens = extract_hr_letter(image_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown doc_type: {doc_type}")
            
        # Draw bounding boxes
        annotated_filename = f"annotated_{os.path.basename(req.image_filename)}"
        annotated_path = os.path.join(req.shared_dir, annotated_filename)
        draw_extracted_boxes(image_path, result, tokens, annotated_path)

        # Build response
        response = {
            "status": "Success",
            "doc_type": doc_type,
            "data": result,
            "annotated_image": annotated_filename,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save JSON output
        out_name = os.path.splitext(req.image_filename)[0] + f"_{doc_type}_result.json"
        with open(os.path.join(req.shared_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
            
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "service": "nbe_extractor_service"}
