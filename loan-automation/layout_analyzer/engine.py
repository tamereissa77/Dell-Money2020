"""
Document Layout Analysis Engine
Precise OCR-based field detection with STRICT BOUNDARY ENFORCEMENT for:
- Egyptian National ID (Front & Back)
- HR Salary Letters
"""

import cv2
import re
import numpy as np
import easyocr
from typing import Dict, List, Tuple, Optional

# Arabic-to-English digit translation
AR2EN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

def ar2en(t: str) -> str:
    return str(t).translate(AR2EN)

class LayoutAnalyzer:
    """Zero-drift spatial localization engine for document entity extraction."""

    def __init__(self, gpu: bool = False):
        self.reader = easyocr.Reader(['ar', 'en'], gpu=gpu, verbose=False)
        self._gender_keywords = {'ذكر', 'ذک', 'انثى', 'أنثى', 'انثي', 'أنثي', 'أنثئ'}
        self._marital_keywords = {'أعزب', 'اعزب', 'متزوج', 'متزوجة', 'مطلق', 'أرمل', 'ارمل',
                                  'أنسة', 'آنسة', 'انسة', 'أنسه', 'انسه'}
        self._religion_keywords = {'مسلم', 'مسلمة', 'مسلمه', 'مسيحي', 'مسيحية', 'مسيحيه'}
        self._job_keywords = [
            'محاسب', 'مهندس', 'مدير', 'أخصائي', 'اخصائي', 'محلل', 'مراجع',
            'مصمم', 'مطور', 'طبيب', 'صيدلي', 'عامل', 'طالب', 'طالبه', 'طالبة',
            'كهرباء', 'مبيعات', 'تسويق', 'عمليات', 'مشروعات', 'موارد', 'تأمينات',
            'مشتريات', 'إنتاج', 'شبكات', 'جرافيك', 'ويب', 'برمجيات', 'مالي'
        ]

    def _ocr_image(self, img: np.ndarray, min_conf: float = 0.15) -> List[dict]:
        """Run EasyOCR and return structured token list with bounding boxes."""
        raw = self.reader.readtext(img, detail=1, paragraph=False,
                                   text_threshold=0.3, low_text=0.15,
                                   width_ths=0.5, add_margin=0.05)
        tokens = []
        for bbox, text, conf in raw:
            t = text.strip()
            if not t or conf < min_conf:
                continue
            x1 = int(min(p[0] for p in bbox))
            y1 = int(min(p[1] for p in bbox))
            x2 = int(max(p[0] for p in bbox))
            y2 = int(max(p[1] for p in bbox))
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            tokens.append({
                'text': t, 'conf': conf,
                'bbox': [x1, y1, x2, y2],
                'cx': cx, 'cy': cy
            })
        tokens.sort(key=lambda t: (t['cy'], -t['cx']))  # top→bottom, RTL
        return tokens

    def _merge_boxes(self, boxes: List[List[int]]) -> List[int]:
        """Merge multiple boxes into a single enclosing bounding box."""
        if not boxes:
            return [0, 0, 0, 0]
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return [x1, y1, x2, y2]

    def _extract_sub_box(self, tok: dict, target_str: str, is_rtl: bool = True) -> List[int]:
        """
        Shrink-wrap: extract the proportional bounding box for a target substring within a token.
        """
        txt = tok['text']
        if target_str not in txt:
            # Maybe the target_str is English but txt has Arabic numerals. 
            # We'll do a simple character length proportion if we can't find exact match.
            # But normally we find the exact match.
            txt_en = ar2en(txt)
            if target_str in txt_en:
                start_idx = txt_en.index(target_str)
            else:
                return tok['bbox']
        else:
            start_idx = txt.index(target_str)
            
        full_len = max(len(txt), 1)
        target_len = len(target_str)
        
        ratio_start = start_idx / full_len
        ratio_end = (start_idx + target_len) / full_len
        
        bx1, by1, bx2, by2 = tok['bbox']
        box_w = bx2 - bx1
        
        if is_rtl:
            # RTL text: string starts from right
            sx2 = int(bx2 - box_w * ratio_start)
            sx1 = int(bx2 - box_w * ratio_end)
        else:
            # LTR text
            sx1 = int(bx1 + box_w * ratio_start)
            sx2 = int(bx1 + box_w * ratio_end)
            
        # Add a tiny padding to not clip ink, but keep it tight
        sx1 = max(bx1, sx1 - 2)
        sx2 = min(bx2, sx2 + 2)
        
        return [sx1, by1, sx2, by2]

    def _detect_signature_stamp(self, img: np.ndarray) -> Optional[List[int]]:
        """Detect signature/stamp region in bottom portion of image."""
        h, w = img.shape[:2]
        bottom_region = img[int(h * 0.55):, :]
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, (90, 30, 30), (140, 255, 255))
        dark_mask = cv2.inRange(gray, 0, 120)
        combined = cv2.bitwise_or(blue_mask, dark_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(combined, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        all_points = np.vstack(contours)
        x, y, bw, bh = cv2.boundingRect(all_points)
        min_area = (h * 0.05) * (w * 0.05)
        if bw * bh < min_area:
            return None
        return [x, y + int(h * 0.55), x + bw, y + int(h * 0.55) + bh]

    def _find_nid_sequence(self, tokens: List[dict], y_filter=None, x_filter=None) -> Optional[Dict]:
        """Finds a 14-digit National ID with date-stripping logic."""
        filtered_toks = []
        for tok in tokens:
            if y_filter and not y_filter(tok): continue
            if x_filter and not x_filter(tok): continue
            filtered_toks.append(tok)
            
        full_txt = " ".join(ar2en(t['text']) for t in filtered_toks)
        # Step 1: strip dates to prevent fake NID from concatenation
        txt_nodates = re.sub(r'\d{4}[/\-\.]\d{1,2}(?:[/\-\.]\d{1,2})?', ' ', full_txt)
        
        for source in [txt_nodates, full_txt]:
            digits = re.sub(r'[^\d]', '', source)
            for i in range(len(digits) - 13):
                cand = digits[i:i+14]
                if cand[0] in ('2', '3') and 1 <= int(cand[3:5]) <= 12 and 1 <= int(cand[5:7]) <= 31:
                    # Find tokens that likely contributed to this ID
                    return {'bbox': self._merge_boxes([t['bbox'] for t in filtered_toks]), 'value': cand, 'tokens': filtered_toks}

        return None

    # ═══════════════════════════════════════════════════════════════
    # TASK 1: ID FRONT
    # ═══════════════════════════════════════════════════════════════
    def analyze_id_front(self, image_path: str) -> Dict:
        """Detect fields on Egyptian National ID front side with STRICT BOUNDARY ENFORCEMENT."""
        img = cv2.imread(image_path)
        if img is None: raise FileNotFoundError(f"Cannot load: {image_path}")
        h, w = img.shape[:2]
        tokens = self._ocr_image(img)
        results = {}

        HEADER_WORDS = {'جمهورية', 'مصر', 'العربية', 'بطاقة', 'تحقيق', 'الشخصية'}
        ADDR_WORDS = {'قرية', 'مركز', 'محافظة', 'شارع', 'حي', 'كفر', 'الشيخ', 'الحامول', 'البرلس', 'ناحية', 'عزبة', 'محطة', 'مدينة', 'برج', 'عزبه'}
        
        name_tokens = []
        addr_tokens = []

        for tok in tokens:
            y_f = tok['cy'] / h
            x_f = tok['cx'] / w
            txt = tok['text']
            
            if y_f < 0.15: continue
            words = set(txt.split())
            if words <= HEADER_WORDS: continue

            arabic_chars = len(re.findall(r'[\u0600-\u06FF]', txt))
            total_chars = len(txt.replace(' ', ''))
            arabic_ratio = arabic_chars / max(total_chars, 1)
            has_addr_word = bool(words & ADDR_WORDS)
            has_numbers = bool(re.search(r'\d{3,}', ar2en(txt)))

            # NAME zone: y=14–57%, right half, no address words, no long digits, Arabic
            if (0.14 <= y_f <= 0.57 and x_f > 0.33
                    and arabic_ratio > 0.60
                    and not has_addr_word
                    and not has_numbers
                    and total_chars >= 2):
                name_tokens.append(tok)

            # ADDRESS zone: y=42–72%, right half, OR has address word
            elif (0.42 <= y_f <= 0.72 and x_f > 0.30) or has_addr_word:
                if arabic_ratio > 0.20 and total_chars >= 2 and not has_numbers:
                    addr_tokens.append(tok)

        if name_tokens:
            results['Full_Name'] = {
                'bbox': self._merge_boxes([t['bbox'] for t in name_tokens]),
                'value': ' '.join(t['text'] for t in name_tokens),
                'tokens': name_tokens
            }
        if addr_tokens:
            results['Address'] = {
                'bbox': self._merge_boxes([t['bbox'] for t in addr_tokens]),
                'value': ' '.join(t['text'] for t in addr_tokens),
                'tokens': addr_tokens
            }

        # National_ID_Front: Exactly 14 digits.
        nid_res = self._find_nid_sequence(tokens, y_filter=lambda t: t['cy'] / h >= 0.60, x_filter=lambda t: t['cx'] / w > 0.35)
        if nid_res:
            results['National_ID_Front'] = nid_res

        return {'doc_type': 'id_front', 'image_size': [w, h], 'fields': results, 'all_tokens': tokens}

    # ═══════════════════════════════════════════════════════════════
    # TASK 2: ID BACK
    # ═══════════════════════════════════════════════════════════════
    def analyze_id_back(self, image_path: str) -> Dict:
        """Detect fields on Egyptian National ID back side with STRICT BOUNDARY ENFORCEMENT."""
        img = cv2.imread(image_path)
        if img is None: raise FileNotFoundError(f"Cannot load: {image_path}")
        h, w = img.shape[:2]
        tokens = self._ocr_image(img)
        results = {}

        # National_ID_Back: Top row, 14 digits strictly. Strip dates first.
        top_tokens = [t for t in tokens if t['cy'] / h < 0.40]
        top_text = " ".join(ar2en(t['text']) for t in top_tokens)
        top_text_nodates = re.sub(r'\d{4}[/\-\.]\d{1,2}(?:[/\-\.]\d{1,2})?', ' ', top_text)
        
        nid_res = self._find_nid_sequence(tokens, y_filter=lambda t: t['cy'] / h < 0.4)
        if nid_res:
            # Re-verify with date-stripped text if possible
            digits_raw = re.sub(r'[^\d]', '', top_text_nodates)
            if len(digits_raw) >= 14:
                nid_res['value'] = digits_raw[:14]
            results['National_ID_Back'] = nid_res

        # Expiry_Date: Date format near "البطاقة سارية حتى"
        expiry_patterns = [
            r'سارية\s*حتى\s*([\d/\-\.\s]+)', r'سارية\s*حتي\s*([\d/\-\.\s]+)',
            r'ساريه\s*حتى\s*([\d/\-\.\s]+)', r'صارية\s*حتى\s*([\d/\-\.\s]+)',
            r'صالحة\s*حتى\s*([\d/\-\.\s]+)', r'حتى\s+([\d/\-\.]+)',
        ]
        full_text_en = " ".join(ar2en(t['text']) for t in tokens)
        for pat in expiry_patterns:
            m = re.search(pat, full_text_en)
            if m:
                val_raw = re.sub(r'\s+', '', m.group(1))
                # Find which token has this date
                for t in tokens:
                    c = ar2en(t['text'])
                    if val_raw in c.replace('/', '').replace('-', '').replace('.', ''):
                        results['Expiry_Date'] = {'bbox': t['bbox'], 'value': val_raw, 'tokens': [t]}
                        break
                if 'Expiry_Date' in results: break

        # Gender: Exact word or spaced
        full_text_raw = " ".join(t['text'] for t in tokens)
        t_nospace = full_text_raw.replace(' ', '')
        gender_val = ''
        if re.search(r'ذك[رولا]', full_text_raw) or re.search(r'ذك[رولا]', t_nospace):
            gender_val = 'ذكر'
        elif re.search(r'[أا]نث[ىيه]', full_text_raw) or re.search(r'[أا]نث[ىيه]', t_nospace):
            gender_val = 'أنثى'
        
        if gender_val:
            for tok in tokens:
                if any(k in tok['text'] for k in ['ذك', 'انث', 'أنث']):
                    results['Gender'] = {'bbox': tok['bbox'], 'value': gender_val, 'tokens': [tok]}
                    break
        elif 'National_ID_Back' in results:
            nid = results['National_ID_Back']['value']
            results['Gender'] = {'bbox': results['National_ID_Back']['bbox'], 'value': 'ذكر' if int(nid[12]) % 2 == 1 else 'أنثى', 'tokens': []}

        # Marital_Status
        for tok in tokens:
            if 'Marital_Status' in results: break
            txt = tok['text']
            for kw in self._marital_keywords:
                if kw in txt:
                    is_rtl = bool(re.search(r'[\u0600-\u06FF]', txt))
                    bbox = self._extract_sub_box(tok, kw, is_rtl=is_rtl)
                    results['Marital_Status'] = {'bbox': bbox, 'value': kw, 'tokens': [tok]}
                    break
        
        # Religion
        for tok in tokens:
            if 'Religion' in results: break
            txt = tok['text']
            for kw in self._religion_keywords:
                if kw in txt:
                    is_rtl = bool(re.search(r'[\u0600-\u06FF]', txt))
                    bbox = self._extract_sub_box(tok, kw, is_rtl=is_rtl)
                    results['Religion'] = {'bbox': bbox, 'value': kw, 'tokens': [tok]}
                    break

        # Job_Profession: Top right area, below ID.
        for tok in tokens:
            words = tok['text'].split()
            for w_str in words:
                if any(kw == w_str for kw in self._job_keywords):
                    bbox = self._extract_sub_box(tok, w_str, is_rtl=True)
                    results['Job_Profession'] = {'bbox': bbox, 'value': w_str, 'tokens': [tok]}
                    break
            if 'Job_Profession' in results: break

        return {'doc_type': 'id_back', 'image_size': [w, h], 'fields': results, 'all_tokens': tokens}

    # ═══════════════════════════════════════════════════════════════
    # TASK 3: HR LETTER
    # ═══════════════════════════════════════════════════════════════
    def analyze_hr_letter(self, image_path: str) -> Dict:
        """Detect fields on HR salary certificate with INLINE ISOLATION."""
        img = cv2.imread(image_path)
        if img is None: raise FileNotFoundError(f"Cannot load: {image_path}")
        h, w = img.shape[:2]
        tokens = self._ocr_image(img, min_conf=0.2)
        results = {}
        
        full_text = " ".join(t['text'] for t in tokens)
        ascii_ratio = len(re.findall(r'[A-Za-z]', full_text)) / max(len(full_text), 1)
        is_english = ascii_ratio > 0.40

        if is_english:
            # English Logic
            for tok in tokens:
                line = tok['text'].strip()
                if (len(line) > 5 and re.match(r'[A-Z]', line) and ':' not in line and 'Department' not in line and len(line.split()) >= 2):
                    results['Company_Name'] = {'bbox': tok['bbox'], 'value': line, 'tokens': [tok]}
                    break
            for tok in tokens:
                m = re.search(r'(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Za-z][A-Za-z\s]{3,40}?)', tok['text'], re.IGNORECASE)
                if m:
                    results['Employee_Name'] = {'bbox': self._extract_sub_box(tok, m.group(1), is_rtl=False), 'value': m.group(1), 'tokens': [tok]}
                    break
            for tok in tokens:
                m = re.search(r'(?:capacity of|position of|role of|as a)\s+([A-Za-z][A-Za-z\s]{1,40}?)', tok['text'], re.IGNORECASE)
                if m:
                    results['Job_Title'] = {'bbox': self._extract_sub_box(tok, m.group(1), is_rtl=False), 'value': m.group(1), 'tokens': [tok]}
                    break
        else:
            # Arabic Logic
            m_comp = re.search(r'((?:شركة|مؤسسة|هيئة|بنك|مصنع)[\u0600-\u06FF\s\w]{2,60}?)(?=\s+(?:بأن|تشهد|تُقر|يشهد)|\n|$)', full_text, re.MULTILINE)
            if m_comp:
                results['Company_Name'] = {'bbox': self._merge_boxes([t['bbox'] for t in tokens if m_comp.group(1)[:10] in t['text']]), 'value': m_comp.group(1), 'tokens': []}
            
            for tok in tokens:
                m = re.search(r'السيد[اةهً]?\s*/?\s*([\u0600-\u06FF][\u0600-\u06FF\s]{3,30}?)\s+(?:والذي|يحمل|يعمل|يعمل لدى|تاريخ التعيين)', tok['text'])
                if m:
                    results['Employee_Name'] = {'bbox': self._extract_sub_box(tok, m.group(1), is_rtl=True), 'value': m.group(1), 'tokens': [tok]}
                    break
            for tok in tokens:
                m = re.search(r'بوظيفة\s+([\u0600-\u06FF][\u0600-\u06FF\s\w]{1,40}?)(?=\s*[.\n،]|\s+وأن\b|\s+وقد\b|\s+ويحمل\b|$)', tok['text'])
                if m:
                    results['Job_Title'] = {'bbox': self._extract_sub_box(tok, m.group(1), is_rtl=True), 'value': m.group(1), 'tokens': [tok]}
                    break

        # Common fields (NID, Salary, Dates)
        results['HR_National_ID'] = self._find_nid_sequence(tokens)
        
        for tok in tokens:
            cleaned = ar2en(tok['text'])
            m = re.search(r'([\d,،\.]+)', cleaned)
            if m and any(kw in tok['text'].lower() for kw in ['راتب', 'صافي', 'salary', 'egp']):
                val = m.group(1).strip(',.،')
                if len(re.sub(r'[^\d]', '', val)) >= 3:
                    results['Salary_Amount'] = {'bbox': self._extract_sub_box(tok, val, is_rtl=not is_english), 'value': val, 'tokens': [tok]}
                    break
        
        for tok in tokens:
            cleaned = ar2en(tok['text'])
            m = re.search(r'(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})', cleaned)
            if m:
                results['Issue_Date'] = {'bbox': self._extract_sub_box(tok, m.group(1), is_rtl=not is_english), 'value': m.group(1), 'tokens': [tok]}
                break

        # Signature_Stamp
        sig_box = self._detect_signature_stamp(img)
        if sig_box:
            results['Signature_Stamp'] = {'bbox': sig_box, 'value': 'detected', 'tokens': []}

        return {'doc_type': 'hr_letter', 'image_size': [w, h], 'fields': results, 'all_tokens': tokens}
