# build_dataset.py
"""Build JSONL training/eval/test files from real extraction results.

- Reads ground-truth labels from ``bulk_results_50/`` JSON files.
- Matches each JSON to its source image in ``data/``.
- Runs EasyOCR (languages: ['ar', 'en']) on GPU for each image.
- Generates two preprocessing variants (original + CLAHE) and unions OCR tokens.
- Normalises Arabic-Indic digits to Western digits.
- Writes JSONL with: ``{doc_type, image_path, ocr_text, system, question, answer}``
- Splits 70/15/15 into ``train.jsonl``, ``eval.jsonl``, ``test.jsonl``.
- Skips samples with empty detections or missing images.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import easyocr

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path("d:/projects/loan")
BULK_RESULTS_DIR = BASE_DIR / "bulk_results_50"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = Path("d:/projects/loan/nbe_finetune") / "dataset"

SEED = 42
TRAIN_RATIO = 0.70
EVAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remainder)

# Fields to exclude from ground truth (non-text detections)
EXCLUDE_FIELDS = {"Signature_Stamp"}

# Minimum confidence threshold — skip fields below this
MIN_CONFIDENCE = 0.1

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_digits(text: str) -> str:
    return text.translate(DIGIT_MAP)


def preprocess_variants(image_path: Path) -> List[np.ndarray]:
    """Return two image variants: original and CLAHE-enhanced."""
    img = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image {image_path}")
    variants = [img]
    # CLAHE (on LAB Y channel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    variants.append(cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR))
    return variants


def ocr_image(reader: easyocr.Reader, img: np.ndarray) -> List[str]:
    """Run EasyOCR on a BGR image and return token strings."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = reader.readtext(rgb, detail=0, paragraph=False)
    return [normalize_digits(tok.strip()) for tok in result if tok.strip()]


def union_ocr_tokens(token_lists: List[List[str]]) -> List[str]:
    """Union tokens from multiple variants preserving order."""
    seen = set()
    ordered = []
    for lst in token_lists:
        for tok in lst:
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)
    return ordered


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "أنت مساعد لاستخراج المعلومات المنظمة من نصوص OCR للوثائق المصرية. "
    "أجب فقط باستخدام كائن JSON صالح يطابق المخطط المطلوب. "
    "لا تضف أي تفسيرات أو نصوص ماركداون أو أحرف إضافية."
)

QUESTIONS = {
    "id_front": (
        "من واقع نص OCR للوجه الأمامي لبطاقة الرقم القومي المصرية، "
        "استخرج الحقول التالية: Full_Name, Address, National_ID_Front, Birthdate. "
        "أرجع كائن JSON يحتوي على هذه المفاتيح بدقة."
    ),
    "id_back": (
        "من واقع نص OCR للوجه الخلفي لبطاقة الرقم القومي المصرية، "
        "استخرج: National_ID_Back, Expiry_Date, Gender, Marital_Status, Religion, Job_Profession. "
        "أرجع كائن JSON يحتوي على هذه المفاتيح بدقة."
    ),
    "hr_letter": (
        "من واقع نص OCR لخطاب مفردات مرتب، "
        "استخرج: Company_Name, Hire_Date, Employee_Name, Job_Title, HR_National_ID, Issue_Date, Salary_Amount. "
        "أرجع كائن JSON يحتوي على هذه المفاتيح بدقة."
    ),
}


# ---------------------------------------------------------------------------
# Parsing bulk_results_50
# ---------------------------------------------------------------------------
def parse_bulk_results() -> List[Dict]:
    """Scan bulk_results_50 and extract labeled samples."""
    records = []
    json_files = sorted(BULK_RESULTS_DIR.glob("*.json"))

    for jf in json_files:
        name = jf.name
        # Skip summary file
        if name == "bulk_test_summary.json":
            continue

        # Determine doc_type from suffix
        if name.endswith("_id_front.json"):
            doc_type = "id_front"
            image_base = name[: -len("_id_front.json")]
        elif name.endswith("_id_back.json"):
            doc_type = "id_back"
            image_base = name[: -len("_id_back.json")]
        elif name.endswith("_hr_letter.json"):
            doc_type = "hr_letter"
            image_base = name[: -len("_hr_letter.json")]
        else:
            continue

        # Parse JSON
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse {jf.name}: {e}")
            continue

        detections = data.get("detections", {})
        if not detections:
            continue  # Skip empty detections

        # Build answer dict from detections (exclude non-text fields)
        answer = {}
        for field_name, field_data in detections.items():
            if field_name in EXCLUDE_FIELDS:
                continue
            value = field_data.get("value", "")
            confidence = field_data.get("confidence", 0.0)
            if value and confidence >= MIN_CONFIDENCE:
                answer[field_name] = normalize_digits(str(value).strip())

        if not answer:
            continue  # No valid fields after filtering

        # Find source image
        image_path = find_source_image(doc_type, image_base)
        if image_path is None:
            logger.warning(f"No source image found for {doc_type}/{image_base}")
            continue

        records.append({
            "doc_type": doc_type,
            "image_path": str(image_path),
            "image_base": image_base,
            "answer": answer,
        })

    return records


def find_source_image(doc_type: str, image_base: str) -> Optional[Path]:
    """Find the source image for a given doc_type and image base name."""
    if doc_type == "hr_letter":
        # HR images are in data/hr_letter/ as .png
        candidates = [
            DATA_DIR / "hr_letter" / f"{image_base}.png",
            DATA_DIR / "hr_letter" / f"{image_base}.jpg",
        ]
    elif doc_type == "id_front":
        candidates = [
            DATA_DIR / "id_front" / "images" / f"{image_base}.jpg",
            DATA_DIR / "id_front" / "images" / f"{image_base}.png",
        ]
    elif doc_type == "id_back":
        candidates = [
            DATA_DIR / "id_back" / "images" / f"{image_base}.jpg",
            DATA_DIR / "id_back" / "images" / f"{image_base}.png",
        ]
    else:
        return None

    for c in candidates:
        if c.is_file():
            return c
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse all labeled results
    logger.info("Scanning bulk_results_50 for labeled samples...")
    records = parse_bulk_results()
    logger.info(f"Found {len(records)} samples with non-empty detections")

    # Count per doc_type
    counts = {}
    for r in records:
        counts[r["doc_type"]] = counts.get(r["doc_type"], 0) + 1
    for dt, cnt in sorted(counts.items()):
        logger.info(f"  {dt}: {cnt} samples")

    if not records:
        logger.error("No valid samples found! Check paths and data.")
        return

    # Step 2: Initialize EasyOCR
    logger.info("Initializing EasyOCR reader (ar + en, GPU)...")
    reader = easyocr.Reader(["ar", "en"], gpu=True)

    # Step 3: Run OCR on each image and build JSONL records
    jsonl_records = []
    skipped = 0

    for i, rec in enumerate(records):
        img_path = Path(rec["image_path"])
        doc_type = rec["doc_type"]

        try:
            variants = preprocess_variants(img_path)
            all_tokens = [ocr_image(reader, var) for var in variants]
            unioned = union_ocr_tokens(all_tokens)
            ocr_text = " ".join(unioned)

            if not ocr_text.strip():
                logger.warning(f"[{i+1}/{len(records)}] Empty OCR for {img_path.name}, skipping")
                skipped += 1
                continue

            answer_json = json.dumps(rec["answer"], ensure_ascii=False)

            jsonl_record = {
                "doc_type": doc_type,
                "image_path": str(img_path),
                "ocr_text": ocr_text,
                "system": SYSTEM_PROMPT,
                "question": QUESTIONS[doc_type],
                "answer": answer_json,
            }
            jsonl_records.append(jsonl_record)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(records)} images...")

        except Exception as e:
            logger.exception(f"[{i+1}/{len(records)}] Error processing {img_path}: {e}")
            skipped += 1
            continue

    logger.info(f"Successfully processed {len(jsonl_records)} samples ({skipped} skipped)")

    # Step 4: Split train/eval/test
    random.shuffle(jsonl_records)
    n = len(jsonl_records)
    train_end = int(n * TRAIN_RATIO)
    eval_end = train_end + int(n * EVAL_RATIO)

    splits = {
        "train": jsonl_records[:train_end],
        "eval": jsonl_records[train_end:eval_end],
        "test": jsonl_records[eval_end:],
    }

    for split_name, split_records in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in split_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Count per doc_type in split
        split_counts = {}
        for r in split_records:
            split_counts[r["doc_type"]] = split_counts.get(r["doc_type"], 0) + 1
        logger.info(f"Wrote {len(split_records)} records to {out_path} — {split_counts}")

    logger.info("Dataset build complete!")


if __name__ == "__main__":
    main()
