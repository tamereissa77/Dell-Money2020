import sys
import json
import logging
import argparse
from pathlib import Path

import torch
import cv2
import numpy as np
import easyocr
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# --- CONFIG ---
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
BASE_DIR = Path(__file__).parent
BEST_DIR = BASE_DIR / "output" / "checkpoints" / "best"

logging.basicConfig(level=logging.ERROR) # Only errors to stderr, output to stdout

# --- UTILS ---
def normalize_digits(text: str) -> str:
    trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    return text.translate(trans)

def get_ocr_text(image_path: Path):
    reader = easyocr.Reader(['ar', 'en'], gpu=True)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Simple single-pass OCR for production inference
    results = reader.readtext(img, detail=0)
    ocr_text = " ".join([normalize_digits(t.strip()) for t in results if t.strip()])
    return ocr_text

def parse_json_robustly(text: str) -> dict:
    import re
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'\s*```', '', text)
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return {}

def main():
    parser = argparse.ArgumentParser(description="Extract JSON from Egyptian Document")
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("doc_type", type=str, choices=["id_front", "id_back", "hr_letter"], help="Type of document")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}", file=sys.stderr)
        sys.exit(1)

    if not BEST_DIR.exists():
        print(f"Error: Fine-tuned adapter not found at {BEST_DIR}. Run training first.", file=sys.stderr)
        sys.exit(1)

    try:
        # 1. OCR
        ocr_text = get_ocr_text(img_path)
        if not ocr_text:
            print("Error: No OCR text extracted from image.", file=sys.stderr)
            sys.exit(1)

        # 2. Model Inference
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, BEST_DIR)
        model.eval()

        # Build prompt
        system = "أنت مساعد لاستخراج المعلومات المنظمة من نصوص OCR للوثائق المصرية. أجب فقط باستخدام كائن JSON صالح يطابق المخطط المطلوب. لا تضف أي تفسيرات أو نصوص ماركداون أو أحرف إضافية."
        questions = {
            "id_front": "من واقع نص OCR للوجه الأمامي لبطاقة الرقم القومي المصرية، استخرج الحقول التالية: full_name_ar, address, governorate, national_id, birth_date, document_number. أرجع كائن JSON يحتوي على هذه المفاتيح بدقة.",
            "id_back": "من واقع نص OCR للوجه الخلفي لبطاقة الرقم القومي المصرية، استخرج: national_id, profession (job_title_ar), gender, marital_status, expiry_date, birth_date. أرجع كائن JSON يحتوي على هذه المفاتيح بدقة.",
            "hr_letter": "من واقع نص OCR لخطاب مفردات مرتب (بالعربية والإنجليزية)، استخرج: employee_name_ar, employee_name_en, national_id, monthly_salary, job_title, company_name, issue_date, hire_date, has_signature, has_stamp. أرجع كائن JSON يحتوي على هذه المفاتيح بدقة."
        }
        
        img_path_str = f"file://{img_path}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "image", "image": img_path_str, "max_pixels": 800000},
                {"type": "text", "text": f"{questions[args.doc_type]}\n{ocr_text}"}
            ]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([messages])
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=400,
                repetition_penalty=1.0,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        result_json = parse_json_robustly(response)
        
        if result_json:
            print(json.dumps(result_json, indent=2, ensure_ascii=False))
            sys.exit(0)
        else:
            print("Error: Model output could not be parsed as JSON.", file=sys.stderr)
            print(f"Raw output: {response}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
