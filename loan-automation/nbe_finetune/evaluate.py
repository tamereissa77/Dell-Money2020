import json
import re
import logging
import sys
from pathlib import Path
from difflib import SequenceMatcher

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

# --- CONFIG ---
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR.parent / "data" / "vl_dataset_v2" / "val.jsonl"
BEST_DIR = BASE_DIR / "output" / "checkpoints" / "best"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- UTILS ---
def normalize_arabic(text: str) -> str:
    if not isinstance(text, str): return str(text)
    # ا/أ/إ/آ -> ا
    text = re.sub(r'[أإآ]', 'ا', text)
    # ى -> ي
    text = text.replace('ى', 'ي')
    # ة -> ه
    text = text.replace('ة', 'ه')
    # Arabic-Indic -> Western
    trans = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    text = text.translate(trans)
    return text.strip()

def fuzzy_match(a: str, b: str, threshold=0.85) -> bool:
    a = normalize_arabic(a).lower()
    b = normalize_arabic(b).lower()
    return SequenceMatcher(None, a, b).ratio() >= threshold

def strict_match(a: str, b: str) -> bool:
    return normalize_arabic(str(a)) == normalize_arabic(str(b))

def numeric_match(a, b, tolerance=1.0) -> bool:
    try:
        return abs(float(a) - float(b)) <= tolerance
    except:
        return False

def bool_match(a, b) -> bool:
    def to_bool(val):
        if isinstance(val, bool): return val
        if str(val).lower() in ['true', '1', 'yes']: return True
        return False
    return to_bool(a) == to_bool(b)

def parse_json_robustly(text: str) -> dict:
    # Remove markdown fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'\s*```', '', text)
    # Extract outermost { ... }
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return {}

# --- EVALUATION ---
def evaluate():
    if not BEST_DIR.exists():
        logger.error(f"Best adapter not found at {BEST_DIR}")
        return

    logger.info("Loading model and adapter...")
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

    eval_data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line))

    results = []
    field_stats = {}

    logger.info(f"Starting evaluation on {len(eval_data)} samples...")

    for item in tqdm(eval_data):
        img_path = BASE_DIR.parent / item["image"]
        img_path_str = f"file://{img_path}"
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": item["system"]}]},
            {"role": "user", "content": [
                {"type": "image", "image": img_path_str, "max_pixels": 800000},
                {"type": "text", "text": f"{item['question']}\n{item['ocr_text']}"}
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
        
        # Decode only the new tokens
        response = processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_json = parse_json_robustly(response)
        true_json = json.loads(item["answer"])

        sample_results = {"correct": 0, "total": 0}
        
        # Rules mapping — updated for vl_dataset_v2 schema
        strict_fields = [
            "national_id", "date_of_birth", "expiry_date", "issue_date", "hire_date",
            "gender", "marital_status", "religion"
        ]
        fuzzy_fields = [
            "name", "address", "profession", "company_name", "employee_name", "job_title"
        ]
        numeric_fields = ["salary", "monthly_salary"]
        bool_fields = ["signatured", "has_signature", "has_stamp"]

        all_keys = set(true_json.keys())
        for key in all_keys:
            if key not in field_stats:
                field_stats[key] = {"correct": 0, "total": 0}
            
            field_stats[key]["total"] += 1
            val_true_obj = true_json.get(key)
            val_pred_obj = pred_json.get(key)
            
            # Extract just the 'value', handling nested dicts from v2 dataset
            val_true = val_true_obj.get("value") if isinstance(val_true_obj, dict) else val_true_obj
            val_pred = val_pred_obj.get("value") if isinstance(val_pred_obj, dict) else val_pred_obj
            
            is_correct = False
            if val_pred is not None:
                if key in strict_fields:
                    is_correct = strict_match(val_true, val_pred)
                elif key in fuzzy_fields:
                    is_correct = fuzzy_match(val_true, val_pred)
                elif key in numeric_fields:
                    is_correct = numeric_match(val_true, val_pred)
                elif key in bool_fields:
                    is_correct = bool_match(val_true, val_pred)
                else:
                    is_correct = strict_match(val_true, val_pred)
            
            if is_correct:
                field_stats[key]["correct"] += 1

    # Report
    logger.info("\n" + "="*50)
    logger.info(f"{'Field':<25} | {'Accuracy %':<10} | {'N':<5} | Status")
    logger.info("-" * 55)
    
    failed_fields = []
    for field, stats in sorted(field_stats.items()):
        acc = (stats["correct"] / stats["total"]) * 100
        status = "✅" if acc >= 90 else ("⚠️" if acc >= 70 else "❌")
        logger.info(f"{field:<25} | {acc:>10.1f} | {stats['total']:>5} | {status}")
        if acc < 90:
            failed_fields.append((field, acc))

    logger.info("="*50)
    if failed_fields:
        logger.info("\nDECISION BLOCK: RECOMMENDED REMEDIATION")
        for field, acc in failed_fields:
            if acc < 70:
                logger.info(f"- {field}: CRITICAL FAILURE. Add 50+ diverse samples and reduce LR.")
            else:
                logger.info(f"- {field}: WEAK PERFORMANCE. Add 20+ targeted samples or increase LoRA rank.")
    else:
        logger.info("\nDECISION BLOCK: ALL FIELDS PASSED (≥90%). Model ready for production.")

if __name__ == "__main__":
    evaluate()
