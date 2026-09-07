"""Visual test: run inference on test images and draw extracted fields + bounding boxes.

Uses the best checkpoint from training to run on test.jsonl samples.
Outputs annotated images to nbe_finetune/output/visual_test/
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
BEST_DIR = Path("d:/projects/loan/nbe_finetune/output/checkpoints/best")
TEST_JSONL = Path("d:/projects/loan/data/vl_dataset_v2/val.jsonl")
BULK_DIR = Path("d:/projects/loan/data/labeled")
OUTPUT_DIR = Path("d:/projects/loan/nbe_finetune/output/visual_test")

# Color palette for bounding boxes (BGR)
COLORS = {
    "Full_Name": (0, 255, 0),       # Green
    "Address": (255, 165, 0),        # Orange
    "National_ID_Front": (255, 0, 0), # Blue
    "Birthdate": (0, 255, 255),       # Yellow
    "National_ID_Back": (255, 0, 0),  # Blue
    "Expiry_Date": (0, 0, 255),       # Red
    "Gender": (255, 0, 255),          # Magenta
    "Marital_Status": (128, 0, 128),  # Purple
    "Religion": (0, 128, 255),        # Light blue
    "Job_Profession": (128, 128, 0),  # Teal
    "Company_Name": (0, 200, 0),      # Dark Green
    "Employee_Name": (0, 255, 0),     # Green
    "Job_Title": (255, 165, 0),       # Orange
    "HR_National_ID": (255, 0, 0),    # Blue
    "Issue_Date": (0, 0, 255),        # Red
    "Hire_Date": (0, 200, 200),       # Gold
    "Salary_Amount": (0, 255, 255),   # Yellow
}
DEFAULT_COLOR = (200, 200, 200)


def load_model():
    """Load base model + LoRA adapter from best checkpoint."""
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("[INFO] Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Try loading LoRA adapter
    adapter_config = BEST_DIR / "adapter_config.json"
    if adapter_config.exists():
        print(f"[INFO] Loading LoRA adapter from {BEST_DIR}...")
        model = PeftModel.from_pretrained(model, str(BEST_DIR))
        model = model.merge_and_unload()
    else:
        print(f"[WARN] No adapter found at {BEST_DIR}, using base model")

    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, system_prompt, question, ocr_text):
    """Run the model on a single sample and return the predicted JSON string."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question}\n\nOCR Text:\n{ocr_text}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
        )

    # Decode only generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return result


def find_bulk_result(image_path, doc_type):
    """Find the corresponding bulk_results_50 JSON for this image."""
    img_name = Path(image_path).stem
    # Try various naming patterns
    candidates = [
        BULK_DIR / f"{img_name}_{doc_type}.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def draw_annotations(image_path, pred_json, true_json, bulk_data, doc_type, sample_idx):
    """Draw bounding boxes and field values on the image."""
    img = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Cannot read image: {image_path}")
        return None

    h, w = img.shape[:2]

    # Create side-by-side: original | annotated
    annotated = img.copy()

    # Draw bounding boxes from bulk_results_50 (ground truth boxes)
    if bulk_data and "detections" in bulk_data:
        for field_name, field_data in bulk_data["detections"].items():
            if field_name == "Signature_Stamp":
                continue
            bbox = field_data.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                color = COLORS.get(field_name, DEFAULT_COLOR)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label
                label = f"{field_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Create comparison panel
    panel_w = max(600, w)
    panel_h = 40 + len(pred_json) * 28 + len(true_json) * 28 + 80
    panel = np.zeros((max(h, panel_h), panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark background

    y_offset = 30
    cv2.putText(panel, f"Doc Type: {doc_type} | Sample #{sample_idx}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_offset += 35

    cv2.putText(panel, "PREDICTED:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 25

    for key, val in pred_json.items():
        color = COLORS.get(key, DEFAULT_COLOR)
        text = f"  {key}: {val}"
        cv2.putText(panel, text[:80], (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 22

    y_offset += 15
    cv2.putText(panel, "GROUND TRUTH:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    y_offset += 25

    for key, val in true_json.items():
        color = COLORS.get(key, DEFAULT_COLOR)
        # Check match
        pred_val = pred_json.get(key, "")
        match = "OK" if str(pred_val).strip() == str(val).strip() else "MISS"
        match_color = (0, 255, 0) if match == "OK" else (0, 0, 255)
        text = f"  [{match}] {key}: {val}"
        cv2.putText(panel, text[:80], (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, match_color, 1)
        y_offset += 22

    # Resize panel to match annotated height
    if panel.shape[0] != annotated.shape[0]:
        target_h = max(panel.shape[0], annotated.shape[0])
        if annotated.shape[0] < target_h:
            pad = np.zeros((target_h - annotated.shape[0], annotated.shape[1], 3), dtype=np.uint8)
            annotated = np.vstack([annotated, pad])
        if panel.shape[0] < target_h:
            pad = np.zeros((target_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])

    combined = np.hstack([annotated, panel])
    return combined


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("[INFO] Loading test samples...")
    test_samples = []
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))

    print(f"[INFO] Found {len(test_samples)} test samples")

    # Limit to first 10 for quick testing
    MAX_SAMPLES = 10
    if len(test_samples) > MAX_SAMPLES:
        import random
        random.seed(42)
        test_samples = random.sample(test_samples, MAX_SAMPLES)
        print(f"[INFO] Testing on {MAX_SAMPLES} random samples")

    # Load model
    model, tokenizer = load_model()

    # Process each test sample
    results = []
    for i, sample in enumerate(test_samples):
        doc_type = sample.get("doc_type", "unknown")
        ocr_text = sample.get("ocr_text", "")
        true_json = json.loads(sample["answer"])

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(test_samples)}] Document Type: {doc_type}")
        print(f"{'='*60}")

        # Run inference
        pred_str = run_inference(model, tokenizer, sample["system"], sample["question"], ocr_text)

        # Parse prediction
        try:
            pred_str_clean = pred_str.strip()
            if pred_str_clean.startswith("```"):
                pred_str_clean = pred_str_clean.split("```")[1]
                if pred_str_clean.startswith("json"):
                    pred_str_clean = pred_str_clean[4:]
            pred_json = json.loads(pred_str_clean)
        except json.JSONDecodeError:
            try:
                import re
                match = re.search(r'\{[^{}]*\}', pred_str, re.DOTALL)
                if match:
                    pred_json = json.loads(match.group())
                else:
                    pred_json = {"_error": "Failed to parse model output"}
            except:
                pred_json = {"_error": pred_str[:200]}

        # Compare fields
        print(f"\n  {'Field':<25} | {'Predicted':<30} | {'Truth':<30} | Match")
        print(f"  {'-'*25}-+-{'-'*30}-+-{'-'*30}-+------")

        correct = 0
        total = len(true_json)
        for key in true_json:
            val_true = str(true_json[key]).strip()
            val_pred = str(pred_json.get(key, "MISSING")).strip()
            match = val_pred == val_true
            if match:
                correct += 1
            icon = "✅" if match else "❌"
            print(f"  {key:<25} | {val_pred[:30]:<30} | {val_true[:30]:<30} | {icon}")

        acc = 100 * correct / max(total, 1)
        print(f"\n  Accuracy: {correct}/{total} ({acc:.0f}%)")

        results.append({"doc_type": doc_type, "correct": correct, "total": total})

    # Summary
    print(f"\n{'='*60}")
    print("VISUAL TEST SUMMARY")
    print(f"{'='*60}")
    total_correct = sum(r["correct"] for r in results)
    total_fields = sum(r["total"] for r in results)
    print(f"Overall: {total_correct}/{total_fields} fields correct ({100*total_correct/max(total_fields,1):.1f}%)")

    by_type = {}
    for r in results:
        dt = r["doc_type"]
        if dt not in by_type:
            by_type[dt] = {"correct": 0, "total": 0}
        by_type[dt]["correct"] += r["correct"]
        by_type[dt]["total"] += r["total"]

    for dt, stats in sorted(by_type.items()):
        pct = 100 * stats["correct"] / max(stats["total"], 1)
        print(f"  {dt}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

