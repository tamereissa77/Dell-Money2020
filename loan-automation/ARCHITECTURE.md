# Egyptian Document Extraction — Qwen2.5-VL-7B Fine-Tuning Plan

> Owner: AI engineering. Status: design v1. Target model: `Qwen/Qwen2.5-VL-7B-Instruct`. Target hardware: single RTX 5090 32 GB (Blackwell, sm_120, CUDA 12.8+).

This document is the source of truth for fine-tuning Qwen2.5-VL on three Egyptian document types: National ID (front), National ID (back), and HR / salary-letter. It supersedes the design baked into `nbe_finetune/train.py`.

---

## 1. Goals & non-goals

**In scope.** Per-document structured field extraction matching the user-provided JSON schema, plus per-field bounding boxes (element extraction) in pixel coordinates. One model handles all three document types via prompt routing.

**Out of scope.** Document classification (we assume the caller already knows the doc type — there's a separate classifier upstream in `layout_analyzer/`). Multi-page documents. Live capture / video.

**Definition of done.** On a held-out test set of ≥150 images (50/doc type):

| Metric | Target |
|---|---|
| Field-level exact match (numeric / date / id) | ≥ 97 % |
| Field-level normalized fuzzy match (names, address, profession) | ≥ 93 % |
| `signatured` boolean accuracy | ≥ 98 % |
| bbox IoU ≥ 0.5 hit-rate, per field | ≥ 90 % |
| Document-level "all fields correct" rate | ≥ 85 % |

---

## 2. Why this is a from-scratch redesign (audit of existing pipeline)

The existing pipeline under `nbe_finetune/` is built on three mistakes that block your stated goals. They are listed here so the rationale for the new code is explicit.

**(a) Vision tower is unused.** `train.py` loads `Qwen2_5_VLForConditionalGeneration` but the dataset only tokenizes text (`prompt + ocr_text + answer`). The `visual` submodule is explicitly frozen and never receives pixels. Fine-tuning is happening in language-model mode only. This caps Arabic OCR accuracy at EasyOCR's level, which on the sample I inspected produced `"50 ذلزم ذافر . متزوجة مسلفةً فضل مدمد هسين هباش"` — unrecoverable noise. A Qwen2.5-VL fine-tune that actually consumes the image will beat this with no OCR step in the loop.

**(b) Element extraction is structurally impossible in the current setup.** Bounding boxes can't be supervised when the model never sees the image — there's no coordinate frame. The fix is to feed the image through `Qwen2VLProcessor` and have the model emit `bbox_2d: [x1, y1, x2, y2]` per field in **absolute pixel coordinates of the processed image**. (Qwen2.5-VL is trained to emit bbox_2d natively; no special tokens.)

**(c) Labels are silently truncated.** `build_dataset.py` drops every field whose layout-analyzer confidence is below `MIN_CONFIDENCE = 0.1`, which means the model is being trained on samples like `{"Marital_Status": "متزوج", "Religion": "مسلم"}` when the source ID actually has six fields. The model learns to omit fields it should be reading. Replace with: keep all fields, mark unreadable ones explicitly as `null`, and gate the sample on **human review**, not OCR confidence.

**(d) Schema drift.** Existing schema has `Religion` on id_back (not requested) and lacks `signatured` on hr_letter (requested). Schema is canonicalized below in §4.

The new pipeline lives at the repo root under `src/`, `configs/`, and `scripts/`. `nbe_finetune/` is preserved but deprecated — keep it for reference for one or two iterations, then delete.

---

## 3. System architecture

```
                  ┌──────────────────────┐
   raw images ──► │  bulk_generator.py   │ ──► bulk_<n>/<id>_<type>.json   (predictions for QA)
                  │  (LayoutAnalyzer)    │     bulk_<n>/<id>_<type>_ann.png (overlay)
                  └──────────────────────┘
                            │
                            ▼ human reviewer corrects each JSON in Label Studio
                            │
                  ┌──────────────────────┐
                  │  verified labels/    │ ──► canonical ground truth (schema in configs/schema.py)
                  └──────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  build_vl_dataset.py │ ──► dataset/{train,val,test}.jsonl  (Qwen2.5-VL convo format)
                  └──────────────────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │  train_qwen_vl.py    │ ──► output/lora/best/  (PEFT adapter)
                  │  (QLoRA + image)     │
                  └──────────────────────┘
                            │
              ┌─────────────┴────────────┐
              ▼                          ▼
      ┌───────────────┐         ┌─────────────────┐
      │ eval_vl.py    │         │ extract.py      │
      │ field+bbox    │         │ (vLLM serve)    │
      │ metrics       │         └─────────────────┘
      └───────────────┘
```

The key swap vs. today: the model receives **the image plus a short instruction**, not the image plus an OCR transcription. The OCR pipeline becomes optional and is only used for pre-labeling (§5).

---

## 4. Canonical schema (single source of truth)

Defined in code at `configs/schema.py`. Mirrors the user-provided schema exactly, plus a `bbox_2d` per field for element extraction. Pixel coordinates are in the **processed image space** that Qwen2.5-VL sees (smart-resize to a 28-multiple, max ~1280px on the long side).

```python
# Top-level discriminator: doc_type ∈ {"id_front", "id_back", "hr_letter"}

ID_FRONT = {
    "name":          {"value": str | None, "bbox_2d": [x1, y1, x2, y2] | None},
    "address":       {"value": str | None, "bbox_2d": [...] | None},
    "national_id":   {"value": str | None, "bbox_2d": [...] | None},  # 14 digits
    "date_of_birth": {"value": str | None, "bbox_2d": [...] | None},  # YYYY-MM-DD
}

ID_BACK = {
    "national_id":    {"value": str | None, "bbox_2d": [...] | None},
    "expiry_date":    {"value": str | None, "bbox_2d": [...] | None},  # YYYY-MM-DD
    "profession":     {"value": str | None, "bbox_2d": [...] | None},
    "gender":         {"value": str | None, "bbox_2d": [...] | None},  # "M" | "F"
    "marital_status": {"value": str | None, "bbox_2d": [...] | None},  # normalized vocab
}

HR_LETTER = {
    "name":         {"value": str | None, "bbox_2d": [...] | None},
    "national_id":  {"value": str | None, "bbox_2d": [...] | None},
    "company_name": {"value": str | None, "bbox_2d": [...] | None},
    "hire_date":    {"value": str | None, "bbox_2d": [...] | None},
    "issue_date":   {"value": str | None, "bbox_2d": [...] | None},
    "salary":       {"value": str | None, "bbox_2d": [...] | None},  # numeric string, no commas
    "job_title":    {"value": str | None, "bbox_2d": [...] | None},
    "signatured":   {"value": bool,       "bbox_2d": [...] | None},  # bbox is the signature region
}
```

**Normalization rules** (applied during label QA and during eval-time comparison; never visible to the model — the model is trained on already-normalized values):

- All Arabic-Indic digits → Western digits (`٠١٢٣٤٥٦٧٨٩` → `0123456789`).
- Dates: ISO `YYYY-MM-DD`. The line `هذه البطاقة سارية حتى 2030/01/15` → `"2030-01-15"`.
- `gender`: extract `ذكر`/`أنثى` and emit `"M"` / `"F"` to make downstream consumption stable.
- `marital_status`: closed vocab `أعزب | متزوج | مطلق | أرمل` (normalize `انسة → أعزب`, `متزوجة → متزوج` etc., as a feminine→masculine fold for storage).
- `salary`: digits only, e.g. `"7,500.00 EGP"` → `"7500"`. EGP is implied.
- Empty / unreadable field → `"value": null`, `"bbox_2d": null`. **Never an empty string.**

---

## 5. Data pipeline & annotation strategy

Three phases. Each one feeds the next.

### Phase 1 — Bootstrap (now, manual)

Take **50 raw images per doc type** (150 total), run `bulk_generator.py`, then review every JSON in Label Studio. Goal is a clean, high-quality seed set, not coverage.

Output: `data/labeled/bulk_50/`. This is your "bulk 50" deliverable.

### Phase 2 — Active-learning expansion (after first training run)

Run `bulk_generator.py` on the next 500 images per doc type → produce predictions JSON + annotated PNG → human reviewer fixes/confirms in Label Studio. With v0 model predictions as a starting point, throughput per reviewer goes from ~15/hour (cold) to ~80/hour (correction-only).

Output: `data/labeled/bulk_500/` (cumulative; the bulk_50 set is included).

### Phase 3 — Synthetic augmentation for HR letters only

HR letters have far more layout variance than the templated ID cards. Build a synthetic generator (`generator_service/` already exists — wire it into `data_prep/synthetic_hr.py`) that emits 2–3k HR letters with known field positions, varied templates, varied signature positions, varied stamp/no-stamp, real Arabic fonts (Cairo, Tajawal). Mix at **40 % synthetic / 60 % real** for HR. **Do not** synthesize ID cards — the layout is too rigid and the model will overfit to the synthesizer's quirks.

### Train / val / test split

Stratified by doc_type. **70/15/15**, seed=42. Critically: the test set must contain **only real-world images** (no synthetic). Hold these out from day one and never touch them during iteration.

---

## 6. Training configuration (RTX 5090 32 GB)

The 5090 is Blackwell (sm_120). PyTorch must be a CUDA 12.8+ build. Flash-Attention 2 doesn't yet have stable Blackwell wheels — start with `attn_implementation="sdpa"`, swap to FA2 once a wheel for sm_120 lands.

| Knob | Value | Why |
|---|---|---|
| Quantization | NF4 4-bit (bitsandbytes) | 7B + vision tower fits comfortably in 32 GB even with grad checkpointing on. |
| Vision tower | **Trainable** in phase 1 — LoRA on `merger.mlp` only. Full ViT updates only in phase 2 if needed. | Most of the Arabic-OCR-specific knowledge sits in the language head; ViT is general. Save VRAM. |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` on the LLM. Also `qkv, proj` on the visual merger. | Standard. |
| LoRA r / α | 64 / 128 | r=64 is the empirical sweet spot for 7B doc extraction; α=2r per Qwen recipe. |
| LoRA dropout | 0.05 | |
| use_rslora | True | Stabilizes large r. |
| Precision | bf16 (compute), nf4 (weights) | bf16 is native on Blackwell, no loss-scale headaches. |
| Effective batch | 16 | per_device=1, grad_accum=16. Per-device must be 1 because each sample includes one ~1280px image — VRAM for activations dominates. |
| Image resolution | `min_pixels=256*28*28`, `max_pixels=1280*28*28` (default) | Lets Qwen2.5-VL pick. ID cards land near 896², HR letters near 1280². |
| LR | 1 e-4 (LoRA), cosine, 3 % warmup | |
| Epochs | 5 with early stopping (patience=4 evals) | Past 3 epochs you start memorizing. |
| Eval every | 50 optimizer steps | |
| Seq len | 4096 | Comfortably fits an HR letter prompt + JSON answer (~1.5k tokens) with margin. |
| Gradient checkpointing | On | Trades ~25 % step time for ~30 % VRAM. |
| Loss masking | Mask prompt tokens (`-100`), mask vision tokens, supervise only assistant JSON. | Done correctly by the data collator in `train_qwen_vl.py`. |

Expected step time on a 5090: ~3.5 s / optimizer step at effective batch 16. For 1500 samples × 5 epochs × (16 microbatches / batch) ≈ 470 optimizer steps → **~30 minutes per epoch**, **~2.5 hours total** including eval. Well within reach of an overnight run plus iteration.

---

## 7. Prompts (frozen)

Prompts are defined in `configs/prompts.py` and **must not vary between training and inference**. Drift here is the single most common silent regression in VL fine-tuning. The eval harness asserts that the prompt used at eval time hashes to the same value as the prompt baked into the training JSONL.

System prompt is shared across doc types and tells the model to emit strict JSON with `bbox_2d` and refuse markdown. Each doc-type prompt names the exact keys to emit (using the schema names in §4) and shows one mini-example. Full text is in code.

---

## 8. Evaluation harness

`src/eval/eval_vl.py` runs greedy decoding on the held-out test set, then `src/eval/metrics.py` computes:

- **Per-field exact match** after applying the normalization in §4.
- **Per-field normalized edit distance** (1 − Levenshtein / max-len, Unicode NFC).
- **Per-field bbox IoU** vs. ground truth, threshold 0.5.
- **Document-level all-fields-correct** rate (binary).
- **Calibration of `null`**: false-positive rate (model emitted a value when the truth was null), false-negative rate (model emitted null when the truth had a value).

Report is grouped by doc_type and emits a markdown table and a JSON file. The harness also writes the 20 worst predictions per doc type to `output/eval/failures/` as image + side-by-side JSON for error analysis.

---

## 9. Inference / deployment

Two paths:

- **vLLM**: production. `vllm serve Qwen2.5-VL-7B-Instruct --enable-lora --lora-modules ours=output/lora/best`. Throughput on a single 5090 should hit ~10–15 docs/s at INT8.
- **Transformers**: dev / debugging. `src/inference/extract.py` is a thin CLI that takes an image + doc_type and prints the parsed JSON.

The existing `extractor_service/` FastAPI service should be repointed at the new adapter once benchmarks pass.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Blackwell ecosystem still maturing — FA2 / bnb wheels lag. | Start with `sdpa` + bnb 0.43+. Have a fallback path to running training on a cloud A100 if local wheels are broken. |
| Small training set (≤150 in phase 1) → overfitting. | Heavy augmentation: random rotation ±5°, hue ±10, JPEG quality 60–95, perspective warp ±5 %. Early-stop on val loss. Phase 1 model is a **labeling assistant**, not the production model — accept the noise. |
| Arabic handwritten signatures vary wildly. | `signatured` field is binary; train a small auxiliary head later if accuracy < 95 %. For now, the bbox supervision on the signature region gives the model a strong attention prior. |
| Date format ambiguity (`15/01/2030` vs `01/15/2030`). | Normalize to ISO during labeling; require the model to emit ISO. Egyptian IDs use DD/MM/YYYY — encode that as a labeling rule. |
| Label leakage across train/test. | Hash each source image's filename into the split. Same physical image can never appear in two splits even after augmentation. |
| `bbox_2d` coordinate frame confusion. | Always store bboxes in the **post-resize Qwen image space**, not the raw image space. Both `build_vl_dataset.py` and `eval_vl.py` apply the same `smart_resize` from `qwen_vl_utils`. |

---

## 11. File layout (what gets created)

```
loan/
├── ARCHITECTURE.md                  ← this file
├── requirements.txt                 ← pinned for Blackwell / CUDA 12.8
├── configs/
│   ├── schema.py                    ← canonical schema + normalization rules
│   ├── prompts.py                   ← frozen system / user prompts per doc_type
│   └── train_config.yaml            ← all training hyperparams
├── src/
│   ├── annotation/
│   │   └── bulk_generator.py        ← upgrade of bulk_test_50.py: produces image + JSON + ann.png
│   ├── data_prep/
│   │   ├── build_vl_dataset.py      ← labels JSON → Qwen2.5-VL JSONL (replaces nbe_finetune/build_dataset.py)
│   │   └── split.py                 ← stratified train/val/test split
│   ├── training/
│   │   └── train_qwen_vl.py         ← QLoRA training, images consumed via processor
│   ├── eval/
│   │   ├── eval_vl.py               ← end-to-end eval harness
│   │   └── metrics.py               ← field + bbox metrics
│   └── inference/
│       └── extract.py               ← CLI inference
└── scripts/
    └── setup.ps1                    ← env bootstrap for Windows + Blackwell
```

Existing `nbe_finetune/`, `bulk_test_50.py`, `bulk_results_50/` are left in place for reference but are no longer the active pipeline.

---

## 12. Execution order

1. `pip install -r requirements.txt` (or run `scripts/setup.ps1`).
2. `python -m src.annotation.bulk_generator --doc-type id_front --limit 50 --out data/labeled/bulk_50` — produce predicted labels for first 50 of each doc type. Repeat for `id_back`, `hr_letter`.
3. **Human review** every JSON in `data/labeled/bulk_50/`. Fix values, fix or add bboxes, set `signatured`. This is the gating step — quality of the trained model is bounded by quality here.
4. `python -m src.data_prep.build_vl_dataset --in data/labeled/bulk_50 --out dataset/v1` — produces `train.jsonl`, `val.jsonl`, `test.jsonl`.
5. `python -m src.training.train_qwen_vl --config configs/train_config.yaml --dataset dataset/v1` — train v1.
6. `python -m src.eval.eval_vl --adapter output/lora/best --dataset dataset/v1/test.jsonl` — measure.
7. Iterate: bulk_500 → v2 → bulk_2000 → v3.
