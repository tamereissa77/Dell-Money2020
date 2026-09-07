"""
model_handler.py
================
Handles loading and inference of the Chandra VLM (Vision Language Model)
for Arabic financial document extraction.

Chandra is built on Qwen3-VL and is loaded via the `chandra-ocr` package.
Supports two modes:
  - OCR mode: faithful HTML reproduction of tables/content
  - Layout mode: full-page OCR with labeled bounding-box blocks
"""

import time

import logging
from typing import List, Dict, Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "datalab-to/chandra"
BBOX_SCALE = 1024  # Chandra normalizes bboxes to 0-1024

# Prompts are loaded lazily from the chandra package at runtime.
_NATIVE_OCR_PROMPT = None
_NATIVE_LAYOUT_PROMPT = None


def _get_native_prompt() -> str:
    """Return Chandra's native OCR prompt (the one it was trained on)."""
    global _NATIVE_OCR_PROMPT
    if _NATIVE_OCR_PROMPT is None:
        try:
            from chandra.prompts import OCR_PROMPT
            _NATIVE_OCR_PROMPT = OCR_PROMPT
        except ImportError:
            _NATIVE_OCR_PROMPT = "OCR this image to HTML."
    return _NATIVE_OCR_PROMPT


def _get_layout_prompt() -> str:
    """Return Chandra's native layout-OCR prompt with bounding boxes."""
    global _NATIVE_LAYOUT_PROMPT
    if _NATIVE_LAYOUT_PROMPT is None:
        try:
            from chandra.prompts import OCR_LAYOUT_PROMPT
            _NATIVE_LAYOUT_PROMPT = OCR_LAYOUT_PROMPT
        except ImportError:
            _NATIVE_LAYOUT_PROMPT = "OCR this image to HTML arranged as layout blocks."
    return _NATIVE_LAYOUT_PROMPT


# Sentinel values for prompt selection
ARABIC_TABLE_PROMPT = "USE_NATIVE"
LAYOUT_PROMPT = "USE_LAYOUT"


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA not available - falling back to CPU. Inference will be slow.")
    return torch.device("cpu")


def load_model(model_checkpoint: str = MODEL_CHECKPOINT):
    """
    Load the Chandra VLM (Qwen3-VL based) model and processor.

    Returns
    -------
    tuple[model, processor, device, float, float]
    """
    start_time = time.perf_counter()
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    device = get_device()

    logger.info("Loading Qwen3VLProcessor from %s ...", model_checkpoint)
    processor = Qwen3VLProcessor.from_pretrained(model_checkpoint)

    logger.info("Loading Qwen3VLForConditionalGeneration from %s ...", model_checkpoint)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=dtype,
        device_map=str(device) if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model = model.eval()
    model.processor = processor

    load_time = time.perf_counter() - start_time
    model_size_bytes = model.get_memory_footprint()
    model_size_gb = model_size_bytes / (1024**3)

    logger.info(
        "Model loaded (dtype=%s, device=%s, time=%.2fs, size=%.2fGB)",
        dtype,
        device,
        load_time,
        model_size_gb,
    )
    return model, processor, device, load_time, model_size_gb



def _scale_to_fit(
    img: Image.Image,
    max_size=(3072, 2048),
    min_size=(28, 28),
) -> Image.Image:
    """Resize image to fit within Chandra's expected input range."""
    import math

    width, height = img.size
    if width == 0 or height == 0:
        return img

    max_w, max_h = max_size
    min_w, min_h = min_size
    current_pixels = width * height
    max_pixels = max_w * max_h
    min_pixels = min_w * min_h

    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        new_w = math.floor(width * scale)
        new_h = math.floor(height * scale)
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5
        new_w = math.ceil(width * scale)
        new_h = math.ceil(height * scale)
    else:
        return img

    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def _run_inference(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 8192,
) -> str:
    """
    Core inference: send image + prompt to the Chandra VLM and return
    the raw generated text.
    """
    from qwen_vl_utils import process_vision_info

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = _scale_to_fit(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return result.strip()


def extract_table_from_image(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    prompt: str = "USE_NATIVE",
    max_new_tokens: int = 8192,
) -> str:
    """
    Run Chandra OCR on a single image and return raw HTML.
    If prompt is "USE_NATIVE", uses Chandra's trained OCR prompt.
    If prompt is "USE_LAYOUT", uses Chandra's layout prompt.
    """
    try:
        if prompt == "USE_NATIVE":
            prompt = _get_native_prompt()
        elif prompt == "USE_LAYOUT":
            prompt = _get_layout_prompt()

        return _run_inference(image, model, processor, device, prompt, max_new_tokens)

    except Exception as exc:
        logger.exception("VLM inference failed: %s", exc)
        return f"[ERROR] VLM inference failed: {exc}"


def extract_page_layout(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    max_new_tokens: int = 12384,
) -> str:
    """
    Run Chandra's layout-OCR on a full page image.
    Returns raw HTML with <div data-bbox="..." data-label="..."> blocks.
    """
    try:
        prompt = _get_layout_prompt()
        return _run_inference(image, model, processor, device, prompt, max_new_tokens)
    except Exception as exc:
        logger.exception("Layout extraction failed: %s", exc)
        return f"[ERROR] Layout extraction failed: {exc}"


def parse_layout_blocks(raw_html: str, page_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Parse Chandra's layout HTML output into a list of structured blocks.

    Each block has:
      - label: str (Table, Text, Section-Header, etc.)
      - bbox: [x0, y0, x1, y1] in pixel coordinates
      - content_html: str (inner HTML of the block)
      - content_text: str (plain text)
    """
    from bs4 import BeautifulSoup
    import json

    soup = BeautifulSoup(raw_html, "html.parser")

    # Find all divs with data-label (layout blocks), regardless of nesting
    top_divs = soup.find_all("div", attrs={"data-label": True})
    if not top_divs:
        # Fallback: try top-level divs
        top_divs = soup.find_all("div", recursive=False)

    width, height = page_image.size
    w_scale = width / BBOX_SCALE
    h_scale = height / BBOX_SCALE

    blocks = []
    for div in top_divs:
        label = div.get("data-label", "Unknown")

        # Parse bbox
        bbox_raw = div.get("data-bbox", "0 0 1 1")
        try:
            bbox = json.loads(bbox_raw)
        except (json.JSONDecodeError, TypeError):
            try:
                bbox = list(map(int, bbox_raw.split()))
            except Exception:
                bbox = [0, 0, 1, 1]

        # Scale to pixel coordinates
        bbox_px = [
            max(0, int(bbox[0] * w_scale)),
            max(0, int(bbox[1] * h_scale)),
            min(int(bbox[2] * w_scale), width),
            min(int(bbox[3] * h_scale), height),
        ]

        content_html = str(div.decode_contents())
        content_text = div.get_text(separator=" ", strip=True)

        blocks.append({
            "label": label,
            "bbox": bbox_px,
            "content_html": content_html,
            "content_text": content_text,
        })

    return blocks