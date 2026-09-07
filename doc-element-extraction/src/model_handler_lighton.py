"""
model_handler_lighton.py
===========================
Handles loading and inference of the LightOnOCR VLM (Vision Language Model).

LightOnOCR is built on transformers and loaded via the `lightonai/LightOnOCR-2-1B`
family of checkpoints.

Core logic is based on the reference implementation at:
  https://github.com/TheAwaken1/LightOnOCR-2-1B-Pinokio/blob/main/app/app.py

Public API used by app.py:
  - load_model()
  - _get_native_prompt()
  - extract_table_from_image()
  - extract_page_layout()
  - parse_layout_blocks()
  - extract_page_layout_with_blocks()
"""

import base64
import logging
import re
import time
from collections import OrderedDict
from io import BytesIO
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "LightOnOCR-2-1B (Best OCR)": {
        "model_id": "lightonai/LightOnOCR-2-1B",
        "has_bbox": False,
        "description": "Best OCR performance (83.2%)",
        "use_case": "General document extraction",
        "accuracy": "83.2%",
        "badge": "⭐ Recommended",
    },
    "LightOnOCR-2-1B-bbox (Best Bbox)": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox",
        "has_bbox": True,
        "description": "Best bounding box detection",
        "use_case": "Image localization",
        "accuracy": None,
        "badge": "📍 Best Bbox",
    },
    "LightOnOCR-2-1B-base": {
        "model_id": "lightonai/LightOnOCR-2-1B-base",
        "has_bbox": False,
        "description": "Supervised baseline (81.8%)",
        "use_case": "Fine-tuning base",
        "accuracy": "81.8%",
        "badge": None,
    },
    "LightOnOCR-2-1B-bbox-base": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox-base",
        "has_bbox": True,
        "description": "Base bbox model",
        "use_case": "Bbox fine-tuning",
        "accuracy": None,
        "badge": None,
    },
    "LightOnOCR-2-1B-ocr-soup": {
        "model_id": "lightonai/LightOnOCR-2-1B-ocr-soup",
        "has_bbox": False,
        "description": "Task-arithmetic merged (82.4%)",
        "use_case": "Alternative OCR",
        "accuracy": "82.4%",
        "badge": None,
    },
    "LightOnOCR-2-1B-bbox-soup": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox-soup",
        "has_bbox": True,
        "description": "OCR-bbox trade-off",
        "use_case": "Balanced performance",
        "accuracy": None,
        "badge": None,
    },
}

# Default checkpoint — used by load_model() and app.get_model()
MODEL_CHECKPOINT = "LightOnOCR-2-1B (Best OCR)"

# Normalisation scale used by LightOnOCR bbox models (0-1000)
BBOX_SCALE = 1000

# Sentinel prompt values — same convention as model_handler.py (Chandra)
ARABIC_TABLE_PROMPT = "USE_NATIVE"
LAYOUT_PROMPT = "USE_LAYOUT"

# Lazy prompt strings
_NATIVE_OCR_PROMPT = None
_NATIVE_LAYOUT_PROMPT = None

# Bbox annotation pattern: ![image](image_N.png)x1,y1,x2,y2
BBOX_PATTERN = r"!\[image\]\((image_\d+\.png)\)\s*(\d+),(\d+),(\d+),(\d+)"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _get_native_prompt() -> str:
    """Return LightOnOCR's default OCR prompt."""
    global _NATIVE_OCR_PROMPT
    if _NATIVE_OCR_PROMPT is None:
        _NATIVE_OCR_PROMPT = "Extract all readable text and table content from this image."
    return _NATIVE_OCR_PROMPT


def _get_layout_prompt() -> str:
    """Return layout-aware OCR prompt."""
    global _NATIVE_LAYOUT_PROMPT
    if _NATIVE_LAYOUT_PROMPT is None:
        _NATIVE_LAYOUT_PROMPT = (
            "Perform layout-aware OCR. Preserve reading order and include "
            "bounding-box-aware structure for page elements."
        )
    return _NATIVE_LAYOUT_PROMPT


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> tuple[torch.device, torch.dtype, str]:
    """Return best available device + dtype + attention implementation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        attn_implementation = "sdpa"
        logger.info("Using CUDA with bfloat16 and sdpa attention")
        return device, dtype, attn_implementation

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
        attn_implementation = "eager"
        logger.info("Using MPS with float32 and eager attention")
        return device, dtype, attn_implementation

    logger.warning("No GPU found — falling back to CPU. Inference will be slow.")
    return torch.device("cpu"), torch.float32, "eager"


def _resolve_model_id(label: str) -> str:
    """Resolve a registry label or a raw HF repo id."""
    return MODEL_REGISTRY[label]["model_id"] if label in MODEL_REGISTRY else label


# ---------------------------------------------------------------------------
# ModelManager  — LRU cache with GPU memory management
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages model loading with LRU caching and GPU memory management."""

    def __init__(self, max_cached: int = 2):
        # {model_id: (model, processor, device, dtype)}
        self._cache: OrderedDict = OrderedDict()
        self._max_cached = max_cached

    def get_model(self, model_name: str = MODEL_CHECKPOINT):
        """Return (model, processor, device, load_time_s, size_gb)."""
        config = MODEL_REGISTRY.get(model_name)
        if config is None:
            raise ValueError(f"Unknown model: {model_name!r}")

        model_id = config["model_id"]

        # Cache hit
        if model_id in self._cache:
            self._cache.move_to_end(model_id)
            logger.info("Using cached model: %s", model_name)
            model, processor, device, _dtype = self._cache[model_id]
            return model, processor, device, 0.0, model.get_memory_footprint() / (1024 ** 3)

        # Evict LRU entries until we have a free slot
        while len(self._cache) >= self._max_cached:
            evicted_id, (evicted_model, _proc, evicted_device, _dtype) = (
                self._cache.popitem(last=False)
            )
            logger.info("Evicting model: %s", evicted_id)
            del evicted_model
            if evicted_device.type == "cuda":
                torch.cuda.empty_cache()
            elif evicted_device.type == "mps":
                torch.mps.empty_cache()

        # Load
        t0 = time.perf_counter()
        device, dtype, attn_impl = get_device()

        logger.info("Loading %s (%s) …", model_name, model_id)
        model = (
            LightOnOcrForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=attn_impl,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
        processor = LightOnOcrProcessor.from_pretrained(model_id, trust_remote_code=True)
        model.processor = processor

        self._cache[model_id] = (model, processor, device, dtype)
        load_time = time.perf_counter() - t0
        size_gb = model.get_memory_footprint() / (1024 ** 3)
        logger.info(
            "Loaded %s  dtype=%s  device=%s  %.2fs  %.2fGB",
            model_name, dtype, device, load_time, size_gb,
        )
        return model, processor, device, load_time, size_gb

    def get_model_info(self, model_name: str) -> dict | None:
        return MODEL_REGISTRY.get(model_name)


# Singleton
model_manager = ModelManager(max_cached=2)
logger.info("ModelManager initialised — models load on first use.")


# ---------------------------------------------------------------------------
# load_model()  — standalone loader for app.py's get_model() cache
# ---------------------------------------------------------------------------

def load_model(model_checkpoint: str = MODEL_CHECKPOINT):
    """
    Load the LightOnOCR model and processor.
    Returns (model, processor, device, load_time_s, size_gb).
    """
    t0 = time.perf_counter()
    model_id = _resolve_model_id(model_checkpoint)
    device, dtype, attn_impl = get_device()

    processor = LightOnOcrProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = (
        LightOnOcrForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    model.processor = processor

    load_time = time.perf_counter() - t0
    size_gb = model.get_memory_footprint() / (1024 ** 3)
    logger.info("load_model: %s  %.2fs  %.2fGB", model_id, load_time, size_gb)
    return model, processor, device, load_time, size_gb


# ---------------------------------------------------------------------------
# Text cleaning  (reference: clean_output_text)
# ---------------------------------------------------------------------------

def clean_output_text(text: str) -> str:
    """Remove common chat-template artefacts from generated output."""
    markers = {"system", "user", "assistant"}
    lines = [l for l in text.split("\n") if l.strip().lower() not in markers]
    cleaned = "\n".join(lines).strip()

    # Keep only what follows the last 'assistant' marker if present
    if "assistant" in text.lower():
        parts = text.split("assistant", 1)
        if len(parts) > 1:
            after = parts[1].strip(" :\n\t")
            if after:
                cleaned = after

    return cleaned


# ---------------------------------------------------------------------------
# Core inference  (reference: extract_text_from_image, local branch)
# ---------------------------------------------------------------------------

def _prepare_inputs(processor, image: Image.Image, device: torch.device, dtype: torch.dtype):
    """Build tokenised inputs in the correct device/dtype (reference pattern)."""
    chat = [{"role": "user", "content": [{"type": "image", "url": image}]}]

    raw = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Cast each tensor to the correct device + dtype
    return {
        k: (
            v.to(device=device, dtype=dtype)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.to(device)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in raw.items()
    }


def _run_inference(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    prompt: str | None = None,   # kept for API parity with model_handler.py
    max_new_tokens: int = 4096,
) -> str:
    """
    Core single-call inference: image → cleaned text.

    LightOnOCR is a pure-vision model; the ``prompt`` parameter is accepted
    for API compatibility with the Chandra handler but is not used.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    _, dtype, _ = get_device()
    inputs = _prepare_inputs(processor, image, device, dtype)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode full output and clean artefacts (reference approach)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return clean_output_text(text).strip()


# ---------------------------------------------------------------------------
# Bbox helpers  (reference: parse_bbox_output, crop_from_bbox, image_to_data_uri)
# ---------------------------------------------------------------------------

def parse_bbox_output(text: str):
    """Return (cleaned_text, [{ref, coords}]) from a bbox model output."""
    detections = []
    for m in re.finditer(BBOX_PATTERN, text or ""):
        ref, x1, y1, x2, y2 = m.groups()
        detections.append({"ref": ref, "coords": (int(x1), int(y1), int(x2), int(y2))})
    cleaned = re.sub(BBOX_PATTERN, r"![image](\1)", text or "")
    return cleaned, detections


def crop_from_bbox(source_image: Image.Image, bbox: dict, padding: int = 5) -> Image.Image:
    """Crop a region from image using normalised [0-1000] bbox coords."""
    w, h = source_image.size
    x1, y1, x2, y2 = bbox["coords"]
    px1 = max(0, int(x1 * w / BBOX_SCALE) - padding)
    py1 = max(0, int(y1 * h / BBOX_SCALE) - padding)
    px2 = min(w, int(x2 * w / BBOX_SCALE) + padding)
    py2 = min(h, int(y2 * h / BBOX_SCALE) + padding)
    return source_image.crop((px1, py1, px2, py2))


def image_to_data_uri(image: Image.Image) -> str:
    """Convert PIL image to base64 PNG data URI."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def render_bbox_with_crops(raw_output: str, source_image: Image.Image) -> str:
    """Replace markdown image placeholders with embedded cropped image data URIs."""
    cleaned, detections = parse_bbox_output(raw_output)
    for bbox in detections:
        try:
            data_uri = image_to_data_uri(crop_from_bbox(source_image, bbox))
            cleaned = cleaned.replace(
                f"![image]({bbox['ref']})", f"![Cropped region]({data_uri})"
            )
        except Exception as exc:
            logger.warning("Error cropping bbox %s: %s", bbox, exc)
    return cleaned


# ---------------------------------------------------------------------------
# app.py compatibility — extract_table_from_image / extract_page_layout
# ---------------------------------------------------------------------------

def extract_table_from_image(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    prompt: str = "USE_NATIVE",
    max_new_tokens: int = 4096,
) -> str:
    """Run LightOnOCR OCR on a single image. Returns raw text/HTML."""
    try:
        return _run_inference(image, model, processor, device, prompt, max_new_tokens)
    except Exception as exc:
        logger.exception("extract_table_from_image failed: %s", exc)
        return f"[ERROR] VLM inference failed: {exc}"


def extract_page_layout(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    max_new_tokens: int = 4096,
) -> str:
    """
    Run full-page OCR and return raw text as a string.

    For the Full Page OCR mode in app.py prefer ``extract_page_layout_with_blocks``
    which does per-strip inference and guarantees bbox ↔ content alignment.
    """
    try:
        return _run_inference(image, model, processor, device, None, max_new_tokens)
    except Exception as exc:
        logger.exception("extract_page_layout failed: %s", exc)
        return f"[ERROR] Layout extraction failed: {exc}"


# ---------------------------------------------------------------------------
# Layout block parsing helpers
# ---------------------------------------------------------------------------

def _markdown_table_to_html(md_table: str) -> str:
    """Convert a markdown table string to an HTML <table>."""
    lines = md_table.strip().split("\n")
    if not lines:
        return ""
    html = ["<table border='1'>"]
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or set(line.replace("|", "").strip()) <= set("-:"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        tag = "th" if i == 0 else "td"
        html.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
    html.append("</table>")
    return "\n".join(html)


def _detect_content_strips(
    page_image: Image.Image,
    min_gap_px: int = 12,
    white_threshold: int = 245,
    margin: int = 4,
) -> List[tuple]:
    """
    Find horizontal content bands by scanning for whitespace gaps.
    Returns [(y_top, y_bottom), …].
    """
    import numpy as np

    arr = np.array(page_image.convert("L"))
    H = arr.shape[0]
    blank = arr.min(axis=1) >= white_threshold

    strips: List[tuple] = []
    in_content = False
    start = 0
    i = 0

    while i < H:
        if not blank[i]:
            if not in_content:
                start = i
                in_content = True
        elif in_content:
            gap_start = i
            while i < H and blank[i]:
                i += 1
            if (i - gap_start) >= min_gap_px:
                strips.append((max(0, start - margin), min(H, gap_start + margin)))
                in_content = False
            continue
        i += 1

    if in_content:
        strips.append((max(0, start - margin), H))

    return strips or [(0, H)]


def _classify_strip_content(raw_text: str) -> str:
    """Heuristically label a strip's OCR output as Table/Image/Section-Header/Text."""
    from bs4 import BeautifulSoup

    if re.search(r"!\[.*?\]\(.*?\)", raw_text):
        return "Image"

    if BeautifulSoup(raw_text, "html.parser").find("table"):
        return "Table"

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        return "Table" if line.startswith("|") else ("Section-Header" if line.startswith("#") else "Text")

    return "Text"


def _format_strip_content(raw_text: str, label: str) -> str:
    """Render a strip's OCR output as an HTML string."""
    if label == "Table":
        from bs4 import BeautifulSoup
        table = BeautifulSoup(raw_text, "html.parser").find("table")
        return str(table) if table else _markdown_table_to_html(raw_text)
    if label == "Image":
        return f"<p>{raw_text}</p>"
    return f"<pre>{raw_text}</pre>"


def parse_layout_blocks(raw_html: str, page_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Parse LightOnOCR's raw output into structured layout blocks.

    Image elements receive their native per-element bbox from the model output.
    Non-image blocks receive bboxes from PIL-based whitespace-gap segmentation.

    Block schema: {label, bbox:[x0,y0,x1,y1], content_html, content_text}
    """
    from bs4 import BeautifulSoup

    width, height = page_image.size
    ws = width / float(BBOX_SCALE)
    hs = height / float(BBOX_SCALE)
    blocks: List[Dict[str, Any]] = []

    # ── 1. Image bboxes from model output
    GEN_PAT = r"!\[(.*?)\]\((.*?)\)\s*(\d+),(\d+),(\d+),(\d+)"
    cleaned = raw_html
    for m in re.finditer(GEN_PAT, raw_html):
        alt, ref, x1, y1, x2, y2 = m.groups()
        blocks.append({
            "label": "Image",
            "bbox": [
                max(0, int(int(x1) * ws)), max(0, int(int(y1) * hs)),
                min(width, int(int(x2) * ws)), min(height, int(int(y2) * hs)),
            ],
            "content_html": f'<img src="{ref}" alt="{alt}" />',
            "content_text": f"![{alt}]({ref})",
        })
        cleaned = cleaned.replace(m.group(0), f"![{alt}]({ref})")

    # ── 2. HTML tables
    soup = BeautifulSoup(cleaned, "html.parser")
    for table in soup.find_all("table"):
        blocks.append({
            "label": "Table",
            "bbox": [0, 0, width, height],
            "content_html": str(table),
            "content_text": table.get_text(separator=" ", strip=True),
        })
        table.extract()

    # ── 2b. Markdown table fallback
    remaining = str(soup)
    MD_PAT = r"(^\|(?:.*?\|)+\s*\n\|(?:[-\s:|]+)+\|\s*\n(?:\|(?:.*?\|)+\s*\n?)+)"
    for m in re.finditer(MD_PAT, remaining, re.MULTILINE):
        md = m.group(1).strip()
        blocks.append({
            "label": "Table",
            "bbox": [0, 0, width, height],
            "content_html": _markdown_table_to_html(md),
            "content_text": md,
        })
        remaining = remaining.replace(m.group(1), "")

    # ── 3. Remaining text
    final = BeautifulSoup(remaining, "html.parser").get_text(separator="\n", strip=True)
    if final:
        blocks.append({
            "label": "Text",
            "bbox": [0, 0, width, height],
            "content_html": f"<pre>{final}</pre>",
            "content_text": final,
        })

    # ── 4. Assign whitespace-gap strips to non-image blocks
    non_img = [i for i, b in enumerate(blocks) if b["label"] != "Image"]
    if non_img:
        try:
            strips = _detect_content_strips(page_image)
            n_b, n_s = len(non_img), len(strips)
            if n_s >= n_b:
                ss = n_s / n_b
                for rank, idx in enumerate(non_img):
                    fi = int(round(rank * ss))
                    li = min(int(round((rank + 1) * ss)) - 1, n_s - 1)
                    blocks[idx]["bbox"] = [0, strips[fi][0], width, strips[li][1]]
            else:
                seg = height // n_b
                for rank, idx in enumerate(non_img):
                    blocks[idx]["bbox"] = [
                        0, rank * seg, width,
                        (rank + 1) * seg if rank < n_b - 1 else height,
                    ]
        except Exception as exc:
            logger.warning("Strip assignment failed: %s", exc)

    return blocks


# ---------------------------------------------------------------------------
# Per-strip OCR — guarantees exact bbox ↔ content correspondence
# ---------------------------------------------------------------------------

def extract_page_layout_with_blocks(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    max_new_tokens: int = 4096,
):
    """
    Segment the page into horizontal content strips, OCR each strip
    individually, and return (blocks, combined_raw_html).

    Each block's ``content_html`` is extracted from exactly its ``bbox``
    crop, guaranteeing correct alignment.

    Returns
    -------
    blocks : List[Dict]
    combined_raw_html : str   – all per-strip outputs joined (for the UI viewer)
    """
    width, height = image.size
    blocks: List[Dict[str, Any]] = []
    raw_parts: List[str] = []

    # LightOnOCR's vision encoder uses successive 2× downsampling.
    # Crops below MIN_CROP_PX in either dimension will cause:
    #   "spatial size (1,1), kernel_size=(2,2) … components must be at least one"
    MIN_CROP_PX = 64

    STRIP_PAT = r"!\[(.*?)\]\((.*?)\)\s*(\d+),(\d+),(\d+),(\d+)"

    # ── 1. Detect content strips
    try:
        strips = _detect_content_strips(image)
    except Exception as exc:
        logger.warning("Strip detection failed (%s); using full page.", exc)
        strips = [(0, height)]

    logger.info("%d strip(s) on %dx%d page", len(strips), width, height)

    # ── 2. Per-strip inference
    for y0, y1 in strips:
        if y1 - y0 <= 0:
            continue

        crop = image.crop((0, y0, width, y1))
        cw, ch = crop.size

        # Upscale tiny crops to satisfy the encoder's minimum spatial size
        if cw < MIN_CROP_PX or ch < MIN_CROP_PX:
            scale = max(MIN_CROP_PX / cw, MIN_CROP_PX / ch)
            nw = max(MIN_CROP_PX, int(cw * scale))
            nh = max(MIN_CROP_PX, int(ch * scale))
            logger.debug("Upscaling strip [%d,%d] %dx%d → %dx%d", y0, y1, cw, ch, nw, nh)
            crop = crop.resize((nw, nh), Image.Resampling.LANCZOS)
            cw, ch = nw, nh

        try:
            raw = _run_inference(crop, model, processor, device, None, max_new_tokens)
        except Exception as exc:
            logger.warning("Strip OCR failed y=[%d,%d]: %s — skipping.", y0, y1, exc)
            continue

        if not raw or not raw.strip():
            continue

        raw_parts.append(raw.strip())
        cleaned = raw

        # ── 3. Translate in-strip image bboxes to full-page coords
        for m in re.finditer(STRIP_PAT, raw):
            alt, ref, sx1, sy1, sx2, sy2 = m.groups()
            blocks.append({
                "label": "Image",
                "bbox": [
                    max(0, int(int(sx1) * cw / BBOX_SCALE)),
                    max(0, int(int(sy1) * ch / BBOX_SCALE)) + y0,
                    min(width, int(int(sx2) * cw / BBOX_SCALE)),
                    min(height, int(int(sy2) * ch / BBOX_SCALE) + y0),
                ],
                "content_html": f'<img src="{ref}" alt="{alt}" />',
                "content_text": f"![{alt}]({ref})",
            })
            cleaned = cleaned.replace(m.group(0), f"![{alt}]({ref})")

        cleaned = cleaned.strip()
        if not cleaned:
            continue

        label = _classify_strip_content(cleaned)
        # Don't duplicate blocks already stored as Image
        if label == "Image" and re.fullmatch(r"!\[.*?\]\(.*?\)", cleaned):
            continue

        blocks.append({
            "label": label,
            "bbox": [0, y0, width, y1],
            "content_html": _format_strip_content(cleaned, label),
            "content_text": cleaned,
        })

    # ── 4. Fallback: single full-page OCR
    if not blocks:
        logger.warning("No strip output — falling back to full-page OCR.")
        try:
            raw_full = _run_inference(image, model, processor, device, None, max_new_tokens)
            if raw_full and not raw_full.startswith("[ERROR]"):
                raw_parts.append(raw_full.strip())
                blocks.append({
                    "label": "Text",
                    "bbox": [0, 0, width, height],
                    "content_html": f"<pre>{raw_full.strip()}</pre>",
                    "content_text": raw_full.strip(),
                })
        except Exception as exc:
            logger.exception("Fallback full-page OCR failed: %s", exc)

    combined_raw_html = "\n\n<!-- strip separator -->\n\n".join(raw_parts)
    return blocks, combined_raw_html


# ---------------------------------------------------------------------------
# PDF utilities
# ---------------------------------------------------------------------------

def render_pdf_page(page, max_resolution: int = 1540, scale: float = 2.77) -> Image.Image:
    """Render a PDF page to a PIL Image."""
    w, h = page.get_size()
    rf = min(1, max_resolution / (w * scale), max_resolution / (h * scale))
    return page.render(scale=scale * rf, rev_byteorder=True).to_pil()


def process_pdf(pdf_path: str, page_num: int = 1):
    """Extract a specific page from a PDF as a PIL Image."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError("pypdfium2 is required for process_pdf().") from exc
    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    idx = min(max(int(page_num) - 1, 0), total - 1)
    img = render_pdf_page(pdf[idx])
    pdf.close()
    return img, total, idx + 1
