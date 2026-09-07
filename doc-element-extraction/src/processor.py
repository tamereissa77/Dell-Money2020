"""
processor.py
============
Document segmentation using the *unstructured* library.

Handles:
  - PDF → image conversion (via pdf2image / poppler).
  - Element partitioning (text, tables, images, etc.).
  - Cropping detected table regions from the source page image.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def file_bytes_to_images(file_bytes: bytes, filename: str) -> List[Image.Image]:
    """
    Convert uploaded file bytes into a list of PIL Images (one per page).

    Supports PDF, PNG, JPG/JPEG, TIFF.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        images = convert_from_bytes(file_bytes, dpi=200)
        return images

    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return [img]

    raise ValueError(f"Unsupported file type: {suffix}")


def partition_document(file_bytes: bytes, filename: str) -> list:
    """
    Use *unstructured* to partition a document into typed elements.

    Returns the raw list of Element objects.
    """
    # Lazy import so the heavy unstructured deps are only loaded when needed.
    from unstructured.partition.auto import partition  # noqa: E402

    suffix = Path(filename).suffix.lower()

    # Write bytes to a temp file because `partition` expects a file path.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    logger.info("Partitioning %s (%d bytes) …", filename, len(file_bytes))

    elements = partition(
        filename=tmp_path,
        strategy="hi_res",          # Use layout model for table detection
        languages=["ara", "eng"],   # Arabic + English OCR
        include_page_breaks=True,
    )

    logger.info("Partitioned into %d elements.", len(elements))

    # Clean up
    Path(tmp_path).unlink(missing_ok=True)
    return elements


def extract_table_elements(elements: list) -> list:
    """
    Filter the partitioned elements to keep only Table (and optionally
    Image) elements.

    Returns a list of element objects whose category is 'Table' or 'Image'.
    """
    from unstructured.documents.elements import (  # noqa: E402
        Table,
        Image as UnstructuredImage,
    )

    tables = [
        el for el in elements
        if isinstance(el, (Table, UnstructuredImage))
    ]
    logger.info("Found %d table/image elements.", len(tables))
    return tables


def crop_element_from_page(
    page_image: Image.Image,
    element,
) -> Image.Image | None:
    """
    Crop the bounding-box region of an *unstructured* element from the
    full-page image.

    The element must carry coordinate metadata
    (``element.metadata.coordinates``).

    Returns *None* if coordinates are unavailable.
    """
    coords = getattr(element.metadata, "coordinates", None)
    if coords is None:
        logger.warning("Element has no coordinate metadata – skipping crop.")
        return None

    # `coordinates.points` is a list of (x, y) tuples describing the polygon.
    points = coords.points
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Determine the axis-aligned bounding box.
    left = max(int(min(xs)), 0)
    upper = max(int(min(ys)), 0)
    right = min(int(max(xs)), page_image.width)
    lower = min(int(max(ys)), page_image.height)

    if right <= left or lower <= upper:
        logger.warning("Degenerate bounding box – skipping crop.")
        return None

    cropped = page_image.crop((left, upper, right, lower))
    return cropped


def process_document(
    file_bytes: bytes,
    filename: str,
) -> List[Tuple[Image.Image, object]]:
    """
    End-to-end pipeline:
      1. Convert file to page images.
      2. Partition with Unstructured.
      3. For each Table/Image element, crop the region from the page.

    Returns
    -------
    list of (cropped_image, element) tuples.
    """
    page_images = file_bytes_to_images(file_bytes, filename)
    elements = partition_document(file_bytes, filename)
    table_elements = extract_table_elements(elements)

    results: List[Tuple[Image.Image, object]] = []

    for el in table_elements:
        # Determine which page this element belongs to.
        page_num = getattr(el.metadata, "page_number", 1) or 1
        page_idx = page_num - 1  # 0-based

        if page_idx >= len(page_images):
            logger.warning(
                "Element references page %d but only %d pages available.",
                page_num,
                len(page_images),
            )
            continue

        cropped = crop_element_from_page(page_images[page_idx], el)
        if cropped is not None:
            results.append((cropped, el))

    logger.info("Extracted %d cropped table images.", len(results))
    return results
