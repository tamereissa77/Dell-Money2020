"""
object_store.py
===============
MinIO (S3-compatible) object storage handler for cropped element images.

Stores cropped images with their text reference metadata so they can be
retrieved later for citation purposes.

Each stored object has:
  - The cropped image (PNG)
  - Metadata: document name, page number, element label, bbox, extracted text
"""

import io
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from minio import Minio
from minio.error import S3Error
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "extracted-elements"
MINIO_SECURE = False


class ObjectStoreHandler:
    """Manages image storage in MinIO with text reference metadata."""

    def __init__(
        self,
        endpoint: str = MINIO_ENDPOINT,
        access_key: str = MINIO_ACCESS_KEY,
        secret_key: str = MINIO_SECRET_KEY,
        bucket: str = MINIO_BUCKET,
        secure: bool = MINIO_SECURE,
    ):
        self.bucket = bucket
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._ensure_bucket()

    # ------------------------------------------------------------------
    # Bucket setup
    # ------------------------------------------------------------------
    def _ensure_bucket(self):
        """Create the bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info("Created MinIO bucket: %s", self.bucket)
            else:
                logger.info("MinIO bucket '%s' already exists.", self.bucket)
        except S3Error as exc:
            logger.error("MinIO bucket check failed: %s", exc)

    # ------------------------------------------------------------------
    # Store image with metadata
    # ------------------------------------------------------------------
    def store_element_image(
        self,
        image: Image.Image,
        document_name: str,
        page_number: int,
        element_index: int,
        element_label: str,
        bbox: list,
        extracted_text: str = "",
        content_html: str = "",
    ) -> Optional[str]:
        """
        Store a cropped element image in MinIO with text reference metadata.

        Returns the object key (path) in MinIO, or None on failure.
        """
        # Generate a unique object key — use ASCII-safe path components
        import re as _re
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        obj_id = uuid.uuid4().hex[:8]
        # Replace non-ASCII and special chars with underscores for the path
        safe_doc = _re.sub(r'[^a-zA-Z0-9._\-]', '_', document_name)[:50]
        if not safe_doc or safe_doc == '_':
            safe_doc = f"doc_{obj_id}"
        safe_label = _re.sub(r'[^a-zA-Z0-9_\-]', '_', element_label.lower())
        object_key = (
            f"{safe_doc}/page_{page_number}/"
            f"{safe_label}_{element_index}_{obj_id}.png"
        )

        # Convert image to PNG bytes
        buf = io.BytesIO()
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        image_size = buf.getbuffer().nbytes

        # Build metadata — S3/MinIO metadata headers only support ASCII.
        # URL-encode any non-ASCII values (e.g. Arabic document names).
        from urllib.parse import quote as url_quote

        def _safe_meta(value: str, max_len: int = 2000) -> str:
            """Encode value to be safe for S3 metadata headers (ASCII only)."""
            return url_quote(value[:max_len], safe="")

        metadata = {
            "x-amz-meta-document-name": _safe_meta(document_name),
            "x-amz-meta-page-number": str(page_number),
            "x-amz-meta-element-index": str(element_index),
            "x-amz-meta-element-label": _safe_meta(element_label),
            "x-amz-meta-bbox": json.dumps(bbox),
            "x-amz-meta-timestamp": timestamp,
        }

        try:
            self.client.put_object(
                self.bucket,
                object_key,
                buf,
                length=image_size,
                content_type="image/png",
                metadata=metadata,
            )
            logger.info(
                "Stored image: %s/%s (%d bytes, label=%s)",
                self.bucket, object_key, image_size, element_label,
            )
            return object_key
        except S3Error as exc:
            logger.error("Failed to store image in MinIO: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Store the text reference alongside the image
    # ------------------------------------------------------------------
    def store_text_reference(
        self,
        object_key: str,
        document_name: str,
        page_number: int,
        element_index: int,
        element_label: str,
        bbox: list,
        extracted_text: str,
        content_html: str = "",
        table_id: str = "",
    ) -> Optional[str]:
        """
        Store a JSON text reference file alongside the image for citation.

        Returns the reference object key, or None on failure.
        """
        ref_key = object_key.rsplit(".", 1)[0] + "_reference.json"

        reference = {
            "image_key": object_key,
            "document_name": document_name,
            "page_number": page_number,
            "element_index": element_index,
            "element_label": element_label,
            "bbox": bbox,
            "extracted_text": extracted_text,
            "content_html": content_html,
            "table_id": table_id,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }

        ref_bytes = json.dumps(reference, ensure_ascii=False, indent=2).encode("utf-8")
        buf = io.BytesIO(ref_bytes)

        try:
            self.client.put_object(
                self.bucket,
                ref_key,
                buf,
                length=len(ref_bytes),
                content_type="application/json",
            )
            logger.info("Stored text reference: %s/%s", self.bucket, ref_key)
            return ref_key
        except S3Error as exc:
            logger.error("Failed to store text reference: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Combined: store image + reference
    # ------------------------------------------------------------------
    def store_element(
        self,
        image: Image.Image,
        document_name: str,
        page_number: int,
        element_index: int,
        element_label: str,
        bbox: list,
        extracted_text: str = "",
        content_html: str = "",
        table_id: str = "",
    ) -> Dict[str, Optional[str]]:
        """
        Store both the cropped image and its text reference.

        Returns {"image_key": ..., "reference_key": ...}
        """
        image_key = self.store_element_image(
            image=image,
            document_name=document_name,
            page_number=page_number,
            element_index=element_index,
            element_label=element_label,
            bbox=bbox,
            extracted_text=extracted_text,
            content_html=content_html,
        )

        ref_key = None
        if image_key:
            ref_key = self.store_text_reference(
                object_key=image_key,
                document_name=document_name,
                page_number=page_number,
                element_index=element_index,
                element_label=element_label,
                bbox=bbox,
                extracted_text=extracted_text,
                content_html=content_html,
                table_id=table_id,
            )

        return {"image_key": image_key, "reference_key": ref_key}

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def get_image(self, object_key: str) -> Optional[Image.Image]:
        """Retrieve an image from MinIO."""
        try:
            response = self.client.get_object(self.bucket, object_key)
            img = Image.open(io.BytesIO(response.read()))
            response.close()
            response.release_conn()
            return img
        except S3Error as exc:
            logger.error("Failed to retrieve image: %s", exc)
            return None

    def get_reference(self, ref_key: str) -> Optional[dict]:
        """Retrieve a text reference JSON from MinIO."""
        try:
            response = self.client.get_object(self.bucket, ref_key)
            data = json.loads(response.read().decode("utf-8"))
            response.close()
            response.release_conn()
            return data
        except S3Error as exc:
            logger.error("Failed to retrieve reference: %s", exc)
            return None

    def list_elements(
        self, document_name: Optional[str] = None, prefix: str = ""
    ) -> List[dict]:
        """List stored elements, optionally filtered by document."""
        search_prefix = prefix
        if document_name:
            safe_doc = document_name.replace("/", "_").replace(" ", "_")[:50]
            search_prefix = f"{safe_doc}/"

        elements = []
        try:
            objects = self.client.list_objects(
                self.bucket, prefix=search_prefix, recursive=True
            )
            for obj in objects:
                if obj.object_name.endswith("_reference.json"):
                    ref = self.get_reference(obj.object_name)
                    if ref:
                        elements.append(ref)
        except S3Error as exc:
            logger.error("Failed to list elements: %s", exc)

        return elements

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Get basic stats about stored objects."""
        try:
            count = 0
            total_size = 0
            objects = self.client.list_objects(self.bucket, recursive=True)
            for obj in objects:
                count += 1
                total_size += obj.size or 0

            return {
                "bucket": self.bucket,
                "objects": count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
        except S3Error as exc:
            logger.error("Failed to get stats: %s", exc)
            return {"bucket": self.bucket, "objects": 0, "total_size_mb": 0}
