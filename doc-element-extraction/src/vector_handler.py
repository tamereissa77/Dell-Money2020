"""
vector_handler.py
=================
Vector embedding handler using a local sentence-transformers service
and Qdrant for vector storage.

Embeds extracted text chunks and table content for semantic search.
"""

import logging
import uuid
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SENTENCE_TRANSFORMER_URL = "http://sentence-transformers:80"
QDRANT_URL = "http://qdrant:6333"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
COLLECTION_NAME = "document-embeddings"


class VectorHandler:
    """Manages text embedding and Qdrant vector storage."""

    def __init__(
        self,
        transformer_url: str = SENTENCE_TRANSFORMER_URL,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
        dimension: int = EMBEDDING_DIM,
    ):
        self.transformer_url = transformer_url.rstrip("/")
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection_name = collection_name
        self.dimension = dimension
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Qdrant collection setup
    # ------------------------------------------------------------------
    def _qdrant_request(
        self, method: str, path: str, data: dict = None
    ) -> Optional[dict]:
        url = f"{self.qdrant_url}{path}"
        try:
            resp = requests.request(
                method, url, json=data, timeout=30,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code < 400:
                return resp.json()
            else:
                logger.warning(
                    "Qdrant %s %s → %d: %s",
                    method, path, resp.status_code, resp.text[:200],
                )
                return None
        except Exception as exc:
            logger.error("Qdrant request failed: %s", exc)
            return None

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        info = self._qdrant_request("GET", f"/collections/{self.collection_name}")
        if info and info.get("result"):
            logger.info(
                "Qdrant collection '%s' exists with %d vectors.",
                self.collection_name,
                info["result"].get("points_count", 0),
            )
            return

        result = self._qdrant_request(
            "PUT",
            f"/collections/{self.collection_name}",
            {"vectors": {"size": self.dimension, "distance": "Cosine"}},
        )
        if result:
            logger.info(
                "Created Qdrant collection '%s' (dim=%d).",
                self.collection_name, self.dimension,
            )
        else:
            logger.warning("Failed to create Qdrant collection.")

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Call the sentence-transformers service to generate embeddings."""
        url = f"{self.transformer_url}/embed"
        try:
            resp = requests.post(
                url,
                json={"texts": texts, "batch_size": 32},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embeddings", [])
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------
    @staticmethod
    def chunk_text(
        text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks by sentence boundaries."""
        if not text or not text.strip():
            return []

        sentences = []
        for s in text.replace('\n', ' ').split('.'):
            s = s.strip()
            if s:
                sentences.append(s + '.')

        if not sentences:
            return [text.strip()] if text.strip() else []

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) > chunk_size and current:
                chunks.append(current.strip())
                # Overlap: keep last N chars
                words = current.split()
                overlap_words = words[-overlap:] if len(words) > overlap else words
                current = " ".join(overlap_words)
            current += " " + sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    # ------------------------------------------------------------------
    # Store embeddings
    # ------------------------------------------------------------------
    def _string_to_uuid(self, s: str) -> str:
        """Generate a deterministic UUID from a string."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

    def embed_and_store(
        self,
        text: str,
        document_name: str = "",
        page_number: int = 1,
        element_label: str = "Text",
        element_index: int = 0,
        chunk_size: int = 500,
    ) -> int:
        """
        Chunk text, generate embeddings, and store in Qdrant.

        Returns the number of vectors stored.
        """
        chunks = self.chunk_text(text, chunk_size=chunk_size)
        if not chunks:
            return 0

        embeddings = self._generate_embeddings(chunks)
        if not embeddings or len(embeddings) != len(chunks):
            logger.warning(
                "Embedding count mismatch: %d chunks, %d embeddings.",
                len(chunks), len(embeddings),
            )
            return 0

        # Build Qdrant points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = self._string_to_uuid(
                f"{document_name}_{page_number}_{element_index}_{i}"
            )
            points.append({
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "text": chunk,
                    "document_name": document_name,
                    "page_number": page_number,
                    "element_label": element_label,
                    "element_index": element_index,
                    "chunk_index": i,
                },
            })

        # Batch upsert
        batch_size = 100
        stored = 0
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            result = self._qdrant_request(
                "PUT",
                f"/collections/{self.collection_name}/points",
                {"points": batch},
            )
            if result:
                stored += len(batch)

        logger.info(
            "Stored %d/%d vectors for '%s' page %d element %d.",
            stored, len(points), document_name, page_number, element_index,
        )
        return stored

    def embed_table(
        self,
        df,
        document_name: str = "",
        page_number: int = 1,
        element_index: int = 0,
    ) -> int:
        """
        Embed a table's content (serialized as text) into the vector store.
        """
        # Serialize table to text
        text_parts = []
        columns = list(df.columns)
        text_parts.append("Table columns: " + ", ".join(str(c) for c in columns))

        for idx, row in df.iterrows():
            row_text = " | ".join(f"{col}: {val}" for col, val in row.items())
            text_parts.append(row_text)

        full_text = "\n".join(text_parts)
        return self.embed_and_store(
            text=full_text,
            document_name=document_name,
            page_number=page_number,
            element_label="Table",
            element_index=element_index,
            chunk_size=800,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self, query: str, top_k: int = 10
    ) -> List[Dict]:
        """Semantic search across stored embeddings."""
        embeddings = self._generate_embeddings([query])
        if not embeddings:
            return []

        result = self._qdrant_request(
            "POST",
            f"/collections/{self.collection_name}/points/query",
            {
                "query": embeddings[0],
                "limit": top_k,
                "with_payload": True,
            },
        )

        if not result or "result" not in result:
            return []

        return [
            {
                "score": point.get("score", 0),
                "text": point.get("payload", {}).get("text", ""),
                "document_name": point.get("payload", {}).get("document_name", ""),
                "page_number": point.get("payload", {}).get("page_number", 0),
                "element_label": point.get("payload", {}).get("element_label", ""),
            }
            for point in result["result"]
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Get Qdrant collection statistics."""
        info = self._qdrant_request("GET", f"/collections/{self.collection_name}")
        if info and info.get("result"):
            r = info["result"]
            return {
                "collection": self.collection_name,
                "vectors": r.get("points_count", 0),
                "status": r.get("status", "unknown"),
                "dimension": self.dimension,
            }
        return {"collection": self.collection_name, "vectors": 0, "status": "unavailable"}
