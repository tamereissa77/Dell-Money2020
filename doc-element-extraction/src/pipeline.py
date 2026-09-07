"""
pipeline.py
===========
Post-extraction pipeline orchestrator.

After Chandra VLM extracts elements from a document page, this module
routes each element to the appropriate storage backends.

**Page-by-page processing:**
  - KG triple extraction is done ONCE per page (all text concatenated)
  - Vector embedding is done ONCE per page (full page text)
  - Table storage (PostgreSQL) is per-element
  - Image storage (MinIO) is per-element

This ensures:
  1. Better KG quality — Ollama sees full page context, not fragments
  2. Fewer LLM calls — one Ollama call per page instead of per-element
  3. Coherent embeddings — page-level vectors capture cross-element relationships
  4. Per-element citation — each cropped image is still stored individually
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Orchestrates post-extraction storage across all backends.

    Primary method: process_page() — processes an entire page at once,
    doing KG and embedding at the page level, and element storage per-element.
    """

    def __init__(self):
        self._pg = None
        self._kg = None
        self._vec = None
        self._obj = None
        self._init_errors = {}

    # ------------------------------------------------------------------
    # Lazy initialization of backends
    # ------------------------------------------------------------------
    @property
    def pg(self):
        """PostgreSQL handler (lazy init)."""
        if self._pg is None and "pg" not in self._init_errors:
            try:
                from db_handler import PostgresHandler
                self._pg = PostgresHandler()
            except Exception as exc:
                self._init_errors["pg"] = str(exc)
                logger.error("PostgreSQL init failed: %s", exc)
        return self._pg

    @property
    def kg(self):
        """Knowledge Graph handler (lazy init)."""
        if self._kg is None and "kg" not in self._init_errors:
            try:
                from kg_handler import KnowledgeGraphHandler
                self._kg = KnowledgeGraphHandler()
            except Exception as exc:
                self._init_errors["kg"] = str(exc)
                logger.error("KG handler init failed: %s", exc)
        return self._kg

    @property
    def vec(self):
        """Vector handler (lazy init)."""
        if self._vec is None and "vec" not in self._init_errors:
            try:
                from vector_handler import VectorHandler
                self._vec = VectorHandler()
            except Exception as exc:
                self._init_errors["vec"] = str(exc)
                logger.error("Vector handler init failed: %s", exc)
        return self._vec

    @property
    def obj(self):
        """Object store handler (lazy init)."""
        if self._obj is None and "obj" not in self._init_errors:
            try:
                from object_store import ObjectStoreHandler
                self._obj = ObjectStoreHandler()
            except Exception as exc:
                self._init_errors["obj"] = str(exc)
                logger.error("Object store init failed: %s", exc)
        return self._obj

    # ==================================================================
    # PAGE-LEVEL PROCESSING (primary entry point)
    # ==================================================================
    def process_page(
        self,
        blocks: List[Dict[str, Any]],
        page_image: Image.Image,
        document_name: str,
        page_number: int,
        dataframes_map: Optional[Dict[int, List[pd.DataFrame]]] = None,
    ) -> Dict[str, Any]:
        """
        Process an entire page's extracted elements at once.

        This is the primary entry point. It ensures:
          - KG triple extraction happens ONCE for the full page text
          - Vector embedding happens ONCE for the full page text
          - Table → PostgreSQL is per-element
          - Image → MinIO is per-element

        Parameters
        ----------
        blocks : list of dicts with keys: label, bbox, content_html, content_text
        page_image : full page PIL Image for cropping
        document_name : source document filename
        page_number : 1-based page number
        dataframes_map : mapping of block_index → list of DataFrames (for tables)

        Returns
        -------
        dict with keys:
          - page_number: int
          - document_name: str
          - page_kg: dict (page-level KG result)
          - page_vectors: dict (page-level embedding result)
          - elements: list of per-element result dicts
        """
        dataframes_map = dataframes_map or {}

        page_result = {
            "page_number": page_number,
            "document_name": document_name,
            "page_kg": {},
            "page_vectors": {},
            "elements": [],
        }

        # ==============================================================
        # STEP 1: Collect all text from the page (for page-level KG + embedding)
        # ==============================================================
        page_text_parts = []
        kg_eligible_parts = []  # Only text-like elements for KG

        for idx, block in enumerate(blocks):
            label = block.get("label", "Unknown")
            content_text = block.get("content_text", "").strip()

            if content_text:
                page_text_parts.append(f"[{label}] {content_text}")

                # KG-eligible: text-like elements (not tables, images, etc.)
                if label in ("Text", "Section-Header", "Caption", "Footnote",
                             "List-Group", "Page-Header", "Page-Footer"):
                    kg_eligible_parts.append(content_text)

        full_page_text = "\n\n".join(page_text_parts)
        kg_page_text = "\n\n".join(kg_eligible_parts)

        # Also include serialized table content in the page text for embedding
        for idx, dfs in dataframes_map.items():
            for df in dfs:
                if df is not None and not df.empty:
                    cols = ", ".join(str(c) for c in df.columns)
                    rows_text = "; ".join(
                        " | ".join(f"{col}: {val}" for col, val in row.items())
                        for _, row in df.head(20).iterrows()  # Limit rows for embedding
                    )
                    page_text_parts.append(f"[Table] Columns: {cols}. Rows: {rows_text}")

        full_page_text_with_tables = "\n\n".join(page_text_parts)

        logger.info(
            "Page %d: %d blocks, %d chars page text, %d chars KG text",
            page_number, len(blocks), len(full_page_text_with_tables), len(kg_page_text),
        )

        # ==============================================================
        # STEP 2: Page-level KG triple extraction (ONE Ollama call)
        # ==============================================================
        if kg_page_text and len(kg_page_text.strip()) > 20:
            page_result["page_kg"] = self._store_to_kg(
                text=kg_page_text,
                document_name=document_name,
                page_number=page_number,
            )
        else:
            page_result["page_kg"] = {"status": "skipped", "reason": "No KG-eligible text on page"}

        # ==============================================================
        # STEP 3: Page-level vector embedding (ONE embedding call)
        # ==============================================================
        if full_page_text_with_tables and len(full_page_text_with_tables.strip()) > 10:
            page_result["page_vectors"] = self._store_to_vector_db(
                text=full_page_text_with_tables,
                document_name=document_name,
                page_number=page_number,
                element_label="Page",
                element_index=0,
            )
        else:
            page_result["page_vectors"] = {"status": "skipped", "reason": "No text on page"}

        # ==============================================================
        # STEP 4: Per-element storage (PostgreSQL + MinIO)
        # ==============================================================
        for idx, block in enumerate(blocks):
            label = block.get("label", "Unknown")
            bbox = block.get("bbox", [0, 0, 0, 0])
            content_text = block.get("content_text", "")
            content_html = block.get("content_html", "")

            element_result = {
                "label": label,
                "element_index": idx,
                "page_number": page_number,
                "backends": {},
            }

            # Crop the element image
            cropped = None
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                try:
                    cropped = page_image.crop(bbox)
                except Exception:
                    pass

            # 4a. Object Store — store cropped image + reference
            if cropped is not None:
                element_result["backends"]["object_store"] = self._store_to_object_store(
                    image=cropped,
                    document_name=document_name,
                    page_number=page_number,
                    element_index=idx,
                    element_label=label,
                    bbox=bbox,
                    extracted_text=content_text,
                    content_html=content_html,
                )

            # 4b. PostgreSQL — store tables
            dfs = dataframes_map.get(idx, [])
            if label == "Table" and dfs:
                element_result["backends"]["postgres"] = self._store_to_postgres(
                    dataframes=dfs,
                    document_name=document_name,
                    page_number=page_number,
                    element_index=idx,
                    element_label=label,
                )

            page_result["elements"].append(element_result)

        return page_result

    # ==================================================================
    # TABLE EXTRACTION MODE: process tables grouped by page
    # ==================================================================
    def process_tables_by_page(
        self,
        tables: List[Dict[str, Any]],
        document_name: str,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process extracted tables grouped by page number.

        Each table dict should have:
          - image: PIL Image (cropped table)
          - raw_text: str (VLM output)
          - raw_html: str (VLM output)
          - dataframes: list of DataFrames
          - page_number: int
          - element_index: int
          - bbox: list

        KG and embedding are done once per page (aggregating all table text).

        Returns dict mapping page_number → page result.
        """
        # Group tables by page
        pages: Dict[int, List[Dict[str, Any]]] = {}
        for table in tables:
            pn = table.get("page_number", 1)
            pages.setdefault(pn, []).append(table)

        results = {}

        for page_number, page_tables in sorted(pages.items()):
            logger.info(
                "Processing %d table(s) from page %d of '%s'",
                len(page_tables), page_number, document_name,
            )

            page_result = {
                "page_number": page_number,
                "document_name": document_name,
                "page_kg": {},
                "page_vectors": {},
                "elements": [],
            }

            # Collect all table text for page-level KG + embedding
            page_text_parts = []
            for table in page_tables:
                raw_text = table.get("raw_text", "").strip()
                if raw_text:
                    page_text_parts.append(f"[Table] {raw_text}")

                # Also serialize DataFrames
                for df in table.get("dataframes", []):
                    if df is not None and not df.empty:
                        cols = ", ".join(str(c) for c in df.columns)
                        rows_text = "; ".join(
                            " | ".join(f"{col}: {val}" for col, val in row.items())
                            for _, row in df.head(20).iterrows()
                        )
                        page_text_parts.append(f"Columns: {cols}. Rows: {rows_text}")

            full_page_text = "\n\n".join(page_text_parts)

            # Page-level KG (ONE Ollama call for all tables on this page)
            if full_page_text and len(full_page_text.strip()) > 20:
                page_result["page_kg"] = self._store_to_kg(
                    text=full_page_text,
                    document_name=document_name,
                    page_number=page_number,
                )
            else:
                page_result["page_kg"] = {"status": "skipped", "reason": "No text"}

            # Page-level embedding (ONE embedding call for all tables on this page)
            if full_page_text and len(full_page_text.strip()) > 10:
                page_result["page_vectors"] = self._store_to_vector_db(
                    text=full_page_text,
                    document_name=document_name,
                    page_number=page_number,
                    element_label="Page-Tables",
                    element_index=0,
                )
            else:
                page_result["page_vectors"] = {"status": "skipped", "reason": "No text"}

            # Per-table: PostgreSQL + MinIO
            for table in page_tables:
                element_result = {
                    "label": "Table",
                    "element_index": table.get("element_index", 0),
                    "page_number": page_number,
                    "backends": {},
                }

                # Object Store
                image = table.get("image")
                if image is not None:
                    bbox = table.get("bbox", [0, 0, image.width, image.height])
                    element_result["backends"]["object_store"] = self._store_to_object_store(
                        image=image,
                        document_name=document_name,
                        page_number=page_number,
                        element_index=table.get("element_index", 0),
                        element_label="Table",
                        bbox=bbox,
                        extracted_text=table.get("raw_text", ""),
                        content_html=table.get("raw_html", ""),
                    )

                # PostgreSQL
                dfs = table.get("dataframes", [])
                if dfs:
                    element_result["backends"]["postgres"] = self._store_to_postgres(
                        dataframes=dfs,
                        document_name=document_name,
                        page_number=page_number,
                        element_index=table.get("element_index", 0),
                        element_label="Table",
                    )

                page_result["elements"].append(element_result)

            results[page_number] = page_result

        return results

    # ------------------------------------------------------------------
    # Backend-specific storage methods (error-isolated)
    # ------------------------------------------------------------------
    def _store_to_object_store(
        self,
        image: Image.Image,
        document_name: str,
        page_number: int,
        element_index: int,
        element_label: str,
        bbox: list,
        extracted_text: str,
        content_html: str,
    ) -> dict:
        """Store image + reference in MinIO."""
        try:
            handler = self.obj
            if handler is None:
                return {"status": "error", "error": self._init_errors.get("obj", "Not initialized")}

            result = handler.store_element(
                image=image,
                document_name=document_name,
                page_number=page_number,
                element_index=element_index,
                element_label=element_label,
                bbox=bbox,
                extracted_text=extracted_text,
                content_html=content_html,
            )
            return {"status": "ok", **result}
        except Exception as exc:
            logger.error("Object store failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _store_to_postgres(
        self,
        dataframes: List[pd.DataFrame],
        document_name: str,
        page_number: int,
        element_index: int,
        element_label: str,
    ) -> dict:
        """Store parsed tables in PostgreSQL."""
        try:
            handler = self.pg
            if handler is None:
                return {"status": "error", "error": self._init_errors.get("pg", "Not initialized")}

            table_ids = []
            for df in dataframes:
                if df is not None and not df.empty:
                    tid = handler.store_table(
                        df=df,
                        document_name=document_name,
                        page_number=page_number,
                        element_index=element_index,
                        element_label=element_label,
                    )
                    table_ids.append(tid)

            return {"status": "ok", "table_ids": table_ids, "count": len(table_ids)}
        except Exception as exc:
            logger.error("PostgreSQL storage failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _store_to_kg(
        self,
        text: str,
        document_name: str,
        page_number: int,
    ) -> dict:
        """Extract triples and store in ArangoDB (ONE call for the full text)."""
        try:
            handler = self.kg
            if handler is None:
                return {"status": "error", "error": self._init_errors.get("kg", "Not initialized")}

            triples = handler.text_to_knowledge_graph(
                text=text,
                document_name=document_name,
                page_number=page_number,
            )
            return {
                "status": "ok",
                "triples_extracted": len(triples),
                "triples": triples[:10],  # Return first 10 for display
            }
        except Exception as exc:
            logger.error("Knowledge graph storage failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _store_to_vector_db(
        self,
        text: str,
        document_name: str,
        page_number: int,
        element_label: str,
        element_index: int,
    ) -> dict:
        """Embed text and store in Qdrant (ONE call for the full text)."""
        try:
            handler = self.vec
            if handler is None:
                return {"status": "error", "error": self._init_errors.get("vec", "Not initialized")}

            count = handler.embed_and_store(
                text=text,
                document_name=document_name,
                page_number=page_number,
                element_label=element_label,
                element_index=element_index,
            )
            return {"status": "ok", "vectors_stored": count}
        except Exception as exc:
            logger.error("Vector DB storage failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_all_stats(self) -> Dict[str, Any]:
        """Get stats from all backends."""
        stats = {}

        if self.pg:
            try:
                stats["postgres"] = {"status": "connected", "tables": len(self.pg.list_tables())}
            except Exception as exc:
                stats["postgres"] = {"status": "error", "error": str(exc)}
        else:
            stats["postgres"] = {"status": "not_initialized", "error": self._init_errors.get("pg", "")}

        if self.kg:
            try:
                stats["knowledge_graph"] = self.kg.get_graph_stats()
                stats["knowledge_graph"]["status"] = "connected"
            except Exception as exc:
                stats["knowledge_graph"] = {"status": "error", "error": str(exc)}
        else:
            stats["knowledge_graph"] = {"status": "not_initialized", "error": self._init_errors.get("kg", "")}

        if self.vec:
            try:
                stats["vector_db"] = self.vec.get_stats()
                stats["vector_db"]["status"] = "connected"
            except Exception as exc:
                stats["vector_db"] = {"status": "error", "error": str(exc)}
        else:
            stats["vector_db"] = {"status": "not_initialized", "error": self._init_errors.get("vec", "")}

        if self.obj:
            try:
                stats["object_store"] = self.obj.get_stats()
                stats["object_store"]["status"] = "connected"
            except Exception as exc:
                stats["object_store"] = {"status": "error", "error": str(exc)}
        else:
            stats["object_store"] = {"status": "not_initialized", "error": self._init_errors.get("obj", "")}

        return stats
