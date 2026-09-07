"""
kg_handler.py
=============
Knowledge Graph builder using Ollama (local LLM) for triple extraction
and ArangoDB for graph storage.

Pipeline:
  1. Text chunks → Ollama LLM → subject-predicate-object triples
  2. Triples → ArangoDB graph (nodes + edges)

Uses the NVIDIA txt2kg pattern: Ollama for extraction, ArangoDB for storage.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL = "llama3.1:8b"

ARANGO_URL = "http://arangodb:8529"
ARANGO_DB = "extraction_kg"
ARANGO_GRAPH = "knowledge_graph"
ARANGO_NODE_COLLECTION = "entities"
ARANGO_EDGE_COLLECTION = "relationships"

# ---------------------------------------------------------------------------
# Triple extraction prompt
# ---------------------------------------------------------------------------
TRIPLE_EXTRACTION_PROMPT = """You are a knowledge graph builder specializing in Arabic and English financial documents.

Extract subject-predicate-object triples from the following text. Focus on:
- Financial entities (companies, accounts, amounts, dates, percentages)
- Relationships between entities (owns, reports, increased, decreased, equals)
- Hierarchical structures (parent-child, category-subcategory)

Return ONLY a valid JSON array of objects with keys: "subject", "predicate", "object".
Do NOT include any explanation or markdown formatting.

Example output:
[{{"subject": "Net Income", "predicate": "increased_by", "object": "15%"}}, {{"subject": "Company A", "predicate": "owns", "object": "Company B"}}]

Text to extract from:
{text}
"""


class KnowledgeGraphHandler:
    """Manages triple extraction via Ollama and storage in ArangoDB."""

    def __init__(
        self,
        ollama_url: str = OLLAMA_BASE_URL,
        ollama_model: str = OLLAMA_MODEL,
        arango_url: str = ARANGO_URL,
        arango_db: str = ARANGO_DB,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.arango_url = arango_url.rstrip("/")
        self.arango_db = arango_db
        self._ensure_arango_db()

    # ------------------------------------------------------------------
    # ArangoDB setup
    # ------------------------------------------------------------------
    def _arango_request(
        self, method: str, path: str, data: Any = None, db: Optional[str] = None
    ) -> Optional[dict]:
        """Make a request to the ArangoDB HTTP API."""
        base = f"{self.arango_url}/_db/{db or self.arango_db}"
        url = f"{base}{path}"
        try:
            resp = requests.request(
                method, url, json=data, timeout=30,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code < 400:
                return resp.json()
            else:
                logger.warning("ArangoDB %s %s -> %d: %s", method, path, resp.status_code, resp.text[:200])
                return None
        except Exception as exc:
            logger.error("ArangoDB request failed: %s", exc)
            return None

    def _ensure_arango_db(self):
        """Create the database, collections, and graph if they don't exist."""
        try:
            requests.post(
                f"{self.arango_url}/_api/database",
                json={"name": self.arango_db},
                timeout=10,
            )
        except Exception:
            pass

        self._arango_request("POST", "/_api/collection", {
            "name": ARANGO_NODE_COLLECTION, "type": 2,
        })
        self._arango_request("POST", "/_api/collection", {
            "name": ARANGO_EDGE_COLLECTION, "type": 3,
        })
        self._arango_request("POST", "/_api/gharial", {
            "name": ARANGO_GRAPH,
            "edgeDefinitions": [{
                "collection": ARANGO_EDGE_COLLECTION,
                "from": [ARANGO_NODE_COLLECTION],
                "to": [ARANGO_NODE_COLLECTION],
            }],
        })
        logger.info("ArangoDB graph '%s' ensured in db '%s'.", ARANGO_GRAPH, self.arango_db)

    # ------------------------------------------------------------------
    # Ollama triple extraction
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call Ollama's chat completion API."""
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge graph builder. Extract subject-predicate-object "
                        "triples from text and return them as a JSON array. Return ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
            return result.get("message", {}).get("content", "")
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            return ""

    def extract_triples(self, text: str) -> List[Dict[str, str]]:
        """Extract knowledge triples from text using Ollama."""
        if not text or len(text.strip()) < 10:
            return []

        prompt = TRIPLE_EXTRACTION_PROMPT.format(text=text)
        raw = self._call_ollama(prompt)
        if not raw:
            return []

        triples = self._parse_triples(raw)
        logger.info("Extracted %d triples from text (%d chars).", len(triples), len(text))
        return triples

    def _parse_triples(self, raw: str) -> List[Dict[str, str]]:
        """Parse triples from LLM response, handling various formats."""

        def _normalize_triple(t: dict) -> Optional[Dict[str, str]]:
            """Normalize a dict into a standard triple, or return None."""
            if not isinstance(t, dict):
                return None
            normalized = {}
            for k, v in t.items():
                normalized[k.lower().strip()] = str(v).strip() if v is not None else ""
            s = normalized.get("subject", "")
            p = normalized.get("predicate", "") or normalized.get("relation", "") or normalized.get("relationship", "")
            o = normalized.get("object", "")
            if s and p and o:
                return {"subject": s, "predicate": p, "object": o}
            return None

        # Attempt 1: Find a JSON array in the response
        try:
            json_match = re.search(r'\[[\s\S]*?\]', raw)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    valid = [_normalize_triple(t) for t in parsed]
                    valid = [t for t in valid if t is not None]
                    if valid:
                        return valid
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("JSON array parse failed: %s", exc)

        # Attempt 2: Find individual JSON objects
        triples = []
        for obj_match in re.finditer(r'\{[^{}]+\}', raw):
            try:
                obj = json.loads(obj_match.group())
                t = _normalize_triple(obj)
                if t:
                    triples.append(t)
            except (json.JSONDecodeError, TypeError):
                pass
        if triples:
            return triples

        # Attempt 3: Line-by-line text parsing (Subject - Predicate - Object)
        for line in raw.split("\n"):
            line = line.strip()
            if not line or len(line) < 5:
                continue
            parts = re.split(r'\s*[\-\|→]\s*', line.lstrip("*-0123456789. "))
            if len(parts) >= 3:
                s, p, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if s and p and o:
                    triples.append({"subject": s, "predicate": p, "object": o})

        return triples

    # ------------------------------------------------------------------
    # Store triples in ArangoDB
    # ------------------------------------------------------------------
    def _node_key(self, name: str) -> str:
        """Generate a deterministic key for a node."""
        clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', name.strip())
        if not clean or clean == '_':
            clean = f"node_{uuid.uuid4().hex[:8]}"
        return clean[:250]

    def _ensure_node(self, name: str, metadata: Optional[dict] = None) -> str:
        """Create or update a node in the entities collection."""
        key = self._node_key(name)
        doc = {"_key": key, "name": name, **(metadata or {})}

        result = self._arango_request(
            "POST", f"/_api/document/{ARANGO_NODE_COLLECTION}", doc,
        )
        if result is None:
            self._arango_request(
                "PATCH", f"/_api/document/{ARANGO_NODE_COLLECTION}/{key}", doc,
            )
        return f"{ARANGO_NODE_COLLECTION}/{key}"

    def store_triples(
        self, triples: List[Dict[str, str]],
        document_name: str = "", page_number: int = 1,
    ) -> int:
        """Store extracted triples in ArangoDB. Returns edges created."""
        stored = 0
        for triple in triples:
            subj = triple.get("subject", "").strip()
            pred = triple.get("predicate", "").strip()
            obj = triple.get("object", "").strip()
            if not subj or not pred or not obj:
                continue

            from_id = self._ensure_node(subj, {"type": "entity"})
            to_id = self._ensure_node(obj, {"type": "entity"})

            edge_key = f"{self._node_key(subj)}__{self._node_key(pred)}__{self._node_key(obj)}"
            edge_doc = {
                "_key": edge_key[:250],
                "_from": from_id, "_to": to_id,
                "predicate": pred,
                "document_name": document_name,
                "page_number": page_number,
            }
            result = self._arango_request(
                "POST", f"/_api/document/{ARANGO_EDGE_COLLECTION}", edge_doc,
            )
            if result:
                stored += 1

        logger.info("Stored %d/%d triples in ArangoDB.", stored, len(triples))
        return stored

    # ------------------------------------------------------------------
    # Combined: extract + store
    # ------------------------------------------------------------------
    def text_to_knowledge_graph(
        self, text: str, document_name: str = "", page_number: int = 1,
    ) -> List[Dict[str, str]]:
        """End-to-end: extract triples from text and store in ArangoDB."""
        triples = self.extract_triples(text)
        if triples:
            self.store_triples(triples, document_name, page_number)
        return triples

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_graph_stats(self) -> dict:
        """Get basic stats about the knowledge graph."""
        node_count = self._arango_request(
            "POST", "/_api/cursor",
            {"query": f"RETURN LENGTH({ARANGO_NODE_COLLECTION})"},
        )
        edge_count = self._arango_request(
            "POST", "/_api/cursor",
            {"query": f"RETURN LENGTH({ARANGO_EDGE_COLLECTION})"},
        )
        return {
            "nodes": node_count.get("result", [0])[0] if node_count else 0,
            "edges": edge_count.get("result", [0])[0] if edge_count else 0,
            "graph": ARANGO_GRAPH,
            "database": self.arango_db,
        }

    def query_entity(self, entity_name: str, depth: int = 1) -> dict:
        """Query relationships for a given entity."""
        key = self._node_key(entity_name)
        start_vertex = f"{ARANGO_NODE_COLLECTION}/{key}"
        aql = (
            f"FOR v, e, p IN 1..{depth} ANY @start "
            f"GRAPH '{ARANGO_GRAPH}' "
            f"RETURN {{\"from\": p.vertices[0].name, \"predicate\": e.predicate, \"to\": p.vertices[1].name}}"
        )
        result = self._arango_request(
            "POST", "/_api/cursor",
            {"query": aql, "bindVars": {"start": start_vertex}},
        )
        return {
            "entity": entity_name,
            "relationships": result.get("result", []) if result else [],
        }
