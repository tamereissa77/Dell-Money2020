"""
retrieval_agent.py
==================
Agentic Router Pattern for Arabic Financial Document Retrieval.

Architecture:
    Query → QueryRouter → [SQLAgent | GraphAgent | VectorAgent] → MasterRetriever → Answer + Visual Evidence

This module implements a hybrid retrieval system that intelligently routes
user questions to the correct storage backend:
    - SQL_MODE   → PostgreSQL (exact numbers, tables, aggregations)
    - GRAPH_MODE → ArangoDB   (entity relationships, ownership, board members)
    - VECTOR_MODE→ Qdrant     (narrative context, risks, strategies)
    - HYBRID_MODE→ SQL + Vector (numbers with explanations)

Each answer includes MinIO image URLs for visual evidence / auditability.

Design decisions:
    - No LangChain: Pure Python for full control over Arabic prompts.
    - Fallback chain: SQL → Vector if SQL generation fails.
    - Read-only SQL: LLM-generated queries are sandboxed.
    - Visual evidence: Every answer includes MinIO image URLs.
"""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SearchParams,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("retrieval_agent")


# ---------------------------------------------------------------------------
# Configuration — all from environment variables with sane defaults
# ---------------------------------------------------------------------------

@dataclass
class ServiceConfig:
    """Centralised connection config — reads from env vars at init time."""

    # Ollama LLM
    ollama_url: str = ""
    ollama_model: str = ""

    # Qdrant Vector DB
    qdrant_url: str = ""
    qdrant_collection: str = ""

    # ArangoDB Knowledge Graph
    arangodb_url: str = ""
    arangodb_db: str = ""
    arangodb_graph: str = ""
    arangodb_node_collection: str = ""
    arangodb_edge_collection: str = ""

    # PostgreSQL
    pg_host: str = ""
    pg_port: int = 5432
    pg_db: str = ""
    pg_user: str = ""
    pg_password: str = ""

    # MinIO Object Store
    minio_endpoint: str = ""
    minio_public_url: str = ""
    minio_bucket: str = ""

    # Sentence Transformers Embedding Service
    sentence_transformer_url: str = ""

    def __post_init__(self):
        """Populate from environment variables, falling back to defaults."""
        self.ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

        self.qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_collection = os.environ.get("QDRANT_COLLECTION", "document-embeddings")

        self.arangodb_url = os.environ.get("ARANGO_URL", "http://arangodb:8529")
        self.arangodb_db = os.environ.get("ARANGO_DB", "extraction_kg")
        self.arangodb_graph = os.environ.get("ARANGO_GRAPH", "knowledge_graph")
        self.arangodb_node_collection = os.environ.get("ARANGO_NODE_COLLECTION", "entities")
        self.arangodb_edge_collection = os.environ.get("ARANGO_EDGE_COLLECTION", "relationships")

        self.pg_host = os.environ.get("POSTGRES_HOST", "postgres")
        self.pg_port = int(os.environ.get("POSTGRES_PORT", "5432"))
        self.pg_db = os.environ.get("POSTGRES_DB", "extraction_db")
        self.pg_user = os.environ.get("POSTGRES_USER", "extract_user")
        self.pg_password = os.environ.get("POSTGRES_PASSWORD", "extract_pass")

        self.minio_endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
        self.minio_public_url = os.environ.get("MINIO_PUBLIC_URL", "http://localhost:9000")
        self.minio_bucket = os.environ.get("MINIO_BUCKET", "extracted-elements")

        self.sentence_transformer_url = os.environ.get(
            "SENTENCE_TRANSFORMER_URL", "http://sentence-transformers:80"
        )


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class QueryIntent(str, Enum):
    SQL_MODE = "SQL_MODE"
    GRAPH_MODE = "GRAPH_MODE"
    VECTOR_MODE = "VECTOR_MODE"
    HYBRID_MODE = "HYBRID_MODE"


@dataclass
class RoutingDecision:
    """Output of the QueryRouter."""
    intent: QueryIntent
    extracted_entities: List[str]
    reasoning: str
    confidence: float  # 0.0 – 1.0
    original_query: str


@dataclass
class VisualEvidence:
    """MinIO image URL(s) that back the answer."""
    image_urls: List[str] = field(default_factory=list)
    document_name: str = ""
    page_numbers: List[int] = field(default_factory=list)
    bounding_boxes: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_urls": self.image_urls,
            "document_name": self.document_name,
            "page_numbers": self.page_numbers,
            "bounding_boxes": self.bounding_boxes,
        }


@dataclass
class AgentResult:
    """Uniform result returned by every agent."""
    success: bool
    data: Any
    data_text: str
    visual_evidence: VisualEvidence
    agent_name: str
    execution_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "data_text": self.data_text,
            "visual_evidence": self.visual_evidence.to_dict(),
            "agent_name": self.agent_name,
            "execution_time": self.execution_time,
            "error": self.error,
        }


@dataclass
class RetrievalResponse:
    """Final response returned to the caller."""
    answer: str
    intent_used: QueryIntent
    visual_evidence: VisualEvidence
    raw_data: Any
    routing_reasoning: str
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "intent_used": self.intent_used.value,
            "visual_evidence": self.visual_evidence.to_dict(),
            "routing_reasoning": self.routing_reasoning,
            "total_time": round(self.total_time, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Shared Helper Clients
# ═══════════════════════════════════════════════════════════════════════════

class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(self, config: ServiceConfig):
        self.url = config.ollama_url.rstrip("/")
        self.model = config.ollama_model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        timeout: int = 90,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


class EmbeddingClient:
    """Talks to the sentence-transformers microservice."""

    def __init__(self, config: ServiceConfig):
        self.url = config.sentence_transformer_url.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            f"{self.url}/embed",
            json={"texts": texts, "batch_size": 32},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════
# 1. QueryRouter — Intent Classification
# ═══════════════════════════════════════════════════════════════════════════

class QueryRouter:
    """
    Uses Llama 3.1 to classify the user query into one of four intents:
    SQL_MODE, GRAPH_MODE, VECTOR_MODE, or HYBRID_MODE.
    """

    SYSTEM_PROMPT = """\
You are a query classifier for an Arabic financial document retrieval system.
You classify user questions into exactly ONE of four intents.

INTENTS:
- SQL_MODE: Questions about specific numbers, totals, comparisons, dates,
  financial figures, table lookups, aggregations.
  Examples: "ما هو صافي الدخل في 2023؟", "What was total revenue?",
  "Compare assets between 2022 and 2023", "كم بلغت الأرباح؟"

- GRAPH_MODE: Questions about relationships, ownership, board members,
  subsidiaries, parent companies, organisational structure.
  Examples: "من يملك الشركة؟", "Who are the board members?",
  "What subsidiaries does X have?", "ما هي الشركات التابعة؟"

- VECTOR_MODE: Questions about narratives, strategies, risks, policies,
  descriptions, summaries, qualitative information.
  Examples: "ما هي المخاطر الرئيسية؟", "Summarise the audit opinion.",
  "What is the company's strategy?", "ما هي السياسات المحاسبية؟"

- HYBRID_MODE: Questions that need BOTH a precise number AND an explanation.
  Examples: "Why did revenue decline in 2023?", "Explain the increase in provisions.",
  "لماذا انخفض صافي الربح مقارنة بالعام السابق؟"

RULES:
1. Return ONLY valid JSON — no markdown, no explanation outside the JSON.
2. Extract key entities (company names, years, financial terms) from the query.
   Handle Arabic morphology: الشركة / شركة / للشركة are all "company".
   Handle Arabic financial terms: صافي الدخل = Net Income, الإيرادات = Revenue.
3. Provide a confidence score from 0.0 to 1.0.

OUTPUT FORMAT (strict JSON):
{
  "intent": "SQL_MODE",
  "extracted_entities": ["entity1", "entity2"],
  "reasoning": "Brief explanation of classification",
  "confidence": 0.92
}"""

    def __init__(self, llm: OllamaClient):
        self.llm = llm

    def route(self, query: str) -> RoutingDecision:
        """Classify a user query into an intent."""
        raw = self.llm.generate(
            prompt=f"Classify this query:\n\n{query}",
            system=self.SYSTEM_PROMPT,
            temperature=0.05,
        )

        parsed = self._parse_json(raw)

        intent_str = parsed.get("intent", "VECTOR_MODE").upper().strip()
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            logger.warning("Unknown intent '%s', falling back to VECTOR_MODE", intent_str)
            intent = QueryIntent.VECTOR_MODE

        return RoutingDecision(
            intent=intent,
            extracted_entities=parsed.get("extracted_entities", []),
            reasoning=parsed.get("reasoning", ""),
            confidence=float(parsed.get("confidence", 0.5)),
            original_query=query,
        )

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Extract the first JSON object from LLM output, robustly."""
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find JSON between braces
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error("Failed to parse router JSON: %.300s", text)
        return {
            "intent": "VECTOR_MODE",
            "extracted_entities": [],
            "reasoning": "JSON parse failure — defaulting to VECTOR_MODE",
            "confidence": 0.3,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. SQLAgent — 100% Numerical Accuracy
# ═══════════════════════════════════════════════════════════════════════════

class SQLAgent:
    """
    Generates and executes read-only SQL against PostgreSQL.

    Safety:
        - Read-only connection (set_session readonly=True).
        - Only SELECT statements are allowed (regex blocklist).
        - Schema is fetched dynamically and fed to the LLM.
    """

    SQL_SYSTEM_PROMPT = """\
You are an expert SQL generator for a PostgreSQL database that stores
financial table data extracted from Arabic annual reports.

DATABASE DESIGN:
- Table "extracted_tables" holds table metadata:
  id (UUID), document_name, page_number, element_index, element_label,
  column_headers (JSONB array of header names), row_count, created_at.
- Table "extracted_table_rows" holds actual data:
  id, table_id (FK), row_index, row_data (JSONB object — keys are column headers).

To query actual data you almost always need to JOIN these two tables.
Example: SELECT t.document_name, t.page_number, r.row_data
         FROM extracted_tables t
         JOIN extracted_table_rows r ON r.table_id = t.id
         WHERE t.document_name ILIKE '%annual%';

To query a specific field inside row_data use:  r.row_data->>'column_name'
To cast to numeric:  (r.row_data->>'column_name')::numeric

RULES:
1. Generate ONLY a single SELECT statement. No INSERT, UPDATE, DELETE, DROP, ALTER.
2. Return ONLY raw SQL — no markdown fences, no explanation, no comments.
3. Use ILIKE for text matching to handle Arabic case/form variations.
4. Always include t.document_name and t.page_number for source citation.
5. For aggregations (SUM, AVG), still GROUP BY t.document_name when meaningful.
6. Limit results to 50 rows unless the user asks for all.
7. Handle Arabic column names by using the row_data JSONB accessor.
"""

    _FORBIDDEN = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|EXECUTE)\b",
        re.IGNORECASE,
    )

    def __init__(self, config: ServiceConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm

    def execute(self, query: str, entities: List[str]) -> AgentResult:
        """Generate SQL, execute it, and return structured results."""
        t0 = time.time()
        evidence = VisualEvidence()

        try:
            schema = self._fetch_schema()
            if not schema:
                return AgentResult(
                    success=False, data=None, data_text="",
                    visual_evidence=evidence, agent_name="SQLAgent",
                    execution_time=time.time() - t0,
                    error="No tables found in PostgreSQL.",
                )

            # Also fetch sample data to help the LLM understand column names
            sample = self._fetch_sample_data()

            sql = self._generate_sql(query, entities, schema, sample)
            logger.info("SQLAgent generated SQL:\n%s", sql)

            rows, columns = self._run_sql(sql)
            data_text, evidence = self._format_results(rows, columns)

            return AgentResult(
                success=True,
                data={"columns": columns, "rows": rows, "sql": sql},
                data_text=data_text,
                visual_evidence=evidence,
                agent_name="SQLAgent",
                execution_time=time.time() - t0,
            )

        except Exception as exc:
            logger.exception("SQLAgent failed")
            return AgentResult(
                success=False, data=None, data_text="",
                visual_evidence=evidence, agent_name="SQLAgent",
                execution_time=time.time() - t0,
                error=str(exc),
            )

    # ---- Connection -------------------------------------------------------

    def _get_connection(self):
        conn = psycopg2.connect(
            host=self.config.pg_host,
            port=self.config.pg_port,
            dbname=self.config.pg_db,
            user=self.config.pg_user,
            password=self.config.pg_password,
        )
        conn.set_session(readonly=True, autocommit=True)
        return conn

    # ---- Schema introspection ---------------------------------------------

    def _fetch_schema(self) -> str:
        """Return a DDL-style description of all public tables."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name, column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    ORDER BY table_name, ordinal_position;
                """)
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return ""

        tables: Dict[str, List[str]] = {}
        for table, col, dtype in rows:
            tables.setdefault(table, []).append(f"  {col} ({dtype})")

        parts = []
        for table, cols in tables.items():
            parts.append(f"TABLE: {table}\n" + "\n".join(cols))
        return "\n\n".join(parts)

    def _fetch_sample_data(self) -> str:
        """Fetch sample column_headers and row_data to show the LLM real column names."""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get sample column headers from extracted_tables
                cur.execute("""
                    SELECT document_name, column_headers
                    FROM extracted_tables
                    ORDER BY created_at DESC
                    LIMIT 5;
                """)
                tables = cur.fetchall()

                # Get sample row_data keys
                cur.execute("""
                    SELECT DISTINCT jsonb_object_keys(row_data) AS key
                    FROM extracted_table_rows
                    LIMIT 30;
                """)
                keys = [r["key"] for r in cur.fetchall()]
        finally:
            conn.close()

        parts = []
        if tables:
            parts.append("SAMPLE TABLE METADATA:")
            for t in tables:
                parts.append(f"  Document: {t['document_name']}, Headers: {t['column_headers']}")
        if keys:
            parts.append(f"\nAVAILABLE ROW_DATA KEYS: {json.dumps(keys, ensure_ascii=False)}")

        return "\n".join(parts) if parts else ""

    # ---- SQL generation ---------------------------------------------------

    def _generate_sql(
        self, query: str, entities: List[str], schema: str, sample: str
    ) -> str:
        prompt = (
            f"DATABASE SCHEMA:\n{schema}\n\n"
        )
        if sample:
            prompt += f"SAMPLE DATA:\n{sample}\n\n"
        prompt += (
            f"DETECTED ENTITIES: {json.dumps(entities, ensure_ascii=False)}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Generate the SQL SELECT query:"
        )
        raw = self.llm.generate(prompt, system=self.SQL_SYSTEM_PROMPT, temperature=0.0)

        # Strip markdown fences
        sql = re.sub(r"```(?:sql)?", "", raw).strip().rstrip(";") + ";"

        # Safety: refuse dangerous statements
        if self._FORBIDDEN.search(sql):
            raise ValueError(f"Refused dangerous SQL: {sql[:200]}")

        # Basic sanity: must start with SELECT
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError(f"Generated query is not a SELECT: {sql[:100]}")

        return sql

    # ---- Execution --------------------------------------------------------

    def _run_sql(self, sql: str) -> Tuple[List[Dict], List[str]]:
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchall()]
                columns = [desc[0] for desc in cur.description] if cur.description else []
        finally:
            conn.close()
        return rows, columns

    # ---- Formatting -------------------------------------------------------

    def _format_results(
        self, rows: List[Dict], columns: List[str]
    ) -> Tuple[str, VisualEvidence]:
        evidence = VisualEvidence()

        if not rows:
            return "Query returned no results.", evidence

        # Build human-readable table
        lines = [" | ".join(str(c) for c in columns)]
        lines.append("-" * len(lines[0]))

        for row in rows:
            vals = [str(row.get(c, "")) for c in columns]
            lines.append(" | ".join(vals))

            # Collect visual evidence from result metadata
            doc = row.get("document_name", "")
            page = row.get("page_number")
            if doc:
                evidence.document_name = doc
            if page is not None:
                evidence.page_numbers.append(int(page))

        # Build MinIO image URLs from document + page info
        evidence.page_numbers = sorted(set(evidence.page_numbers))
        for page_num in evidence.page_numbers:
            # Pattern matches how object_store.py stores images
            image_url = (
                f"{self.config.minio_public_url}/{self.config.minio_bucket}/"
                f"{evidence.document_name}/page_{page_num}"
            )
            evidence.image_urls.append(image_url)

        return "\n".join(lines), evidence


# ═══════════════════════════════════════════════════════════════════════════
# 3. GraphAgent — Relational Accuracy
# ═══════════════════════════════════════════════════════════════════════════

class GraphAgent:
    """
    Queries ArangoDB Knowledge Graph for entity relationships.
    Supports LLM-generated AQL and a simple traversal fallback.
    """

    GRAPH_SYSTEM_PROMPT = """\
You are an AQL (ArangoDB Query Language) expert for a knowledge graph that
stores financial entity relationships as triples (Subject -[Predicate]-> Object).

COLLECTIONS:
- Vertex collection: "entities" with field "name" (string).
- Edge collection: "relationships" with fields: "predicate" (string),
  "document_name" (string), "page_number" (integer).
- Graph name: "knowledge_graph".

RULES:
1. Return ONLY the AQL query — no explanation, no markdown fences.
2. Use graph traversal: FOR v, e, p IN 1..2 ANY start_node GRAPH 'knowledge_graph'.
3. Handle Arabic entity names using == or LIKE for partial matching.
4. RETURN both vertex names and edge predicates.
5. Use CONTAINS(LOWER(node.name), LOWER(@entity)) for fuzzy Arabic matching.
"""

    def __init__(self, config: ServiceConfig, llm: OllamaClient):
        self.config = config
        self.llm = llm
        self._base = config.arangodb_url.rstrip("/")
        self._db = config.arangodb_db

    def execute(self, query: str, entities: List[str]) -> AgentResult:
        """Query the knowledge graph for entity relationships."""
        t0 = time.time()
        evidence = VisualEvidence()

        try:
            # Try LLM-generated AQL first
            triples = self._query_with_llm(query, entities)

            # Fallback to simple traversal
            if not triples:
                logger.info("LLM AQL returned nothing, falling back to simple traversal")
                triples = self._simple_traversal(entities)

            data_text = self._format_triples(triples)

            # Build evidence from triple metadata
            for t in triples:
                doc = t.get("document_name", "")
                page = t.get("page_number")
                if doc and not evidence.document_name:
                    evidence.document_name = doc
                if page is not None:
                    evidence.page_numbers.append(int(page))

            evidence.page_numbers = sorted(set(evidence.page_numbers))

            return AgentResult(
                success=bool(triples),
                data=triples,
                data_text=data_text,
                visual_evidence=evidence,
                agent_name="GraphAgent",
                execution_time=time.time() - t0,
            )
        except Exception as exc:
            logger.exception("GraphAgent failed")
            return AgentResult(
                success=False, data=None, data_text="",
                visual_evidence=evidence, agent_name="GraphAgent",
                execution_time=time.time() - t0,
                error=str(exc),
            )

    # ---- ArangoDB HTTP API ------------------------------------------------

    def _run_aql(self, aql: str, bind_vars: Optional[Dict] = None) -> List[Dict]:
        payload: Dict[str, Any] = {"query": aql}
        if bind_vars:
            payload["bindVars"] = bind_vars

        url = f"{self._base}/_db/{self._db}/_api/cursor"
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code not in (200, 201):
                logger.error("AQL failed (%s): %.300s", resp.status_code, resp.text)
                return []
            return resp.json().get("result", [])
        except requests.RequestException as exc:
            logger.error("ArangoDB request failed: %s", exc)
            return []

    # ---- LLM-generated AQL -----------------------------------------------

    def _query_with_llm(self, query: str, entities: List[str]) -> List[Dict]:
        prompt = (
            f"ENTITIES: {json.dumps(entities, ensure_ascii=False)}\n"
            f"USER QUESTION: {query}\n\n"
            f"Generate the AQL query:"
        )
        raw = self.llm.generate(prompt, system=self.GRAPH_SYSTEM_PROMPT, temperature=0.0)

        # Clean up markdown fences
        aql = re.sub(r"```(?:aql)?", "", raw).strip()
        logger.info("GraphAgent generated AQL:\n%s", aql)

        if not aql or len(aql) < 10:
            return []

        return self._run_aql(aql)

    # ---- Simple traversal fallback ----------------------------------------

    def _simple_traversal(self, entities: List[str], max_depth: int = 2) -> List[Dict]:
        """Traverse from each entity up to max_depth hops."""
        all_results: List[Dict] = []

        for entity in entities:
            aql = """
            LET starts = (
                FOR node IN entities
                    FILTER CONTAINS(LOWER(node.name), LOWER(@entity))
                    LIMIT 3
                    RETURN node
            )
            FOR start IN starts
                FOR v, e, p IN 1..@depth ANY start GRAPH @graph
                    LIMIT 50
                    RETURN DISTINCT {
                        subject: p.vertices[0].name,
                        predicate: e.predicate,
                        object: p.vertices[-1].name,
                        document_name: e.document_name,
                        page_number: e.page_number
                    }
            """
            results = self._run_aql(aql, {
                "entity": entity,
                "depth": max_depth,
                "graph": self.config.arangodb_graph,
            })
            all_results.extend(results)

        # Deduplicate
        seen = set()
        unique = []
        for t in all_results:
            key = (t.get("subject"), t.get("predicate"), t.get("object"))
            if key not in seen:
                seen.add(key)
                unique.append(t)

        return unique

    # ---- Formatting -------------------------------------------------------

    @staticmethod
    def _format_triples(triples: List[Dict]) -> str:
        if not triples:
            return "No relationships found in the knowledge graph."

        lines = ["Knowledge Graph Relationships:"]
        for t in triples:
            s = t.get("subject", "?")
            p = t.get("predicate", "?")
            o = t.get("object", "?")
            doc = t.get("document_name", "")
            page = t.get("page_number", "")
            source = f"  [source: {doc} p.{page}]" if doc else ""
            lines.append(f"  • {s} —[{p}]→ {o}{source}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 4. VectorAgent — Narrative Context
# ═══════════════════════════════════════════════════════════════════════════

class VectorAgent:
    """
    Dense semantic search in Qdrant for narrative / qualitative content.
    Returns text chunks with MinIO image URLs for visual evidence.
    """

    def __init__(self, config: ServiceConfig, embedder: EmbeddingClient):
        self.config = config
        self.embedder = embedder
        self.client = QdrantClient(url=config.qdrant_url, timeout=30)
        self.collection = config.qdrant_collection

    def execute(
        self,
        query: str,
        entities: List[str],
        top_k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Perform dense vector search in Qdrant."""
        t0 = time.time()
        evidence = VisualEvidence()

        try:
            query_vector = self.embedder.embed_single(query)

            qdrant_filter = self._build_filter(filters) if filters else None

            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
                search_params=SearchParams(hnsw_ef=128, exact=False),
            )

            chunks: List[Dict[str, Any]] = []
            for hit in results:
                payload = hit.payload or {}
                chunk = {
                    "text": payload.get("text", ""),
                    "score": round(hit.score, 4),
                    "document_name": payload.get("document_name", ""),
                    "page_number": payload.get("page_number", 0),
                    "element_type": payload.get("element_label", "text"),
                    "element_index": payload.get("element_index", 0),
                }
                chunks.append(chunk)

                # Collect evidence
                doc = chunk["document_name"]
                page = chunk["page_number"]
                if doc:
                    evidence.document_name = doc
                if page:
                    evidence.page_numbers.append(int(page))

            evidence.page_numbers = sorted(set(evidence.page_numbers))

            # Build MinIO image URLs
            for chunk in chunks:
                if chunk["document_name"] and chunk["page_number"]:
                    image_url = (
                        f"{self.config.minio_public_url}/{self.config.minio_bucket}/"
                        f"{chunk['document_name']}/page_{chunk['page_number']}"
                    )
                    if image_url not in evidence.image_urls:
                        evidence.image_urls.append(image_url)

            data_text = self._format_chunks(chunks)

            return AgentResult(
                success=bool(chunks),
                data=chunks,
                data_text=data_text,
                visual_evidence=evidence,
                agent_name="VectorAgent",
                execution_time=time.time() - t0,
            )
        except Exception as exc:
            logger.exception("VectorAgent failed")
            return AgentResult(
                success=False, data=None, data_text="",
                visual_evidence=evidence, agent_name="VectorAgent",
                execution_time=time.time() - t0,
                error=str(exc),
            )

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> Optional[Filter]:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _format_chunks(chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant text chunks found."

        lines = ["Retrieved Text Chunks:"]
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"\n--- Chunk {i} (score: {c['score']}, "
                f"doc: {c['document_name']}, "
                f"page: {c['page_number']}, type: {c['element_type']}) ---\n"
                f"{c['text']}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 5. MasterRetriever — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class MasterRetriever:
    """
    Orchestrates:  Query → Router → Agent(s) → LLM Synthesis → Answer + Evidence.

    Fallback logic:
        SQL_MODE  :  SQLAgent → (on failure) VectorAgent
        GRAPH_MODE:  GraphAgent → (on failure) VectorAgent
        VECTOR_MODE: VectorAgent
        HYBRID_MODE: SQLAgent + VectorAgent (both run, results merged)
    """

    SYNTHESIS_SYSTEM_PROMPT = """\
You are a financial analyst assistant for Arabic annual reports and financial documents.

RULES:
1. Answer the user's question based ONLY on the provided retrieved data.
2. If the data contains exact numbers, quote them precisely — NEVER round or approximate.
3. If the question is in Arabic, respond in Arabic. If in English, respond in English.
   If mixed, match the language of the question.
4. Cite the source document and page number when available.
   Format: (المصدر: اسم_المستند، صفحة X) or (Source: doc_name, page X).
5. If the data is insufficient, say so explicitly — do NOT hallucinate or make up numbers.
6. Format currency values with proper separators.
7. For table data, present results in a clear structured format.
8. When visual evidence is available, mention that the source image can be viewed.
"""

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()

        # Shared clients
        self.llm = OllamaClient(self.config)
        self.embedder = EmbeddingClient(self.config)

        # Agents
        self.router = QueryRouter(self.llm)
        self.sql_agent = SQLAgent(self.config, self.llm)
        self.graph_agent = GraphAgent(self.config, self.llm)
        self.vector_agent = VectorAgent(self.config, self.embedder)

        logger.info(
            "MasterRetriever initialised — Ollama: %s, Qdrant: %s, PG: %s:%s/%s, ArangoDB: %s/%s",
            self.config.ollama_url,
            self.config.qdrant_url,
            self.config.pg_host, self.config.pg_port, self.config.pg_db,
            self.config.arangodb_url, self.config.arangodb_db,
        )

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResponse:
        """
        Main entry point — takes a natural-language query, routes it,
        retrieves data, synthesises an answer, and returns visual evidence.
        """
        t0 = time.time()

        # ── Step 1: Route ─────────────────────────────────────────────
        routing = self.router.route(query)
        logger.info(
            "Router: intent=%s confidence=%.2f entities=%s reason=%s",
            routing.intent.value, routing.confidence,
            routing.extracted_entities, routing.reasoning,
        )

        # Low confidence → upgrade to HYBRID_MODE for safety
        if routing.confidence < 0.6 and routing.intent != QueryIntent.HYBRID_MODE:
            logger.info("Low confidence (%.2f), upgrading to HYBRID_MODE", routing.confidence)
            routing.intent = QueryIntent.HYBRID_MODE

        # ── Step 2: Execute agent(s) ──────────────────────────────────
        agent_results: List[AgentResult] = []

        if routing.intent == QueryIntent.SQL_MODE:
            result = self.sql_agent.execute(query, routing.extracted_entities)
            if not result.success:
                logger.warning("SQL failed (%s), falling back to VectorAgent", result.error)
                result = self.vector_agent.execute(
                    query, routing.extracted_entities, filters=filters
                )
            agent_results.append(result)

        elif routing.intent == QueryIntent.GRAPH_MODE:
            result = self.graph_agent.execute(query, routing.extracted_entities)
            if not result.success:
                logger.warning("Graph failed (%s), falling back to VectorAgent", result.error)
                result = self.vector_agent.execute(
                    query, routing.extracted_entities, filters=filters
                )
            agent_results.append(result)

        elif routing.intent == QueryIntent.VECTOR_MODE:
            result = self.vector_agent.execute(
                query, routing.extracted_entities, filters=filters
            )
            agent_results.append(result)

        elif routing.intent == QueryIntent.HYBRID_MODE:
            # Run both SQL and Vector
            sql_result = self.sql_agent.execute(query, routing.extracted_entities)
            vec_result = self.vector_agent.execute(
                query, routing.extracted_entities, filters=filters
            )
            if sql_result.success:
                agent_results.append(sql_result)
            agent_results.append(vec_result)

        # ── Step 3: Merge evidence ────────────────────────────────────
        merged_evidence = self._merge_evidence(agent_results)
        merged_data_text = "\n\n".join(r.data_text for r in agent_results if r.data_text)

        # ── Step 4: Synthesise answer ─────────────────────────────────
        answer = self._synthesise(query, merged_data_text, routing)

        total_time = time.time() - t0
        logger.info("Retrieval complete in %.2fs (intent=%s)", total_time, routing.intent.value)

        return RetrievalResponse(
            answer=answer,
            intent_used=routing.intent,
            visual_evidence=merged_evidence,
            raw_data=[r.to_dict() for r in agent_results],
            routing_reasoning=routing.reasoning,
            total_time=total_time,
        )

    # ---- Health check for all backends ------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Check connectivity to all backend services."""
        checks = {}

        # Ollama
        checks["ollama"] = self.llm.health_check()

        # Sentence Transformers
        checks["sentence_transformers"] = self.embedder.health_check()

        # Qdrant
        try:
            info = self.vector_agent.client.get_collection(self.config.qdrant_collection)
            checks["qdrant"] = True
            checks["qdrant_vectors"] = info.points_count
        except Exception:
            checks["qdrant"] = False

        # PostgreSQL
        try:
            conn = self.sql_agent._get_connection()
            conn.close()
            checks["postgres"] = True
        except Exception:
            checks["postgres"] = False

        # ArangoDB
        try:
            resp = requests.get(
                f"{self.config.arangodb_url}/_api/version", timeout=5
            )
            checks["arangodb"] = resp.status_code == 200
        except Exception:
            checks["arangodb"] = False

        # MinIO
        try:
            resp = requests.get(
                f"http://{self.config.minio_endpoint}/minio/health/live", timeout=5
            )
            checks["minio"] = resp.status_code == 200
        except Exception:
            checks["minio"] = False

        checks["all_healthy"] = all(
            v for k, v in checks.items()
            if k not in ("qdrant_vectors", "all_healthy") and isinstance(v, bool)
        )
        return checks

    # ---- Internal helpers -------------------------------------------------

    def _synthesise(
        self, query: str, data_text: str, routing: RoutingDecision
    ) -> str:
        """Use Ollama to synthesise a final answer from retrieved data."""
        if not data_text.strip():
            return (
                "لم يتم العثور على بيانات كافية للإجابة على هذا السؤال.\n"
                "No sufficient data found to answer this question."
            )

        prompt = (
            f"USER QUESTION: {query}\n\n"
            f"RETRIEVED DATA:\n{data_text}\n\n"
            f"ROUTING: intent={routing.intent.value}, "
            f"entities={json.dumps(routing.extracted_entities, ensure_ascii=False)}\n\n"
            f"Provide a clear, accurate answer based ONLY on the retrieved data above."
        )

        return self.llm.generate(
            prompt,
            system=self.SYNTHESIS_SYSTEM_PROMPT,
            temperature=0.2,
            timeout=120,
        )

    @staticmethod
    def _merge_evidence(results: List[AgentResult]) -> VisualEvidence:
        """Merge visual evidence from multiple agent results."""
        merged = VisualEvidence()
        for r in results:
            ev = r.visual_evidence
            merged.image_urls.extend(ev.image_urls)
            merged.page_numbers.extend(ev.page_numbers)
            merged.bounding_boxes.extend(ev.bounding_boxes)
            if ev.document_name and not merged.document_name:
                merged.document_name = ev.document_name

        # Deduplicate
        merged.image_urls = list(dict.fromkeys(merged.image_urls))
        merged.page_numbers = sorted(set(merged.page_numbers))
        return merged


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_retriever(**overrides) -> MasterRetriever:
    """Factory — create a MasterRetriever with optional config overrides."""
    config = ServiceConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return MasterRetriever(config)
