"""
db_handler.py
=============
PostgreSQL handler for storing extracted table data.

Tables are stored with metadata (document name, page number, element index,
extraction timestamp) and the full DataFrame content as rows.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, execute_values

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PG_CONFIG = {
    "host": "postgres",
    "port": 5432,
    "dbname": "extraction_db",
    "user": "extract_user",
    "password": "extract_pass",
}


class PostgresHandler:
    """Manages PostgreSQL connections and table storage."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or DEFAULT_PG_CONFIG
        self._conn = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.config)
            self._conn.autocommit = True
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------
    def _ensure_schema(self):
        """Create the metadata and data tables if they don't exist."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS extracted_tables (
                    id              UUID PRIMARY KEY,
                    document_name   TEXT NOT NULL,
                    page_number     INT,
                    element_index   INT,
                    element_label   TEXT DEFAULT 'Table',
                    column_headers  JSONB,
                    row_count       INT,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS extracted_table_rows (
                    id          BIGSERIAL PRIMARY KEY,
                    table_id    UUID REFERENCES extracted_tables(id) ON DELETE CASCADE,
                    row_index   INT NOT NULL,
                    row_data    JSONB NOT NULL
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_table_rows_table_id
                ON extracted_table_rows(table_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_extracted_tables_doc
                ON extracted_tables(document_name);
            """)
        logger.info("PostgreSQL schema ensured.")

    # ------------------------------------------------------------------
    # Store a DataFrame
    # ------------------------------------------------------------------
    def store_table(
        self,
        df: pd.DataFrame,
        document_name: str,
        page_number: int = 1,
        element_index: int = 0,
        element_label: str = "Table",
    ) -> str:
        """
        Store a parsed DataFrame into PostgreSQL.

        Returns the UUID of the stored table record.
        """
        table_id = str(uuid.uuid4())
        conn = self._get_conn()

        # Replace NaN/NaT with None so JSON serialization produces null (not NaN)
        df_clean = df.where(df.notna(), None)
        columns = list(df_clean.columns)
        rows = df_clean.to_dict(orient="records")
        # Extra safety: replace any remaining float('nan') with None
        for row in rows:
            for k, v in row.items():
                if isinstance(v, float) and (v != v):  # NaN check
                    row[k] = None

        with conn.cursor() as cur:
            # Insert metadata
            cur.execute(
                """
                INSERT INTO extracted_tables
                    (id, document_name, page_number, element_index,
                     element_label, column_headers, row_count, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    table_id,
                    document_name,
                    page_number,
                    element_index,
                    element_label,
                    Json(columns),
                    len(rows),
                    datetime.now(timezone.utc),
                ),
            )

            # Bulk-insert rows
            if rows:
                values = [
                    (table_id, idx, Json(row))
                    for idx, row in enumerate(rows)
                ]
                execute_values(
                    cur,
                    """
                    INSERT INTO extracted_table_rows (table_id, row_index, row_data)
                    VALUES %s
                    """,
                    values,
                    template="(%s, %s, %s)",
                )

        logger.info(
            "Stored table %s (%d cols × %d rows) for doc '%s' page %d",
            table_id, len(columns), len(rows), document_name, page_number,
        )
        return table_id

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def get_table(self, table_id: str) -> Optional[pd.DataFrame]:
        """Retrieve a stored table as a DataFrame."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_headers FROM extracted_tables WHERE id = %s",
                (table_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            columns = row[0]

            cur.execute(
                """
                SELECT row_data FROM extracted_table_rows
                WHERE table_id = %s ORDER BY row_index
                """,
                (table_id,),
            )
            rows = [r[0] for r in cur.fetchall()]

        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns)

    def list_tables(self, document_name: Optional[str] = None) -> list:
        """List stored table metadata, optionally filtered by document."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            if document_name:
                cur.execute(
                    """
                    SELECT id, document_name, page_number, element_index,
                           element_label, row_count, created_at
                    FROM extracted_tables WHERE document_name = %s
                    ORDER BY created_at DESC
                    """,
                    (document_name,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, document_name, page_number, element_index,
                           element_label, row_count, created_at
                    FROM extracted_tables ORDER BY created_at DESC
                    """
                )
            rows = cur.fetchall()

        return [
            {
                "id": str(r[0]),
                "document_name": r[1],
                "page_number": r[2],
                "element_index": r[3],
                "element_label": r[4],
                "row_count": r[5],
                "created_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]
