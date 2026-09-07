-- PostgreSQL initialization script
-- Creates the extraction database and tables

CREATE DATABASE extraction_db;

\c extraction_db;

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

CREATE TABLE IF NOT EXISTS extracted_table_rows (
    id          BIGSERIAL PRIMARY KEY,
    table_id    UUID REFERENCES extracted_tables(id) ON DELETE CASCADE,
    row_index   INT NOT NULL,
    row_data    JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_table_rows_table_id
ON extracted_table_rows(table_id);

CREATE INDEX IF NOT EXISTS idx_extracted_tables_doc
ON extracted_tables(document_name);
