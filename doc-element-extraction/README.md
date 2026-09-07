# 📊 Arabic Financial Document Extractor

A GPU-accelerated, fully containerized platform for extracting structured data from Arabic financial documents. Built with **Streamlit**, **Chandra VLM** (Qwen3-VL), and the **Unstructured** library, with a complete post-extraction pipeline that stores data across **PostgreSQL**, **ArangoDB** (Knowledge Graph), **Qdrant** (Vector DB), and **MinIO** (Object Store).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Overview

This application processes Arabic (and English) financial documents — such as annual reports, balance sheets, income statements, and regulatory filings — and extracts structured data from them. It leverages a multi-service architecture running entirely on an **NVIDIA DGX Spark / GB10**:

- **[Chandra VLM](https://huggingface.co/datalab-to/chandra)** — Vision Language Model for document OCR and layout understanding
- **[Unstructured](https://github.com/Unstructured-IO/unstructured)** — Intelligent document segmentation and table detection
- **[Ollama](https://ollama.com/)** — Local LLM (Llama 3.1 8B) for knowledge graph triple extraction
- **[PostgreSQL](https://www.postgresql.org/)** — Structured storage for extracted table data
- **[ArangoDB](https://www.arangodb.com/)** — Graph database for knowledge triples (txt2kg pattern)
- **[Qdrant](https://qdrant.tech/)** — Vector database for semantic search embeddings
- **[MinIO](https://min.io/)** — S3-compatible object store for cropped images with citation references
- **[Sentence Transformers](https://www.sbert.net/)** — Embedding microservice (all-MiniLM-L6-v2)

---

## ✨ Features

### 🔹 Document Extraction
- **Table Extraction Mode** — Unstructured detects tables → Chandra VLM OCR → structured DataFrames
- **Full Page OCR Mode** — Chandra identifies all elements with bounding boxes (Tables, Text, Headers, Captions, Footnotes, Images, Forms, etc.)
- **RTL Arabic text rendering** with proper font support
- **Multi-page PDF support** with per-page processing
- **CSV and Excel export** for all extracted tables

### 🔹 Interactive Table Editor
When the VLM extracts a table, it is displayed in a fully interactive editor that lets you fix OCR errors and structural issues before storing:

- **Editable headers** — Each column header is a text input; rename columns directly (e.g., fix Arabic header text the model got wrong)
- **Add columns** (`➕ Col`) — If the model detected fewer columns than the original table (e.g., 2 instead of 3), add the missing column and redistribute data
- **Remove columns** (`🗑`) — Drop unwanted columns via a dropdown selector
- **Editable cells** — Click any cell to edit its value; add or delete rows with the built-in toolbar
- **Automatic type handling** — All values are treated as text to avoid type conflicts between numeric and string data
- **Persistent state** — Edits are preserved across Streamlit reruns; the final edited table is what gets stored to PostgreSQL and exported to CSV/Excel

### 🔹 Post-Extraction Storage Pipeline
After extraction, each element is automatically routed to the appropriate backends:

| Element Type | PostgreSQL | Knowledge Graph | Vector DB | Object Store |
|-------------|:----------:|:---------------:|:---------:|:------------:|
| **Table** | ✅ Rows + metadata | — | ✅ Embedded | ✅ Image + ref |
| **Text** | — | ✅ Triples via Ollama | ✅ Embedded | ✅ Image + ref |
| **Section-Header** | — | ✅ Triples | ✅ Embedded | ✅ Image + ref |
| **Caption** | — | ✅ Triples | ✅ Embedded | ✅ Image + ref |
| **Footnote** | — | ✅ Triples | ✅ Embedded | ✅ Image + ref |
| **Image/Figure** | — | — | — | ✅ Image + ref |

### 🔹 Knowledge Graph (txt2kg Pattern)
- Text elements are sent to **Ollama** (Llama 3.1 8B) for subject-predicate-object triple extraction
- Triples are stored in **ArangoDB** as a graph (entities + relationships)
- Specialized prompt for Arabic financial document entities
- Follows the [NVIDIA txt2kg](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/txt2kg) architecture

### 🔹 Vector Embeddings
- All text content is chunked and embedded via **sentence-transformers** (all-MiniLM-L6-v2, 384 dims)
- Embeddings stored in **Qdrant** for semantic similarity search
- Tables are serialized to text and embedded for cross-modal search

### 🔹 Image Citation Store
- Every extracted element's cropped image is stored in **MinIO** (S3-compatible)
- Each image has a companion JSON reference file containing:
  - Document name, page number, element label, bounding box
  - Extracted text and HTML content
  - Timestamp for audit trail
- Enables citation: "This data came from this region of this page"

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (8501)                    │
│                         src/app.py                               │
├─────────────────────────────────────────────────────────────────┤
│                    Extraction Pipeline                           │
│                      src/pipeline.py                             │
├────��─────┬──────────┬──────────┬──────────┬────────────────────┤
│          │          │          │          │                      │
│  Chandra │ Unstruc- │ Ollama   │ Sentence │                     │
│  VLM     │ tured    │ LLM      │ Trans-   │                     │
│  (GPU)   │          │ (GPU)    │ formers  │                     │
│          │          │          │          │                      │
├──────────┴──────────┴──────────┴──────────┴────────────────────┤
│                     Storage Backends                             │
│                                                                  │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌───────────────┐ │
│  │ PostgreSQL │ │  ArangoDB  │ │  Qdrant  │ │     MinIO     │ │
│  ��   (5432)   │ │   (8529)   │ │  (6333)  │ │  (9000/9001)  │ │
│  │            │ │            │ │          │ │               │ │
│  │  Tables &  │ │ Knowledge  │ │  Vector  │ │ Cropped Images│ │
│  │  Metadata  │ │   Graph    │ │Embeddings│ │ + References  │ │
│  └────────────┘ └────────────┘ └──────────┘ └───────────────┘ │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│              NVIDIA GB10 GPU (CUDA) / Docker Compose             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
EXTRACT-elements/
├── src/
│   ├── app.py              # Streamlit UI + main application logic
│   ├── model_handler.py    # Chandra VLM loading, inference, layout parsing
│   ├── processor.py        # Document segmentation via Unstructured
│   ├── pipeline.py         # Post-extraction orchestrator (page-by-page)
│   ├── db_handler.py       # PostgreSQL client for table storage
│   ├── kg_handler.py       # Knowledge Graph (Ollama + ArangoDB)
│   ├── vector_handler.py   # Vector embeddings (sentence-transformers + Qdrant)
│   └── object_store.py     # MinIO object storage for images + citations
├── services/
│   ├── txt2kg-frontend/    # NVIDIA txt2kg Next.js visualization (3D graph)
│   │   ├── app/            # Next.js app router (pages + API routes)
│   │   ├── components/     # React UI components
│   │   ├── lib/            # ArangoDB, Qdrant, graph utilities
│   │   ├── Dockerfile
│   │   └── package.json
│   ├── sentence-transformers/
│   │   ├── app.py          # Flask embedding microservice
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── ollama/
│   │   ├── Dockerfile
│   │   └── entrypoint.sh   # Auto-pulls llama3.1:8b on first start
│   └── postgres/
│       └── init.sql        # Database schema initialization
├── docker-compose.yml      # Full 9-service stack with GPU support
├── Dockerfile              # Main app (CUDA + Python)
├── Makefile                # make ingestion, make stop, make logs, etc.
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys, passwords)
├── .gitignore
└── README.md
```

### Module Details

| Module | Responsibility |
|--------|---------------|
| **`src/app.py`** | Streamlit frontend — file upload, mode selection, extraction, interactive table editor (editable headers + add/remove columns), pipeline integration, result display |
| **`src/pipeline.py`** | Orchestrates post-extraction routing to all backends with error isolation |
| **`src/db_handler.py`** | PostgreSQL — stores tables as metadata + JSONB rows, with retrieval and listing |
| **`src/kg_handler.py`** | Ollama LLM triple extraction → ArangoDB graph storage (entities + relationships) |
| **`src/vector_handler.py`** | Sentence-transformers embedding → Qdrant vector storage with semantic search |
| **`src/object_store.py`** | MinIO — stores cropped PNG images + JSON citation references |
| **`src/model_handler.py`** | Chandra VLM (Qwen3-VL) loading, OCR and layout inference |
| **`src/processor.py`** | PDF/image conversion, Unstructured partitioning, table region cropping |

---

## 🚀 Getting Started

### Prerequisites

- **NVIDIA DGX Spark / GB10** (or any NVIDIA GPU with 8GB+ VRAM)
- **Docker** and **Docker Compose** v2+
- **NVIDIA Container Toolkit** (`nvidia-docker2`)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/iPulse-AI/element-extraction-test.git
cd element-extraction-test

# Configure environment (edit API keys if needed)
cp .env.example .env  # or edit .env directly

# Build and start the full ingestion pipeline (all 9 services)
make ingestion

# Watch logs
make logs
```

### Service Endpoints

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| **Streamlit App** | 8501 | http://localhost:8501 | Document extraction UI |
| **txt2kg Frontend** | 3001 | http://localhost:3001 | 3D Knowledge Graph visualization |
| **PostgreSQL** | 5432 | `psql -h localhost -U extract_user -d extraction_db` | Table data |
| **ArangoDB** | 8529 | http://localhost:8529 | Knowledge graph DB console |
| **Qdrant** | 6333 | http://localhost:6333/dashboard | Vector DB dashboard |
| **MinIO Console** | 9001 | http://localhost:9001 | Object store UI (minioadmin/minioadmin) |
| **MinIO API** | 9000 | http://localhost:9000 | S3-compatible API |
| **Ollama** | 11434 | http://localhost:11434 | LLM API |
| **Embeddings** | 8000 | http://localhost:8000/health | Embedding service |

### Makefile Targets

```bash
make ingestion      # Build & start the full stack
make stop           # Stop all services
make restart        # Restart all services
make status         # Show service status
make logs           # Tail logs from all services
make clean          # Stop and remove all containers + volumes
make build          # Build all images without starting
make rebuild        # Force rebuild and restart

# Per-service logs
make logs-app       # Streamlit app logs
make logs-txt2kg    # txt2kg frontend logs
make logs-ollama    # Ollama LLM logs
make logs-kg        # ArangoDB logs
make logs-pg        # PostgreSQL logs

# Shell access
make shell-app      # Shell into app container
make shell-pg       # psql into PostgreSQL
make shell-arango   # arangosh into ArangoDB
```

### First Run Notes

1. **Chandra VLM** weights (~4-8 GB) download on first extraction request
2. **Ollama** automatically pulls `llama3.1:8b` on first start (~4.7 GB)
3. **Sentence-transformers** model (`all-MiniLM-L6-v2`) is pre-downloaded during Docker build
4. All model caches are persisted in Docker volumes across restarts

---

## 🖥️ Usage

### 1. Upload a Document

Upload a financial document: PDF, PNG, JPG, TIFF, BMP, or WebP.

### 2. Choose Extraction Mode

#### Table Extraction Mode
1. Unstructured detects table regions automatically
2. Select/deselect tables with checkboxes
3. Click **Extract** → Chandra VLM OCR → parsed DataFrames
4. **Edit the table** — fix headers, add/remove columns, correct cell values
5. Download as CSV/Excel, then click **✅ Confirm & Store** to send to all backends

#### Full Page OCR Mode
1. Select pages → Click **Run Full Page OCR**
2. Chandra identifies all elements with bounding boxes
3. Annotated page displayed with color-coded elements
4. **Edit extracted tables** — each Table block has an interactive editor with editable headers and add/remove column controls
5. Click **✅ Confirm & Store All Pages** to send to all backends

### 3. Storage Pipeline

Toggle **"Enable post-extraction storage"** in the sidebar. When enabled:

- **🐘 PostgreSQL** — Tables stored with metadata (document, page, columns, rows)
- **🔗 Knowledge Graph** — Text → Ollama → triples → ArangoDB
- **🔍 Vector DB** — All text → embeddings → Qdrant
- **📦 Object Store** — Cropped images + JSON references → MinIO

Each element shows colored badges indicating which backends received its data.

### 4. Access Stored Data

- **PostgreSQL**: Connect with any SQL client to query `extracted_tables` and `extracted_table_rows`
- **ArangoDB**: Open http://localhost:8529 → database `extraction_kg` → graph `knowledge_graph`
- **Qdrant**: Open http://localhost:6333/dashboard → collection `document-embeddings`
- **MinIO**: Open http://localhost:9001 → bucket `extracted-elements`

---

## 🐳 Docker Services

### Full Stack (7 services)

```yaml
services:
  arabic-table-extractor    # Streamlit + Chandra VLM (GPU)
  postgres                  # PostgreSQL 16
  arangodb                  # ArangoDB (graph DB)
  arangodb-init             # DB initialization (one-shot)
  ollama                    # Ollama + Llama 3.1 8B (GPU)
  qdrant                    # Qdrant vector DB
  qdrant-init               # Collection initialization (one-shot)
  sentence-transformers     # all-MiniLM-L6-v2 embedding service
  minio                     # S3-compatible object store
```

### Resource Usage (GB10)

| Service | GPU | RAM (approx) | Disk |
|---------|-----|-------------|------|
| Chandra VLM | ✅ Shared | ~8 GB | ~8 GB (model) |
| Ollama (Llama 3.1 8B) | ✅ Shared | ~5 GB | ~5 GB (model) |
| PostgreSQL | — | ~256 MB | Variable |
| ArangoDB | — | ~512 MB | Variable |
| Qdrant | — | ~256 MB | Variable |
| Sentence Transformers | — | ~512 MB | ~500 MB (model) |
| MinIO | — | ~256 MB | Variable |

### Persistent Volumes

| Volume | Purpose |
|--------|---------|
| `hf_cache` | Hugging Face model weights (Chandra VLM) |
| `postgres_data` | PostgreSQL database files |
| `arangodb_data` | ArangoDB graph data |
| `ollama_data` | Ollama model weights |
| `qdrant_data` | Qdrant vector indices |
| `minio_data` | MinIO object storage |

---

## ⚙️ Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | — | NVIDIA NGC API key |
| `POSTGRES_USER` | `extract_user` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `extract_pass` | PostgreSQL password |
| `POSTGRES_DB` | `extraction_db` | PostgreSQL database name |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO admin username |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO admin password |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model for triple extraction |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce `Max new tokens`, or stop Ollama when running Chandra (`docker compose stop ollama`) |
| **Ollama model not ready** | Wait 5-10 min on first start for model download; check `docker logs extraction-ollama` |
| **PostgreSQL connection refused** | Ensure postgres is healthy: `docker compose ps postgres` |
| **ArangoDB not responding** | Wait for init container: `docker compose logs extraction-arangodb-init` |
| **Qdrant collection missing** | Check init: `docker compose logs extraction-qdrant-init` |
| **MinIO bucket not found** | Bucket is auto-created on first use by the app |
| **Pipeline badges show ❌** | Click "Check Backend Status" in sidebar; some services may still be starting |
| **Slow triple extraction** | Ollama uses GPU; if Chandra is also loaded, they share VRAM |

### Useful Commands

```bash
# Check all service status
docker compose ps

# View logs for a specific service
docker compose logs -f arabic-table-extractor
docker compose logs -f extraction-ollama

# Restart a single service
docker compose restart arabic-table-extractor

# Stop everything
docker compose down

# Stop and remove all data
docker compose down -v

# Flush GPU memory (DGX Spark)
sudo sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

---

## 📄 License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Chandra VLM](https://huggingface.co/datalab-to/chandra)** by Datalab — Document OCR Vision Language Model
- **[NVIDIA txt2kg](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/txt2kg)** — Knowledge graph extraction pattern
- **[Qwen3-VL](https://github.com/QwenLM/Qwen-VL)** by Alibaba Cloud — Base vision-language architecture
- **[Unstructured](https://github.com/Unstructured-IO/unstructured)** — Document preprocessing
- **[Ollama](https://ollama.com/)** — Local LLM inference
- **[ArangoDB](https://www.arangodb.com/)** — Multi-model graph database
- **[Qdrant](https://qdrant.tech/)** — Vector similarity search engine
- **[MinIO](https://min.io/)** — S3-compatible object storage
- **[Streamlit](https://streamlit.io/)** — ML/AI web application framework
