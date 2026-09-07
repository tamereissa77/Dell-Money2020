"""
Sentence Transformers embedding microservice.

Provides /embed and /embeddings endpoints for generating text embeddings
using the all-MiniLM-L6-v2 model (384 dimensions).
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model_name = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
logger.info(f"Loading model: {model_name}")

start_time = time.time()
try:
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": model_name})


@app.route("/embed", methods=["POST"])
def embed():
    """Generate embeddings for a list of texts."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        texts = data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        batch_size = data.get("batch_size", 32)

        start = time.time()
        embeddings = model.encode(texts, batch_size=batch_size).tolist()
        processing_time = time.time() - start

        logger.info(f"Processed {len(texts)} texts in {processing_time:.2f}s")

        return jsonify({
            "embeddings": embeddings,
            "model": model_name,
            "processing_time": processing_time,
        })
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/embeddings", methods=["POST"])
def embeddings():
    """OpenAI-compatible embeddings endpoint."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        texts = data.get("input", [])
        if not texts:
            return jsonify({"error": "No input texts provided"}), 400

        batch_size = data.get("batch_size", 32)

        start = time.time()
        embs = model.encode(texts, batch_size=batch_size).tolist()
        processing_time = time.time() - start

        response_data = {
            "data": [{"embedding": emb} for emb in embs],
            "model": model_name,
            "processing_time": processing_time,
        }

        logger.info(f"Processed {len(texts)} texts in {processing_time:.2f}s")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
