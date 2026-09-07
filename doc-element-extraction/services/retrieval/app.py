"""
app.py
======
Flask REST API for the Retrieval Agent microservice.

Endpoints:
    POST /retrieve       — Main retrieval endpoint (query → answer + evidence)
    POST /route          — Route-only (classify query intent without retrieval)
    GET  /health         — Health check for all backend services
    GET  /health/simple  — Quick liveness probe

Environment variables are read by ServiceConfig in retrieval_agent.py.
"""

import logging
import os
import time

from flask import Flask, jsonify, request

from retrieval_agent import MasterRetriever, ServiceConfig, create_retriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("retrieval_api")

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Lazy-initialise the retriever (first request triggers backend connections)
# ---------------------------------------------------------------------------
_retriever: MasterRetriever = None  # type: ignore


def get_retriever() -> MasterRetriever:
    global _retriever
    if _retriever is None:
        logger.info("Initialising MasterRetriever...")
        _retriever = create_retriever()
        logger.info("MasterRetriever ready.")
    return _retriever


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════


@app.route("/health/simple", methods=["GET"])
def health_simple():
    """Quick liveness probe — does NOT check backends."""
    return jsonify({"status": "alive", "service": "retrieval-agent"})


@app.route("/health", methods=["GET"])
def health():
    """Deep health check — verifies connectivity to all backends."""
    try:
        retriever = get_retriever()
        checks = retriever.health_check()
        status_code = 200 if checks.get("all_healthy") else 503
        return jsonify(checks), status_code
    except Exception as exc:
        logger.exception("Health check failed")
        return jsonify({"error": str(exc), "all_healthy": False}), 503


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """
    Main retrieval endpoint.

    Request JSON:
    {
        "query": "ما هو صافي الدخل في 2023؟",
        "filters": {"document_name": "annual_report_2023.pdf"}  // optional
    }

    Response JSON:
    {
        "answer": "...",
        "intent_used": "SQL_MODE",
        "visual_evidence": {
            "image_urls": [...],
            "document_name": "...",
            "page_numbers": [...]
        },
        "routing_reasoning": "...",
        "total_time": 2.35
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No 'query' field provided"}), 400

    filters = data.get("filters")

    try:
        retriever = get_retriever()
        response = retriever.retrieve(query=query, filters=filters)
        return jsonify(response.to_dict())
    except Exception as exc:
        logger.exception("Retrieval failed for query: %s", query)
        return jsonify({"error": str(exc), "query": query}), 500


@app.route("/route", methods=["POST"])
def route_only():
    """
    Route-only endpoint — classify the query intent without executing retrieval.
    Useful for debugging or building custom retrieval flows.

    Request JSON:
    {
        "query": "Who are the board members?"
    }

    Response JSON:
    {
        "intent": "GRAPH_MODE",
        "extracted_entities": ["board members"],
        "reasoning": "...",
        "confidence": 0.95
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No 'query' field provided"}), 400

    try:
        retriever = get_retriever()
        routing = retriever.router.route(query)
        return jsonify({
            "intent": routing.intent.value,
            "extracted_entities": routing.extracted_entities,
            "reasoning": routing.reasoning,
            "confidence": routing.confidence,
        })
    except Exception as exc:
        logger.exception("Routing failed for query: %s", query)
        return jsonify({"error": str(exc)}), 500


@app.route("/agents/sql", methods=["POST"])
def sql_agent():
    """
    Direct SQL agent endpoint — bypass the router.

    Request JSON:
    {
        "query": "What was total revenue in 2023?",
        "entities": ["revenue", "2023"]
    }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("query"):
        return jsonify({"error": "No 'query' field provided"}), 400

    try:
        retriever = get_retriever()
        result = retriever.sql_agent.execute(
            data["query"], data.get("entities", [])
        )
        return jsonify(result.to_dict())
    except Exception as exc:
        logger.exception("SQLAgent direct call failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/agents/graph", methods=["POST"])
def graph_agent():
    """
    Direct Graph agent endpoint — bypass the router.

    Request JSON:
    {
        "query": "Who owns Company X?",
        "entities": ["Company X"]
    }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("query"):
        return jsonify({"error": "No 'query' field provided"}), 400

    try:
        retriever = get_retriever()
        result = retriever.graph_agent.execute(
            data["query"], data.get("entities", [])
        )
        return jsonify(result.to_dict())
    except Exception as exc:
        logger.exception("GraphAgent direct call failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/agents/vector", methods=["POST"])
def vector_agent():
    """
    Direct Vector agent endpoint — bypass the router.

    Request JSON:
    {
        "query": "What are the primary risks?",
        "entities": ["risks"],
        "top_k": 5,
        "filters": {}
    }
    """
    data = request.get_json(silent=True)
    if not data or not data.get("query"):
        return jsonify({"error": "No 'query' field provided"}), 400

    try:
        retriever = get_retriever()
        result = retriever.vector_agent.execute(
            data["query"],
            data.get("entities", []),
            top_k=data.get("top_k", 8),
            filters=data.get("filters"),
        )
        return jsonify(result.to_dict())
    except Exception as exc:
        logger.exception("VectorAgent direct call failed")
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8100))
    app.run(host="0.0.0.0", port=port, debug=False)
