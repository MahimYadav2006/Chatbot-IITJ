#!/usr/bin/env python3
"""
Flask Web Application for IIT Jammu Multi-Department GraphRAG Chatbot.

Provides a modern web interface for querying department-specific knowledge
graphs with dynamic, isolated session retrieval.
"""

import os
import logging
import time
from typing import Optional

from flask import Flask, render_template, request, jsonify, session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Dynamic registry mapping: department_code -> HybridRetriever instance
retrievers = {}
llm = None


def _log_retrieval_provenance(dept_code: str, query: str, provenance: dict, answerability: Optional[dict] = None):
    """Log which retrieval channels contributed to the answer."""
    answerability = answerability or {}
    logger.info(
        f"[{dept_code.upper()}] Provenance | mode={provenance.get('source_mode', 'unknown')} "
        f"| route={provenance.get('route', 'unknown')} "
        f"| graph_items={provenance.get('graph', {}).get('items', 0)} "
        f"(words={provenance.get('graph', {}).get('word_count', 0)}) "
        f"| vector_items={provenance.get('vector', {}).get('items', 0)} "
        f"(words={provenance.get('vector', {}).get('word_count', 0)}) "
        f"| community_items={provenance.get('community', {}).get('items', 0)} "
        f"(words={provenance.get('community', {}).get('word_count', 0)}) "
        f"| answerable={answerability.get('answerable', True)}"
    )
    if answerability.get("reason"):
        logger.info(f"[{dept_code.upper()}] Answerability note: {answerability['reason']}")


def get_retriever(dept_code: str = "ee"):
    """Get or dynamically load retriever for the specified department."""
    global retrievers
    dept_code = dept_code.lower().strip()
    
    if dept_code not in retrievers:
        from graphrag.retriever import load_retriever
        from departments import get_data_dir
        
        data_dir = get_data_dir(dept_code)
        # Check if the department has been ingested
        graph_path = os.path.join(data_dir, "graph.pkl")
        if not os.path.exists(graph_path):
            logger.warning(f"No ingested knowledge base found for department '{dept_code}' at {data_dir}.")
            if dept_code == "ee":
                raise FileNotFoundError("EE Department knowledge base not found. Please run ingest first.")
            return None
            
        logger.info(f"Loading GraphRAG retriever for department '{dept_code}'...")
        retrievers[dept_code] = load_retriever(dept_code=dept_code)
        logger.info(f"✅ Retriever for department '{dept_code}' loaded!")
        
    return retrievers[dept_code]


def init_app():
    """Initialize the LLM and preload the default EE retriever."""
    global llm
    from graphrag.llm import OllamaLLM

    logger.info("Initializing GraphRAG LLM service...")
    llm = OllamaLLM()
    
    try:
        # Preload the default department (EE)
        get_retriever("ee")
    except Exception as e:
        logger.warning(f"Could not preload EE department retriever: {e}")
        
    logger.info("✅ Multi-tenant system initialized!")


@app.route("/")
def index():
    """Serve the main chat page."""
    return render_template("index.html")


@app.route("/api/departments")
def get_departments_api():
    """Return the list of registered departments and their ingestion status."""
    from departments import DEPARTMENTS, get_data_dir
    result = []
    for code, info in DEPARTMENTS.items():
        data_dir = get_data_dir(code)
        ingested = os.path.exists(os.path.join(data_dir, "graph.pkl"))
        result.append({
            "code": code,
            "name": info["name"],
            "full_name": info["full_name"],
            "ingested": ingested
        })
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages with dynamic multi-tenant routing."""
    from graphrag.llm import get_system_prompt, build_chat_prompt, sanitize_response

    data = request.get_json() or {}
    query = data.get("message", "").strip()
    dept_code = data.get("department", "ee").strip().lower()

    if not query:
        return jsonify({"error": "Empty message"}), 400

    try:
        start = time.time()

        # Route dynamically to the requested department's retriever
        dept_retriever = get_retriever(dept_code)
        if not dept_retriever:
            return jsonify({
                "response": f"The knowledge base for the **{dept_code.upper()}** department is not ingested yet. "
                            f"Please run ingestion first using: `python ingest.py --dept {dept_code}`.",
                "retrieval_time": 0.0,
                "total_time": 0.0,
                "not_ingested": True
            })

        # Try to retrieve dynamic direct answers (rosters, count queries)
        direct_response = dept_retriever.get_direct_answer(query)
        if direct_response:
            total_time = time.time() - start
            provenance = {
                "route": "direct_graph",
                "source_mode": "graph",
                "graph": {"direct": True, "items": 1, "avg_score": 1.0, "labels": {}, "word_count": len(direct_response.split())},
                "vector": {"items": 0, "avg_score": 0.0, "sources": [], "word_count": 0},
                "community": {"items": 0, "avg_score": 0.0, "word_count": 0},
            }
            _log_retrieval_provenance(
                dept_code,
                query,
                provenance,
                {"answerable": True, "reason": "", "matched_terms": [], "missing_concepts": []},
            )
            logger.info(f"[{dept_code.upper()}] Query: '{query[:50]}...' | Direct graph answer: {total_time:.2f}s")
            return jsonify({
                "response": sanitize_response(direct_response),
                "retrieval_time": round(total_time, 2),
                "total_time": round(total_time, 2),
                "direct": True,
                "retrieval_provenance": provenance,
                "answerability": {"answerable": True},
            })

        # Multi-hop hybrid graph-vector retrieval
        retrieval_bundle = dept_retriever.retrieve_bundle(
            query,
            local_top_k=5,
            vector_top_k=5,
            global_top_k=3,
            max_context_words=4500,
        )
        context = retrieval_bundle["context"]
        provenance = retrieval_bundle["provenance"]
        answerability = retrieval_bundle["answerability"]

        retrieval_time = time.time() - start

        _log_retrieval_provenance(dept_code, query, provenance, answerability)

        if not answerability.get("answerable", True):
            total_time = time.time() - start
            logger.info(
                f"[{dept_code.upper()}] Query: '{query[:50]}...' | "
                f"Unavailable fallback: {total_time:.2f}s"
            )
            return jsonify({
                "response": sanitize_response(retrieval_bundle["fallback_response"]),
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "direct": False,
                "retrieval_provenance": provenance,
                "answerability": answerability,
                "information_available": False,
            })

        # Generate scoped and grounded response via local Llama 3.1
        prompt = build_chat_prompt(query, context, dept_code=dept_code)
        system_prompt = get_system_prompt(dept_code=dept_code)
        response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1400)

        total_time = time.time() - start

        logger.info(f"[{dept_code.upper()}] Query: '{query[:50]}...' | "
                    f"Retrieval: {retrieval_time:.2f}s | Total: {total_time:.2f}s")

        return jsonify({
            "response": response,
            "retrieval_time": round(retrieval_time, 2),
            "total_time": round(total_time, 2),
            "retrieval_provenance": provenance,
            "answerability": answerability,
            "information_available": True,
        })

    except Exception as e:
        logger.error(f"Error processing query in {dept_code.upper()} tenant: {e}", exc_info=True)
        return jsonify({
            "error": "An error occurred while processing your request.",
            "response": "I'm sorry, I encountered an error. Please try again."
        }), 500


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "loaded_departments": list(retrievers.keys()),
        "llm_loaded": llm is not None,
    })


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5050, debug=False)
