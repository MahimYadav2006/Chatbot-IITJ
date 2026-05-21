#!/usr/bin/env python3
"""
Flask Web Application for IIT Jammu EE GraphRAG Chatbot.

Provides a modern web interface for querying the knowledge graph
with real-time response generation.
"""

import os
import logging
import time

from flask import Flask, render_template, request, jsonify, session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global retriever and LLM instances (loaded once at startup)
retriever = None
llm = None


def init_app():
    """Initialize the retriever and LLM."""
    global retriever, llm

    from graphrag.retriever import load_retriever
    from graphrag.llm import OllamaLLM, SYSTEM_PROMPT, build_chat_prompt

    logger.info("Initializing GraphRAG chatbot...")
    retriever = load_retriever()
    llm = OllamaLLM()
    logger.info("✅ Chatbot ready!")


@app.route("/")
def index():
    """Serve the main chat page."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages."""
    from graphrag.llm import SYSTEM_PROMPT, build_chat_prompt, sanitize_response

    data = request.get_json()
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"error": "Empty message"}), 400

    try:
        start = time.time()

        direct_response = retriever.get_direct_answer(query)
        if direct_response:
            total_time = time.time() - start
            logger.info(f"Query: '{query[:50]}...' | Direct graph answer: {total_time:.2f}s")
            return jsonify({
                "response": sanitize_response(direct_response),
                "retrieval_time": round(total_time, 2),
                "total_time": round(total_time, 2),
                "direct": True,
            })

        # Retrieve context
        context = retriever.retrieve(
            query,
            local_top_k=5,
            vector_top_k=5,
            global_top_k=3,
            max_context_words=4500,
        )

        retrieval_time = time.time() - start

        # Generate response
        prompt = build_chat_prompt(query, context)
        response = llm.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=1400)

        total_time = time.time() - start

        logger.info(f"Query: '{query[:50]}...' | "
                   f"Retrieval: {retrieval_time:.2f}s | Total: {total_time:.2f}s")

        return jsonify({
            "response": response,
            "retrieval_time": round(retrieval_time, 2),
            "total_time": round(total_time, 2),
        })

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({
            "error": "An error occurred while processing your request.",
            "response": "I'm sorry, I encountered an error. Please try again."
        }), 500


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "retriever_loaded": retriever is not None,
        "llm_loaded": llm is not None,
    })


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5050, debug=False)
