#!/usr/bin/env python3
"""
Flask Web Application for IIT Jammu Unified GraphRAG Chatbot.

Provides a single, unified interface for querying all department knowledge
graphs with automatic department detection and routing.
"""

import os
import logging
import time
from typing import Optional, Dict

from flask import Flask, render_template, request, jsonify
from env_config import load_env_file

load_env_file()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global instances
retrievers: Dict = {}          # dept_code → HybridRetriever
multi_retriever = None          # MultiDepartmentRetriever
router = None                   # DepartmentRouter
llm = None                      # Active response LLM provider
verifier = None                 # ResponseVerifier


def _log_retrieval_provenance(dept_codes, query, provenance, answerability=None):
    """Log which retrieval channels contributed to the answer."""
    answerability = answerability or {}
    dept_str = ",".join(d.upper() for d in dept_codes) if dept_codes else "BROADCAST"

    if isinstance(provenance, dict) and "route" in provenance:
        # Single department provenance
        logger.info(
            f"[{dept_str}] Provenance | mode={provenance.get('source_mode', 'unknown')} "
            f"| route={provenance.get('route', 'unknown')} "
            f"| graph_items={provenance.get('graph', {}).get('items', 0)} "
            f"| vector_items={provenance.get('vector', {}).get('items', 0)} "
            f"| answerable={answerability.get('answerable', True)}"
        )
    else:
        # Multi-department provenance (dict of dicts)
        logger.info(f"[{dept_str}] Multi-dept provenance | answerable={answerability.get('answerable', True)}")

    if answerability.get("reason"):
        logger.info(f"[{dept_str}] Answerability note: {answerability['reason']}")


def get_retriever(dept_code: str):
    """Get or dynamically load retriever for the specified department."""
    global retrievers
    dept_code = dept_code.lower().strip()

    if dept_code not in retrievers:
        from graphrag.retriever import load_retriever
        from departments import get_data_dir

        data_dir = get_data_dir(dept_code)
        graph_path = os.path.join(data_dir, "graph.pkl")
        if not os.path.exists(graph_path):
            logger.warning(f"No ingested knowledge base found for department '{dept_code}'.")
            return None

        logger.info(f"Loading GraphRAG retriever for department '{dept_code}'...")
        retrievers[dept_code] = load_retriever(dept_code=dept_code)
        logger.info(f"✅ Retriever for department '{dept_code}' loaded!")

    return retrievers[dept_code]


def init_app():
    """Initialize the LLM, router, verifier, and preload all ingested retrievers."""
    global llm, multi_retriever, router, verifier

    from graphrag.llm import create_llm_from_env
    from graphrag.verifier import ResponseVerifier
    from graphrag.multi_retriever import MultiDepartmentRetriever
    from dept_router import DepartmentRouter
    from departments import DEPARTMENTS, get_data_dir

    logger.info("Initializing IIT Jammu Unified Chatbot...")

    # Initialize LLM
    llm = create_llm_from_env()
    logger.info(
        f"Using LLM provider '{getattr(llm, 'provider', 'unknown')}' "
        f"with model '{getattr(llm, 'model', 'unknown')}'"
    )

    # Initialize router
    router = DepartmentRouter()

    # Initialize verifier
    verifier = ResponseVerifier(llm)

    # Preload all ingested department retrievers
    loaded_count = 0
    for code in DEPARTMENTS:
        data_dir = get_data_dir(code)
        if os.path.exists(os.path.join(data_dir, "graph.pkl")):
            try:
                get_retriever(code)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Could not preload '{code}' retriever: {e}")

    # Initialize multi-department retriever
    multi_retriever = MultiDepartmentRetriever(retrievers)

    logger.info(f"✅ Unified system initialized! {loaded_count} departments loaded.")


@app.route("/")
def index():
    """Serve the main chat page."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Handle chat messages with automatic department routing.

    Flow:
      1. Router detects target department(s) from query text
      2. MultiDepartmentRetriever fetches from the right scope
      3. LLM generates response with appropriate system prompt
      4. Verifier checks faithfulness before returning
    """
    from graphrag.llm import (
        get_system_prompt, get_unified_system_prompt,
        build_chat_prompt, build_multi_dept_chat_prompt,
        sanitize_response,
    )

    data = request.get_json() or {}
    query = data.get("message", "").strip()

    if not query:
        return jsonify({"error": "Empty message"}), 400

    try:
        start = time.time()

        # Step 1: Route query to department(s)
        route_result = router.route(query)
        logger.info(f"Routing: '{query[:60]}...' → {route_result.confidence} | {route_result.reason}")

        # Step 2: Retrieve context
        if route_result.confidence == "exact" and len(route_result.departments) == 1:
            # Single department — delegate directly
            dept_code = route_result.departments[0]
            dept_retriever = get_retriever(dept_code)

            if not dept_retriever:
                return jsonify({
                    "response": f"The knowledge base for **{dept_code}** is not available yet.",
                    "routed_departments": [dept_code],
                    "routing_reason": route_result.reason,
                    "retrieval_time": 0.0,
                    "total_time": 0.0,
                })

            # Check for direct graph answers first
            direct_response = dept_retriever.get_direct_answer(query)
            if direct_response:
                total_time = time.time() - start
                _log_retrieval_provenance(
                    [dept_code], query,
                    {"route": "direct_graph", "source_mode": "graph"},
                    {"answerable": True},
                )
                return jsonify({
                    "response": sanitize_response(direct_response),
                    "routed_departments": [dept_code],
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(total_time, 2),
                    "total_time": round(total_time, 2),
                    "direct": True,
                })

            # Full hybrid retrieval
            bundle = dept_retriever.retrieve_bundle(
                query, local_top_k=5, vector_top_k=5, global_top_k=3, max_context_words=4500
            )
            context = bundle["context"]
            answerability = bundle["answerability"]
            retrieval_time = time.time() - start

            _log_retrieval_provenance([dept_code], query, bundle.get("provenance", {}), answerability)

            if not answerability.get("answerable", True):
                total_time = time.time() - start
                return jsonify({
                    "response": sanitize_response(bundle["fallback_response"]),
                    "routed_departments": [dept_code],
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": False,
                })

            # Generate scoped response
            prompt = build_chat_prompt(query, context, dept_code=dept_code)
            system_prompt = get_system_prompt(dept_code=dept_code)
            response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1400)

            # Verify faithfulness (Layer 4)
            verification = verifier.verify(query, context, response)
            if not verification.faithful:
                logger.warning(f"Response failed faithfulness check. {verification.reason}")
                from departments import get_department
                dept_config = get_department(dept_code)
                base_url = dept_config.get("base_url", "https://iitjammu.ac.in")
                response = (
                    f"I don't have verified information to answer that question accurately. "
                    f"Please check the official {dept_config['name']} department website at {base_url} for details."
                )

            total_time = time.time() - start
            logger.info(f"[{dept_code.upper()}] Query: '{query[:50]}...' | Total: {total_time:.2f}s")

            return jsonify({
                "response": response,
                "routed_departments": [dept_code],
                "routing_reason": route_result.reason,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "information_available": True,
            })

        elif route_result.confidence == "exact" and len(route_result.departments) > 1:
            # Multi-department query
            bundle = multi_retriever.retrieve_multi(query, route_result.departments)
            context = bundle["context"]
            answerability = bundle.get("answerability", {})
            retrieval_time = time.time() - start

            _log_retrieval_provenance(route_result.departments, query, bundle.get("provenance", {}), answerability)

            if not answerability.get("answerable", True):
                total_time = time.time() - start
                return jsonify({
                    "response": sanitize_response(bundle.get("fallback_response", "I don't have that information.")),
                    "routed_departments": route_result.departments,
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": False,
                })

            # Generate with unified prompt for cross-department queries
            dept_contexts = bundle.get("dept_contexts", {})
            prompt = build_multi_dept_chat_prompt(query, dept_contexts)
            system_prompt = get_unified_system_prompt()
            response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1800)

            # Verify faithfulness
            verification = verifier.verify(query, context, response)
            if not verification.faithful:
                logger.warning(f"Cross-dept response failed faithfulness check. {verification.reason}")
                response = (
                    "I don't have verified information to answer that question accurately across those departments. "
                    "Please check the official IIT Jammu website for details."
                )

            total_time = time.time() - start
            logger.info(f"[MULTI] Query: '{query[:50]}...' | Depts: {route_result.departments} | Total: {total_time:.2f}s")

            return jsonify({
                "response": response,
                "routed_departments": bundle.get("departments", route_result.departments),
                "routing_reason": route_result.reason,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "information_available": True,
            })

        else:
            # Broadcast — no department signal detected
            bundle = multi_retriever.retrieve_broadcast(query, top_n=len(retrievers))
            context = bundle["context"]
            answerability = bundle.get("answerability", {})
            routed_depts = bundle.get("departments", [])
            retrieval_time = time.time() - start

            _log_retrieval_provenance(routed_depts, query, bundle.get("provenance", {}), answerability)

            # Handle direct answers from broadcast
            if bundle.get("direct"):
                if len(routed_depts) <= 1:
                    # Single department direct answer — return as-is
                    total_time = time.time() - start
                    return jsonify({
                        "response": sanitize_response(context),
                        "routed_departments": routed_depts,
                        "routing_reason": route_result.reason,
                        "retrieval_time": round(retrieval_time, 2),
                        "total_time": round(total_time, 2),
                        "direct": True,
                    })
                else:
                    # Multi-department direct answers — run through LLM for
                    # proper cross-department synthesis and labeling
                    dept_contexts = bundle.get("dept_contexts", {})
                    if dept_contexts:
                        prompt = build_multi_dept_chat_prompt(query, dept_contexts)
                    else:
                        prompt = build_chat_prompt(query, context, dept_code="ee")
                    system_prompt = get_unified_system_prompt()
                    response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1800)
                    total_time = time.time() - start
                    return jsonify({
                        "response": sanitize_response(response),
                        "routed_departments": routed_depts,
                        "routing_reason": route_result.reason,
                        "retrieval_time": round(retrieval_time, 2),
                        "total_time": round(total_time, 2),
                        "direct": True,
                    })

            if not answerability.get("answerable", True):
                total_time = time.time() - start
                return jsonify({
                    "response": sanitize_response(bundle.get("fallback_response",
                        "I don't have that specific information. Try mentioning a department like CSE, EE, or Physics.")),
                    "routed_departments": routed_depts,
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": False,
                })

            # Generate with appropriate prompt
            if len(routed_depts) == 1:
                dept_code = routed_depts[0]
                prompt = build_chat_prompt(query, context, dept_code=dept_code)
                system_prompt = get_system_prompt(dept_code=dept_code)
            else:
                dept_contexts = bundle.get("dept_contexts", {})
                if dept_contexts:
                    prompt = build_multi_dept_chat_prompt(query, dept_contexts)
                else:
                    # Even without structured dept_contexts, use unified prompt
                    prompt = build_multi_dept_chat_prompt(query, {
                        code: {"name": __import__('departments').DEPARTMENTS.get(code, {}).get("full_name", code), "context": context}
                        for code in routed_depts
                    }) if routed_depts else build_chat_prompt(query, context, dept_code="ee")
                system_prompt = get_unified_system_prompt()

            response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1400)

            # Verify faithfulness
            verification = verifier.verify(query, context, response)
            if not verification.faithful:
                logger.warning(f"Broadcast response failed faithfulness check. {verification.reason}")
                response = (
                    "I don't have verified information to answer that question accurately. "
                    "Please try asking about a specific department (e.g., CSE, EE, Physics) for better results."
                )

            total_time = time.time() - start
            logger.info(f"[BROADCAST] Query: '{query[:50]}...' | Routed: {routed_depts} | Total: {total_time:.2f}s")

            return jsonify({
                "response": response,
                "routed_departments": routed_depts,
                "routing_reason": route_result.reason,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "information_available": True,
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
        "loaded_departments": list(retrievers.keys()),
        "llm_loaded": llm is not None,
        "router_loaded": router is not None,
        "verifier_loaded": verifier is not None,
    })


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5050, debug=False)
