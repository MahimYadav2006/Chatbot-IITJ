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
section_retrievers: Dict = {}  # sec_code → SectionRetriever
person_index = None            # GlobalPersonIndex
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


def get_section_retriever(sec_code: str):
    """Get or dynamically load retriever for the specified section."""
    global section_retrievers
    sec_code = sec_code.lower().strip()

    if sec_code not in section_retrievers:
        from graphrag.section_retriever import SectionRetriever
        from graphrag.section_kg_builder import SectionKGBuilder
        from departments import get_section_data_dir

        data_dir = get_section_data_dir(sec_code)
        graph_path = os.path.join(data_dir, "graph.pkl")
        if not os.path.exists(graph_path):
            logger.warning(f"No ingested knowledge base found for section '{sec_code}'.")
            return None

        logger.info(f"Loading GraphRAG retriever for section '{sec_code}'...")
        graph, chunks = SectionKGBuilder.load(data_dir)
        
        from graphrag.embeddings import EmbeddingEngine
        engine = EmbeddingEngine()
        
        section_retrievers[sec_code] = SectionRetriever(
            section_code=sec_code,
            graph=graph,
            chunks=chunks,
            embedding_engine=engine
        )
        logger.info(f"✅ Retriever for section '{sec_code}' loaded!")

    return section_retrievers[sec_code]


def _is_identity_query(query: str) -> bool:
    """Return True only when the query is asking about a person's identity/role/position.

    Returns False for queries about research areas, PhD scholars, publications,
    teaching, supervision, etc. — those need department-specific retrieval.
    """
    q = query.lower().strip()

    # Non-identity intents: if the query contains ANY of these, it is NOT a
    # simple identity lookup and should NOT be short-circuited with a role card.
    non_identity_keywords = (
        # Research / academic
        "research area", "research interest", "academic interest",
        "research work", "research output", "research expertise",
        "research contribution", "research domain", "research topic",
        "research field", "area of research", "field of research",
        "areas of interest", "interest area",
        # PhD / student supervision
        "phd scholar", "phd student", "doctoral student",
        "mtech student", "m.tech student", "master student",
        "supervised by", "supervise", "supervisor", "supervision",
        "scholars under", "students under", "guided by",
        "advise", "advisor", "advisee",
        # Publications / patents
        "publication", "publications", "paper", "papers",
        "journal", "conference", "patent", "patents",
        # Teaching
        "teaching", "courses taught", "courses", "course",
        "subjects taught", "teaches",
        # Qualifications
        "qualification", "education background", "background",
        "specialization", "specialisation", "expertise area",
        # Comparisons
        "compare", "comparison", "versus", "vs",
        "difference between", "similarities",
        # Projects / startups
        "project", "projects", "startup", "startups",
        "lab", "laboratory",
    )

    if any(kw in q for kw in non_identity_keywords):
        return False

    return True


def get_global_person_direct_answer(query: str) -> Optional[str]:
    """Check if query is about a person and return a deterministic direct answer from Global Person Index.

    Only returns a role-card for identity/role queries (e.g. "who is Dr. X?",
    "tell me about Prof. Y").  Queries about research areas, PhD scholars,
    publications, teaching, etc. are deliberately skipped so that the full
    retrieval pipeline (which has actual faculty data) can handle them.
    """
    global person_index
    if not person_index:
        return None

    # Intent guard: skip role-card for non-identity queries
    if not _is_identity_query(query):
        return None

    import re
    q = query.lower().strip()

    # Differentiate roles: if administrative requested, put section roles first. Otherwise, put academic/department roles first.
    admin_keywords = ("dean", "coordinator", "registrar", "ar", "assistant registrar", "associate dean", "head", "incharge", "in charge")
    has_admin_request = any(kw in q for kw in admin_keywords)

    for name in person_index.person_roles.keys():
        # Word boundary check
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, q):
            res = person_index.lookup(name)
            if res:
                roles = res["roles"]
                
                # Sort roles
                def sort_key(role):
                    is_sec = role.get("is_section", False)
                    if has_admin_request:
                        return 0 if is_sec else 1
                    else:
                        return 1 if is_sec else 0

                sorted_roles = sorted(roles, key=sort_key)
                
                lines = [f"**{res['name']}** holds the following role(s) at IIT Jammu:"]
                for r in sorted_roles:
                    source_name = r["source"].replace("_", " ").title()
                    lines.append(f"- **{r['designation']}** under the {source_name} division/department")
                    if r.get("email"):
                        lines.append(f"  - Email: {r['email']}")
                    if r.get("phone"):
                        lines.append(f"  - Phone: {r['phone']}")
                    if r.get("office"):
                        lines.append(f"  - Office: {r['office']}")
                    if r.get("profile_url"):
                        lines.append(f"  - Profile: [Faculty Page]({r['profile_url']})")
                return "\n".join(lines)
    return None


def init_app():
    """Initialize the LLM, router, verifier, and preload all ingested retrievers."""
    global llm, multi_retriever, router, verifier, person_index

    from graphrag.llm import create_llm_from_env
    from graphrag.verifier import ResponseVerifier
    from graphrag.multi_retriever import MultiDepartmentRetriever
    from dept_router import DepartmentRouter
    from departments import DEPARTMENTS, get_data_dir, SECTIONS, get_section_data_dir
    from graphrag.person_index import GlobalPersonIndex

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

    # Preload all ingested section retrievers
    sec_loaded_count = 0
    for code in SECTIONS:
        data_dir = get_section_data_dir(code)
        if os.path.exists(os.path.join(data_dir, "graph.pkl")):
            try:
                get_section_retriever(code)
                sec_loaded_count += 1
            except Exception as e:
                logger.warning(f"Could not preload section '{code}' retriever: {e}")

    # Initialize and populate Global Person Index
    person_index = GlobalPersonIndex()
    for code, ret in retrievers.items():
        person_index.index_graph(ret.graph, code, is_section=False)
    for code, ret in section_retrievers.items():
        person_index.index_graph(ret.graph, code, is_section=True)

    # Initialize multi-department retriever
    multi_retriever = MultiDepartmentRetriever(retrievers, section_retrievers)

    logger.info(f"✅ Unified system initialized! {loaded_count} departments and {sec_loaded_count} sections loaded.")


@app.route("/")
def index():
    """Serve the main chat page."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Handle chat messages with automatic department and section routing.

    Flow:
      1. Global person index lookup to bypass retrieval for direct matches
      2. Router detects target department(s) and/or section(s) from query text
      3. MultiDepartmentRetriever/SectionRetriever fetches from the right scope
      4. LLM generates response with appropriate system prompt
      5. Verifier checks faithfulness before returning
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

        # Step 0: Check global person direct answer first
        person_direct_ans = get_global_person_direct_answer(query)
        if person_direct_ans:
            total_time = time.time() - start
            _log_retrieval_provenance(
                [], query,
                {"route": "direct_graph", "source_mode": "graph"},
                {"answerable": True},
            )
            return jsonify({
                "response": sanitize_response(person_direct_ans),
                "routed_departments": [],
                "routed_sections": [],
                "routing_reason": "Person query resolved via Global Person Index",
                "retrieval_time": round(total_time, 2),
                "total_time": round(total_time, 2),
                "direct": True,
            })

        # Step 1: Route query to department(s) and/or section(s)
        route_result = router.route(query)
        logger.info(f"Routing: '{query[:60]}...' → {route_result.confidence} | depts={route_result.departments} | sections={route_result.sections} | {route_result.reason}")

        # Step 2: Retrieve context based on routing
        
        # Scenario A: Section-only query
        if route_result.sections and not route_result.departments:
            if len(route_result.sections) == 1:
                sec_code = route_result.sections[0]
                sec_retriever = get_section_retriever(sec_code)

                if not sec_retriever:
                    return jsonify({
                        "response": f"The knowledge base for section **{sec_code}** is not available yet.",
                        "routed_departments": [],
                        "routed_sections": [sec_code],
                        "routing_reason": route_result.reason,
                        "retrieval_time": 0.0,
                        "total_time": 0.0,
                    })

                # Check for direct graph answers first
                direct_response = sec_retriever.get_direct_answer(query, global_person_index=person_index)
                if direct_response:
                    total_time = time.time() - start
                    _log_retrieval_provenance(
                        [sec_code], query,
                        {"route": "direct_graph", "source_mode": "graph"},
                        {"answerable": True},
                    )
                    return jsonify({
                        "response": sanitize_response(direct_response),
                        "routed_departments": [],
                        "routed_sections": [sec_code],
                        "routing_reason": route_result.reason,
                        "retrieval_time": round(total_time, 2),
                        "total_time": round(total_time, 2),
                        "direct": True,
                    })

                # Full hybrid retrieval for section
                bundle = sec_retriever.retrieve_bundle(query)
                context = bundle["context"]
                answerability = bundle["answerability"]
                retrieval_time = time.time() - start

                _log_retrieval_provenance([sec_code], query, bundle.get("provenance", {}), answerability)

                if not answerability.get("answerable", True):
                    total_time = time.time() - start
                    return jsonify({
                        "response": sanitize_response(bundle["fallback_response"]),
                        "routed_departments": [],
                        "routed_sections": [sec_code],
                        "routing_reason": route_result.reason,
                        "retrieval_time": round(retrieval_time, 2),
                        "total_time": round(total_time, 2),
                        "information_available": False,
                    })

                # Generate scoped response
                prompt = build_chat_prompt(query, context, dept_code=sec_code)
                system_prompt = get_system_prompt(dept_code=sec_code)
                response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1400)

                # Verify faithfulness
                verification = verifier.verify(query, context, response)
                if not verification.faithful:
                    logger.warning(f"Section response failed faithfulness check. {verification.reason}")
                    from departments import SECTIONS
                    sec_config = SECTIONS[sec_code]
                    base_url = sec_config.get("base_url", "https://iitjammu.ac.in")
                    response = (
                        f"I don't have verified information to answer that question accurately. "
                        f"Please check the official {sec_config['name']} section website at {base_url} for details."
                    )

                total_time = time.time() - start
                logger.info(f"[{sec_code.upper()}] Query: '{query[:50]}...' | Total: {total_time:.2f}s")

                return jsonify({
                    "response": response,
                    "routed_departments": [],
                    "routed_sections": [sec_code],
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": True,
                })
            else:
                # Multi-section query
                bundle = multi_retriever.retrieve_multi(query, route_result.sections)
                context = bundle["context"]
                answerability = bundle.get("answerability", {})
                retrieval_time = time.time() - start

                _log_retrieval_provenance(route_result.sections, query, bundle.get("provenance", {}), answerability)

                if not answerability.get("answerable", True):
                    total_time = time.time() - start
                    return jsonify({
                        "response": sanitize_response(bundle.get("fallback_response", "I don't have that information.")),
                        "routed_departments": [],
                        "routed_sections": route_result.sections,
                        "routing_reason": route_result.reason,
                        "retrieval_time": round(retrieval_time, 2),
                        "total_time": round(total_time, 2),
                        "information_available": False,
                    })

                dept_contexts = bundle.get("dept_contexts", {})
                prompt = build_multi_dept_chat_prompt(query, dept_contexts)
                system_prompt = get_unified_system_prompt()
                response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1800)

                total_time = time.time() - start
                return jsonify({
                    "response": response,
                    "routed_departments": [],
                    "routed_sections": bundle.get("sections", route_result.sections),
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": True,
                })

        # Scenario B: Single Department query
        elif route_result.confidence == "exact" and len(route_result.departments) == 1 and not route_result.sections:
            dept_code = route_result.departments[0]
            dept_retriever = get_retriever(dept_code)

            if not dept_retriever:
                return jsonify({
                    "response": f"The knowledge base for **{dept_code}** is not available yet.",
                    "routed_departments": [dept_code],
                    "routed_sections": [],
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
                    "routed_sections": [],
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
                    "routed_sections": [],
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": False,
                })

            # Generate scoped response
            prompt = build_chat_prompt(query, context, dept_code=dept_code)
            system_prompt = get_system_prompt(dept_code=dept_code)
            response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1400)

            # Verify faithfulness
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
                "routed_sections": [],
                "routing_reason": route_result.reason,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "information_available": True,
            })

        # Scenario C: Multi-system query (combination of departments and/or sections)
        elif (route_result.confidence == "exact" and 
              (len(route_result.departments) + len(route_result.sections)) > 1):
            all_codes = route_result.departments + route_result.sections
            bundle = multi_retriever.retrieve_multi(query, all_codes)
            context = bundle["context"]
            answerability = bundle.get("answerability", {})
            retrieval_time = time.time() - start

            _log_retrieval_provenance(all_codes, query, bundle.get("provenance", {}), answerability)

            if not answerability.get("answerable", True):
                total_time = time.time() - start
                return jsonify({
                    "response": sanitize_response(bundle.get("fallback_response", "I don't have that information.")),
                    "routed_departments": bundle.get("departments", []),
                    "routed_sections": bundle.get("sections", []),
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": False,
                })

            # Generate with unified prompt
            dept_contexts = bundle.get("dept_contexts", {})
            prompt = build_multi_dept_chat_prompt(query, dept_contexts)
            system_prompt = get_unified_system_prompt()
            response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1800)

            # Verify faithfulness
            verification = verifier.verify(query, context, response)
            if not verification.faithful:
                logger.warning(f"Cross-system response failed faithfulness check. {verification.reason}")
                response = (
                    "I don't have verified information to answer that question accurately. "
                    "Please check the official IIT Jammu website for details."
                )

            total_time = time.time() - start
            logger.info(f"[MULTI] Query: '{query[:50]}...' | Total: {total_time:.2f}s")

            return jsonify({
                "response": response,
                "routed_departments": bundle.get("departments", []),
                "routed_sections": bundle.get("sections", []),
                "routing_reason": route_result.reason,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "information_available": True,
            })

        # Scenario D: Broadcast query (fallback)
        else:
            bundle = multi_retriever.retrieve_broadcast(query, top_n=len(retrievers) + len(section_retrievers))
            context = bundle["context"]
            answerability = bundle.get("answerability", {})
            routed_depts = bundle.get("departments", [])
            routed_secs = bundle.get("sections", [])
            retrieval_time = time.time() - start

            _log_retrieval_provenance(routed_depts + routed_secs, query, bundle.get("provenance", {}), answerability)

            # Handle direct answers from broadcast
            if bundle.get("direct"):
                if len(routed_depts) + len(routed_secs) <= 1:
                    total_time = time.time() - start
                    return jsonify({
                        "response": sanitize_response(context),
                        "routed_departments": routed_depts,
                        "routed_sections": routed_secs,
                        "routing_reason": route_result.reason,
                        "retrieval_time": round(retrieval_time, 2),
                        "total_time": round(total_time, 2),
                        "direct": True,
                    })
                else:
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
                        "routed_sections": routed_secs,
                        "routing_reason": route_result.reason,
                        "retrieval_time": round(retrieval_time, 2),
                        "total_time": round(total_time, 2),
                        "direct": True,
                    })

            if not answerability.get("answerable", True):
                total_time = time.time() - start
                return jsonify({
                    "response": sanitize_response(bundle.get("fallback_response",
                        "I don't have that specific information. Try mentioning a department/section for better results.")),
                    "routed_departments": routed_depts,
                    "routed_sections": routed_secs,
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": False,
                })

            # Generate response
            if len(routed_depts) == 1 and not routed_secs:
                dept_code = routed_depts[0]
                prompt = build_chat_prompt(query, context, dept_code=dept_code)
                system_prompt = get_system_prompt(dept_code=dept_code)
            elif len(routed_secs) == 1 and not routed_depts:
                sec_code = routed_secs[0]
                prompt = build_chat_prompt(query, context, dept_code=sec_code)
                system_prompt = get_system_prompt(dept_code=sec_code)
            else:
                dept_contexts = bundle.get("dept_contexts", {})
                if dept_contexts:
                    prompt = build_multi_dept_chat_prompt(query, dept_contexts)
                else:
                    from departments import DEPARTMENTS, SECTIONS
                    prompt = build_multi_dept_chat_prompt(query, {
                        code: {
                            "name": (SECTIONS.get(code, {}).get("name", code.upper()) if code in section_retrievers else DEPARTMENTS.get(code, {}).get("full_name", code)),
                            "context": context
                        }
                        for code in (routed_depts + routed_secs)
                    }) if (routed_depts + routed_secs) else build_chat_prompt(query, context, dept_code="ee")
                system_prompt = get_unified_system_prompt()

            response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=1400)

            # Verify faithfulness
            verification = verifier.verify(query, context, response)
            if not verification.faithful:
                logger.warning(f"Broadcast response failed faithfulness check. {verification.reason}")
                response = (
                    "I don't have verified information to answer that question accurately. "
                    "Please try asking about a specific department or section for better results."
                )

            total_time = time.time() - start
            logger.info(f"[BROADCAST] Query: '{query[:50]}...' | Routed depts: {routed_depts} | Routed secs: {routed_secs} | Total: {total_time:.2f}s")

            return jsonify({
                "response": response,
                "routed_departments": routed_depts,
                "routed_sections": routed_secs,
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
