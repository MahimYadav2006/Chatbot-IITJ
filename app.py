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
response_cache = None           # L1 ResponseCache


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
    Also returns False for admin-role queries (e.g. "who is the dean") that
    should be routed to the administration department's deterministic logic.
    """
    import re
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
        # Section items (clubs, hostels, fests, sports, mous, etc.)
        "club", "clubs", "hostel", "hostels", "fest", "fests", "festival", "festivals",
        "sports", "facility", "facilities", "mou", "mous", "collaboration", "collaborations",
        "medical service", "medical services", "service timings", "hospital", "hospitals",
        "timing", "timings",
        # Academic terms
        "minor", "minors", "credit", "credits", "curriculum", "semester", 
        "eligible", "eligibility", "syllabus", "degree",
        # Office / address queries — need full retrieval, not identity fast-path
        "address", "office address", "office location", "office of",
        "cabin", "room number", "where is the office",
        # Admin role queries — should go through administration dept routing,
        # not person index (which can contain role-placeholder names).
        "dean of", "dean ", "associate dean",
        "registrar", "director of iit",
        "bog chairman", "chairman of",
    )

    matched_non_id = [kw for kw in non_identity_keywords if kw in q]
    if matched_non_id:
        admin_roles = {"dean of", "dean ", "associate dean", "registrar", "director of iit", "bog chairman", "chairman of"}
        if all(kw in admin_roles for kw in matched_non_id) and person_index:
            for name in person_index.person_roles.keys():
                pattern = r"\b" + re.escape(name.lower()) + r"\b"
                if re.search(pattern, q):
                    return True
        return False

    return True


def extract_person_name_from_query(query: str) -> Optional[str]:
    """Extract a name from queries like 'Who is X?' or 'tell me about X'."""
    import re
    q = query.strip()
    patterns = [
        r"(?i)\bwho\s+is\s+(?:dr\.?|prof\.?|mr\.?|ms\.?|sh\.?|shri\.?|smt\.?)?\s*([a-zA-Z\s]+)",
        r"(?i)\btell\s+me\s+about\s+(?:dr\.?|prof\.?|mr\.?|ms\.?|sh\.?|shri\.?|smt\.?)?\s*([a-zA-Z\s]+)",
        r"(?i)\bprofile\s+of\s+(?:dr\.?|prof\.?|mr\.?|ms\.?|sh\.?|shri\.?|smt\.?)?\s*([a-zA-Z\s]+)",
        r"(?i)\binfo\s+on\s+(?:dr\.?|prof\.?|mr\.?|ms\.?|sh\.?|shri\.?|smt\.?)?\s*([a-zA-Z\s]+)",
        r"(?i)\binformation\s+about\s+(?:dr\.?|prof\.?|mr\.?|ms\.?|sh\.?|shri\.?|smt\.?)?\s*([a-zA-Z\s]+)",
        r"(?i)\bwho\s+was\s+(?:dr\.?|prof\.?|mr\.?|ms\.?|sh\.?|shri\.?|smt\.?)?\s*([a-zA-Z\s]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            name = m.group(1).strip()
            name = re.sub(r"[?!.,;:]", "", name).strip()
            if len(name) >= 3 and len(name.split()) <= 4:
                return name
    
    # Fallback: if query is exactly 2 or 3 words and all are alphabetic
    words = q.split()
    if 2 <= len(words) <= 3 and all(w.isalpha() and len(w) >= 3 for w in words):
        return q
    return None


def search_scraped_data_for_person(name: str) -> Optional[str]:
    """Search all .md files in scraped_data for the given person name and extract context."""
    import os
    import re
    scraped_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "scraped_data"))
    if not os.path.exists(scraped_dir):
        return None

    clean_name = name.strip().lower()
    if not clean_name:
        return None

    matches = []
    max_files = 5
    pattern = re.compile(r"\b" + re.escape(clean_name) + r"\b", re.IGNORECASE)

    for root, dirs, files in os.walk(scraped_dir):
        for file in files:
            if not file.endswith(".md"):
                continue
            if "00_combined" in file:
                continue
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if pattern.search(content):
                        lines = content.splitlines()
                        file_matches = []
                        for i, line in enumerate(lines):
                            if pattern.search(line):
                                start_idx = max(0, i - 4)
                                end_idx = min(len(lines), i + 10)
                                snippet = "\n".join(lines[start_idx:end_idx])
                                file_matches.append(snippet)
                                break
                        if file_matches:
                            rel_path = os.path.relpath(filepath, scraped_dir)
                            matches.append((rel_path, file_matches[0]))
                            if len(matches) >= max_files:
                                break
            except Exception as e:
                logger.warning(f"Error reading file {filepath} in markdown fallback: {e}")
        if len(matches) >= max_files:
            break

    if not matches:
        return None

    blocks = []
    for rel_path, snippet in matches:
        norm_path = rel_path.replace("\\", "/")
        source_name = norm_path.split("/")[0].replace("_", " ").title()
        blocks.append(f"### Source: {source_name} ({norm_path})\n\n{snippet}")

    return "\n\n---\n\n".join(blocks)


def get_global_person_direct_answer(query: str) -> Optional[str]:
    """Check if query is about a person and return a deterministic direct answer from Global Person Index.

    Only returns a role-card for identity/role queries (e.g. "who is Dr. X?",
    "tell me about Prof. Y").  Queries about research areas, PhD scholars,
    publications, teaching, etc. are deliberately skipped so that the full
    retrieval pipeline (which has actual faculty data) can handle them.
    """
    global person_index
    if not person_index:
        # Even if person_index is not loaded, we can still attempt raw markdown fallback search
        extracted_name = extract_person_name_from_query(query)
        if extracted_name:
            fallback_ctx = search_scraped_data_for_person(extracted_name)
            if fallback_ctx:
                return f"**{extracted_name}** is mentioned in the following department/division documents:\n\n{fallback_ctx}"
        return None

    # Intent guard: skip role-card for non-identity queries
    if not _is_identity_query(query):
        return None

    import re
    q = query.lower().strip()

    # Differentiate roles: if administrative requested, put section roles first. Otherwise, put academic/department roles first.
    admin_keywords = ("dean", "coordinator", "registrar", "ar", "assistant registrar", "associate dean", "head", "incharge", "in charge")
    has_admin_request = any(kw in q for kw in admin_keywords)

    matched_name = None
    for name in person_index.person_roles.keys():
        # Word boundary check
        pattern = r"\b" + re.escape(name.lower()) + r"\b"
        if re.search(pattern, q):
            matched_name = name
            break

    if not matched_name:
        # Try matching unique/significant parts of the name of length >= 4
        # We explicitly ignore generic terms/titles to prevent false matches on queries like "clubs at IIT Jammu"
        ignored_tokens = {
            "prof", "profs", "professor", "professors", "appointed", "by", "iit", "jammu", 
            "committee", "members", "member", "faculty", "staff", "administration", "administrator", 
            "officer", "officers", "department", "division", "counselling", "counselor", 
            "counselors", "services", "service", "medical", "centre", "center", "health", 
            "unit", "cell", "library", "librarian", "office", "head", "chairperson", 
            "coordinator", "assistant", "associate", "dean", "deans", "director", "registrar",
            "institutes", "institute", "iitj", "kumar", "singh", "sharma", "doctor", "dr",
            "computer", "science", "engineering", "electrical", "mechanical", "civil", 
            "chemical", "biosciences", "bioengineering", "materials", "physics", 
            "chemistry", "mathematics", "humanities", "social",
            "gupta", "yadav", "verma", "patel", "das", "devi", "prasad", "lal", 
            "choudhary", "sen", "roy", "mishra", "pandey", "dutta", "bose"
        }
        for name in person_index.person_roles.keys():
            parts = [p.lower() for p in name.split() if len(p) >= 4 and p.lower() not in ignored_tokens]
            if parts and any(re.search(r"\b" + re.escape(part) + r"\b", q) for part in parts):
                matched_name = name
                break

    if matched_name:
        res = person_index.lookup(matched_name)
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
                
                display_name = res['name']
                is_doc = any(r.get("label") == "MedicalDoctor" for r in sorted_roles)
                is_student = any(r.get("is_student") for r in sorted_roles)
                if is_doc and not display_name.startswith("Dr. ") and not display_name.startswith("Dr "):
                    display_name = f"Dr. {display_name}"

                if is_student:
                    # Build student-specific role card
                    lines = [f"**{display_name}** is a student at IIT Jammu:"]
                    for r in sorted_roles:
                        source_name = r["source"].replace("_", " ").title()
                        program = r.get("program", "")
                        lines.append(f"- **{program} Scholar** in the {source_name} department")
                        if r.get("supervisor"):
                            lines.append(f"  - Supervisor: {r['supervisor']}")
                        if r.get("research_area"):
                            lines.append(f"  - Research Area: {r['research_area']}")
                        if r.get("thesis_title"):
                            lines.append(f"  - Thesis: {r['thesis_title']}")
                        if r.get("email"):
                            lines.append(f"  - Email: {r['email']}")
                else:
                    lines = [f"**{display_name}** holds the following role(s) at IIT Jammu:"]
                    for r in sorted_roles:
                        source_name = r["source"].replace("_", " ").title()
                        lines.append(f"- **{r['designation']}** under the {source_name} division/department")
                        if r.get("qualifications"):
                            lines.append(f"  - Qualifications: {r['qualifications']}")
                        if r.get("experience"):
                            lines.append(f"  - Experience: {r['experience']}")
                        if r.get("email"):
                            lines.append(f"  - Email: {r['email']}")
                        if r.get("phone"):
                            lines.append(f"  - Phone: {r['phone']}")
                        if r.get("office"):
                            lines.append(f"  - Office: {r['office']}")
                        if r.get("profile_url"):
                            lines.append(f"  - Profile: [Faculty Page]({r['profile_url']})")
                return "\n".join(lines)

    # Try raw markdown fallback search before giving up
    extracted_name = extract_person_name_from_query(query)
    if extracted_name:
        fallback_ctx = search_scraped_data_for_person(extracted_name)
        if fallback_ctx:
            return f"**{extracted_name}** is mentioned in the following department/division documents:\n\n{fallback_ctx}"

    return None


def init_app():
    """Initialize the LLM, router, verifier, and preload all ingested retrievers."""
    global llm, multi_retriever, router, verifier, person_index, response_cache

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

    from graphrag.cache import get_response_cache, is_cache_enabled
    if is_cache_enabled():
        response_cache = get_response_cache()

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


    def _finalize_response(resp_dict):
        if response_cache and resp_dict.get("information_available"):
            response_cache.set(cache_key, resp_dict)
        return jsonify(resp_dict)

    from graphrag.cache import normalize_query
    cache_key = normalize_query(query) if response_cache else None

    if response_cache:
        cached_resp = response_cache.get(cache_key)
        if cached_resp:
            logger.info(f"[CACHE HIT] L1 Response Cache for query: {query}")
            return jsonify(cached_resp)

    try:
        start = time.time()

        # Step 0: Check global person index for identity queries
        # For pure identity queries ("Who is Dr. X?"), fast-path via short LLM call.
        # For non-identity queries with person names, save context as supplementary.
        person_context = None
        person_direct_ans = get_global_person_direct_answer(query)
        if person_direct_ans and _is_identity_query(query):
            # Identity fast-path: LLM synthesizes from person context (short response)
            # Use the person's actual department for system prompt scoping
            identity_dept_code = "administration"
            if person_index:
                import re
                q_lower = query.lower().strip()
                for name in person_index.person_roles.keys():
                    pattern = r"\b" + re.escape(name.lower()) + r"\b"
                    if re.search(pattern, q_lower):
                        for r in person_index.person_roles[name]:
                            if r.get("source") and not r.get("is_section", False):
                                identity_dept_code = r["source"]
                                break
                        break
                # Partial name fallback
                if identity_dept_code == "administration":
                    ignored_tokens = {
                        "prof", "professor", "iit", "jammu", "kumar", "singh", "sharma", "dr",
                    }
                    for name in person_index.person_roles.keys():
                        parts = [p.lower() for p in name.split() if len(p) >= 4 and p.lower() not in ignored_tokens]
                        if parts and any(re.search(r"\b" + re.escape(part) + r"\b", q_lower) for part in parts):
                            for r in person_index.person_roles[name]:
                                if r.get("source") and not r.get("is_section", False):
                                    identity_dept_code = r["source"]
                                    break
                            break

            # Extract department code from raw markdown search results if applicable
            if "### Source:" in person_direct_ans:
                import re
                for line in person_direct_ans.splitlines():
                    if "Source:" in line:
                        m = re.search(r"\(([^/]+)/", line)
                        if m:
                            potential_dept = m.group(1)
                            from departments import DEPARTMENTS, SECTIONS
                            if potential_dept in DEPARTMENTS or potential_dept in SECTIONS:
                                identity_dept_code = potential_dept
                                break

            prompt = build_chat_prompt(query, person_direct_ans, dept_code=identity_dept_code)
            system_prompt = get_system_prompt(dept_code=identity_dept_code)
            response = llm.generate(prompt, system_prompt=system_prompt, max_tokens=300)
            total_time = time.time() - start
            _log_retrieval_provenance(
                [], query,
                {"route": "identity_fast_path", "source_mode": "graph+llm"},
                {"answerable": True},
            )
            return _finalize_response({
                "response": sanitize_response(response),
                "routed_departments": [],
                "routed_sections": [],
                "routing_reason": "Identity query resolved via Global Person Index + LLM",
                "retrieval_time": round(total_time, 2),
                "total_time": round(total_time, 2),
                "direct": True,
            })
        elif person_direct_ans:
            # Non-identity query with person name — save as supplementary context
            person_context = person_direct_ans
            logger.info("Person context found but query is not identity — continuing to full retrieval.")

        # Step 1: Route query to department(s) and/or section(s)
        route_result = router.route(query)

        # RC-5: Person-aware department injection for broadcast queries.
        # When a person is matched but the query isn't a pure identity question
        # (e.g., "address of Dr. Rimen Jamatia"), inject the person's department
        # into the route result so retrieval targets the correct department.
        if route_result.confidence == "broadcast" and person_context and person_index:
            # Extract department(s) from the matched person's roles
            import re
            q = query.lower().strip()
            for name in person_index.person_roles.keys():
                pattern = r"\b" + re.escape(name.lower()) + r"\b"
                if re.search(pattern, q):
                    person_roles = person_index.person_roles[name]
                    for r in person_roles:
                        source = r.get("source", "")
                        if source and not r.get("is_section", False):
                            if source not in route_result.departments:
                                route_result.departments.append(source)
                                route_result.confidence = "exact"
                                route_result.reason = f"Person-aware routing: matched '{name}' → {source}"
                                logger.info(f"Person-aware routing: injected dept '{source}' for person '{name}'")
                    break
            # Also try partial name matching (same as get_global_person_direct_answer)
            if route_result.confidence == "broadcast":
                for name in person_index.person_roles.keys():
                    parts = [p.lower() for p in name.split() if len(p) >= 4]
                    if parts and any(re.search(r"\b" + re.escape(part) + r"\b", q) for part in parts):
                        person_roles = person_index.person_roles[name]
                        for r in person_roles:
                            source = r.get("source", "")
                            if source and not r.get("is_section", False):
                                if source not in route_result.departments:
                                    route_result.departments.append(source)
                                    route_result.confidence = "exact"
                                    route_result.reason = f"Person-aware routing: matched '{name}' → {source}"
                                    logger.info(f"Person-aware routing: injected dept '{source}' for person '{name}'")
                        break

        logger.info(f"Routing: '{query[:60]}...' → {route_result.confidence} | depts={route_result.departments} | sections={route_result.sections} | {route_result.reason}")

        # Step 2: Retrieve context based on routing
        
        # Scenario A: Section-only query
        if route_result.sections and not route_result.departments:
            if len(route_result.sections) == 1:
                sec_code = route_result.sections[0]
                sec_retriever = get_section_retriever(sec_code)

                if not sec_retriever:
                    return _finalize_response({
                        "response": f"The knowledge base for section **{sec_code}** is not available yet.",
                        "routed_departments": [],
                        "routed_sections": [sec_code],
                        "routing_reason": route_result.reason,
                        "retrieval_time": 0.0,
                        "total_time": 0.0,
                    })

                # Full hybrid retrieval for section (deterministic context is handled inside retrieve_bundle)
                bundle = sec_retriever.retrieve_bundle(query)
                context = bundle["context"]
                # Prepend person context if available
                if person_context:
                    context = f"## Person Information\n\n{person_context}\n\n---\n\n{context}"
                answerability = bundle["answerability"]
                retrieval_time = time.time() - start

                _log_retrieval_provenance([sec_code], query, bundle.get("provenance", {}), answerability)

                if not answerability.get("answerable", True):
                    total_time = time.time() - start
                    return _finalize_response({
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
                faithfulness_ok = verification.faithful
                if not faithfulness_ok:
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

                return _finalize_response({
                    "response": response,
                    "routed_departments": [],
                    "routed_sections": [sec_code],
                    "routing_reason": route_result.reason,
                    "retrieval_time": round(retrieval_time, 2),
                    "total_time": round(total_time, 2),
                    "information_available": faithfulness_ok,
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
                    return _finalize_response({
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
                return _finalize_response({
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
                return _finalize_response({
                    "response": f"The knowledge base for **{dept_code}** is not available yet.",
                    "routed_departments": [dept_code],
                    "routed_sections": [],
                    "routing_reason": route_result.reason,
                    "retrieval_time": 0.0,
                    "total_time": 0.0,
                })

            # Full hybrid retrieval (deterministic context is handled inside retrieve_bundle)
            bundle = dept_retriever.retrieve_bundle(
                query, local_top_k=5, vector_top_k=5, global_top_k=3, max_context_words=4500
            )
            context = bundle["context"]
            # Prepend person context if available
            if person_context:
                context = f"## Person Information\n\n{person_context}\n\n---\n\n{context}"
            answerability = bundle["answerability"]
            retrieval_time = time.time() - start

            _log_retrieval_provenance([dept_code], query, bundle.get("provenance", {}), answerability)

            if not answerability.get("answerable", True):
                total_time = time.time() - start
                return _finalize_response({
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
            faithfulness_ok = verification.faithful
            if not faithfulness_ok:
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

            return _finalize_response({
                "response": response,
                "routed_departments": [dept_code],
                "routed_sections": [],
                "routing_reason": route_result.reason,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "information_available": faithfulness_ok,
            })

        # Scenario C: Multi-system query (combination of departments and/or sections)
        elif (route_result.confidence == "exact" and 
              (len(route_result.departments) + len(route_result.sections)) > 1):
            all_codes = route_result.departments + route_result.sections
            bundle = multi_retriever.retrieve_multi(query, all_codes)
            context = bundle["context"]
            # Prepend person context if available
            if person_context:
                context = f"## Person Information\n\n{person_context}\n\n---\n\n{context}"
            answerability = bundle.get("answerability", {})
            retrieval_time = time.time() - start

            _log_retrieval_provenance(all_codes, query, bundle.get("provenance", {}), answerability)

            if not answerability.get("answerable", True):
                total_time = time.time() - start
                return _finalize_response({
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
            faithfulness_ok = verification.faithful
            if not faithfulness_ok:
                logger.warning(f"Cross-system response failed faithfulness check. {verification.reason}")
                response = (
                    "I don't have verified information to answer that question accurately. "
                    "Please check the official IIT Jammu website for details."
                )

            total_time = time.time() - start
            logger.info(f"[MULTI] Query: '{query[:50]}...' | Total: {total_time:.2f}s")

            return _finalize_response({
                "response": response,
                "routed_departments": bundle.get("departments", []),
                "routed_sections": bundle.get("sections", []),
                "routing_reason": route_result.reason,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "information_available": faithfulness_ok,
            })

        # Scenario D: Broadcast query (fallback)
        else:
            bundle = multi_retriever.retrieve_broadcast(query, top_n=10)
            context = bundle["context"]
            # Prepend person context if available
            if person_context:
                context = f"## Person Information\n\n{person_context}\n\n---\n\n{context}"
            answerability = bundle.get("answerability", {})
            routed_depts = bundle.get("departments", [])
            routed_secs = bundle.get("sections", [])
            retrieval_time = time.time() - start

            _log_retrieval_provenance(routed_depts + routed_secs, query, bundle.get("provenance", {}), answerability)

            if not answerability.get("answerable", True):
                total_time = time.time() - start
                return _finalize_response({
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
            faithfulness_ok = verification.faithful
            if not faithfulness_ok:
                logger.warning(f"Broadcast response failed faithfulness check. {verification.reason}")
                response = (
                    "I don't have verified information to answer that question accurately. "
                    "Please try asking about a specific department or section for better results."
                )

            total_time = time.time() - start
            logger.info(f"[BROADCAST] Query: '{query[:50]}...' | Routed depts: {routed_depts} | Routed secs: {routed_secs} | Total: {total_time:.2f}s")

            return _finalize_response({
                "response": response,
                "routed_departments": routed_depts,
                "routed_sections": routed_secs,
                "routing_reason": route_result.reason,
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(total_time, 2),
                "information_available": faithfulness_ok,
            })
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return _finalize_response({
            "error": "An error occurred while processing your request.",
            "response": "I'm sorry, I encountered an error. Please try again."
        }), 500


@app.route("/api/llm-status", methods=["GET"])
def llm_status():
    """Return the current LLM provider and whether an API key is configured."""
    global llm
    if llm is None:
        return jsonify({
            "provider": None,
            "has_api_key": False,
            "model": None
        })
    
    provider = getattr(llm, "provider", "unknown")
    api_key = getattr(llm, "api_key", None)
    model = getattr(llm, "model", None)
    
    return jsonify({
        "provider": provider,
        "has_api_key": bool(api_key),
        "model": model
    })


@app.route("/api/set-gemini-key", methods=["POST", "DELETE"])
def set_gemini_key():
    """Set or clear the Gemini API key in-memory."""
    global llm, verifier
    from graphrag.llm import GeminiLLM
    from graphrag.verifier import ResponseVerifier
    
    if request.method == "DELETE":
        logger.info("Reverting Gemini API key to default config")
        llm = GeminiLLM()
        verifier = ResponseVerifier(llm)
        if response_cache: response_cache.clear()
        return jsonify({"ok": True, "message": "Gemini API key reverted to default config"})
        
    data = request.get_json() or {}
    api_key = data.get("api_key", "").strip()
    
    if not api_key:
        return jsonify({"ok": False, "error": "API key cannot be empty"}), 400
        
    valid, error_msg = GeminiLLM.validate_key(api_key)
    if not valid:
        return jsonify({"ok": False, "error": f"Invalid API key: {error_msg}"}), 400
        
    logger.info("Updating global Gemini LLM with user-provided API key")
    llm = GeminiLLM(api_key=api_key)
    verifier = ResponseVerifier(llm)
    if response_cache: response_cache.clear()
    
    return jsonify({
        "ok": True,
        "message": "Gemini API key successfully configured",
        "model": llm.model
    })


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



@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    stats = {}
    if response_cache:
        stats["L1_response_cache"] = response_cache.stats()
    else:
        stats["L1_response_cache"] = "disabled"
        
    stats["L2_bundle_caches"] = {}
    for code, ret in retrievers.items():
        if hasattr(ret, "bundle_cache") and ret.bundle_cache:
            stats["L2_bundle_caches"][code] = ret.bundle_cache.stats()
    for code, ret in section_retrievers.items():
        if hasattr(ret, "bundle_cache") and ret.bundle_cache:
            stats["L2_bundle_caches"][code] = ret.bundle_cache.stats()
            
    return jsonify(stats)

@app.route("/api/cache/clear", methods=["POST"])
def cache_clear():
    cleared = 0
    if response_cache:
        response_cache.clear()
        cleared += 1
    for ret in retrievers.values():
        if hasattr(ret, "bundle_cache") and ret.bundle_cache:
            ret.bundle_cache.clear()
            cleared += 1
    for ret in section_retrievers.values():
        if hasattr(ret, "bundle_cache") and ret.bundle_cache:
            ret.bundle_cache.clear()
            cleared += 1
    return jsonify({"ok": True, "message": f"Cleared {cleared} caches."})

if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5050, debug=False)
