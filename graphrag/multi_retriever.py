"""
Cross-Department Retrieval Orchestrator for GraphRAG.

Coordinates retrieval across multiple department-scoped HybridRetriever
instances. Supports:
  - Single department delegation
  - Multi-department merging with department headers
  - Broadcast search across all departments with relevance ranking
"""

import logging
from typing import Dict, List, Any, Optional

from departments import DEPARTMENTS

logger = logging.getLogger(__name__)


class MultiDepartmentRetriever:
    """Orchestrates retrieval across 1+ department-scoped HybridRetrievers and SectionRetrievers."""

    def __init__(self, retrievers: Dict, section_retrievers: Dict = None):
        """
        Args:
            retrievers: Dict mapping dept_code → HybridRetriever instance.
            section_retrievers: Dict mapping sec_code → SectionRetriever instance.
        """
        self.retrievers = retrievers
        self.section_retrievers = section_retrievers or {}

    def retrieve_single(self, query: str, dept_code: str) -> Dict[str, Any]:
        """Retrieve from a single department. Delegates directly."""
        retriever = self.retrievers.get(dept_code)
        if not retriever:
            return {
                "context": "",
                "provenance": {},
                "answerability": {"answerable": False, "reason": f"Department '{dept_code}' not loaded."},
                "fallback_response": f"The knowledge base for **{dept_code}** is not available.",
                "departments": [dept_code],
            }

        bundle = retriever.retrieve_bundle(
            query, local_top_k=5, vector_top_k=5, global_top_k=3, max_context_words=4500
        )
        bundle["departments"] = [dept_code]
        return bundle

    def retrieve_multi(self, query: str, dept_codes: List[str]) -> Dict[str, Any]:
        """
        Retrieve from multiple departments/sections and merge contexts with headers.
        """
        dept_contexts = {}
        dept_provenances = {}
        all_answerable = False

        for code in dept_codes:
            retriever = self.retrievers.get(code) or self.section_retrievers.get(code)
            if not retriever:
                logger.warning(f"Retriever for '{code}' not available, skipping.")
                continue

            is_sec = code in self.section_retrievers
            if is_sec:
                bundle = retriever.retrieve_bundle(query)
            else:
                bundle = retriever.retrieve_bundle(
                    query, local_top_k=4, vector_top_k=4, global_top_k=2, max_context_words=2500
                )
                
            # Only keep the context if it is actually relevant
            if not self._is_bundle_relevant(query, bundle, is_topic=False):
                logger.info(f"Multi-retrieval: excluding '{code}' as not relevant to query.")
                continue

            if bundle.get("answerability", {}).get("answerable", False):
                all_answerable = True

            if is_sec:
                from departments import SECTIONS
                dept_name = SECTIONS.get(code, {}).get("name", code.upper())
            else:
                dept_name = DEPARTMENTS.get(code, {}).get("full_name", code.upper())
                
            dept_contexts[code] = {
                "name": dept_name,
                "context": bundle["context"],
            }
            dept_provenances[code] = bundle.get("provenance", {})

        if not dept_contexts:
            return {
                "context": "",
                "provenance": {},
                "answerability": {"answerable": False, "reason": "No departments or sections available."},
                "fallback_response": "No relevant knowledge bases are available.",
                "departments": [],
                "sections": [],
                "dept_contexts": {},
            }

        # Merge contexts with headers
        merged_sections = []
        for code in dept_codes:
            if code not in dept_contexts:
                continue
            entry = dept_contexts[code]
            context = entry["context"].strip()
            if context and context != "No relevant information found in the knowledge graph for this query.":
                merged_sections.append(f"## {entry['name']}\n\n{context}")

        if not merged_sections:
            merged_context = "No relevant information found across the queried departments/sections."
            all_answerable = False
        else:
            merged_context = "\n\n---\n\n".join(merged_sections)

        return {
            "context": merged_context,
            "provenance": dept_provenances,
            "answerability": {
                "answerable": all_answerable,
                "reason": "" if all_answerable else "No relevant evidence across queried systems.",
                "matched_terms": [],
                "missing_concepts": [],
            },
            "fallback_response": None if all_answerable else (
                "I don't have enough information across those sections to answer this query."
            ),
            "departments": [c for c in dept_codes if c in dept_contexts and c not in self.section_retrievers],
            "sections": [c for c in dept_codes if c in dept_contexts and c in self.section_retrievers],
            "dept_contexts": dept_contexts,
        }

    @staticmethod
    def _is_topic_query(query: str) -> bool:
        """Detect broad subject/topic queries that need full cross-department coverage.

        Examples that should return True:
            - "Who teaches deep learning?"
            - "Who works on machine learning?"
            - "Faculty working on VLSI"
            - "Which professors do research in robotics?"
            - "Computer vision researchers at IIT Jammu"
            - "Experts in deep learning"
        """
        import re
        q = re.sub(r"\s+", " ", query.lower()).strip()

        # Pattern 1: structural verb-based topic patterns
        topic_patterns = (
            r"who\s+(?:teaches|teach|works?\s+on|research(?:es|ing)?|stud(?:ies|ying))\b",
            # faculty/professor VERBING (working/researching/teaching) or followed by "who" — NOT just "in/on"
            r"(?:faculty|professor|professors|teacher|teachers)\s+(?:working|researching|teaching|who)\b",
            r"which\s+(?:faculty|professor|professors)\s+(?:work|teach|research|are)\b",
            r"who\s+(?:is\s+)?(?:working\s+on|expert\s+in|specialist\s+in|doing|involved\s+in)\b",
            r"who\s+(?:should\s+i\s+)?contact\s+(?:for|about|regarding)\b",
            # "experts in X", "specialists in X"
            r"(?:experts?|specialists?)\s+(?:in|on|for)\s+\w",
            # "X researchers at IIT", "X experts at IIT"
            r"\w+\s+(?:researchers?|experts?)\s+(?:at|in)\s+iit\b",
            # "who is doing X at IIT Jammu"
            r"who\s+is\s+(?:doing|working\s+on|working\s+in)\b",
            # "any faculty working in X", "is there any professor for X"
            r"(?:any|is\s+there\s+any)\s+(?:faculty|professor|researcher)\s+(?:working|who\s+works)\b",
        )
        if any(re.search(p, q) for p in topic_patterns):
            return True

        # Pattern 2: broad subject keywords without a department name
        from departments import DEPARTMENTS
        from dept_router import DEPT_NAME_ALIASES
        all_dept_aliases = set()
        for aliases in DEPT_NAME_ALIASES.values():
            all_dept_aliases.update(a.lower() for a in aliases)
        has_dept_name = any(a in q for a in all_dept_aliases if len(a) > 2)

        broad_subjects = (
            "deep learning", "machine learning", "artificial intelligence",
            "robotics", "vlsi", "signal processing", "control", "antenna",
            "communications", "power systems", "embedded", "iot",
            "data science", "computer vision", "nlp", "natural language",
            "neural network", "renewable energy", "nanotechnology",
            "thermodynamics", "fluid", "structural", "biomedical",
            "image processing", "speech processing", "reinforcement learning",
            "transfer learning", "graph neural", "large language model",
            "computer networks", "cyber security", "cybersecurity",
            "quantum computing", "material science", "structural analysis",
            "finite element", "geotechnical", "water resources",
        )
        if not has_dept_name and any(subj in q for subj in broad_subjects):
            return True

        return False

    def _is_person_or_comparison_query(self, query: str) -> bool:
        """Detect queries about specific people or comparisons between people.

        Used to prevent admin-only direct answers from short-circuiting
        broadcast when richer faculty data is available from departments.
        """
        import re
        q = re.sub(r"\s+", " ", query.lower()).strip()

        # Comparison indicators
        comparison_terms = (
            "compare", "comparison", "versus", "vs", "difference between",
            "similarities", "distinguish",
        )
        if any(term in q for term in comparison_terms):
            return True

        # Person identity queries ("who is Dr. X", "tell me about Prof. X")
        person_patterns = (
            r"\bwho\s+is\s+(?:dr\.?|prof\.?|mr\.?|ms\.?)\s+\w+",
            r"\btell\s+(?:me\s+)?about\s+(?:dr\.?|prof\.?|mr\.?|ms\.?)\s+\w+",
            r"\babout\s+(?:dr\.?|prof\.?)\s+\w+",
        )
        if any(re.search(p, q) for p in person_patterns):
            return True

        # Research-focused queries mentioning people
        research_person_terms = (
            "research work", "research interest", "research area",
            "research expertise", "research contribution", "research domain",
            "research topic", "research field", "academic interest",
            "areas of interest", "area of research", "field of research",
            "publication", "publications", "academic work",
            "specialization", "specialisation", "expertise",
        )
        has_honorific = bool(re.search(r"\b(?:dr\.?|prof\.?)\b", q))
        if has_honorific and any(term in q for term in research_person_terms):
            return True

        # PhD / student supervision queries mentioning people
        supervision_terms = (
            "phd scholar", "phd student", "doctoral student",
            "scholars under", "students under", "supervised by",
            "supervisor", "supervision", "guided by",
            "mtech student", "m.tech student",
        )
        if has_honorific and any(term in q for term in supervision_terms):
            return True

        return False

    def retrieve_broadcast(self, query: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Search ALL loaded departments and sections and return the top-N by relevance.
        """
        all_retrievers = {}
        all_retrievers.update(self.retrievers)
        all_retrievers.update(self.section_retrievers)

        if not all_retrievers:
            return {
                "context": "",
                "provenance": {},
                "answerability": {"answerable": False, "reason": "No retrievers loaded."},
                "fallback_response": "No department or section knowledge bases are available.",
                "departments": [],
            }

        is_topic = self._is_topic_query(query)

        # Phase 1: Collect direct answers from ALL departments.
        # For topic queries (is_topic=True), we do NOT suppress topic matching —
        # we let each department run its own deterministic topic-expert lookup and
        # then merge all results below. suppress_topic_match was previously set to
        # is_topic which accidentally disabled the very feature that produces
        # correct faculty-by-domain answers across departments.
        direct_answers = {}
        dept_scores = []
        dept_bundles = {}

        for code, retriever in all_retrievers.items():
            try:
                is_sec = code in self.section_retrievers
                if is_sec:
                    direct = retriever.get_direct_answer(query)
                else:
                    # Always run topic matching (suppress_topic_match=False).
                    # For topic queries we collect every dept's answer and merge;
                    # we no longer suppress topic matching to "prevent" early return —
                    # instead the merging logic below handles cross-dept aggregation.
                    direct = retriever.get_direct_answer(
                        query, suppress_topic_match=False
                    )
                if direct:
                    direct_answers[code] = direct

                if is_sec:
                    bundle = retriever.retrieve_bundle(query)
                else:
                    bundle = retriever.retrieve_bundle(
                        query, local_top_k=4, vector_top_k=3, global_top_k=1, max_context_words=2000
                    )
                dept_bundles[code] = bundle

                # Score = average of graph + vector item scores
                prov = bundle.get("provenance", {})
                graph_score = prov.get("graph", {}).get("avg_score", 0.0)
                vector_score = prov.get("vector", {}).get("avg_score", 0.0)
                graph_items = prov.get("graph", {}).get("items", 0)
                vector_items = prov.get("vector", {}).get("items", 0)
                total_items = graph_items + vector_items

                if total_items > 0:
                    avg_score = (graph_score * graph_items + vector_score * vector_items) / total_items
                else:
                    avg_score = 0.0

                # Boost score if answerable
                if bundle.get("answerability", {}).get("answerable", False):
                    avg_score += 0.1

                dept_scores.append((code, avg_score, total_items))

            except Exception as e:
                logger.warning(f"Broadcast retrieval failed for '{code}': {e}")
                continue

        # Phase 2: If we have direct answers, handle them.
        #
        # For TOPIC queries (is_topic=True): always merge ALL direct answers from
        # all departments into a single cross-department response.  Even a single
        # match should be returned (topic queries are meant to be global).
        # Guard: skip if the ONLY result is from "administration" (admin nodes
        # don't hold research-domain data and would give a false answer).
        #
        # For NON-TOPIC queries with exactly 1 direct answer: return it directly
        # unless it's an admin answer for a person/comparison query.
        if direct_answers:
            if is_topic:
                # Filter out administration-only results for topic queries
                topic_answers = {
                    code: ans for code, ans in direct_answers.items()
                    if code != "administration"
                }
                if not topic_answers:
                    logger.info("Topic query: only admin direct answer found — falling through to full retrieval")
                else:
                    logger.info(f"Topic query: merging direct answers from {list(topic_answers.keys())}")
                    merged_sections = []
                    for code, answer in topic_answers.items():
                        is_sec = code in self.section_retrievers
                        if is_sec:
                            from departments import SECTIONS
                            dept_name = SECTIONS.get(code, {}).get("name", code.upper())
                        else:
                            dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
                        merged_sections.append(f"## {dept_name}\n\n{answer}")
                    return {
                        "context": "\n\n---\n\n".join(merged_sections),
                        "provenance": {"route": "direct_graph_multi", "source_mode": "graph"},
                        "answerability": {"answerable": True, "reason": "", "matched_terms": [], "missing_concepts": []},
                        "fallback_response": None,
                        "departments": [c for c in topic_answers.keys() if c not in self.section_retrievers],
                        "sections": [c for c in topic_answers.keys() if c in self.section_retrievers],
                        "dept_contexts": {
                            code: {
                                "name": (SECTIONS.get(code, {}).get("name", code.upper()) if code in self.section_retrievers else DEPARTMENTS.get(code, {}).get("full_name", code)),
                                "context": ans
                            }
                            for code, ans in topic_answers.items()
                        },
                        "direct": True,
                    }
            elif len(direct_answers) == 1:
                # Non-topic single department answer
                code = list(direct_answers.keys())[0]
                if code == "administration" and self._is_person_or_comparison_query(query):
                    logger.info("Admin-only direct answer for person/comparison query — falling through to full retrieval")
                else:
                    is_sec = code in self.section_retrievers
                    if is_sec:
                        from departments import SECTIONS
                        dept_name = SECTIONS.get(code, {}).get("name", code.upper())
                    else:
                        dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
                    return {
                        "context": f"## {dept_name}\n\n{direct_answers[code]}",
                        "provenance": {"route": "direct_graph", "source_mode": "graph"},
                        "answerability": {"answerable": True, "reason": "", "matched_terms": [], "missing_concepts": []},
                        "fallback_response": None,
                        "departments": [code] if not is_sec else [],
                        "sections": [code] if is_sec else [],
                        "direct": True,
                    }
            else:
                # Non-topic multiple direct answers — merge them all
                merged_sections = []
                for code, answer in direct_answers.items():
                    is_sec = code in self.section_retrievers
                    if is_sec:
                        from departments import SECTIONS
                        dept_name = SECTIONS.get(code, {}).get("name", code.upper())
                    else:
                        dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
                    merged_sections.append(f"## {dept_name}\n\n{answer}")
                return {
                    "context": "\n\n---\n\n".join(merged_sections),
                    "provenance": {"route": "direct_graph_multi", "source_mode": "graph"},
                    "answerability": {"answerable": True, "reason": "", "matched_terms": [], "missing_concepts": []},
                    "fallback_response": None,
                    "departments": [c for c in direct_answers.keys() if c not in self.section_retrievers],
                    "sections": [c for c in direct_answers.keys() if c in self.section_retrievers],
                    "dept_contexts": {
                        code: {
                            "name": (SECTIONS.get(code, {}).get("name", code.upper()) if code in self.section_retrievers else DEPARTMENTS.get(code, {}).get("full_name", code)),
                            "context": ans
                        }
                        for code, ans in direct_answers.items()
                    },
                    "direct": True,
                }

        if not dept_scores:
            return {
                "context": "",
                "provenance": {},
                "answerability": {"answerable": False, "reason": "All retrievers failed."},
                "fallback_response": "I couldn't find relevant information. Please try rephrasing your query.",
                "departments": [],
            }

        # Phase 3: Sort by score descending, select
        dept_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

        if is_topic:
            top_depts = [code for code, score, items in dept_scores if score > 0.0 or items > 0]

            if top_depts:
                filtered = []
                for code in top_depts:
                    bundle = dept_bundles.get(code, {})
                    ctx = bundle.get("context", "").strip()
                    if not ctx or ctx == "No relevant information found in the knowledge graph for this query.":
                        continue
                    if not self._is_bundle_relevant(query, bundle, is_topic):
                        logger.info(f"Broadcast: skipping '{code}' (not relevant to query focus terms)")
                        continue
                    filtered.append(code)
                if filtered:
                    top_depts = filtered
        else:
            top_depts = []
            for code, score, items in dept_scores[:top_n]:
                if score > 0.0 or items > 0:
                    bundle = dept_bundles.get(code, {})
                    if self._is_bundle_relevant(query, bundle, is_topic):
                        top_depts.append(code)

        if not top_depts:
            top_depts = [dept_scores[0][0]]

        if len(top_depts) == 1:
            code = top_depts[0]
            bundle = dept_bundles[code]
            is_sec = code in self.section_retrievers
            if is_sec:
                from departments import SECTIONS
                dept_name = SECTIONS.get(code, {}).get("name", code.upper())
            else:
                dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
            ctx = bundle["context"].strip()
            if ctx and ctx != "No relevant information found in the knowledge graph for this query.":
                bundle["context"] = f"## {dept_name}\n\n{ctx}"
            bundle["departments"] = [code] if not is_sec else []
            bundle["sections"] = [code] if is_sec else []
            return bundle

        # Merge top
        merged_sections = []
        dept_contexts = {}
        any_answerable = False
        for code in top_depts:
            bundle = dept_bundles.get(code, {})
            ctx = bundle.get("context", "").strip()
            if ctx and ctx != "No relevant information found in the knowledge graph for this query.":
                is_sec = code in self.section_retrievers
                if is_sec:
                    from departments import SECTIONS
                    dept_name = SECTIONS.get(code, {}).get("name", code.upper())
                else:
                    dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
                merged_sections.append(f"## {dept_name}\n\n{ctx}")
                dept_contexts[code] = {"name": dept_name, "context": ctx}
                if bundle.get("answerability", {}).get("answerable", False):
                    any_answerable = True

        if not merged_sections:
            return {
                "context": "",
                "provenance": {},
                "answerability": {"answerable": False, "reason": "No relevant evidence found across all."},
                "fallback_response": "I don't have that specific information. Try mentioning a specific department or section.",
                "departments": [],
            }

        return {
            "context": "\n\n---\n\n".join(merged_sections),
            "provenance": {code: dept_bundles[code].get("provenance", {}) for code in top_depts},
            "answerability": {
                "answerable": any_answerable,
                "reason": "" if any_answerable else "Low relevance across all.",
                "matched_terms": [],
                "missing_concepts": [],
            },
            "fallback_response": None if any_answerable else (
                "I don't have that specific information."
            ),
            "departments": [c for c in top_depts if c not in self.section_retrievers],
            "sections": [c for c in top_depts if c in self.section_retrievers],
            "dept_contexts": dept_contexts,
        }

    def _normalize_token(self, token: str) -> str:
        """Apply lightweight normalization for keyword coverage checks."""
        import re
        cleaned = re.sub(r"[^a-z0-9]+", "", token.lower())
        for suffix in ("ing", "ers", "ies", "es", "s"):
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 2:
                if suffix == "ies":
                    return cleaned[:-3] + "y"
                return cleaned[:-len(suffix)]
        return cleaned

    def _extract_query_focus_terms(self, query: str) -> List[str]:
        """Extract important focus keywords from query, ignoring all stopwords/generic terms.

        Handles compound research phrases (e.g. "computer vision", "deep learning")
        by scanning for known multi-word topics BEFORE splitting into individual
        words. This prevents stop-word filtering on individual component words
        (e.g. "computer" in "computer vision") from erasing the research topic.
        """
        import re
        q_clean = re.sub(r"[^\w\s]", " ", query.lower())

        # Known compound research topic phrases to detect as atomic units.
        # When found, inject their normalized tokens as focus terms so that
        # both words survive into the relevance check even if one word is a
        # stop word by itself (e.g. "computer" alone would be filtered).
        COMPOUND_TOPICS = [
            "computer vision",
            "deep learning",
            "machine learning",
            "artificial intelligence",
            "natural language processing",
            "natural language",
            "signal processing",
            "image processing",
            "speech processing",
            "power systems",
            "power electronics",
            "control systems",
            "embedded systems",
            "renewable energy",
            "neural network",
            "neural networks",
            "reinforcement learning",
            "transfer learning",
            "graph neural",
            "large language",
            "computer networks",
            "computer architecture",
        ]

        focus = []
        remaining_q = q_clean  # We'll blank out matched phrases so they aren't re-split

        for phrase in COMPOUND_TOPICS:
            if phrase in q_clean:
                # Add each word of the phrase individually (as normalized tokens)
                for w in phrase.split():
                    norm = self._normalize_token(w)
                    if norm and norm not in focus:
                        focus.append(norm)
                # Blank out matched phrase so individual word loop doesn't duplicate
                remaining_q = remaining_q.replace(phrase, " ")

        words = remaining_q.split()

        stop_words = {
            "iit", "jammu", "department", "dept", "faculty", "member", "members", "student", "students",
            "phd", "research", "official", "main", "information", "science", "engineering",
            "electrical", "mechanical", "civil", "chemical", "bioscience", "bioengineering", "physics",
            "chemistry", "mathematics", "humanities", "social", "sciences", "ee", "cse", "me", "ce", "che",
            "bsbe", "hss", "idp", "phy", "math", "maths", "chem", "materials",
            "work", "works", "working", "professor", "professors", "teach", "teaches", "teaching",
            "under", "who", "whose", "whom", "what", "where", "when", "why", "how", "list", "show", "find",
            "get", "give", "tell", "discuss", "compare", "comparison", "versus", "vs", "and", "for", "are",
            "the", "with", "from", "in", "on", "at", "of", "to", "is", "a", "an", "this", "that", "these",
            "those", "they", "them", "their", "his", "her", "its", "it", "he", "she", "you", "your", "we",
            "our", "us", "about", "any", "some", "such", "than", "then", "too", "very", "was", "were", "will",
            "would", "should", "could", "can", "may", "might", "must", "shall", "does", "do", "did", "has",
            "have", "had", "having", "been", "being", "be", "other", "another", "domain", "field", "topic",
            "subject", "area", "areas", "related", "tasks", "task", "lab", "labs", "project", "projects",
            "programme", "programmes", "program", "programs", "course", "courses", "class", "classes",
            "specialist", "expert", "experts", "specialists", "people", "person", "someone", "anyone",
            "everyone", "body", "write", "reach", "contact", "email", "phone", "address", "location",
            "office", "website", "page", "profile", "device", "devices",
            # Note: "computer" intentionally NOT in stop_words so "computer vision"
            # still contributes "computer" even when encountered individually.
        }

        for w in words:
            cleaned = w.strip()
            if len(cleaned) <= 2:
                continue
            norm = self._normalize_token(cleaned)
            if norm not in stop_words and cleaned not in stop_words and norm not in focus:
                focus.append(norm)

        return list(set(focus))

    def _is_bundle_relevant(self, query: str, bundle: Dict[str, Any], is_topic: bool) -> bool:
        """Determine if a department bundle has genuinely relevant information."""
        prov = bundle.get("provenance", {})
        
        # 1. If it's a direct/provenance match (highest confidence)
        if prov.get("route") in ("direct_graph", "direct_graph_multi") or prov.get("graph", {}).get("direct", False):
            return True
            
        # 2. Extract query focus terms
        focus_terms = self._extract_query_focus_terms(query)
        
        # If the query has no specific focus terms (e.g. extremely short or generic),
        # we fall back to checking if there is any graph/vector match.
        if not focus_terms:
            graph_items = prov.get("graph", {}).get("items", 0)
            graph_avg = prov.get("graph", {}).get("avg_score", 0.0)
            vector_items = prov.get("vector", {}).get("items", 0)
            vector_avg = prov.get("vector", {}).get("avg_score", 0.0)
            return (graph_items > 0 and graph_avg > 0.8) or (vector_items > 0 and vector_avg >= 0.45)
            
        # 3. Check if any query focus terms are present in the context text
        context = bundle.get("context", "")
        
        # Split context into words, normalize them, and check match
        import re
        context_words = set()
        for word in re.findall(r"[A-Za-z0-9]+", context.lower()):
            norm_word = self._normalize_token(word)
            if norm_word:
                context_words.add(norm_word)
                
        matched_terms = [term for term in focus_terms if term in context_words]
        
        # If we found at least one matching focus term in the context, it is relevant
        if matched_terms:
            return True
            
        # 4. Check vector search similarity score (avg_score >= 0.45)
        # Even if matched_terms is empty, we keep it if vector search has high similarity
        # because vector search handles synonyms/semantics.
        vector_items = prov.get("vector", {}).get("items", 0)
        vector_avg = prov.get("vector", {}).get("avg_score", 0.0)
        if vector_items > 0 and vector_avg >= 0.45:
            return True
            
        # Otherwise, not relevant (even if graph_avg was high due to loose name token match)
        return False
