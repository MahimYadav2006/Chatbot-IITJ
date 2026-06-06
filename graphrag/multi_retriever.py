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
    """Orchestrates retrieval across 1+ department-scoped HybridRetrievers."""

    def __init__(self, retrievers: Dict):
        """
        Args:
            retrievers: Dict mapping dept_code → HybridRetriever instance.
        """
        self.retrievers = retrievers

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
        Retrieve from multiple departments and merge contexts with headers.

        Each department's context is prefixed with its name so the LLM
        can attribute information correctly.
        """
        dept_contexts = {}
        dept_provenances = {}
        all_answerable = False

        for code in dept_codes:
            retriever = self.retrievers.get(code)
            if not retriever:
                logger.warning(f"Retriever for '{code}' not available, skipping.")
                continue

            bundle = retriever.retrieve_bundle(
                query, local_top_k=4, vector_top_k=4, global_top_k=2, max_context_words=2500
            )
            if bundle.get("answerability", {}).get("answerable", False):
                all_answerable = True

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
                "answerability": {"answerable": False, "reason": "No departments available."},
                "fallback_response": "No department knowledge bases are available.",
                "departments": dept_codes,
                "dept_contexts": {},
            }

        # Merge contexts with department headers
        merged_sections = []
        for code in dept_codes:
            if code not in dept_contexts:
                continue
            entry = dept_contexts[code]
            context = entry["context"].strip()
            if context and context != "No relevant information found in the knowledge graph for this query.":
                merged_sections.append(f"## {entry['name']}\n\n{context}")

        if not merged_sections:
            merged_context = "No relevant information found across the queried departments."
            all_answerable = False
        else:
            merged_context = "\n\n---\n\n".join(merged_sections)

        return {
            "context": merged_context,
            "provenance": dept_provenances,
            "answerability": {
                "answerable": all_answerable,
                "reason": "" if all_answerable else "No relevant evidence across queried departments.",
                "matched_terms": [],
                "missing_concepts": [],
            },
            "fallback_response": None if all_answerable else (
                "I don't have enough information across those departments to answer this query."
            ),
            "departments": [c for c in dept_codes if c in dept_contexts],
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
        """
        import re
        q = re.sub(r"\s+", " ", query.lower()).strip()

        # Pattern 1: "who teaches / who works on / who researches <topic>"
        topic_patterns = (
            r"who\s+(?:teaches|teach|works?\s+on|research(?:es|ing)?|stud(?:ies|ying))\b",
            r"(?:faculty|professor|professors|teacher|teachers)\s+(?:working|researching|teaching|for|in|on|who)\b",
            r"which\s+(?:faculty|professor|professors)\s+(?:work|teach|research|are)\b",
            r"who\s+(?:is\s+)?(?:working\s+on|expert\s+in|specialist\s+in)\b",
            r"who\s+(?:should\s+i\s+)?contact\s+(?:for|about|regarding)\b",
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
        )
        if not has_dept_name and any(subj in q for subj in broad_subjects):
            return True

        return False

    def retrieve_broadcast(self, query: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Search ALL loaded departments and return the top-N by relevance.

        Used when no department signal is detected in the query.
        For topic/subject queries, aggregates results from ALL departments
        that have relevant content to prevent single-department bias.
        """
        if not self.retrievers:
            return {
                "context": "",
                "provenance": {},
                "answerability": {"answerable": False, "reason": "No departments loaded."},
                "fallback_response": "No department knowledge bases are available.",
                "departments": [],
            }

        is_topic = self._is_topic_query(query)

        # Phase 1: Collect direct answers from ALL departments (don't short-circuit)
        direct_answers = {}
        dept_scores = []
        dept_bundles = {}

        for code, retriever in self.retrievers.items():
            try:
                # For topic queries, suppress topic matching in direct answers
                # so they go through full retrieval instead of short-circuiting
                direct = retriever.get_direct_answer(
                    query, suppress_topic_match=is_topic
                )
                if direct:
                    direct_answers[code] = direct

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

        # Phase 2: If we have direct answers from multiple departments, merge them
        if direct_answers:
            if len(direct_answers) == 1 and not is_topic:
                # Single department, non-topic query — return directly (old behavior)
                code = list(direct_answers.keys())[0]
                dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
                return {
                    "context": f"## {dept_name}\n\n{direct_answers[code]}",
                    "provenance": {"route": "direct_graph", "source_mode": "graph"},
                    "answerability": {"answerable": True, "reason": "", "matched_terms": [], "missing_concepts": []},
                    "fallback_response": None,
                    "departments": [code],
                    "direct": True,
                }
            elif direct_answers:
                # Multiple departments have direct answers — merge them all
                merged_sections = []
                for code, answer in direct_answers.items():
                    dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
                    merged_sections.append(f"## {dept_name}\n\n{answer}")
                return {
                    "context": "\n\n---\n\n".join(merged_sections),
                    "provenance": {"route": "direct_graph_multi", "source_mode": "graph"},
                    "answerability": {"answerable": True, "reason": "", "matched_terms": [], "missing_concepts": []},
                    "fallback_response": None,
                    "departments": list(direct_answers.keys()),
                    "dept_contexts": {
                        code: {"name": DEPARTMENTS.get(code, {}).get("full_name", code), "context": ans}
                        for code, ans in direct_answers.items()
                    },
                    "direct": True,
                }

        if not dept_scores:
            return {
                "context": "",
                "provenance": {},
                "answerability": {"answerable": False, "reason": "All retrievers failed."},
                "fallback_response": "I couldn't find relevant information. Please try rephrasing your query with a specific department name.",
                "departments": [],
            }

        # Phase 3: Sort by score descending, select departments
        dept_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

        if is_topic:
            # For topic queries, include ALL departments that have relevant results
            # (score > 0 or items > 0) — no top-N cap
            top_depts = [code for code, score, items in dept_scores if score > 0.0 or items > 0]
        else:
            # For non-topic queries, use top-N as before
            top_depts = [code for code, score, items in dept_scores[:top_n] if score > 0.0 or items > 0]

        if not top_depts:
            top_depts = [dept_scores[0][0]]  # At least try the top one

        if len(top_depts) == 1:
            # Single best department
            code = top_depts[0]
            bundle = dept_bundles[code]
            dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
            ctx = bundle["context"].strip()
            if ctx and ctx != "No relevant information found in the knowledge graph for this query.":
                bundle["context"] = f"## {dept_name}\n\n{ctx}"
            bundle["departments"] = [code]
            return bundle

        # Merge top departments
        merged_sections = []
        dept_contexts = {}
        any_answerable = False
        for code in top_depts:
            bundle = dept_bundles.get(code, {})
            ctx = bundle.get("context", "").strip()
            if ctx and ctx != "No relevant information found in the knowledge graph for this query.":
                dept_name = DEPARTMENTS.get(code, {}).get("full_name", code)
                merged_sections.append(f"## {dept_name}\n\n{ctx}")
                dept_contexts[code] = {"name": dept_name, "context": ctx}
                if bundle.get("answerability", {}).get("answerable", False):
                    any_answerable = True

        if not merged_sections:
            return {
                "context": "",
                "provenance": {},
                "answerability": {"answerable": False, "reason": "No relevant evidence found across all departments."},
                "fallback_response": "I don't have that specific information. Try mentioning a specific department (e.g., CSE, EE, Physics).",
                "departments": [],
            }

        return {
            "context": "\n\n---\n\n".join(merged_sections),
            "provenance": {code: dept_bundles[code].get("provenance", {}) for code in top_depts},
            "answerability": {
                "answerable": any_answerable,
                "reason": "" if any_answerable else "Low relevance across all departments.",
                "matched_terms": [],
                "missing_concepts": [],
            },
            "fallback_response": None if any_answerable else (
                "I don't have that specific information. Try mentioning a specific department."
            ),
            "departments": top_depts,
            "dept_contexts": dept_contexts,
        }
