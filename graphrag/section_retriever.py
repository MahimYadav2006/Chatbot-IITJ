"""
Section Retriever Engine for GraphRAG.
Handles Academics, Accounts, Counselling, DI, and E2 sections with direct deterministic answers
and fallback semantic chunk search.
"""

import os
import re
import json
import logging
from collections import Counter
from typing import Any, List, Dict, Tuple, Optional
import networkx as nx

from departments import SECTIONS, get_section_data_dir
from graphrag.kg_builder import normalize_name

logger = logging.getLogger(__name__)

class SectionRetriever:
    def __init__(self, section_code: str, graph: nx.DiGraph, chunks: List[Dict], embedding_engine=None):
        self.section_code = section_code
        self.section_config = SECTIONS[section_code]
        self.graph = graph
        self.chunks = chunks
        self.embeddings = embedding_engine
        
        # Load embedding index if present in data directory
        if self.embeddings:
            try:
                data_dir = get_section_data_dir(self.section_code)
                if os.path.exists(os.path.join(data_dir, "embeddings.faiss")):
                    self.embeddings.load(data_dir)
                    logger.info(f"Loaded embeddings index for section {section_code}")
            except Exception as e:
                logger.warning(f"Failed to load embeddings index for section {section_code}: {e}")

    def _build_provenance(
        self,
        direct: bool,
        local_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        section_word_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Summarize which retrieval channels contributed evidence."""
        graph_used = bool(direct or local_results)
        vector_used = bool(vector_results)

        if direct:
            source_mode = "graph"
            route = "direct_graph"
        elif graph_used and vector_used:
            source_mode = "both"
            route = "graph+vector"
        elif graph_used:
            source_mode = "graph"
            route = "graph"
        elif vector_used:
            source_mode = "vector"
            route = "vector"
        else:
            source_mode = "none"
            route = "none"

        def avg_score(results: List[Dict[str, Any]]) -> float:
            if not results:
                return 0.0
            return round(sum(float(item.get("score", 0.0)) for item in results) / len(results), 3)

        return {
            "route": route,
            "source_mode": source_mode,
            "graph": {
                "direct": direct,
                "items": len(local_results) if not direct else 1,
                "avg_score": avg_score(local_results),
                "labels": dict(Counter(item.get("label", "Unknown") for item in local_results)),
                "word_count": section_word_counts.get("graph", 0),
            },
            "vector": {
                "items": len(vector_results),
                "avg_score": avg_score(vector_results),
                "sources": [item.get("source", "Unknown") for item in vector_results[:5]],
                "word_count": section_word_counts.get("vector", 0),
            },
            "community": {
                "items": 0,
                "avg_score": 0.0,
                "word_count": 0,
            }
        }

    def get_direct_answer(self, query: str, global_person_index=None) -> Optional[str]:
        """Check for direct, deterministic answers based on section-specific entities."""
        q = query.lower().strip()

        # 1. Global Person Cross-Reference Check first
        #    Only for identity queries ("who is X?", "tell me about X").
        #    Skip for research/PhD/publication queries so full retrieval handles them.
        if global_person_index:
            from app import _is_identity_query
            if _is_identity_query(query):
                # Look for names in the query
                for p_name in global_person_index.person_roles.keys():
                    # Word boundary check to avoid partial matching (e.g. "Ajay" matching "Ajay Singh")
                    pattern = r"\b" + re.escape(p_name.lower()) + r"\b"
                    if re.search(pattern, q):
                        res = global_person_index.lookup(p_name)
                        if res:
                            # Build formatted answer
                            roles_lines = []
                            for role in res["roles"]:
                                source_label = role["source"].upper()
                                roles_lines.append(f"- **{role['designation']}** under {source_label}")
                                if role.get("email"):
                                    roles_lines.append(f"  - Email: {role['email']}")
                                if role.get("phone"):
                                    roles_lines.append(f"  - Phone: {role['phone']}")
                                if role.get("office"):
                                    roles_lines.append(f"  - Office: {role['office']}")
                                    
                            return (
                                f"**{res['name']}** holds the following role(s) at IIT Jammu:\n" +
                                "\n".join(roles_lines)
                            )

        # 2. Section Contact Query
        if any(term in q for term in ("contact", "email", "phone", "hours", "timing", "address", "number")):
            contacts = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "SectionContact"]
            if contacts:
                c = contacts[0]
                lines = [f"### Contact Information for {self.section_config['name']}:"]
                if c.get("email"):
                    lines.append(f"- **Email:** {c['email']}")
                if c.get("phone"):
                    lines.append(f"- **Phone:** {c['phone']}")
                if c.get("hours"):
                    lines.append(f"- **Working Hours:** {c['hours']}")
                if c.get("address"):
                    lines.append(f"- **Office Address:** {c['address']}")
                return "\n".join(lines)

        # 3. Section Head Query
        if any(term in q for term in ("head", "dean", "coordinator", "officer in charge", "faculty in charge", "incharge")):
            heads = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "SectionHead"]
            if heads:
                lines = [f"### Head / Coordinator of {self.section_config['name']}:"]
                for h in heads:
                    lines.append(f"- **{h['name']}** ({h['designation']})")
                    if h.get("email"):
                        lines.append(f"  - Email: {h['email']}")
                    if h.get("phone"):
                        lines.append(f"  - Phone: {h['phone']}")
                return "\n".join(lines)

        # 4. List Section Staff
        # Use specific patterns to avoid matching topic queries like "who works on Deep Learning"
        section_name_lower = self.section_config.get("name", "").lower()
        staff_list_triggers = (
            "list people", "list staff", "list members",
            f"who works in {section_name_lower}",
            f"who works at {section_name_lower}",
            f"members in {section_name_lower}",
            f"staff in {section_name_lower}",
            f"team in {section_name_lower}",
            "members in this section", "staff of this section",
            "staff here", "who works here",
        )
        if any(term in q for term in staff_list_triggers):
            people = [d for n, d in self.graph.nodes(data=True) if d.get("label") in ("SectionPerson", "Counselor")]
            if people:
                lines = [f"### Members of {self.section_config['name']}:"]
                for p in sorted(people, key=lambda x: x.get("name", "")):
                    lines.append(f"- **{p['name']}** ({p['designation']})")
                    if p.get("email"):
                        lines.append(f"  - Email: {p['email']}")
                return "\n".join(lines)

        # 5. Counselling Services specific queries
        if self.section_code == "counselling":
            # Counselors bio queries
            counselors = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "Counselor"]
            for c in counselors:
                if c.get("name", "").lower() in q:
                    lines = [f"### Counselor Profile: **{c['name']}**"]
                    lines.append(f"- **Designation:** {c['designation']}")
                    if c.get("email"):
                        lines.append(f"- **Email:** {c['email']}")
                    if c.get("phone"):
                        lines.append(f"- **Phone:** {c['phone']}")
                    if c.get("office"):
                        lines.append(f"- **Office:** {c['office']}")
                    if c.get("bio"):
                        lines.append(f"\n**Bio/Profile:**\n{c['bio']}")
                    return "\n".join(lines)

            if any(term in q for term in ("services", "programs", "workshops", "seminars", "confidentiality", "first visit")):
                # Return counselling about text chunks
                counselling_about = [c["text"] for c in self.chunks if "about-counselling-services" in c.get("metadata", {}).get("doc", "")]
                if counselling_about:
                    return counselling_about[0]

        # 6. Academics spec/course Catalog and Rules check
        if self.section_code == "academics":
            if any(term in q for term in ("specialisation", "specialization", "course", "curriculum", "m.tech in", "m.sc in", "minor in", "micro-specialisation", "honours in")):
                # Retrieve course spec text chunk
                spec_chunks = [c["text"] for c in self.chunks if "specialisation-and-courses" in c.get("metadata", {}).get("doc", "")]
                if spec_chunks:
                    # Filter matching courses/specialisations if specific keywords found
                    # Otherwise return the full overview
                    for line in spec_chunks[0].splitlines():
                        if any(kw in line.lower() for kw in q.split()) and "drive.google.com" in line:
                            return f"Here is the curriculum document link you requested:\n{line}"
                    return spec_chunks[0]

            if any(term in q for term in ("rules", "regulation", "regulations", "manual", "handbook")):
                rule_chunks = [c["text"] for c in self.chunks if "rules-and-regulations" in c.get("metadata", {}).get("doc", "")]
                if rule_chunks:
                    return rule_chunks[0]

            if any(term in q for term in ("function", "responsibility", "work breakdown", "tasks", "mcm", "scholarship", "freeship", "fellowship", "contingency", "stipend")):
                func_chunks = [c["text"] for c in self.chunks if "function-list" in c.get("metadata", {}).get("doc", "")]
                if func_chunks:
                    return func_chunks[0]

        # 7. DI specific queries (Divisions details)
        if self.section_code == "di":
            divisions_keywords = {
                "network": "network-services",
                "software": "software-development",
                "data center": "data-center",
                "datacenter": "data-center",
            }
            for div_kw, filename_kw in divisions_keywords.items():
                if div_kw in q:
                    div_chunks = [c["text"] for c in self.chunks if filename_kw in c.get("metadata", {}).get("doc", "")]
                    if div_chunks:
                        return div_chunks[0]

        # 8. E2 specific queries (Establishment II — HR/admin work functions)
        if self.section_code == "e2":
            # Use specific E2 work terms — avoid bare "work" which matches "works on" in topic queries
            if any(term in q for term in ("function of e2", "responsibility of e2", "work of e2",
                                           "establishment ii", "ccs", "conduct rules", "retirement",
                                           "promotion", "seniority", "hba", "house building",
                                           "leave rules", "pay fixation", "service book",
                                           "what does e2 do", "e2 section")):
                work_chunks = [c["text"] for c in self.chunks if "work-establishment-ii" in c.get("metadata", {}).get("doc", "")]
                if work_chunks:
                    return work_chunks[0]


        return None

    def retrieve_bundle(
        self,
        query: str,
        local_top_k: int = 5,
        vector_top_k: int = 5,
        global_top_k: int = 3,
        max_context_words: int = 4500,
    ) -> Dict[str, Any]:
        """Retrieve relevant context for a section query. Falls back to BM25/vector search over chunks."""
        # 1. Check direct answer
        direct_ans = self.get_direct_answer(query)
        if direct_ans:
            provenance = self._build_provenance(
                direct=True,
                local_results=[],
                vector_results=[],
                section_word_counts={"graph": len(direct_ans.split())},
            )
            return {
                "context": direct_ans,
                "provenance": provenance,
                "answerability": {"answerable": True, "reason": "", "matched_terms": [], "missing_concepts": []},
                "fallback_response": None,
            }

        # 2. Check semantic vector search if Embedding Engine is available
        vector_results = []
        if self.embeddings and self.embeddings.index is not None:
            try:
                search_res = self.embeddings.search(query, top_k=vector_top_k, department_filter=self.section_code)
                for item, score in search_res:
                    vector_results.append({
                        "id": item["id"],
                        "text": item["text"],
                        "score": score,
                        "source": item.get("metadata", {}).get("name", "Unknown")
                    })
            except Exception as e:
                logger.warning(f"Error during section vector search: {e}")

        # 3. Text chunk ranking fallback (simple word overlap matching)
        local_results = []
        q_words = set(re.findall(r"\w+", query.lower()))
        for chunk in self.chunks:
            chunk_text = chunk["text"]
            chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
            overlap = len(q_words.intersection(chunk_words))
            if overlap > 0:
                local_results.append({
                    "id": chunk["id"],
                    "text": chunk_text,
                    "score": overlap / len(q_words),
                    "label": "TextChunk",
                })
        
        # Sort by score descending
        local_results.sort(key=lambda x: x["score"], reverse=True)
        local_results = local_results[:local_top_k]

        # Combine contexts
        combined_blocks = []
        word_count = 0
        
        for item in local_results:
            text_block = item["text"]
            block_words = len(text_block.split())
            if word_count + block_words > max_context_words:
                break
            combined_blocks.append(text_block)
            word_count += block_words

        for item in vector_results:
            text_block = item["text"]
            block_words = len(text_block.split())
            if word_count + block_words > max_context_words:
                break
            if text_block not in combined_blocks:
                combined_blocks.append(text_block)
                word_count += block_words

        context = "\n\n---\n\n".join(combined_blocks)
        
        provenance = self._build_provenance(
            direct=False,
            local_results=local_results,
            vector_results=vector_results,
            section_word_counts={
                "graph": sum(len(r["text"].split()) for r in local_results),
                "vector": sum(len(r["text"].split()) for r in vector_results),
            }
        )

        return {
            "context": context,
            "provenance": provenance,
            "answerability": {
                "answerable": len(combined_blocks) > 0,
                "reason": "" if combined_blocks else "No relevant section documents found.",
                "matched_terms": [],
                "missing_concepts": [],
            },
            "fallback_response": (
                f"I don't have verified information in the {self.section_config['name']} section "
                f"to answer that question. Please visit the official website at {self.section_config['base_url']}."
            )
        }
