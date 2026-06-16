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

    def get_deterministic_context(self, query: str) -> Optional[str]:
        """Return deterministic context from section-specific graph entities.

        The returned text is intended as high-priority context for the LLM,
        NOT as a final response returned directly to the user.
        """
        q = query.lower().strip()

        # Section Contact Query
        if any(term in q for term in ("contact", "email", "phone", "hours", "timing", "address", "number")) and not any(svc in q for svc in ("dental", "physiotherapy", "pharmacy", "ambulance", "ward", "ecg", "laboratory")):
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
        if any(term in q for term in ("head", "dean", "coordinator", "officer in charge", "faculty in charge", "incharge", "chairperson")):
            heads = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "SectionHead"]
            if heads:
                lines = [f"### Head / Coordinator / Chairperson of {self.section_config['name']}:"]
                for h in heads:
                    lines.append(f"- **{h['name']}** ({h['designation']})")
                    if h.get("qualifications"):
                        lines.append(f"  - Qualifications: {h['qualifications']}")
                    if h.get("experience"):
                        lines.append(f"  - Experience: {h['experience']}")
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
            is_link_query = any(term in q for term in ("link", "url", "download", "website", "document", "pdf", "file", "drive"))
            if is_link_query:
                # Scan raw scraped markdown files for direct matching drive/doc links
                import os
                scraped_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scraped_data", "sections", "academics"))
                if os.path.exists(scraped_dir):
                    query_terms = [t.lower() for t in re.findall(r'\w+', q) if len(t) > 2 and t.lower() not in (
                        'what', 'the', 'link', 'for', 'download', 'pdf', 'document', 'url', 'file', 'drive', 'how', 'get', 'give', 'can', 'find', 'show', 'where'
                    )]
                    if query_terms:
                        matches = []
                        for fn in os.listdir(scraped_dir):
                            if not fn.endswith('.md'):
                                continue
                            filepath = os.path.join(scraped_dir, fn)
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        if '[' in line and '](' in line:
                                            link_text = re.findall(r'\[([^\]]+)\]', line)
                                            if link_text:
                                                lt = link_text[0].lower()
                                                matches_count = sum(1 for term in query_terms if term in lt)
                                                if matches_count > 0:
                                                    matches.append((matches_count, line.strip()))
                            except Exception:
                                pass
                        if matches:
                            matches.sort(key=lambda x: x[0], reverse=True)
                            seen = set()
                            result_links = []
                            max_matches = matches[0][0]
                            threshold = max(1, min(2, max_matches))
                            for count, link in matches:
                                if count >= threshold and link not in seen:
                                    seen.add(link)
                                    clean_line = re.sub(r'^\d+\.\s*', '', link)
                                    result_links.append(f"- {clean_line}")
                                    if len(result_links) >= 5:
                                        break
                            if result_links:
                                return "Here are the relevant document links found on the Academics website:\n\n" + "\n".join(result_links)
                
                # Fallback to search self.chunks if directory check was skipped or yielded no results
                query_terms = [t.lower() for t in re.findall(r'\w+', q) if len(t) > 2 and t.lower() not in (
                    'what', 'the', 'link', 'for', 'download', 'pdf', 'document', 'url', 'file', 'drive', 'how', 'get', 'give', 'can', 'find', 'show', 'where'
                )]
                if query_terms:
                    matches = []
                    for chunk in self.chunks:
                        for line in chunk.get("text", "").split("\n"):
                            # Check if line contains any query term and looks like a link
                            if any(term in line.lower() for term in query_terms) and ("http" in line or "[" in line or "drive" in line):
                                matches.append(line.strip())
                    if matches:
                        # Clean prefix bullet if present
                        cleaned_matches = []
                        for m in matches[:5]:
                            m_clean = re.sub(r'^\d+\.\s*|-\s*', '', m)
                            cleaned_matches.append(f"- {m_clean}")
                        return "Here are the relevant document links found on the Academics website:\n\n" + "\n".join(cleaned_matches)

            # If not a link query, check structured academic program/specialization details in graph
            specializations = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "Specialization"]
            programs = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "AcademicProgram"]

            # Helper to find matching specialization/program node
            matched_entity = None
            matched_id = None
            best_score = 0

            # Exact word-based analysis to avoid substring matching bugs (e.g. 'ee' matching 'engineering')
            q_words = set(re.findall(r'\w+', q.lower()))
            q_expanded_words = set(q_words)
            if "cse" in q_words:
                q_expanded_words.update({"computer", "science", "cse"})
            if "ee" in q_words:
                q_expanded_words.update({"electrical", "engineering", "electronics", "ee"})
            if "me" in q_words:
                q_expanded_words.update({"mechanical", "me"})
            if "ce" in q_words:
                q_expanded_words.update({"civil", "ce"})

            stop_words = {
                "what", "is", "are", "the", "for", "of", "in", "and", "a", "to", "i", "can", "do", "offer",
                "offered", "available", "at", "iit", "jammu", "program", "programmes", "programs", "programme",
                "specialization", "specializations", "specialisation", "specialisations", "minor", "minors",
                "micro", "honours", "honors", "course", "courses", "curriculum", "credits", "credit", "graduate",
                "graduation", "student", "students", "require", "required", "requirement", "requirements",
                "how", "many", "engineering", "department", "dept", "academic", "academics", "information",
                "details", "subject", "subjects", "syllabus", "syllabi", "study"
            }
            filtered_q_words = {w for w in q_expanded_words if w not in stop_words and len(w) > 1}

            # Map known abbreviation expansions for precise matching (e.g., CSP -> Communication and Signal Processing)
            ABBREVIATIONS = {
                "csp": {"communication", "signal", "processing"},
                "cse": {"computer", "science", "engineering"},
                "ee": {"electrical", "engineering"},
                "me": {"mechanical", "engineering"},
                "ce": {"civil", "engineering"},
                "ch": {"chemical", "engineering"},
                "chemical": {"chemical", "engineering"},
                "mechanical": {"mechanical", "engineering"},
                "electrical": {"electrical", "engineering"},
                "civil": {"civil", "engineering"},
            }
            expanded_q_words = set(filtered_q_words)
            for w in filtered_q_words:
                if w in ABBREVIATIONS:
                    expanded_q_words.update(ABBREVIATIONS[w])

            # Check if query targets a specific department with word boundary safety
            from departments import DEPARTMENTS
            target_dept = None
            for code, config in DEPARTMENTS.items():
                aliases = [code.lower(), config["name"].lower()] + [a.lower() for a in config.get("aliases", [])]
                if any(re.search(r'\b' + re.escape(a) + r'\b', q.lower()) for a in aliases):
                    target_dept = code
                    break

            is_course_query = any(term in q.lower() for term in ("course", "curriculum", "syllabus", "subject", "credit"))

            if expanded_q_words:
                for node_id, d in self.graph.nodes(data=True):
                    if d.get("label") in ("Specialization", "AcademicProgram"):
                        name_words = set(re.findall(r'\w+', d.get("name", "").lower()))
                        node_expanded_words = set(name_words)
                        for w in name_words:
                            if w in ABBREVIATIONS:
                                node_expanded_words.update(ABBREVIATIONS[w])
                        
                        overlap = len(expanded_q_words.intersection(node_expanded_words))
                        
                        # Apply department match bonus
                        if target_dept and d.get("department") == target_dept:
                            overlap += 2
                            
                        # If node has 'cse' in name, and query has 'cse', boost it
                        if "cse" in q_words and "cse" in name_words:
                            overlap += 1
                        if "ee" in q_words and ("ee" in name_words or "electrical" in name_words):
                            overlap += 1
                        if ("civil" in q_words or "ce" in q_words) and ("civil" in name_words or "ce" in name_words):
                            overlap += 1
                        if ("mechanical" in q_words or "me" in q_words) and ("mechanical" in name_words or "me" in name_words):
                            overlap += 1
                        if ("chemical" in q_words or "ch" in q_words) and ("chemical" in name_words or "ch" in name_words):
                            overlap += 1
                        if "bsbe" in q_words and "bsbe" in name_words:
                            overlap += 1
                        if "hss" in q_words and "hss" in name_words:
                            overlap += 1
                        if ("materials" in q_words or "mt" in q_words or "mty" in q_words) and ("materials" in name_words or "mt" in name_words or "mty" in name_words):
                            overlap += 1

                        # Apply type-matching bonuses (Minor, Honours, Micro Specialization)
                        node_name_lower = d.get("name", "").lower()
                        node_type_lower = str(d.get("type", "")).lower()
                        if "minor" in q_words and ("minor" in node_name_lower or "minor" in node_type_lower):
                            overlap += 3
                        if ("honour" in q_words or "honor" in q_words or "honours" in q_words) and ("honour" in node_name_lower or "honor" in node_name_lower or "honours" in node_name_lower or "honour" in node_type_lower or "honor" in node_type_lower or "honours" in node_type_lower):
                            overlap += 3
                        if "micro" in q_words and ("micro" in node_name_lower or "micro" in node_type_lower):
                            overlap += 3

                        # Apply course-existence bonus if this is a course query
                        if is_course_query:
                            has_courses = any(self.graph.nodes[t].get("label") == "Course" 
                                              for s, t, edge_data in self.graph.out_edges(node_id, data=True) 
                                              if edge_data.get("type") == "OFFERS_COURSE")
                            if has_courses:
                                overlap += 4

                        # Apply level matching bonus
                        node_level = str(d.get("level", "")).lower()
                        is_pg_query = any(w in q_words for w in ("mtech", "m.tech", "postgraduate", "pg", "master", "masters", "phd", "specialization", "specializations"))
                        is_ug_query = any(w in q_words for w in ("btech", "b.tech", "undergraduate", "ug", "bachelor", "bachelors"))
                        if is_pg_query and ("pg" in node_level or "mtech" in node_level or "master" in node_level or "m.tech" in node_name_lower or "mtech" in node_name_lower):
                            overlap += 3
                        elif is_ug_query and ("ug" in node_level or "btech" in node_level or "b.tech" in node_name_lower or "btech" in node_name_lower):
                            overlap += 3
                        elif not is_pg_query and ("ug" in node_level or "btech" in node_level or "b.tech" in node_name_lower or "btech" in node_name_lower):
                            overlap += 1

                        # De-prioritize superseded versions
                        if d.get("superseded", False):
                            overlap -= 2

                        if overlap > best_score:
                            best_score = overlap
                            matched_entity = d
                            matched_id = node_id

            # If we matched an entity with a decent overlap, and the query is asking about courses/credits
            if matched_entity and best_score >= 1 and any(term in q.lower() for term in ("course", "curriculum", "syllabus", "subject", "credit")):
                # Find all course nodes connected to this program/specialization
                courses = []
                out_edges = list(self.graph.out_edges(matched_id, data=True))
                for s, t, edge_data in out_edges:
                    if edge_data.get("type") == "OFFERS_COURSE" and self.graph.nodes[t].get("label") == "Course":
                        c_node = self.graph.nodes[t]
                        courses.append({
                            "name": c_node.get("name"),
                            "code": c_node.get("code"),
                            "ltp": c_node.get("ltp"),
                            "credits": c_node.get("credits"),
                            "semester": edge_data.get("semester"),
                            "category": edge_data.get("category"),
                            "bucket": edge_data.get("bucket")
                        })

                if courses:
                    lines = []
                    if matched_entity.get("total_credits"):
                        lines.append(f"**Total Graduation Credits Requirement:** {matched_entity['total_credits']} credits\n")
                    lines.append(f"### Courses offered in {matched_entity['name']}:")
                    from collections import defaultdict
                    by_sem = defaultdict(list)
                    by_cat = defaultdict(list)
                    other_courses = []

                    for c in courses:
                        if c["semester"]:
                            by_sem[c["semester"]].append(c)
                        elif c["category"]:
                            by_cat[c["category"]].append(c)
                        else:
                            other_courses.append(c)

                    if by_sem:
                        for sem in sorted(by_sem.keys()):
                            lines.append(f"\n#### Semester {sem}:")
                            for c in sorted(by_sem[sem], key=lambda x: (x["code"] or "", x["name"] or "")):
                                c_info = f"- **{c['name']}**"
                                if c['code']:
                                    c_info += f" ({c['code']})"
                                if c['credits'] or c['ltp']:
                                    details = []
                                    if c['ltp']: details.append(f"L-T-P: {c['ltp']}")
                                    if c['credits']: details.append(f"Credits: {c['credits']}")
                                    c_info += f" — {', '.join(details)}"
                                lines.append(c_info)
                    elif by_cat:
                        for cat in sorted(by_cat.keys()):
                            lines.append(f"\n#### {cat}:")
                            for c in sorted(by_cat[cat], key=lambda x: (x["code"] or "", x["name"] or "")):
                                c_info = f"- **{c['name']}**"
                                if c['code']:
                                    c_info += f" ({c['code']})"
                                if c['credits'] or c['ltp']:
                                    details = []
                                    if c['ltp']: details.append(f"L-T-P: {c['ltp']}")
                                    if c['credits']: details.append(f"Credits: {c['credits']}")
                                    c_info += f" — {', '.join(details)}"
                                lines.append(c_info)
                    else:
                        for c in sorted(other_courses, key=lambda x: (x["code"] or "", x["name"] or "")):
                            c_info = f"- **{c['name']}**"
                            if c['code']:
                                c_info += f" ({c['code']})"
                            if c['credits'] or c['ltp']:
                                details = []
                                if c['ltp']: details.append(f"L-T-P: {c['ltp']}")
                                if c['credits']: details.append(f"Credits: {c['credits']}")
                                c_info += f" — {', '.join(details)}"
                            lines.append(c_info)

                    if matched_entity.get("link"):
                        lines.append(f"\nOfficial Curriculum Document: [Download/View Link]({matched_entity['link']})")
                    return "\n".join(lines)

            # General Specialization / Minor Lookup
            from departments import DEPARTMENTS
            target_dept = None
            for code, config in DEPARTMENTS.items():
                aliases = [code.lower(), config["name"].lower()] + [a.lower() for a in config.get("aliases", [])]
                if any(a in q for a in aliases):
                    target_dept = code
                    break

            is_minor_q = any(term in q for term in ("minor", "minors"))
            is_micro_q = any(term in q for term in ("micro", "micros"))
            is_honours_q = any(term in q for term in ("honours", "honor", "honors"))
            is_spec_q = any(term in q for term in ("specialization", "specialisation", "specializations", "specialisations"))
            is_program_q = any(term in q for term in ("program", "programs", "programme", "programmes", "course", "courses"))

            if is_minor_q or is_micro_q or is_honours_q or is_spec_q or is_program_q or target_dept:
                matching_specs = []
                matching_progs = []

                for s in specializations:
                    if target_dept and s.get("department") != target_dept:
                        continue
                    s_type = s.get("type", "").lower()
                    if is_minor_q and "minor" not in s_type and "minor" not in s.get("name", "").lower():
                        continue
                    if is_micro_q and "micro" not in s_type:
                        continue
                    if is_honours_q and "honour" not in s_type and "honor" not in s_type:
                        continue
                    matching_specs.append(s)

                for p in programs:
                    if target_dept and p.get("department") != target_dept:
                        continue
                    matching_progs.append(p)

                lines = []
                if matching_specs:
                    lines.append("### Relevant Academic Specializations & Minors:")
                    for s in sorted(matching_specs, key=lambda x: x.get("name", "")):
                        status = " (Superseded/Old Version)" if s.get("superseded") else ""
                        lines.append(f"- **{s['name']}** ({s.get('type', 'Specialization')}){status}")
                        if s.get("link"):
                            lines.append(f"  - Document Link: {s['link']}")

                if matching_progs:
                    lines.append("\n### Relevant Academic Programs / Curriculum Frameworks:")
                    for p in sorted(matching_progs, key=lambda x: x.get("name", "")):
                        status = " (Superseded/Old Version)" if p.get("superseded") else ""
                        lines.append(f"- **{p['name']}** ({p.get('level', 'UG/PG')}){status}")
                        if p.get("link"):
                            lines.append(f"  - Document Link: {p['link']}")

                if lines:
                    return "\n".join(lines)



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

        # 9. Alumni Affairs specific queries
        if self.section_code == "alumni-affairs":
            if any(term in q for term in ("medalist", "gold medal", "silver medal", "convocation award")):
                medalists = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "AlumniMedalist"]
                if medalists:
                    lines = ["### IIT Jammu Alumni Medalists:"]
                    from collections import defaultdict
                    by_year = defaultdict(list)
                    for m in medalists:
                        by_year[m.get("year", "Unknown")].append(m)
                    
                    for year in sorted(by_year.keys(), reverse=True):
                        lines.append(f"\n#### Batch / Convocation Year: {year}")
                        for m in sorted(by_year[year], key=lambda x: x.get("award", "")):
                            dept_str = f" ({m['department']})" if m.get("department") else ""
                            lines.append(f"- **{m['name']}**: {m['award']}{dept_str}")
                    return "\n".join(lines)
            
            if any(term in q for term in ("award", "initiative award", "alumni award")):
                awards = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "AlumniAward"]
                if awards:
                    lines = ["### IIT Jammu Alumni Awards:"]
                    for a in awards:
                        lines.append(f"- **{a['name']}** ({a.get('degree', '')} {a.get('department', '')}) - {a.get('award_name', '')} in {a.get('year', '')}")
                    return "\n".join(lines)

        # 10. Career Development Services (CDS) specific queries
        if self.section_code == "cds":
            if any(term in q for term in ("recruiter", "companies visiting", "placement companies", "which company", "list of companies")):
                recruiters = [d.get("name") for n, d in self.graph.nodes(data=True) if d.get("label") == "Recruiter"]
                if recruiters:
                    return f"### Past Recruiters at IIT Jammu:\nSome of the prominent companies that visited campus include:\n" + ", ".join(sorted(recruiters))
            
            # Policy check BEFORE statistics — queries about eligibility, offers,
            # upgrade rules, etc. should hit policy first, not stats.
            if any(term in q for term in ("policy", "rule", "cutoff", "cgpa", "attendance",
                                          "dress code", "debar", "withdraw", "core", "non-core",
                                          "upgrade", "eligible", "eligibility", "sit", "offer",
                                          "another company", "can i apply", "category")):
                policies = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "PlacementPolicy"]
                if policies:
                    matched_policies = []
                    for p in policies:
                        p_name_lower = p.get("name", "").lower()
                        p_desc_lower = p.get("description", "").lower()
                        # Match on policy name keywords or description keywords
                        if p_name_lower in q or any(kw in q for kw in p_name_lower.split()):
                            matched_policies.append(p)
                        elif any(kw in p_desc_lower for kw in ["lpa", "ctc", "category"] if kw in q):
                            matched_policies.append(p)
                    
                    # For queries about offers/upgrade/sitting in another company,
                    # specifically match CTC category and upgrade policies
                    if any(term in q for term in ("offer", "sit", "another company", "lpa")):
                        offer_policies = [p for p in policies if any(
                            kw in p.get("category", "").lower() or kw in p.get("name", "").lower()
                            for kw in ("ctc", "upgrade", "application", "core")
                        )]
                        if offer_policies:
                            matched_policies = offer_policies
                    
                    to_show = matched_policies if matched_policies else policies
                    
                    lines = ["### CDS Placement Policy Rules & Guidelines:"]
                    for p in to_show:
                        lines.append(f"- **{p['name']}** ({p.get('category', 'General')}): {p['description']}")
                    return "\n".join(lines)
            
            if any(term in q for term in ("placement statistic", "placement package", "average package", "highest package", "placement percentage", "placement rate", "placement record", "package", "salary", "lpa")):
                stats = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "PlacementStat"]
                if stats:
                    lines = ["### Placement Statistics at IIT Jammu:"]
                    from collections import defaultdict
                    by_year = defaultdict(list)
                    for s in stats:
                        by_year[s.get("year", "Unknown")].append(s)
                        
                    for year in sorted(by_year.keys(), reverse=True):
                        lines.append(f"\n#### Batch / Placement Year: {year}")
                        for s in by_year[year]:
                            if s.get("degree") == "Overall":
                                lines.append(f"- **Overall Placements**: Percentage Placed: {s.get('percentage_placed', 'N/A')}, Highest CTC: {s.get('highest_salary', 'N/A')}, Average CTC: {s.get('avg_salary', 'N/A')}")
                            else:
                                lines.append(f"- **{s.get('degree')} {s.get('department')}**: Registered: {s.get('registered', 'N/A')}, Placed: {s.get('placed', 'N/A')}, Average Salary: {s.get('avg_salary', 'N/A')}")
                    return "\n".join(lines)

        # 11. International Relations (IR) specific queries
        if self.section_code == "ir":
            if any(term in q for term in ("mou", "collaboration", "partnership", "global networks", "exchange program", "international agreement")):
                mous = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "MOU"]
                if mous:
                    filtered_mous = []
                    for m in mous:
                        country_val = m.get("country", "").lower()
                        if country_val in q or country_val.replace(" ", "") in q:
                            filtered_mous.append(m)
                    
                    to_show = filtered_mous if filtered_mous else mous
                    lines = ["### International Collaborations & MOUs at IIT Jammu:"]
                    for m in sorted(to_show, key=lambda x: x.get("name", "")):
                        lines.append(f"- **{m['partner']}** ({m.get('country', '')}): Collaboration type: {m.get('program_type', '')}")
                    return "\n".join(lines)
            
            # Clubs logic
            club_keywords = (
                "club", "coding", "fintech", "robo-sapiens", "robotics", "re4m", "sae", "mesh", 
                "astria", "kritash", "beatstreet", "dance", "malang", "music", "foot", "tinkerers", 
                "culinary", "abhivyakta", "literary", "anisoul", "animation", "ebsb", "dramatizers", 
                "drama", "theatre", "wellness", "nac"
            )
            if any(term in q for term in club_keywords):
                # Guard: reasoning/comparative/opinion queries should go through LLM,
                # not return raw templates. E.g. "which club is better", "why does X exist",
                # "what does X do", "which is more focused on technicality"
                reasoning_indicators = (
                    "better", "best", "worse", "worst", "more focused", "most focused",
                    "why does", "why do", "why is", "why are",
                    "how does", "how do", "how is", "how are",
                    "should i join", "recommend", "suggestion", "opinion",
                    "compare", "comparison", "versus", "vs",
                    "difference between", "which is",
                )
                is_reasoning = any(ind in q for ind in reasoning_indicators)
                
                clubs = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "Club"]
                if clubs:
                    matched_clubs = []
                    for c in clubs:
                        c_name_lower = c["name"].lower()
                        aliases = [c_name_lower]
                        if "robo-sapiens" in c_name_lower:
                            aliases.extend(["robotics club", "robotics"])
                        if "mesh" in c_name_lower:
                            aliases.append("mesh club")
                        if "coding" in c_name_lower:
                            aliases.append("coding club")
                        if "fintech" in c_name_lower:
                            aliases.append("fintech club")
                        if "sae" in c_name_lower:
                            aliases.append("sae club")
                        if "astria" in c_name_lower:
                            aliases.append("astria-za")
                        if "kritash" in c_name_lower:
                            aliases.append("kritash club")
                        if "beatstreet" in c_name_lower:
                            aliases.append("beatstreet club")
                        if "malang" in c_name_lower:
                            aliases.append("malang club")
                        if "foot" in c_name_lower:
                            aliases.append("foot tinkerers")
                        if "abhivyakta" in c_name_lower:
                            aliases.append("abhivyakta club")
                        if "anisoul" in c_name_lower:
                            aliases.append("anisoul club")
                        if "ebsb" in c_name_lower:
                            aliases.append("ebsb club")
                        if "dramatizers" in c_name_lower:
                            aliases.append("dramatizers club")
                        if "wellness" in c_name_lower:
                            aliases.append("wellness club")
                        if "nac" in c_name_lower:
                            aliases.append("nac club")
                            
                        if any(alias in q for alias in aliases):
                            matched_clubs.append(c)
                    
                    # For reasoning/comparative queries with matched clubs,
                    # fall through to LLM with club context instead of raw template
                    if is_reasoning and not any(ind in q for ind in ("list", "all", "names of")):
                        # Still return None so the query goes to LLM-based reasoning
                        pass
                    else:
                        list_indicators = ("list", "give", "name", "show", "all", "what are the", "various", "different", "names of", "clubs")
                        is_list_query = any(ind in q for ind in list_indicators) and len(matched_clubs) != 1
                        
                        if matched_clubs and not is_list_query:
                            lines = ["### Matched Student Club(s) at IIT Jammu:"]
                            for c in sorted(matched_clubs, key=lambda x: x.get("name", "")):
                                lines.append(f"- **{c['name']}** ({c.get('category', '')} Club): {c.get('description', '')}")
                            return "\n".join(lines)
                        else:
                            lines = ["### Student Clubs at IIT Jammu:"]
                            for c in sorted(clubs, key=lambda x: x.get("name", "")):
                                lines.append(f"- **{c['name']}** ({c.get('category', '')} Club): {c.get('description', '')}")
                            return "\n".join(lines)
                    
            if any(term in q for term in ("sports", "facility", "cricket", "gym", "basketball", "football", "fitness")):
                facilities = [d for n, d in self.graph.nodes(data=True) if d.get("label") in ("SportsFacility", "SportsFest")]
                if facilities:
                    lines = ["### Sports Facilities & Events at IIT Jammu:"]
                    for f in sorted(facilities, key=lambda x: x.get("name", "")):
                        lines.append(f"- **{f['name']}**: {f.get('description', '')}")
                    return "\n".join(lines)
                    
            if any(term in q for term in ("hostel", "residential", "canary", "braeg", "fulgar", "dedhar", "egret")):
                hostels = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "Hostel"]
                if hostels:
                    matched_hostels = []
                    for h in hostels:
                        h_name_lower = h["name"].lower()
                        if h_name_lower in q:
                            matched_hostels.append(h)
                            
                    list_indicators = ("list", "give", "name", "show", "all", "what are the", "various", "different", "names of", "hostels")
                    is_list_query = any(ind in q for ind in list_indicators) and len(matched_hostels) != 1
                    
                    if matched_hostels and not is_list_query:
                        lines = ["### Matched Student Hostel(s) at IIT Jammu:"]
                        for h in sorted(matched_hostels, key=lambda x: x.get("name", "")):
                            lines.append(f"- **{h['name']} Hostel** ({h.get('gender', '')}' Hostel): {h.get('description', '')}")
                        return "\n".join(lines)
                    else:
                        lines = ["### Student Hostels at IIT Jammu:"]
                        for h in sorted(hostels, key=lambda x: x.get("name", "")):
                            lines.append(f"- **{h['name']} Hostel** ({h.get('gender', '')}' Hostel): {h.get('description', '')}")
                        return "\n".join(lines)
                    
            if any(term in q for term in ("fest", "festival", "anhad", "pravaah", "convoquer", "nexus", "pragyaan", "udyamitsav")):
                fests = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "Fest"]
                if fests:
                    matched_fests = []
                    for f in fests:
                        f_name_lower = f["name"].lower()
                        if f_name_lower in q:
                            matched_fests.append(f)
                            
                    list_indicators = ("list", "give", "name", "show", "all", "what are the", "various", "different", "names of", "fests", "festivals")
                    is_list_query = any(ind in q for ind in list_indicators) and len(matched_fests) != 1
                    
                    if matched_fests and not is_list_query:
                        lines = ["### Matched Festival(s) & Event(s) at IIT Jammu:"]
                        for f in sorted(matched_fests, key=lambda x: x.get("name", "")):
                            lines.append(f"- **{f['name']}** ({f.get('category', '')}): {f.get('description', '')}")
                        return "\n".join(lines)
                    else:
                        lines = ["### Annual Festivals & Events at IIT Jammu:"]
                        for f in sorted(fests, key=lambda x: x.get("name", "")):
                            lines.append(f"- **{f['name']}** ({f.get('category', '')}): {f.get('description', '')}")
                        return "\n".join(lines)

        # 12. Medical Centre specific queries
        if self.section_code == "medical-centre":
            if any(term in q for term in ("service", "timing", "dental", "physiotherapy", "pharmacy", "ambulance", "ward", "dressing", "ecg", "laboratory")):
                services = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "MedicalService"]
                if services:
                    lines = ["### Medical Services & Working Hours at Health Centre:"]
                    for s in sorted(services, key=lambda x: x.get("name", "")):
                        lines.append(f"- **{s['name']}**: Timings: {s.get('timings', '')} | Description: {s.get('description', '')}")
                    return "\n".join(lines)
            
            if any(term in q for term in ("doctor", "specialist", "medical officer", "dentist", "physiotherapist")):
                # Guard: superlative/ranking/reasoning queries should go through
                # LLM so it can reason about the data, not dump all doctors.
                superlative_indicators = (
                    "highest", "most", "best", "longest", "maximum",
                    "senior", "senior most", "seniormost",
                    "least", "lowest", "youngest", "newest",
                    "rank", "ranking", "top",
                )
                is_superlative = any(ind in q for ind in superlative_indicators)
                
                doctors = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "MedicalDoctor"]
                if doctors and not is_superlative:
                    lines = ["### Doctors at the Health Centre, IIT Jammu:\n"]
                    for doc in sorted(doctors, key=lambda x: x.get("name", "")):
                        doc_name = doc['name']
                        if not doc_name.startswith("Dr. ") and not doc_name.startswith("Dr "):
                            doc_name = f"Dr. {doc_name}"
                        lines.append(f"#### {doc_name}")
                        lines.append(f"- **Designation:** {doc.get('designation', 'Medical Officer')}")
                        if doc.get("qualifications"):
                            lines.append(f"- **Qualifications:** {doc['qualifications']}")
                        if doc.get("experience"):
                            lines.append(f"- **Experience:** {doc['experience']}")
                        if doc.get("email"):
                            lines.append(f"- **Email:** {doc['email']}")
                        if doc.get("phone"):
                            lines.append(f"- **Phone:** {doc['phone']}")
                        lines.append("") # empty line spacing
                    return "\n".join(lines).strip()
                    
            if any(term in q for term in ("hospital", "cghs", "empanel", "narayana", "ascoms", "bee enn", "fortis", "ankur maitrika")):
                hospitals = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "EmpaneledHospital"]
                if hospitals:
                    lines = ["### Empaneled Hospitals at IIT Jammu:"]
                    for h in sorted(hospitals, key=lambda x: x.get("name", "")):
                        lines.append(f"- **{h['name']}** ({h.get('location', '')}): Rates type: {h.get('rate_type', '')}")
                    return "\n".join(lines)

        # 13. Outreach & Skilling Division (OSD) specific queries
        if self.section_code == "osd":
            if any(term in q for term in ("uba", "unnat bharat", "coordinator of uba", "village adoption")):
                ubas = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "UBAProgram"]
                if ubas:
                    u = ubas[0]
                    return f"### {u['name']} (UBA) at IIT Jammu:\n- **Description:** {u.get('description', '')}\n- **Focus Areas:** {u.get('focus_areas', '')}\n- **Coordinator:** {u.get('coordinator', '')}"
            
            if any(term in q for term in ("ces", "essential skill", "skilling course", "vocational training")):
                cess = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "CESProgram"]
                if cess:
                    c = cess[0]
                    return f"### {c['name']} (CES) at IIT Jammu:\n- **Description:** {c.get('description', '')}\n- **Available Courses:** {c.get('courses', '')}"
            
            if any(term in q for term in ("event", "raise", "outreach program", "skilling program", "summer internship")):
                events = [d for n, d in self.graph.nodes(data=True) if d.get("label") == "OSDEvent"]
                if events:
                    lines = ["### OSD Flagship Programs & Events:"]
                    for ev in sorted(events, key=lambda x: x.get("name", "")):
                        desc_str = f" - {ev.get('description')}" if ev.get("description") else ""
                        lines.append(f"- **{ev['name']}** ({ev.get('category', '')}){desc_str}")
                    return "\n".join(lines)

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
        rules_context = ""
        rules_provenance = None
        is_academic_rules_request = False
        
        if self.section_code == "academics":
            from graphrag.rules_retriever import RulesRetriever
            from graphrag.intent_utils import is_academic_rules_query
            rr = RulesRetriever()
            intent = rr.classify_intent(query)
            is_rules_q = is_academic_rules_query(query) or (
                intent["grades"]
                or intent["milestones"]
                or intent["credits"]
                or intent["facts"]
                or any(term in query.lower() for term in (
                    "rule", "regulation", "requirement", "curriculum",
                    "minor", "specialization", "specialisation"
                ))
            )
            is_academic_rules_request = bool(is_rules_q)
            if is_rules_q:
                ret_res = rr.retrieve(query)
                if ret_res["fts_results"] or any(ret_res["structured_data"].values()):
                    rules_context = rr.generate_context(ret_res)
                
                # Scan raw scraped markdown files for matching document links
                # only when the user explicitly asks for links/documents.
                import os
                scraped_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scraped_data", "sections", "academics"))
                is_link_query = any(term in query.lower() for term in (
                    "link", "url", "download", "pdf", "document", "file", "drive"
                ))
                if is_link_query and os.path.exists(scraped_dir):
                    query_terms = [t.lower() for t in re.findall(r'\w+', query) if len(t) > 2 and t.lower() not in (
                        'what', 'the', 'link', 'for', 'download', 'pdf', 'document', 'url', 'file', 'drive', 'how', 'get', 'give', 'can', 'find', 'show', 'where'
                    )]
                    if query_terms:
                        matches = []
                        for fn in os.listdir(scraped_dir):
                            if not fn.endswith('.md'):
                                continue
                            filepath = os.path.join(scraped_dir, fn)
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        if '[' in line and '](' in line:
                                            link_text = re.findall(r'\[([^\]]+)\]', line)
                                            if link_text:
                                                lt = link_text[0].lower()
                                                matches_count = sum(1 for term in query_terms if term in lt)
                                                if matches_count > 0:
                                                    matches.append((matches_count, line.strip()))
                            except Exception:
                                pass
                        if matches:
                            matches.sort(key=lambda x: x[0], reverse=True)
                            seen = set()
                            result_links = []
                            max_matches = matches[0][0]
                            threshold = max(1, min(2, max_matches))
                            for count, link in matches:
                                if count >= threshold and link not in seen:
                                    seen.add(link)
                                    clean_line = re.sub(r'^\d+\.\s*', '', link)
                                    result_links.append(f"- {clean_line}")
                                    if len(result_links) >= 5:
                                        break
                            if result_links:
                                links_block = "Relevant document links found on the Academics website:\n" + "\n".join(result_links)
                                if rules_context:
                                    rules_context = rules_context + "\n\n" + links_block
                                else:
                                    rules_context = links_block
                
                if rules_context:
                    rules_provenance = self._build_provenance(
                        direct=False,
                        local_results=[],
                        vector_results=[],
                        section_word_counts={"graph": len(rules_context.split())}
                    )
                    rules_provenance["route"] = "rules_db"
                    rules_provenance["source_mode"] = "hybrid_db"

        # 1. Deterministic context from graph (prepend to combined blocks, don't return early)
        direct_ctx = self.get_deterministic_context(query)
        if direct_ctx:
            logger.info(f"Section '{self.section_code}': deterministic context found, injecting as priority context.")

        if self.section_code == "academics" and is_academic_rules_request:
            has_course_code = bool(re.search(r"\b[a-zA-Z]{2,3}\s*\d{3}\b", query))
            is_course_query = (
                any(term in query.lower() for term in ("course", "syllabus", "curriculum", "ltp", "l-t-p", "structure of", "specialization", "specialisation", "minor", "credit", "credits", "graduate", "graduation")) 
                and not any(term in query.lower() for term in ("rule", "regulation", "policy", "guideline", "requirement", "criteria", "scheme", "evaluation", "convention", "award", "medal", "medals", "registration", "register", "withdrawal", "withdraw", "allotment", "internship", "amendment", "change"))
            )
            
            if rules_context and not has_course_code and not is_course_query:
                return {
                    "context": rules_context,
                    "provenance": rules_provenance or self._build_provenance(
                        direct=False,
                        local_results=[],
                        vector_results=[],
                        section_word_counts={"graph": len(rules_context.split())},
                    ),
                    "answerability": {"answerable": True, "reason": "", "matched_terms": [], "missing_concepts": []},
                    "fallback_response": None,
                }
            # Fall through to vector/local search if rules_context is empty or if it looks like a course/curriculum query



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
        
        if direct_ctx:
            combined_blocks.append("## Authoritative Section Data\n\n" + direct_ctx)
            word_count += len(direct_ctx.split())

        if rules_context:
            combined_blocks.append(rules_context)
            word_count += len(rules_context.split())
        
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
        
        if rules_provenance:
            # Merge provenances
            provenance["route"] = "rules_db+chunks"
            provenance["source_mode"] = "hybrid_db+chunks"
            provenance["graph"]["word_count"] = provenance["graph"].get("word_count", 0) + rules_provenance["graph"].get("word_count", 0)

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
