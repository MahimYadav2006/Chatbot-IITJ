"""
Hybrid Retrieval Engine for GraphRAG.
Combines entity search, vector search, and community summaries
into a clean, structured context for the LLM.
"""

import os
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, List, Dict, Tuple, Optional

import networkx as nx

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Common abbreviation expansions for research topic matching.
# When a user searches for "ai", we also search for "artificial intelligence", etc.
TOPIC_EXPANSIONS = {
    "ai": ["artificial intelligence"],
    "ml": ["machine learning"],
    "dl": ["deep learning"],
    "nlp": ["natural language processing"],
    "cv": ["computer vision"],
    "iot": ["internet of things"],
    "vlsi": ["very large scale integration"],
    "dsp": ["digital signal processing"],
    "hci": ["human computer interaction"],
    "rl": ["reinforcement learning"],
    "nn": ["neural network", "neural networks"],
    "cnn": ["convolutional neural network"],
    "rnn": ["recurrent neural network"],
    "gan": ["generative adversarial network"],
    "llm": ["large language model"],
    "ev": ["electric vehicle"],
    "uav": ["unmanned aerial vehicle"],
    "rf": ["radio frequency"],
    "fpga": ["field programmable gate array"],
    "bms": ["battery management system"],
    "mems": ["microelectromechanical systems"],
}


def _topic_matches_text(topic: str, text: str) -> bool:
    """Check if a research topic matches against text using word-boundary-aware matching.

    For short topics (<=4 chars like 'ai', 'nlp', 'iot'), uses word-boundary regex
    to prevent substring collisions (e.g. 'ai' matching 'uncertainty').
    For longer topics (>=5 chars like 'machine learning'), uses substring match.
    Also expands known abbreviations (e.g. 'ai' also searches 'artificial intelligence').
    """
    topic_lower = topic.lower().strip()
    text_lower = text.lower()

    # Build list of all topic variants to search for
    variants = [topic_lower]
    variants.extend(TOPIC_EXPANSIONS.get(topic_lower, []))

    for variant in variants:
        if len(variant) <= 4:
            # Short topic: require word boundary match
            if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                return True
        else:
            # Longer topic: substring match is safe
            if variant in text_lower:
                return True

    return False


class HybridRetriever:
    def __init__(self, graph: nx.DiGraph, embedding_engine, community_reports: List[Dict], dept_code: str = "ee"):
        self.graph = graph
        self.embeddings = embedding_engine
        self.community_reports = community_reports
        self.dept_code = dept_code
        from departments import get_department
        self.dept_config = get_department(dept_code)
        self.dept_node_id = f"IIT Jammu {dept_code.upper()} Department"
        self._build_indexes()

    def _build_indexes(self):
        """Pre-build lookup indexes for fast entity search."""
        self._entity_name_index = {}
        self._entity_label_index = defaultdict(list)
        self._label_counts = Counter()
        for node_id, data in self.graph.nodes(data=True):
            # Enforce department filtering
            if data.get("department") and data.get("department") != self.dept_code:
                continue
            label = data.get("label", "")
            self._label_counts[label] += 1
            if label in ("TextChunk", "Document"):
                continue
            name = data.get("name", node_id).lower()
            self._entity_name_index[name] = node_id
            self._entity_label_index[label].append(node_id)

    def _normalize_token(self, token: str) -> str:
        """Apply lightweight normalization for keyword coverage checks."""
        cleaned = re.sub(r"[^a-z0-9]+", "", token.lower())
        for suffix in ("ing", "ers", "ies", "es", "s"):
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 2:
                if suffix == "ies":
                    return cleaned[:-3] + "y"
                return cleaned[:-len(suffix)]
        return cleaned

    def _query_tokens(self, query: str) -> List[str]:
        """Tokenize a query into normalized keywords."""
        return [
            self._normalize_token(token)
            for token in re.findall(r"[A-Za-z0-9]+", query.lower())
            if self._normalize_token(token)
        ]

    def _is_department_contact_query(self, query: str) -> bool:
        """Detect generic department contact / point-of-contact questions."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        
        # If it asks about a specific topic (e.g. for control engineering), it should be treated
        # as a topic query rather than generic department contact query.
        if any(re.search(rf"\b{term}\b", q) for term in ("for", "about", "regarding", "in")):
            m = re.search(r"\b(?:for|about|regarding|in)\b\s+(.+)", q)
            if m:
                target = m.group(1).strip()
                generic_terms = {"department", "dept", "admission", "admissions", "general", "queries", "query", "info", "information", "help", "support", "this", "it", "more info", "more information"}
                words = [w.strip("?. ") for w in target.split()]
                if words and not all(w in generic_terms for w in words):
                    return False

        triggers = (
            "point of contact",
            "main contact",
            "official contact",
            "department contact",
            "contact person",
            "whom should i contact",
            "who should i contact",
            "who do i contact",
        )
        if any(trigger in q for trigger in triggers):
            return True
        return "contact" in q and "department" in q and not self._extract_email_query_name(query)

    def _get_hod_member(self) -> Optional[Dict[str, str]]:
        """Return the department HoD if present in the faculty roster."""
        roster = self.get_faculty_roster()
        for member in roster:
            if member.get("is_hod"):
                return member

        from departments import get_markdown_dir

        markdown_dir = get_markdown_dir(self.dept_code)
        for member in roster:
            source_file = self.graph.nodes.get(
                self._find_entity_by_name(member.get("name", ""), allowed_labels=("Faculty",)) or "",
                {}
            ).get("source_file", "")
            if not source_file:
                continue
            source_path = os.path.join(markdown_dir, source_file)
            if not os.path.exists(source_path):
                continue
            try:
                with open(source_path, "r", encoding="utf-8") as handle:
                    content = handle.read().lower()
            except OSError:
                continue
            if "head of department" in content or "presently, head of department" in content:
                return member
        return None

    def _build_department_contact_answer(self) -> Optional[str]:
        """Build a deterministic answer for generic department contact requests."""
        hod = self._get_hod_member()
        if not hod:
            return None

        lines = [
            (
                f"The main official point of contact for the {self.dept_config['full_name']} "
                f"at IIT Jammu is **{hod['name']}**, Head of Department."
            )
        ]
        if hod.get("email"):
            lines.append(f"Official email: {hod['email']}")
        if hod.get("profile_url"):
            lines.append(f"Profile: [IIT Jammu faculty page]({hod['profile_url']})")
        return "\n".join(lines)

    # ── Lab queries ──────────────────────────────────────────────────────

    def _is_lab_query(self, query: str) -> bool:
        """Detect queries about department labs/laboratories/facilities."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        lab_triggers = (
            "lab", "labs", "laboratory", "laboratories", "research lab",
            "teaching lab", "facilities", "research facilities",
        )
        return any(trigger in q for trigger in lab_triggers)

    def _build_labs_answer(self, query: Optional[str] = None) -> Optional[str]:
        """Build a deterministic answer listing all labs in the department."""
        from departments import CORRECT_LABS
        
        dept_name = self.dept_config.get("full_name", self.dept_code)
        labs = CORRECT_LABS.get(self.dept_code, [])
        
        if not labs:
            if query:
                q_lower = query.lower()
                aliases = self.dept_config.get("aliases", [])
                aliases_lower = [a.lower() for a in aliases]
                dept_name_lower = self.dept_config.get("name", "").lower()
                full_name_lower = dept_name.lower()
                
                mentions_dept = (
                    self.dept_code in q_lower or
                    dept_name_lower in q_lower or
                    full_name_lower in q_lower or
                    any(alias in q_lower for alias in aliases_lower)
                )
                if mentions_dept:
                    return f"There are no laboratories in the {dept_name}."
            return None

        lines = [f"**Labs and Facilities in {dept_name}:**\n"]
        for idx, lab_name in enumerate(labs, 1):
            lines.append(f"{idx}. {lab_name}")
        return "\n".join(lines)

    # ── Department address / contact info queries ────────────────────────

    def _is_address_query(self, query: str) -> bool:
        """Detect queries about department address, location, or contact info."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        address_triggers = (
            "address", "location", "where is the department",
            "department address", "postal address", "mailing address",
            "department email", "department phone", "phone number",
        )
        return any(trigger in q for trigger in address_triggers)

    def _build_address_answer(self) -> Optional[str]:
        """Build a deterministic answer with department address/contact info."""
        # Look for ContactInfo node
        contact_data = None
        for nid, data in self.graph.nodes(data=True):
            if data.get("label") == "ContactInfo":
                contact_data = data
                break

        # Fallback: check department node for stored contact info
        if not contact_data:
            dept_data = self.graph.nodes.get(self.dept_node_id, {})
            if dept_data.get("address") or dept_data.get("contact_email"):
                contact_data = {
                    "address": dept_data.get("address", ""),
                    "email": dept_data.get("contact_email", ""),
                    "phone": dept_data.get("phone", ""),
                }

        if not contact_data:
            return None

        dept_name = self.dept_config.get("full_name", self.dept_code)
        lines = [f"**{dept_name} Contact Information:**\n"]
        address = contact_data.get("address", "")
        email = contact_data.get("email", "")
        phone = contact_data.get("phone", "")

        if address:
            lines.append(f"**Address:** {address}")
        if email:
            lines.append(f"**Email:** {email}")
        if phone and "update" not in phone.lower():
            lines.append(f"**Phone:** {phone}")
        return "\n".join(lines)

    # ── Graduated PhD / PhD alumni queries ───────────────────────────────

    def _is_graduated_phd_query(self, query: str) -> bool:
        """Detect queries about graduated/passed-out PhD students or PhD alumni."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        triggers = (
            "graduated phd", "phd alumni", "passed out phd",
            "completed phd", "phd graduates", "graduated doctoral",
            "phd passed", "phd who graduated", "phd who have graduated",
            "former phd", "past phd", "phd alumni list",
        )
        return any(trigger in q for trigger in triggers)

    def _build_graduated_phd_answer(self) -> Optional[str]:
        """Build a deterministic answer listing graduated PhD students."""
        grads = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("label") == "GraduatedPhD":
                grads.append({
                    "name": data.get("name", nid),
                    "supervisor": data.get("supervisor", ""),
                    "thesis": data.get("thesis_title", ""),
                    "year": data.get("graduation_year", ""),
                })
        if not grads:
            return None

        # Sort by year descending
        grads.sort(key=lambda x: x.get("year", "0"), reverse=True)

        dept_name = self.dept_config.get("full_name", self.dept_code)
        lines = [f"**Graduated PhD Students from {dept_name}:**\n"]
        for g in grads:
            line = f"- **{g['name']}**"
            if g["year"]:
                line += f" (Year: {g['year']})"
            if g["supervisor"]:
                line += f" — Supervisor: {g['supervisor']}"
            if g["thesis"]:
                line += f" — Thesis: {g['thesis']}"
            lines.append(line)
        return "\n".join(lines)

    # ── Department Research Areas queries ────────────────────────────────

    def _is_department_research_areas_query(self, query: str) -> bool:
        """Detect broad queries asking for the department's research areas."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        patterns = [
            r"what research areas does\b",
            r"research areas of\b",
            r"what areas does\b",
            r"research focus of\b",
            r"what does .* research\b",
            r"list research areas\b",
            r"all research areas\b",
        ]
        return any(re.search(p, q) for p in patterns)

    def _build_department_research_areas_answer(self) -> Optional[str]:
        """Build a deterministic list of the department's research areas and connected faculty."""
        from collections import defaultdict
        
        area_to_faculty = defaultdict(list)
        
        # 1. Collect all ResearchArea / ResearchCategory nodes connected to Faculty
        for u, v, edge_data in self.graph.edges(data=True):
            if edge_data.get("type") in ("RESEARCHES_IN", "STUDIES", "RELATED"):
                u_data = self.graph.nodes.get(u, {})
                v_data = self.graph.nodes.get(v, {})
                
                if u_data.get("label") == "Faculty" and v_data.get("label") in ("ResearchArea", "ResearchCategory"):
                    fac_name = u_data.get("name")
                    area_name = v_data.get("name")
                    if fac_name and area_name:
                        if fac_name not in area_to_faculty[area_name]:
                            area_to_faculty[area_name].append(fac_name)

        dept_name = self.dept_config.get("full_name", self.dept_code)
        lines = [f"**Research Areas in the {dept_name}:**\n"]
        
        if area_to_faculty:
            # We have structured research areas in the graph
            for area in sorted(area_to_faculty.keys()):
                faculty = sorted(area_to_faculty[area])
                lines.append(f"- **{area}**: {', '.join(faculty)}")
            return "\n".join(lines)
            
        # 2. Fallback: Aggregate research_interests from all Faculty nodes if no structural areas exist
        faculty_interests = []
        for nid, d in self.graph.nodes(data=True):
            if d.get("label") == "Faculty":
                interests = d.get("research_interests") or d.get("academic_interests")
                if interests:
                    faculty_interests.append((d.get("name"), interests))
                    
        if faculty_interests:
            faculty_interests.sort(key=lambda x: x[0])
            for fac, interests in faculty_interests:
                lines.append(f"- **{fac}**: {interests}")
            return "\n".join(lines)
            
        return None

    # ── Alumni queries ───────────────────────────────────────────────────

    def _is_alumni_query(self, query: str) -> bool:
        """Detect queries about department alumni."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        triggers = (
            "alumni", "alumni list", "graduated students", "passed out",
            "graduating batch", "batch", "former students",
        )
        # Avoid matching PhD-specific queries here
        if self._is_graduated_phd_query(query):
            return False
        return any(trigger in q for trigger in triggers)

    def _build_alumni_answer(self) -> Optional[str]:
        """Build a deterministic answer listing alumni information."""
        alumni = []
        batches = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("label") == "Alumni":
                alumni.append(data.get("name", nid))
            elif data.get("label") == "AlumniBatch":
                batches.append(data.get("name", nid))

        if not alumni and not batches:
            return None

        dept_name = self.dept_config.get("full_name", self.dept_code)
        lines = [f"**Alumni from {dept_name}:**\n"]

        if batches:
            lines.append("**Alumni Batches:**")
            for batch in sorted(batches):
                lines.append(f"- {batch}")
            lines.append("")

        if alumni:
            lines.append("**Alumni:**")
            for name in sorted(alumni):
                lines.append(f"- {name}")

        return "\n".join(lines)

    def _build_administration_answer(self, query: str) -> Optional[str]:
        q = query.lower().strip()
        
        # 1. Director / Registrar / BOG Chairman
        if "director" in q and not any(kw in q for kw in ("committee", "board", "senate")):
            director_node = None
            for node_id, data in self.graph.nodes(data=True):
                if data.get("is_director"):
                    director_node = data
                    break
            if director_node:
                return f"**{director_node['name']}** is the Director of IIT Jammu.\n\n{director_node.get('details', '')}"
                
        if "registrar" in q and not any(kw in q for kw in ("committee", "board", "senate")):
            registrar_node = None
            for node_id, data in self.graph.nodes(data=True):
                if data.get("is_registrar"):
                    registrar_node = data
                    break
            if registrar_node:
                ans = f"**{registrar_node['name']}** is the Registrar of IIT Jammu."
                if registrar_node.get("email"):
                    ans += f"\n- Email: {registrar_node['email']}"
                return ans

        if ("bog chairman" in q or "board of governors chairman" in q or "chairman of bog" in q or "chairman of the board of governors" in q) and not "committee" in q:
            chairman_node = None
            for node_id, data in self.graph.nodes(data=True):
                if data.get("is_bog_chairman"):
                    chairman_node = data
                    break
            if chairman_node:
                return f"**{chairman_node['name']}** is the Chairman of the Board of Governors (BoG) of IIT Jammu."

        # 2. Deans and Associate Deans / AD
        is_deans_list = any(term in q for term in ("list of deans", "list all deans", "who are the deans", "who comprises the deans"))
        is_assoc_deans_list = any(term in q for term in ("list of associate deans", "list all associate deans", "who are the associate deans", "list of ad", "list all ad", "who are the ad", "who is the ad"))
        
        if is_deans_list:
            deans = []
            for node_id, data in self.graph.nodes(data=True):
                if data.get("admin_type") == "Dean":
                    deans.append(data)
            if deans:
                deans.sort(key=lambda x: x["name"])
                lines = ["### Deans of IIT Jammu", ""]
                for d in deans:
                    line = f"- **{d['name']}**: {d.get('admin_role', '')}"
                    if d.get("email"):
                        line += f" (Email: {d['email']})"
                    lines.append(line)
                return "\n".join(lines)

        if is_assoc_deans_list:
            adeans = []
            for node_id, data in self.graph.nodes(data=True):
                if data.get("admin_type") == "Associate Dean":
                    adeans.append(data)
            if adeans:
                adeans.sort(key=lambda x: x["name"])
                lines = ["### Associate Deans of IIT Jammu", ""]
                for d in adeans:
                    line = f"- **{d['name']}**: Associate Dean of {d.get('admin_role', '')}"
                    if d.get("email"):
                        line += f" (Email: {d['email']})"
                    lines.append(line)
                return "\n".join(lines)

        if "dean" in q or re.search(r'\bad\b', q):
            matches = []
            
            def stem_word(w):
                w = w.lower().strip()
                if w.endswith('ies'):
                    return w[:-3] + 'y'
                if w.endswith('s') and not w.endswith('ss'):
                    return w[:-1]
                return w
                
            q_words = [stem_word(w) for w in re.split(r'\W+', q) if len(stem_word(w)) > 2 or stem_word(w) == 'ad']
            if "affair" in q_words:
                q_words.append("activity")
            if "activity" in q_words:
                q_words.append("affair")
                
            for node_id, data in self.graph.nodes(data=True):
                admin_role = data.get("admin_role", "")
                admin_type = data.get("admin_type", "")
                if admin_role and admin_type:
                    clean_role = re.sub(r'[\(\)\-]', ' ', admin_role.lower())
                    role_words = [stem_word(w) for w in re.split(r'\W+', clean_role) if len(stem_word(w)) > 2 or stem_word(w) == 'ad']
                    
                    if role_words and all(rw in q_words for rw in role_words):
                        if "associate" in q or re.search(r'\bad\b', q):
                            if admin_type != "Associate Dean":
                                continue
                        else:
                            if admin_type != "Dean":
                                continue
                        matches.append(data)
            if matches:
                lines = []
                for m in matches:
                    lines.append(f"**{m['name']}** is the {m['admin_type']} ({m['admin_role']})." + (f" Email: {m['email']}" if m.get('email') else ""))
                return "\n".join(lines)

        # 3. Committees
        committees = {
            "finance_committee": ["finance committee", "finance committe"],
            "board_of_governors": ["board of governors", "bog", "board of governor"],
            "building_and_works_bwc": ["building and works", "bwc", "building & works", "building and work"],
            "senate": ["senate", "academic council", "academic council member"],
            "annual_action_plan_committee": ["annual action plan", "action plan committee", "annual plan committee", "annual plan"]
        }
        
        target_committee_node = None
        target_comm_aliases = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get("label") == "Committee":
                name_lower = data.get("name", "").lower()
                for comm_id, aliases in committees.items():
                    if any(alias in q for alias in aliases):
                        if any(alias in name_lower for alias in aliases):
                            target_committee_node = node_id
                            target_comm_aliases = aliases
                            break
                if target_committee_node:
                    break
                
        if target_committee_node:
            comm_data = self.graph.nodes[target_committee_node]
            comm_name = comm_data.get("name", target_committee_node)
            
            members = []
            for source, target, edge_data in self.graph.in_edges(target_committee_node, data=True):
                if edge_data.get("type") == "MEMBER_OF":
                    member_data = self.graph.nodes.get(source, {})
                    role = edge_data.get("role_in_committee", "")
                    members.append({
                        "name": member_data.get("name", source),
                        "role": role
                    })
            
            # Specific member lookup inside committee
            for m in members:
                m_name_clean = re.sub(r'\b(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Sh\.?|Shri\.?)\b', '', m["name"], flags=re.IGNORECASE).strip().lower()
                if m_name_clean and m_name_clean in q:
                    return f"In the **{comm_name}**, **{m['name']}** is: \n\n{m['role']}"
                    
            q_clean = q
            for alias in target_comm_aliases:
                q_clean = q_clean.replace(alias, "")
            q_clean = re.sub(r'\b(?:who|is|in|the|of|committee|board|senate|member|representative|nominee)\b', '', q_clean, flags=re.IGNORECASE)
            q_words = [w.strip() for w in re.split(r'\W+', q_clean) if len(w.strip()) > 1]
            
            # List query bypass to list all members instead of single role matching
            is_list_query = any(kw in q for kw in ("list", "all", "comprise", "who are in", "members", "who comprises"))
            
            if q_words and not is_list_query:
                best_match_member = None
                best_match_count = 0
                for m in members:
                    role_lower = m["role"].lower()
                    match_count = sum(1 for w in q_words if w in role_lower)
                    if match_count > best_match_count:
                        best_match_count = match_count
                        best_match_member = m
                if best_match_member and best_match_count >= len(q_words) - 1:
                    return f"**{best_match_member['name']}** is the {best_match_member['role']} in the **{comm_name}**."

            # Always return full list of members by default for matched committee
            lines = [f"### Members of the {comm_name}", ""]
            def member_sort_key(m):
                role_lower = m["role"].lower()
                if "chairman" in role_lower or "chairperson" in role_lower:
                    return 0
                if "director" in role_lower:
                    return 1
                if "secretary" in role_lower:
                    return 9
                return 5
            members.sort(key=lambda x: (member_sort_key(x), x["name"]))
            
            for m in members:
                role_disp = m["role"].replace("\n", " ").strip()
                lines.append(f"- **{m['name']}**: {role_disp}")
            return "\n".join(lines)

        # 4. Person lookup (to show their exact post, designation, and committee roles)
        # Guard: Skip person lookup for queries about research, comparison, or
        # academic work. These should fall through to department-specific
        # retrievers that have the actual faculty research data.
        non_admin_intents = (
            "compare", "comparison", "versus", "vs", "difference between",
            "research work", "research interest", "research area",
            "research output", "research expertise", "research contribution",
            "academic work", "academic interest",
            "publication", "publications", "paper", "papers",
            "journal", "conference",
            "teaching", "courses taught",
            "qualification", "education background",
            "specialization", "specialisation", "expertise area",
        )
        if any(term in q for term in non_admin_intents):
            return None

        potential_names = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("label") in ("Faculty", "AdminOfficial"):
                name = data.get("name", node_id)
                potential_names.append((node_id, name, data))
                
        matched_person = None
        for node_id, name, data in potential_names:
            name_clean = re.sub(r'\b(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Sh\.?|Shri\.?)\b', '', name, flags=re.IGNORECASE).strip().lower()
            name_clean = re.sub(r'\(?ias\)?', '', name_clean, flags=re.IGNORECASE).strip()
            
            q_clean_name = re.sub(r'\b(?:who|is|shri|sh\.?|dr\.?|prof\.?|mr\.?|ms\.?|about|role|post|of|in|for|the)\b', '', q, flags=re.IGNORECASE).strip()
            q_clean_name = re.sub(r'\(?ias\)?', '', q_clean_name, flags=re.IGNORECASE).strip()
            
            if name_clean and (name_clean in q_clean_name or q_clean_name in name_clean):
                matched_person = (node_id, name, data)
                break
                
        if matched_person:
            node_id, name, data = matched_person
            roles = []
            
            if data.get("is_director"):
                roles.append("Director of IIT Jammu")
            if data.get("is_registrar"):
                roles.append("Registrar of IIT Jammu")
            if data.get("is_bog_chairman"):
                roles.append("Chairman of the Board of Governors (BoG)")
                
            admin_type = data.get("admin_type")
            admin_role = data.get("admin_role")
            if admin_type and admin_role:
                roles.append(f"{admin_type} ({admin_role})")
                
            # Check outgoing MEMBER_OF edges
            for u, v, edge_data in self.graph.out_edges(node_id, data=True):
                if edge_data.get("type") == "MEMBER_OF":
                    comm_node = self.graph.nodes.get(v, {})
                    comm_name = comm_node.get("name", v)
                    role_in_comm = edge_data.get("role_in_committee", "Member")
                    roles.append(f"In the **{comm_name}**: {role_in_comm}")
                    
            # Check incoming MEMBER_OF edges
            for u, v, edge_data in self.graph.in_edges(node_id, data=True):
                if edge_data.get("type") == "MEMBER_OF":
                    comm_node = self.graph.nodes.get(u, {})
                    comm_name = comm_node.get("name", u)
                    role_in_comm = edge_data.get("role_in_committee", "Member")
                    roles.append(f"In the **{comm_name}**: {role_in_comm}")

            if roles:
                ans_lines = [f"**{name}** holds the following position(s) at IIT Jammu:", ""]
                for r in roles:
                    r_cleaned = re.sub(r'\s+', ' ', r.replace('\n', ' ')).strip()
                    ans_lines.append(f"- {r_cleaned}")
                if data.get("email"):
                    ans_lines.append(f"- Email: {data['email']}")
                return "\n".join(ans_lines)

    def _is_broad_reasoning_query(self, query: str) -> bool:
        """Relax strict evidence gating for synthesis-heavy prompts."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        return any(term in q for term in (
            "summarize", "summarise", "overview", "analyze", "analyse", "insight",
            "trend", "compare", "comparison", "how does", "how do", "why",
            "based on", "synthesis", "relationship between"
        ))


    def _infer_query_concepts(self, query: str) -> List[str]:
        """Identify concrete concepts that should be explicitly supported by evidence."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        concept_aliases = {
            "startup": ("startup", "startups", "incubated", "venture", "company", "companies"),
            "patent": ("patent", "patents", "invented", "invention"),
            "project": ("project", "projects", "funded project", "funding", "grant", "sponsored project"),
            "lab": ("lab", "labs", "laboratory", "laboratories", "research lab", "teaching lab", "facilities"),
            "placement": ("placement", "placements", "salary", "package", "higher studies", "higher study"),
            "contact": ("contact", "point of contact", "official contact"),
            "publication": ("publication", "publications", "paper", "papers", "journal", "conference"),
            "award": ("award", "awards", "honor", "honors", "honour", "honours", "recogni"),
            "course": ("course", "courses", "programme", "programmes", "program", "programs", "curriculum", "btech", "mtech"),
            "staff": ("staff", "laboratory assistant", "lab assistant", "technician", "officer"),
            "alumni": ("alumni", "alumni list", "former students", "passed out students"),
            "graduated_phd": ("phd alumni", "graduated phd", "phd graduates", "completed phd"),
            "address": ("address", "location", "postal address", "department address", "department email", "department phone"),
        }
        matched = []
        for concept, aliases in concept_aliases.items():
            if any(alias in q for alias in aliases):
                matched.append(concept)
        return matched

    def _concept_supported_by_graph(self, concept: str) -> bool:
        """Check whether the department graph contains the concept structurally."""
        if concept == "contact":
            return self._get_hod_member() is not None or self._label_counts.get("ContactInfo", 0) > 0

        if concept == "publication":
            return self._label_counts.get("Publication", 0) > 0 or any(
                self.graph.nodes[n].get("publications")
                for n in self.graph.nodes
                if self.graph.nodes[n].get("label") == "Faculty"
            )

        if concept == "award":
            return self._label_counts.get("Award", 0) > 0 or any(
                self.graph.nodes[n].get("awards")
                for n in self.graph.nodes
                if self.graph.nodes[n].get("label") == "Faculty"
            )

        if concept == "course":
            return self._label_counts.get("Course", 0) > 0 or self._label_counts.get("Program", 0) > 0

        if concept == "staff":
            return self._label_counts.get("Staff", 0) > 0

        label_map = {
            "startup": ("Startup",),
            "patent": ("Patent",),
            "project": ("Project",),
            "lab": ("Lab",),
            "placement": ("PlacementData", "HigherStudiesData"),
            "alumni": ("Alumni", "AlumniBatch"),
            "graduated_phd": ("GraduatedPhD",),
            "address": ("ContactInfo",),
        }
        labels = label_map.get(concept, ())
        return any(self._label_counts.get(label, 0) > 0 for label in labels)

    def _build_unavailable_response(self, query: str, reason: Optional[str] = None) -> str:
        """Return a safe, department-scoped unavailable-information response."""
        if self._is_department_contact_query(query):
            contact_answer = self._build_department_contact_answer()
            if contact_answer:
                return contact_answer

        base = (
            f"I don't have that specific information for the {self.dept_config['full_name']} "
            f"at IIT Jammu."
        )
        if reason:
            base += f" {reason}"
        base += (
            f" You can check the IIT Jammu {self.dept_config['name']} website at "
            f"{self.dept_config['base_url']} for more details."
        )
        base += " If you're looking for information from a specific department, try mentioning the department name in your query."
        return base

    def _build_provenance(
        self,
        direct: bool,
        local_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        global_results: List[Dict[str, Any]],
        section_word_counts: Dict[str, int],
    ) -> Dict[str, Any]:
        """Summarize which retrieval channels contributed evidence."""
        graph_used = bool(direct or local_results)
        vector_used = bool(vector_results)
        community_used = bool(global_results)

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
        elif community_used:
            source_mode = "community"
            route = "community"
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
                "items": len(global_results),
                "avg_score": avg_score(global_results),
                "word_count": section_word_counts.get("community", 0),
            },
        }

    def _assess_answerability(
        self,
        query: str,
        local_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        global_results: List[Dict[str, Any]],
        context: str,
    ) -> Dict[str, Any]:
        """Decide whether the retrieved evidence can safely support an answer."""
        concepts = self._infer_query_concepts(query)
        if self._is_broad_reasoning_query(query):
            return {
                "answerable": True,
                "reason": "",
                "matched_terms": [],
                "missing_concepts": [],
            }

        # Concept aliases map to check if concept is mentioned in retrieved context
        concept_aliases = {
            "startup": ("startup", "startups", "incubated", "incubation", "spin-off", "spinoff"),
            "patent": ("patent", "patents", "invented", "invention"),
            "project": ("project", "projects", "funded project", "funding", "grant"),
            "lab": ("lab", "labs", "laboratory", "laboratories"),
            "placement": ("placement", "placements", "salary", "package", "higher studies", "higher study"),
            "contact": ("contact", "point of contact", "official contact", "hod", "head"),
            "publication": ("publication", "publications", "paper", "papers", "journal", "conference"),
            "award": ("award", "awards", "honor", "honors", "honour", "honours", "recogni"),
            "course": ("course", "courses", "programme", "programmes", "program", "programs", "curriculum", "btech", "mtech"),
            "staff": ("staff", "laboratory assistant", "lab assistant", "technician", "officer"),
        }

        missing_concepts = []
        for concept in concepts:
            supported = self._concept_supported_by_graph(concept)
            if not supported:
                # Also check if present in retrieved context (soft check)
                aliases = concept_aliases.get(concept, (concept,))
                context_lower = context.lower()
                if any(alias in context_lower for alias in aliases):
                    supported = True
            if not supported:
                missing_concepts.append(concept)

        if missing_concepts:
            concept_text = ", ".join(sorted(missing_concepts))
            reason = (
                f"The available department data does not contain grounded {concept_text} information "
                "for this query."
            )
            return {
                "answerable": False,
                "reason": reason,
                "matched_terms": [],
                "missing_concepts": missing_concepts,
            }

        factoid_starters = (
            "who", "what", "which", "where", "when", "name", "list", "give me", "show me",
            "tell me", "startups", "startup", "contact", "email",
        )
        q_normalized = re.sub(r"\s+", " ", query.lower()).strip()
        is_factoid = q_normalized.startswith(factoid_starters)
        if not is_factoid:
            return {
                "answerable": True,
                "reason": "",
                "matched_terms": [],
                "missing_concepts": [],
            }

        stop_tokens = {
            "iit", "jammu", "department", "dept", "faculty", "member", "members", "student", "students",
            "phd", "research", "official", "main", "information", "computer", "science", "engineering",
            "electrical", "mechanical", "civil", "chemical", "bioscience", "bioengineering", "physics",
            "chemistry", "mathematics", "humanities", "social", "sciences",
        }
        focus_terms = [
            token
            for token in self._query_tokens(query)
            if len(token) >= 4 and token not in stop_tokens
        ]
        # Build set of normalized words from the context to avoid substring collisions
        context_words = set()
        for word in re.findall(r"[A-Za-z0-9]+", context.lower()):
            norm_word = self._normalize_token(word)
            if norm_word:
                context_words.add(norm_word)
        matched_terms = [term for term in focus_terms if term in context_words]

        # Redesigned answerability gate: if we have retrieved vector or local results,
        # we trust the retriever and consider it answerable (even if matched_terms is empty,
        # since vector search handles synonym / semantic matching).
        # We only block if there is absolutely NO retrieved content.
        if not (local_results or vector_results or global_results):
            return {
                "answerable": False,
                "reason": "The retriever did not find relevant evidence for this query.",
                "matched_terms": [],
                "missing_concepts": [],
            }

        return {
            "answerable": True,
            "reason": "",
            "matched_terms": matched_terms[:10],
            "missing_concepts": [],
        }

    def _is_faculty_roster_query(self, query: str) -> bool:
        """Detect department-level faculty count/list requests."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        has_faculty_term = any(term in q for term in (
            "faculty", "faculties", "professor", "professors"
        ))
        if not has_faculty_term:
            return False

        count_intent = any(term in q for term in (
            "how many", "count", "total", "number of", "strength"
        ))
        roster_intent = any(term in q for term in (
            "faculty list", "list of faculty", "list all faculty",
            "all faculty", "all faculties", "names of faculty",
            "list of faculties", "faculties list", "faculty names",
            "full list",
            "faculty members", "faculties in the department",
            "faculty in the department", "count and list",
            "professor list", "list of professors", "list all professors",
            "all professors", "names of professors", "professors list",
            "professor names", "professors in the department", "professor in the department"
        ))

        if not (count_intent or roster_intent):
            return False

        # Avoid hijacking filtered questions such as "Which faculty work on VLSI?"
        filtered_intent = any(term in q for term in (
            "work on", "working on", "research area", "research interest",
            "supervis", "patent", "project", "startup", "publication"
        ))
        if filtered_intent and not any(term in q for term in (
            "all faculty", "all faculties", "count and list", "faculty list",
            "list of faculty", "all professors", "list all professors", "professor list",
            "list of professors"
        )):
            return False

        return True

    def _get_faculty_structured_fields(self) -> List[str]:
        """Return non-empty structured fields available on faculty nodes."""
        dept_data = self.graph.nodes.get(self.dept_node_id, {})
        fields = dept_data.get("faculty_structured_fields")
        if isinstance(fields, list) and fields:
            return fields

        structured_fields = set()
        for _, data in self.graph.nodes(data=True):
            if data.get("label") != "Faculty":
                continue
            for key, value in data.items():
                if key in {"label", "department", "name"}:
                    continue
                if value in (None, "", [], {}):
                    continue
                structured_fields.add(key)
        return sorted(structured_fields)

    def _extract_requested_faculty_attribute(self, query: str) -> Optional[str]:
        """Map analytic faculty questions to a structured attribute."""
        q = re.sub(r"\s+", " ", query.lower()).strip()

        def has_alias(alias: str) -> bool:
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            return re.search(pattern, q) is not None

        attribute_aliases = {
            "gender": (
                "gender", "male", "female", "men", "women", "man", "woman",
                "boys", "girls", "ladies", "gents"
            ),
            "designation": (
                "designation", "designations", "rank", "ranks",
                "assistant professor", "associate professor", "professor", "director"
            ),
            "is_hod": (
                "hod", "head of department", "department head"
            ),
        }

        for attribute, aliases in attribute_aliases.items():
            if any(has_alias(alias) for alias in aliases):
                return attribute
        return None

    def _is_faculty_analytics_query(self, query: str) -> bool:
        """Detect faculty analytics/breakdown queries that should be answered structurally."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        has_faculty_anchor = any(term in q for term in (
            "faculty", "faculties", "professor", "professors", self.dept_code.replace("_", " ")
        ))
        if not has_faculty_anchor:
            return False

        attribute = self._extract_requested_faculty_attribute(q)
        if not attribute:
            return False

        analytic_terms = (
            "ratio", "breakdown", "distribution", "split", "proportion",
            "calculate", "compare", "comparison", "versus", "vs", "group by",
            "grouped by", "how many", "count", "number of", "total"
        )
        return any(term in q for term in analytic_terms) or attribute == "gender"

    def _normalize_designation(self, designation: str) -> str:
        """Collapse free-form designations into stable analytic buckets."""
        text = (designation or "").strip()
        lowered = text.lower()
        if "associate professor" in lowered:
            return "Associate Professor"
        if "assistant professor" in lowered:
            return "Assistant Professor"
        if "professor" in lowered and "director" in lowered:
            return "Professor (Director)"
        if "professor" in lowered:
            return "Professor"
        return text or "Unknown"

    def _compute_faculty_breakdown(self, attribute: str) -> Dict[str, int]:
        """Compute deterministic faculty breakdowns for supported structured attributes."""
        roster = self.get_faculty_roster()
        counts = Counter()

        if attribute == "designation":
            for member in roster:
                counts[self._normalize_designation(member.get("designation", ""))] += 1
            return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

        if attribute == "is_hod":
            for member in roster:
                key = "Head of Department" if member.get("is_hod") else "Non-HoD Faculty"
                counts[key] += 1
            return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

        return {}

    def _faculty_listing_url(self) -> str:
        """Return the canonical faculty listing page for the department."""
        template = self.dept_config.get("template", "A")
        suffix = "faculty-list" if template == "B" else "faculty.html"
        return f"{self.dept_config['base_url']}/{suffix}"

    def _build_faculty_analytics_answer(self, query: str) -> Optional[str]:
        """Answer faculty analytics queries without falling through to fuzzy retrieval/LLM counting."""
        if not self._is_faculty_analytics_query(query):
            return None

        attribute = self._extract_requested_faculty_attribute(query)
        roster = self.get_faculty_roster()
        total = len(roster)
        available_fields = self._get_faculty_structured_fields()

        if attribute == "gender":
            fields_text = ", ".join(available_fields) if available_fields else "name"
            return (
                "I can't calculate a male-to-female ratio for the faculty from the IIT Jammu source data "
                f"because gender is not stored in the ingested faculty records for {self.dept_config['full_name']}. "
                f"The authoritative faculty roster currently has **{total} members**, but the structured faculty fields "
                f"available are: {fields_text}. To avoid guessing a sensitive attribute from names, I won't infer gender."
            )

        breakdown = self._compute_faculty_breakdown(attribute or "")
        if not breakdown:
            return None

        lines = [
            f"Structured faculty breakdown for the {self.dept_config['full_name']} at IIT Jammu:",
            "",
        ]
        for label, count in breakdown.items():
            lines.append(f"- {label}: **{count}**")
        lines.extend([
            "",
            f"Total faculty members: **{total}**",
            "",
            f"Source: [IIT Jammu {self.dept_config['name']} Faculty]({self._faculty_listing_url()})",
        ])
        return "\n".join(lines)

    def _query_has_count_intent(self, query: str) -> bool:
        q = re.sub(r"\s+", " ", query.lower()).strip()
        return any(term in q for term in (
            "how many", "count", "total", "number of", "strength"
        ))

    def _query_has_list_intent(self, query: str) -> bool:
        q = re.sub(r"\s+", " ", query.lower()).strip()
        return any(term in q for term in (
            "list", "all", "names", "roster", "full list", "show me"
        ))

    def _is_exact_count_query(self, query: str) -> bool:
        """Detect queries where approximate community summaries should not drive the answer."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        return self._query_has_count_intent(q) and not any(term in q for term in (
            "approx", "approximately", "around", "estimate", "roughly"
        ))

    def _is_phd_roster_query(self, query: str) -> bool:
        """Detect department-level PhD scholar/student count/list requests."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        has_phd_term = any(term in q for term in (
            "phd", "ph.d", "doctoral"
        ))
        has_student_term = any(term in q for term in (
            "student", "students", "scholar", "scholars", "research scholar", "research scholars"
        ))
        if not (has_phd_term or has_student_term):
            return False

        if not (self._query_has_count_intent(q) or self._query_has_list_intent(q)):
            return False

        # Avoid hijacking faculty-specific supervision questions.
        if any(term in q for term in (
            "supervis", "advisor", "advises", "under ", "working with", "works with",
            "guided by", "co-supervis", "co supervis"
        )):
            return False

        # Avoid hijacking summarization/analysis queries that mention PhD/students
        if any(term in q for term in (
            "summarize", "summarise", "primary domain", "analyze", "analyse",
            "based on", "overview", "insight", "reflect", "trend",
            "infer", "suggest", "what do"
        )):
            return False

        return True

    def _is_mtech_roster_query(self, query: str) -> bool:
        """Detect department-level M.Tech student count/list requests."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        has_mtech_term = any(term in q for term in (
            "mtech", "m.tech", "master", "masters"
        ))
        if any(term in q for term in ("phd", "ph.d", "doctoral")):
            return False
            
        has_student_term = any(term in q for term in (
            "student", "students", "scholar", "scholars"
        ))
        if not (has_mtech_term or (has_student_term and "mtech" in q)):
            return False

        if not (self._query_has_count_intent(q) or self._query_has_list_intent(q)):
            return False

        # Avoid hijacking faculty-specific supervision questions.
        if any(term in q for term in (
            "supervis", "advisor", "advises", "under ", "working with", "works with",
            "guided by", "co-supervis", "co supervis"
        )):
            return False

        # Avoid hijacking summarization/analysis queries that mention MTech/students
        if any(term in q for term in (
            "summarize", "summarise", "primary domain", "analyze", "analyse",
            "based on", "overview", "insight", "reflect", "trend",
            "infer", "suggest", "what do"
        )):
            return False

        return True

    def _extract_supervisor_query_name(self, query: str) -> Optional[str]:
        """Extract the student name from supervisor/advisor style questions."""
        q = re.sub(r"\s+", " ", query.strip())
        patterns = (
            r"^who supervises (?P<name>.+?)\??$",
            r"^who is (?P<name>.+?) supervised by\??$",
            r"^who is the supervisor of (?P<name>.+?)\??$",
            r"^who is (?P<name>.+?)'?s supervisor\??$",
            r"^who is (?P<name>.+?)'?s advisor\??$",
            r"^who advises (?P<name>.+?)\??$",
            r"^who is the advisor of (?P<name>.+?)\??$",
        )
        for pattern in patterns:
            match = re.match(pattern, q, flags=re.IGNORECASE)
            if match:
                return match.group("name").strip(" ?.")
        return None

    def _extract_research_area_query_name(self, query: str) -> Optional[str]:
        """Extract the student name from research area style questions."""
        q = re.sub(r"\s+", " ", query.strip())
        patterns = (
            r"^what is the research area of phd student (?P<name>.+?)\??$",
            r"^what is the research area of (?P<name>.+?)\??$",
            r"^what research area is (?P<name>.+?) working on\??$",
            r"^what is (?P<name>.+?)'?s research area\??$",
        )
        for pattern in patterns:
            match = re.match(pattern, q, flags=re.IGNORECASE)
            if match:
                return match.group("name").strip(" ?.")
        return None

    def _extract_email_query_name(self, query: str) -> Optional[str]:
        """Extract the entity name from email/contact style questions."""
        q = re.sub(r"\s+", " ", query.strip())
        patterns = (
            r"^what is the email of (?P<name>.+?)\??$",
            r"^what is (?P<name>.+?)'?s email\??$",
            r"^email of (?P<name>.+?)\??$",
            r"^contact of (?P<name>.+?)\??$",
            r"^contact details of (?P<name>.+?)\??$",
            r"^how can i contact (?P<name>.+?)\??$",
        )
        for pattern in patterns:
            match = re.match(pattern, q, flags=re.IGNORECASE)
            if match:
                return match.group("name").strip(" ?.")
        return None

    def _extract_supervisor_from_students_query(self, query: str) -> Optional[str]:
        """Extract the supervisor's name from query seeking their students."""
        q = re.sub(r"\s+", " ", query.strip())
        patterns = (
            r"^(?:who are the |list of )?ph\.?d\.? (?:students|studetns|scholars) (?:under|working under|supervised by|guided by) (?P<name>.+?)\??$",
            r"^ph\.?d\.? (?:students|studetns|scholars) of (?P<name>.+?)\??$",
            r"^(?:who are )?(?P<name>.+?)'?s ph\.?d\.? (?:students|studetns|scholars)\??$",
        )
        for pattern in patterns:
            match = re.match(pattern, q, flags=re.IGNORECASE)
            if match:
                return match.group("name").strip(" ?.")
        return None

    def _find_entity_by_name(self, raw_name: str, allowed_labels: Optional[Tuple[str, ...]] = None) -> Optional[str]:
        """Resolve an entity name to a graph node id using exact then fuzzy matching."""
        if not raw_name:
            return None

        normalized = re.sub(r"\s+", " ", raw_name.lower()).strip(" ?.")
        direct = self._entity_name_index.get(normalized)
        if direct:
            if not allowed_labels or self.graph.nodes[direct].get("label") in allowed_labels:
                return direct

        for node_id in self._name_match(normalized):
            if not allowed_labels or self.graph.nodes[node_id].get("label") in allowed_labels:
                return node_id
        return None

    def get_faculty_roster(self) -> List[Dict]:
        """Return the authoritative department faculty roster from graph nodes."""
        dept_id = self.dept_node_id
        roster = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("label") != "Faculty":
                continue
            if not (
                self.graph.has_edge(node_id, dept_id)
                or data.get("profile_url")
                or str(data.get("source_file", "")).startswith(f"{self.dept_code}_faculty")
            ):
                continue
            order = data.get("faculty_order")
            roster.append({
                "name": data.get("name", node_id),
                "designation": data.get("designation", ""),
                "email": data.get("email", ""),
                "profile_url": data.get("profile_url", ""),
                "is_hod": bool(data.get("is_hod")),
                "order": order if isinstance(order, int) else 9999,
            })

        # Stable sort: canonical hod first, then alphabetically
        return sorted(roster, key=lambda x: (not x["is_hod"], x["order"], x["name"].lower()))

    def get_phd_roster(self) -> List[Dict]:
        """Return the authoritative PhD scholar roster from graph nodes and supervision edges."""
        roster = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("label") != "PhDStudent":
                continue
            if "phd-list" not in str(data.get("source_file", "")).lower():
                continue

            supervisors = []
            for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                if edge_data.get("type") != "SUPERVISED_BY":
                    continue
                target_data = self.graph.nodes.get(target, {})
                supervisors.append(target_data.get("name", target))

            roster.append({
                "name": data.get("name", node_id),
                "supervisors": list(dict.fromkeys(supervisors)),
                "research_area": data.get("research_area", ""),
                "email": data.get("email", ""),
            })

        return sorted(roster, key=lambda item: item["name"].lower())

    def _faculty_roster_context(self) -> str:
        """Build a complete roster context block for the LLM."""
        roster = self.get_faculty_roster()
        dept_data = self.graph.nodes.get(self.dept_node_id, {})
        count = dept_data.get("faculty_count") or len(roster)

        lines = [
            "## Authoritative Faculty Roster",
            (
                "Use this complete roster for department-level faculty count "
                f"and list questions. The {self.dept_config['full_name']} "
                f"at IIT Jammu has {count} faculty members."
            ),
            "",
        ]
        for idx, member in enumerate(roster, start=1):
            role = " (Head of Department)" if member["is_hod"] else ""
            details = [f"{idx}. {member['name']}{role}"]
            if member["designation"]:
                details.append(member["designation"])
            if member["email"]:
                details.append(f"Email: {member['email']}")
            if member["profile_url"]:
                details.append(f"Profile: {member['profile_url']}")
            lines.append(" - ".join(details))
        return "\n".join(lines)

    def _phd_roster_context(self) -> str:
        """Build a complete PhD roster context block for deterministic answering."""
        roster = self.get_phd_roster()
        dept_data = self.graph.nodes.get(self.dept_node_id, {})
        count = dept_data.get("phd_student_count") or len(roster)

        supervisor_counts = defaultdict(int)
        for scholar in roster:
            for supervisor in scholar["supervisors"]:
                supervisor_counts[supervisor] += 1

        lines = [
            "## Authoritative PhD Scholar Roster",
            (
                "Use this complete roster for department-level PhD scholar count "
                f"and list questions. The {self.dept_config['full_name']} at IIT Jammu "
                f"has {count} PhD scholars listed on the official PhD page."
            ),
            "",
            "### Supervisor Breakdown",
        ]

        for supervisor, total in sorted(supervisor_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {supervisor}: {total} scholar(s)")

        lines.extend(["", "### Full Scholar List"])
        for idx, scholar in enumerate(roster, start=1):
            details = [f"{idx}. {scholar['name']}"]
            if scholar["supervisors"]:
                details.append(f"Supervisor(s): {', '.join(scholar['supervisors'])}")
            if scholar["research_area"]:
                details.append(f"Research Area: {scholar['research_area']}")
            if scholar["email"]:
                details.append(f"Email: {scholar['email']}")
            lines.append(" - ".join(details))

        return "\n".join(lines)

    def get_mtech_roster(self) -> List[Dict]:
        """Return the authoritative M.Tech student roster from graph nodes and supervision edges."""
        roster = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("label") != "MTechStudent":
                continue
            if "mtech-list" not in str(data.get("source_file", "")).lower():
                continue

            supervisors = []
            for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                if edge_data.get("type") != "SUPERVISED_BY":
                    continue
                target_data = self.graph.nodes.get(target, {})
                supervisors.append(target_data.get("name", target))

            roster.append({
                "name": data.get("name", node_id),
                "supervisors": list(dict.fromkeys(supervisors)),
                "research_area": data.get("research_area", ""),
                "email": data.get("email", ""),
            })

        return sorted(roster, key=lambda item: item["name"].lower())

    def _mtech_roster_context(self) -> str:
        """Build a complete M.Tech roster context block for deterministic answering."""
        roster = self.get_mtech_roster()
        dept_data = self.graph.nodes.get(self.dept_node_id, {})
        count = len(roster)

        supervisor_counts = defaultdict(int)
        for scholar in roster:
            for supervisor in scholar["supervisors"]:
                supervisor_counts[supervisor] += 1

        lines = [
            "## Authoritative M.Tech Student Roster",
            (
                "Use this complete roster for department-level M.Tech student count "
                f"and list questions. The {self.dept_config['full_name']} at IIT Jammu "
                f"has {count} M.Tech students listed on the official M.Tech roster page."
            ),
            "",
            "### Supervisor Breakdown",
        ]

        for supervisor, total in sorted(supervisor_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {supervisor}: {total} student(s)")

        lines.extend(["", "### Full M.Tech Student List"])
        for idx, scholar in enumerate(roster, start=1):
            details = [f"{idx}. {scholar['name']}"]
            if scholar["supervisors"]:
                details.append(f"Supervisor(s): {', '.join(scholar['supervisors'])}")
            if scholar["research_area"]:
                details.append(f"Research Area: {scholar['research_area']}")
            if scholar["email"]:
                details.append(f"Email: {scholar['email']}")
            lines.append(" - ".join(details))

        return "\n".join(lines)

    def get_deterministic_context(self, query: str, suppress_topic_match: bool = False) -> Optional[str]:
        """Return deterministic context from graph data for queries with authoritative answers.

        The returned text is intended to be used as high-priority context for the LLM,
        NOT as a final response returned directly to the user.

        Args:
            query: The user's question.
            suppress_topic_match: If True, skip topic/expert contact matching.
                Used during broadcast to prevent a single department from
                short-circuiting cross-department topic queries.
        """
        ql = query.lower()
        q_cleaned = re.sub(r"\s+", " ", query.strip().lower()).strip(" ?.")

        if self.dept_code == "administration":
            admin_answer = self._build_administration_answer(query)
            if admin_answer:
                return admin_answer

        if self._is_department_contact_query(query) or any(kw in q_cleaned for kw in ("who is the hod", "who is the head of department", "who is the head of the department", "who is the department head", "head of department name", "name of the hod", "name of hod", "name of head of department")):
            contact_answer = self._build_department_contact_answer()
            if contact_answer:
                return contact_answer

        # Lab queries → deterministic lab listing
        if self._is_lab_query(query):
            labs_answer = self._build_labs_answer(query)
            if labs_answer:
                return labs_answer

        # Department address / contact info queries
        if self._is_address_query(query):
            address_answer = self._build_address_answer()
            if address_answer:
                return address_answer

        # Graduated PhD / PhD alumni queries
        if self._is_graduated_phd_query(query):
            grad_answer = self._build_graduated_phd_answer()
            if grad_answer:
                return grad_answer

        # Alumni queries
        if self._is_alumni_query(query):
            alumni_answer = self._build_alumni_answer()
            if alumni_answer:
                return alumni_answer

        # Department Research Areas Listing
        if self._is_department_research_areas_query(query):
            research_areas_answer = self._build_department_research_areas_answer()
            if research_areas_answer:
                return research_areas_answer

        # Check for topic contact / expert queries
        # When suppress_topic_match is True (broadcast mode), skip this entire
        # block so the query goes through full retrieval across ALL departments.
        is_topic_query = False
        topic = None

        if not suppress_topic_match:
            contact_patterns = [
                # "who should I contact for X", "who to contact about X"
                r"who\s+(?:should\s+i\s+|to\s+)?contact\s+(?:for|about|regarding)\s+(.+)",
                # "who works on X", "which faculty working on X", "who researches X"
                r"(?:who|which)\s+(?:faculty|member|professor|people|person|individual|scholars?|students?|is\s+)?(?:working\s+on|works\s+on|researching|expert\s+in|specialist\s+in|does\s+research\s+in|do\s+research\s+in)\s+(.+)",
                # "who to reach out to for X"
                r"who\s+(?:to\s+)?(?:reach\s+out\s+to|write\s+to)\s+(?:for|about|regarding)\s+(.+)",
                # "find/list faculty working on X"
                r"(?:find|search|list|get|show)\s+(?:faculty|member|professor|people|person|individual|scholars?|students?)\s+(?:working\s+on|who\s+work\s+on|researching|expert\s+in|specialist\s+in|in\s+the\s+area\s+of|in\s+the\s+field\s+of)\s+(.+)",
                # "experts in X", "specialists in X"
                r"(?:experts?|specialists?)\s+(?:in|on|for)\s+(.+)",
                # "faculty for X", "professors for X"
                r"(?:faculty|professors?|researchers?)\s+(?:for|in|on)\s+(.+)\s+(?:at\s+iit\s+jammu|iit\s+jammu|department)",
                # "X researchers at IIT Jammu", "X experts at IIT Jammu"
                r"(.+?)\s+(?:researchers?|experts?|faculty|professors?)\s+(?:at\s+iit|in\s+iit|at\s+the\s+iit)",
                # "X research at IIT Jammu" (e.g. "computer vision research at IIT Jammu")
                r"(.+?)\s+research\s+(?:at|in)\s+(?:iit|the\s+department)",
                # "who is doing X", "who is working in X"
                r"who\s+(?:is\s+)?(?:doing|working\s+in|involved\s+in)\s+(.+)",
                # "faculty in the area of X", "faculty in the field of X"
                r"faculty\s+(?:in\s+the\s+area\s+of|in\s+the\s+field\s+of|specializing\s+in|with\s+expertise\s+in)\s+(.+)",
                # "which professor works in X" / "any professor working in X"
                r"(?:any|is\s+there\s+any)\s+(?:faculty|professor|researcher)\s+(?:working\s+on|working\s+in|who\s+works\s+on|who\s+works\s+in)\s+(.+)",
            ]

            for pat in contact_patterns:
                m = re.search(pat, q_cleaned, re.IGNORECASE)
                if m:
                    is_topic_query = True
                    topic = m.group(1).strip()
                    topic = re.sub(r"\s*(?:related\s+tasks|tasks|work|research|lab|projects|area|field|topics|subject|course|class|\?|\.)+$", "", topic, flags=re.IGNORECASE).strip()
                    # Strip trailing department/context qualifiers that were captured
                    topic = re.sub(r"\s+(?:at\s+iit\s+jammu|at\s+iit|in\s+iit|iit\s+jammu|in\s+the\s+department|at\s+the\s+department)$", "", topic, flags=re.IGNORECASE).strip()
                    if not topic:
                        is_topic_query = False
                    break

        if is_topic_query and topic:
            matching_faculty = []      # (node_data, matched_area_description)
            matching_phd_scholars = [] # (node_data, matched_area_description, supervisors)
            topic_lower = topic.lower()

            # Step A: Check for structural ResearchArea nodes (strongest signal)
            # Faculty with RESEARCHES_IN edges to matching areas
            for u, v, edge_data in self.graph.edges(data=True):
                if edge_data.get("type") not in ("RESEARCHES_IN", "STUDIES", "RELATED"):
                    continue
                u_data = self.graph.nodes.get(u, {})
                v_data = self.graph.nodes.get(v, {})
                area_name = v_data.get("name", "")
                if not _topic_matches_text(topic_lower, area_name):
                    continue

                if u_data.get("label") == "Faculty" and v_data.get("label") in ("ResearchArea", "ResearchCategory"):
                    matching_faculty.append((u_data, area_name.lower()))
                elif u_data.get("label") in ("PhDStudent", "MTechStudent") and v_data.get("label") in ("ResearchArea",):
                    # Collect PhD scholar matches too
                    supervisors = []
                    for _, sup_target, sup_edge in self.graph.out_edges(u, data=True):
                        if sup_edge.get("type") == "SUPERVISED_BY":
                            sup_data = self.graph.nodes.get(sup_target, {})
                            supervisors.append(sup_data.get("name", sup_target))
                    matching_phd_scholars.append((u_data, area_name, supervisors))

            # Step B: Check authoritative text fields on Faculty
            # ONLY search research_interests and academic_interests — NOT publications,
            # education, or research_experience which contain too much incidental text
            # and cause false positives (e.g. 'ai' matching 'uncertainty' in publications).
            already_matched_names = {f.get("name") for f, _ in matching_faculty}
            for nid, d in self.graph.nodes(data=True):
                if d.get("label") != "Faculty":
                    continue
                if d.get("name") in already_matched_names:
                    continue
                matched_field = None
                for field in ("research_interests", "academic_interests"):
                    field_val = d.get(field, "")
                    if isinstance(field_val, str) and field_val and _topic_matches_text(topic_lower, field_val):
                        matched_field = field
                        break
                if matched_field:
                    matching_faculty.append((d, f"their listed {matched_field.replace('_', ' ')}"))

            # Step B2: Also check PhD scholars' research_area text field
            already_matched_scholars = {s.get("name") for s, _, _ in matching_phd_scholars}
            for nid, d in self.graph.nodes(data=True):
                if d.get("label") not in ("PhDStudent", "MTechStudent"):
                    continue
                if d.get("name") in already_matched_scholars:
                    continue
                area = d.get("research_area", "")
                if area and _topic_matches_text(topic_lower, area):
                    supervisors = []
                    for _, sup_target, sup_edge in self.graph.out_edges(nid, data=True):
                        if sup_edge.get("type") == "SUPERVISED_BY":
                            sup_data = self.graph.nodes.get(sup_target, {})
                            supervisors.append(sup_data.get("name", sup_target))
                    matching_phd_scholars.append((d, area, supervisors))

            if matching_faculty or matching_phd_scholars:
                # Deduplicate faculty
                seen_names = set()
                unique_faculty = []
                for fac, source in matching_faculty:
                    name = fac.get("name")
                    if name and name not in seen_names:
                        seen_names.add(name)
                        unique_faculty.append((fac, source))

                # Deduplicate PhD scholars
                seen_scholar_names = set()
                unique_scholars = []
                for scholar, area, sups in matching_phd_scholars:
                    name = scholar.get("name")
                    if name and name not in seen_scholar_names:
                        seen_scholar_names.add(name)
                        unique_scholars.append((scholar, area, sups))

                lines = []

                # Faculty section (listed first)
                if unique_faculty:
                    lines.append(f"**Faculty members** working in areas related to **{topic.title()}**:")
                    lines.append("")
                    for fac, source in unique_faculty:
                        name = fac.get("name")
                        email = fac.get("email", "Not specified")
                        desig = fac.get("designation", "Faculty Member")
                        name_display = name if name.startswith(("Dr.", "Prof.")) else f"Dr. {name}"
                        line = f"- **{name_display}** ({desig})"
                        if email and email != "Not specified":
                            line += f" - Email: {email}"
                        lines.append(line)

                # PhD scholars section (listed after faculty)
                if unique_scholars:
                    lines.append("")
                    lines.append(f"**PhD/M.Tech scholars** working in areas related to **{topic.title()}**:")
                    lines.append("")
                    for scholar, area, sups in unique_scholars:
                        name = scholar.get("name")
                        label = scholar.get("label", "PhDStudent")
                        prog = "PhD Scholar" if label == "PhDStudent" else "M.Tech Student"
                        sup_str = ", ".join(sups) if sups else "Unknown"
                        line = f"- **{name}** ({prog}) — Research: {area} — Supervisor(s): {sup_str}"
                        lines.append(line)

                lines.append("")
                lines.append(f"You can reach out to the faculty for tasks or queries related to {topic}.")
                return "\n".join(lines)

        faculty_analytics_answer = self._build_faculty_analytics_answer(query)
        if faculty_analytics_answer:
            return faculty_analytics_answer

        entity_email_name = self._extract_email_query_name(query)
        if entity_email_name:
            entity_id = self._find_entity_by_name(
                entity_email_name,
                allowed_labels=("Faculty", "PhDStudent", "MTechStudent", "ExternalPerson"),
            )
            if entity_id and self.graph.has_node(entity_id):
                entity = self.graph.nodes[entity_id]
                email = entity.get("email", "")
                entity_name = entity.get("name", entity_email_name)
                if email:
                    return f"{entity_name}'s official email is {email}."
                return f"I couldn't find an official email address for {entity_name} in the department records."

        # 1. PhD/M.Tech Student supervisor queries (dynamic, fallback)
        student_name = self._extract_supervisor_query_name(query)
        if student_name:
            student_id = self._find_entity_by_name(student_name, allowed_labels=("PhDStudent", "MTechStudent"))
            if student_id and self.graph.has_node(student_id):
                student = self.graph.nodes[student_id]
                supervisor_names = []
                for _, target, edge_data in self.graph.out_edges(student_id, data=True):
                    if edge_data.get("type") != "SUPERVISED_BY":
                        continue
                    target_data = self.graph.nodes.get(target, {})
                    supervisor_names.append(target_data.get("name", target))

                if supervisor_names:
                    unique_supervisors = list(dict.fromkeys(supervisor_names))
                    s_name = student.get('name', student_id)
                    s_name_title = ' '.join(w.capitalize() for w in s_name.split())
                    
                    formatted_sups = []
                    for name in unique_supervisors:
                        name_display = name
                        if name == "Anup Shukla":
                            name_display = "Anup kumar Shukla"
                        if not name_display.startswith("Dr. "):
                            formatted_sups.append(f"Dr. {name_display}")
                        else:
                            formatted_sups.append(name_display)
                            
                    if len(formatted_sups) == 1:
                        sups_str = formatted_sups[0]
                    elif len(formatted_sups) == 2:
                        sups_str = f"{formatted_sups[0]} and {formatted_sups[1]}"
                    else:
                        sups_str = ", ".join(formatted_sups[:-1]) + f", and {formatted_sups[-1]}"
                        
                    research_area = student.get("research_area", "")
                    label = student.get("label", "PhDStudent")
                    prog = "PhD scholar" if label == "PhDStudent" else "M.Tech student"
                    ans = f"{s_name_title} is a {prog} supervised by {sups_str}."
                    if research_area:
                        ans += f" Research area: {research_area}."
                    return ans

        # 2.5 PhD/M.Tech students under a specific supervisor query (dynamic, fallback)
        supervisor_query_name = self._extract_supervisor_from_students_query(query)
        if supervisor_query_name:
            sup_id = self._find_entity_by_name(supervisor_query_name, allowed_labels=("Faculty", "ExternalPerson"))
            if sup_id and self.graph.has_node(sup_id):
                sup_node = self.graph.nodes[sup_id]
                sup_display_name = sup_node.get("name", sup_id)
                if not sup_display_name.startswith("Dr. "):
                    sup_display_name = f"Dr. {sup_display_name}"
                    
                students = []
                for source, _, edge_data in self.graph.in_edges(sup_id, data=True):
                    if edge_data.get("type") != "SUPERVISED_BY":
                        continue
                    s_data = self.graph.nodes.get(source, {})
                    if s_data.get("label") in ("PhDStudent", "MTechStudent"):
                        students.append({
                            "name": s_data.get("name", source),
                            "research_area": s_data.get("research_area", ""),
                            "label": s_data.get("label", "PhDStudent")
                        })
                
                if students:
                    students = sorted(students, key=lambda x: x["name"].lower())
                    lines = [
                        f"The following students are working under the guidance of **{sup_display_name}**:",
                        ""
                    ]
                    for idx, s in enumerate(students, start=1):
                        prog = "PhD Scholar" if s["label"] == "PhDStudent" else "M.Tech Student"
                        details = [f"{idx}. **{s['name']}** ({prog})"]
                        if s["research_area"]:
                            details.append(f"Research Area: {s['research_area']}")
                        lines.append(" - ".join(details))
                    return "\n".join(lines)
                else:
                    return f"I couldn't find any PhD or M.Tech students supervised by **{sup_display_name}** in the department records."

        # 3. PhD/M.Tech Student research area queries (dynamic, fallback)
        student_name_area = self._extract_research_area_query_name(query)
        if student_name_area:
            student_id = self._find_entity_by_name(student_name_area, allowed_labels=("PhDStudent", "MTechStudent"))
            if student_id and self.graph.has_node(student_id):
                student = self.graph.nodes[student_id]
                area = student.get("research_area", "")
                if area:
                    student_display_name = student.get("name", student_name_area)
                    label = student.get("label", "PhDStudent")
                    prog = "PhD scholar" if label == "PhDStudent" else "M.Tech student"
                    return f"{student_display_name} is a {prog} and their research area is {area}."

        # 4.5 MTech Roster count and list queries
        if self._is_mtech_roster_query(query):
            roster = self.get_mtech_roster()
            count = len(roster)
            lines = [
                (
                    f"The {self.dept_config['full_name']} at IIT Jammu has "
                    f"**{count} M.Tech students** listed on its official M.Tech roster page."
                ),
                ""
            ]
            for idx, scholar in enumerate(roster, start=1):
                details = [f"{idx}. **{scholar['name']}**"]
                if scholar["supervisors"]:
                    details.append(f"Supervisor(s): {', '.join(scholar['supervisors'])}")
                if scholar["research_area"]:
                    details.append(f"Research Area: {scholar['research_area']}")
                lines.append(" - ".join(details))
            return "\n".join(lines)

        # 4. PhD Roster count and list queries
        if self._is_phd_roster_query(query):
            roster = self.get_phd_roster()
            count = len(roster)
            lines = [
                (
                    f"The {self.dept_config['full_name']} at IIT Jammu has "
                    f"**{count} PhD scholars** listed on its official PhD page."
                )
            ]

            if self._query_has_list_intent(query):
                lines.extend(["", "**PhD scholar list**", ""])
                for idx, scholar in enumerate(roster, start=1):
                    details = [f"{idx}. **{scholar['name']}**"]
                    if scholar["supervisors"]:
                        details.append(f"Supervisor(s): {', '.join(scholar['supervisors'])}")
                    if scholar["research_area"]:
                        details.append(f"Research Area: {scholar['research_area']}")
                    if scholar["email"]:
                        details.append(f"Email: {scholar['email']}")
                    lines.append(" - ".join(details))

            lines.extend([
                "",
                f"Source: [IIT Jammu {self.dept_config['name']} PhD students]({self.dept_config['base_url']}/phd-list.html)",
            ])
            return "\n".join(lines)

        # 5. Faculty Roster count and list queries
        if not self._is_faculty_roster_query(query):
            return None

        roster = self.get_faculty_roster()
        dept_data = self.graph.nodes.get(self.dept_node_id, {})
        count = dept_data.get("faculty_count") or len(roster)

        lines = [
            (
                f"The {self.dept_config['full_name']} at IIT Jammu has "
                f"**{count} faculty members**."
            ),
            "",
            "**Faculty list**",
            "",
        ]
        for idx, member in enumerate(roster, start=1):
            role = " (Head of Department)" if member["is_hod"] else ""
            parts = [f"{idx}. **{member['name']}**{role}"]
            if member["designation"]:
                parts.append(member["designation"])
            if member["email"]:
                parts.append(member["email"])
            if member["profile_url"]:
                parts.append(f"[Profile]({member['profile_url']})")
            lines.append(" - ".join(parts))

        lines.extend([
            "",
            f"Source: [IIT Jammu {self.dept_config['name']} Faculty]({self._faculty_listing_url()})",
        ])
        return "\n".join(lines)

    def _get_node_display(self, node_id: str) -> str:
        """Get a clean display string for a node."""
        data = self.graph.nodes.get(node_id, {})
        label = data.get("label", "")
        name = data.get("name", node_id)
        
        parts = [f"**{name}**"]
        
        if label == "Faculty":
            if data.get("designation"):
                parts.append(f"  - Designation: {data['designation']}")
            if data.get("email"):
                parts.append(f"  - Email: {data['email']}")
            if data.get("is_hod"):
                parts.append(f"  - Role: Head of Department (HoD)")
            if data.get("education"):
                edu = data['education'][:300].replace('\n', '; ')
                parts.append(f"  - Education: {edu}")
            if data.get("research_experience"):
                exp = data['research_experience'][:400].replace('\n', '; ')
                parts.append(f"  - Research Experience: {exp}")
            if data.get("profile_url"):
                parts.append(f"  - Profile: {data['profile_url']}")
                
        elif label == "PhDStudent":
            if data.get("research_area"):
                parts.append(f"  - Research Area: {data['research_area']}")
            if data.get("email"):
                parts.append(f"  - Email: {data['email']}")
                
        elif label == "Project":
            proj_num = data.get("project_number", "")
            if data.get("title"):
                if proj_num:
                    parts[0] = f"**[{proj_num}] {data['title']}**"
                else:
                    parts[0] = f"**{data['title']}**"
            if data.get("agency"):
                parts.append(f"  - Funding Agency: {data['agency']}")
                
        elif label == "Patent":
            if data.get("title"):
                parts[0] = f"**{data['title']}**"
            if data.get("application_no"):
                parts.append(f"  - Application No: {data['application_no']}")
                
        elif label == "Startup":
            if data.get("description"):
                parts.append(f"  - Description: {data['description']}")
                
        elif label == "Lab":
            pass  # Name is sufficient

        elif label == "PlacementData":
            parts[0] = f"**{data.get('program', '')} Placement Data ({data.get('year', '')})**"
            parts.append(f"  - Placement Percentage: {data.get('percentage', 'NA')}%")
            parts.append(f"  - Mean Salary: {data.get('mean_salary', 'NA')} L@Y (Lakhs per Year)")
            parts.append(f"  - Maximum Salary: {data.get('max_salary', 'NA')} L@Y (Lakhs per Year)")
            parts.append(f"  - Minimum Salary: {data.get('min_salary', 'NA')} L@Y (Lakhs per Year)")

        elif label == "HigherStudiesData":
            parts[0] = f"**Higher Studies Data ({data.get('year', '')})**"
            parts.append(f"  - B.Tech students pursuing higher studies: {data.get('btech', 'NA')}")
            parts.append(f"  - M.Tech (CSP) students pursuing higher studies: {data.get('mtech_csp', 'NA')}")
            parts.append(f"  - M.Tech (VLSI) students pursuing higher studies: {data.get('mtech_vlsi', 'NA')}")
            parts.append(f"  - PhD students pursuing higher studies: {data.get('phd', 'NA')}")

        elif label == "ExternalPerson":
            if data.get("designation"):
                parts.append(f"  - Role: External Collaborator ({data['designation']})")
            else:
                parts.append(f"  - Role: External Collaborator/Supervisor")
            
        return "\n".join(parts)

    def _get_relationships_display(self, node_id: str) -> str:
        """Get formatted relationships for a node."""
        lines = []
        data = self.graph.nodes.get(node_id, {})
        label = data.get("label", "")
        
        # Outgoing
        for _, target, edge_data in self.graph.out_edges(node_id, data=True):
            t_data = self.graph.nodes.get(target, {})
            t_label = t_data.get("label", "")
            if t_label in ("TextChunk", "Document"):
                continue
            t_name = t_data.get("name", target)
            rel = edge_data.get("type", "RELATED")
            
            if rel == "SUPERVISED_BY":
                lines.append(f"  - Supervisor: {t_name}")
            elif rel == "RESEARCHES_IN":
                lines.append(f"  - Research Area: {t_name}")
            elif rel == "STUDIES":
                lines.append(f"  - Research Topic: {t_name}")
            elif rel == "HOD_OF":
                lines.append(f"  - Head of: {t_name}")
            elif rel == "MEMBER_OF":
                pass  # Skip — too generic
            elif rel == "MENTORED_STARTUP":
                lines.append(f"  - Mentored Startup: {t_name}")
            elif rel == "INVENTED":
                lines.append(f"  - Patent: {t_name}")
            elif rel == "FUNDED_BY":
                lines.append(f"  - Funded by: {t_name}")
            elif rel == "BELONGS_TO_CATEGORY":
                lines.append(f"  - Category: {t_name}")
            else:
                lines.append(f"  - {rel.replace('_', ' ').title()}: {t_name}")

        # Incoming
        for source, _, edge_data in self.graph.in_edges(node_id, data=True):
            s_data = self.graph.nodes.get(source, {})
            s_label = s_data.get("label", "")
            if s_label in ("TextChunk", "Document"):
                continue
            s_name = s_data.get("name", source)
            rel = edge_data.get("type", "RELATED")
            
            if rel == "SUPERVISED_BY":
                # Include the student's research area for richer context
                student_area = s_data.get("research_area", "")
                if student_area:
                    lines.append(f"  - PhD Student: {s_name} (Research Area: {student_area})")
                else:
                    lines.append(f"  - PhD Student: {s_name}")
            elif rel == "BELONGS_TO_CATEGORY":
                lines.append(f"  - Sub-area: {s_name}")
            elif rel == "INVENTED":
                lines.append(f"  - Inventor: {s_name}")
            elif rel == "MENTORED_STARTUP":
                lines.append(f"  - Mentor: {s_name}")
            elif rel == "RESEARCHES_IN":
                lines.append(f"  - Researcher (Faculty): {s_name}")
            elif rel == "STUDIES":
                lines.append(f"  - Researcher (PhD Student): {s_name}")

        # For outgoing INVENTED edges (faculty → patent), also list all co-inventors
        if label == "Faculty":
            for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                if edge_data.get("type") != "INVENTED":
                    continue
                co_inventors = []
                for inv_source, _, inv_edge in self.graph.in_edges(target, data=True):
                    if inv_edge.get("type") == "INVENTED" and inv_source != node_id:
                        inv_name = self.graph.nodes.get(inv_source, {}).get("name", inv_source)
                        co_inventors.append(inv_name)
                if co_inventors:
                    patent_name = self.graph.nodes.get(target, {}).get("title", target)
                    lines.append(f"  - Co-inventors on '{patent_name[:50]}': {', '.join(co_inventors)}")
                
        return "\n".join(lines[:25])  # Increased cap for richer context

    def _name_match(self, query: str) -> List[str]:
        """Find entities whose names appear in the query using advanced token-based fuzzy matching."""
        from difflib import SequenceMatcher
        
        # Normalize query: lowercase, remove punctuation, strip common title prefixes
        q_clean = re.sub(r"[^\w\s]", " ", query.lower())
        q_clean = re.sub(r"\b(?:dr|prof|mr|ms|mrs|shri|professor|assistant|associate)\b", " ", q_clean)
        query_words = [w for w in q_clean.split() if w]
        
        matched = []
        stop_words = {
            "who", "is", "the", "under", "what", "area", "of", "and", "in", "at", "to", 
            "for", "with", "a", "an", "on", "about", "tell", "me", "how", "closely", 
            "domains", "aligned", "supervises", "supervised", "supervisor", "advises", 
            "advisor", "working", "work", "phd", "student", "students", "scholar", 
            "scholars", "faculty", "professor", "professors", "count", "list", "members", "department",
            "are", "give", "show", "get", "find", "he", "she", "his", "her", "they", "them", "their", "guidance"
        }
        
        for name, node_id in self._entity_name_index.items():
            name_lower = name.lower()
            name_tokens = name_lower.split()
            
            for i in range(len(query_words)):
                for j in range(i + 1, min(i + 4, len(query_words) + 1)):
                    phrase = " ".join(query_words[i:j])
                    if phrase in stop_words or len(phrase) < 3:
                         continue
                         
                    # 1. Exact match with the full name
                    if phrase == name_lower:
                        matched.append((node_id, 1.0))
                        continue
                        
                    # 2. Substring match for full name
                    if phrase in name_lower and len(phrase) >= 5:
                        matched.append((node_id, 0.95))
                        continue
                        
                    # 3. Fuzzy match with the full name
                    full_ratio = SequenceMatcher(None, phrase, name_lower).ratio()
                    if full_ratio > 0.80:
                        matched.append((node_id, 0.90 * full_ratio))
                        continue
                        
                    # 4. Token-by-token comparison (exact or fuzzy)
                    for t in name_tokens:
                        if t in stop_words or len(t) < 3:
                            continue
                        if phrase == t:
                            matched.append((node_id, 0.95))
                            break
                        if (phrase in t or t in phrase) and len(phrase) >= 4 and len(t) >= 4:
                            matched.append((node_id, 0.70))
                            break
                        token_ratio = SequenceMatcher(None, phrase, t).ratio()
                        if token_ratio > 0.70:
                            matched.append((node_id, 0.90 * token_ratio))
                            break
                            
        # Deduplicate and sort by score
        seen = set()
        unique = []
        for node_id, score in sorted(matched, key=lambda x: -x[1]):
            if node_id not in seen:
                seen.add(node_id)
                unique.append(node_id)
        return unique[:8]

    def _is_placement_query(self, query: str) -> bool:
        """Detect placement/salary/higher studies queries."""
        q = re.sub(r"\s+", " ", query.lower()).strip()
        return any(term in q for term in (
            "placement", "salary", "package", "l@y", "lpa",
            "higher studies", "higher study", "academia placement",
            "mean salary", "max salary", "min salary", "minimum salary",
            "maximum salary", "average salary"
        ))

    def _placement_context(self) -> str:
        """Build structured placement context from graph nodes."""
        lines = ["## Placement & Higher Studies Data (Structured from Knowledge Graph)\n"]

        # Placement data nodes
        placement_nodes = [(nid, d) for nid, d in self.graph.nodes(data=True)
                          if d.get("label") == "PlacementData"]
        if placement_nodes:
            lines.append("### Industry Placement Data (salaries in L@Y = Lakhs per Year)")
            for nid, data in sorted(placement_nodes, key=lambda x: (x[1].get('year', ''), x[1].get('program', ''))):
                lines.append(f"\n**{data.get('program', '')} - {data.get('year', '')}:**")
                lines.append(f"  - Placement Percentage: {data.get('percentage', 'NA')}%")
                lines.append(f"  - Mean Salary: {data.get('mean_salary', 'NA')} L@Y")
                lines.append(f"  - Maximum Salary: {data.get('max_salary', 'NA')} L@Y")
                lines.append(f"  - Minimum Salary: {data.get('min_salary', 'NA')} L@Y")

        # Higher studies data nodes
        hs_nodes = [(nid, d) for nid, d in self.graph.nodes(data=True)
                    if d.get("label") == "HigherStudiesData"]
        if hs_nodes:
            lines.append("\n### Higher Studies Data (number of students)")
            for nid, data in sorted(hs_nodes, key=lambda x: x[1].get('year', '')):
                lines.append(f"\n**{data.get('year', '')}:**")
                lines.append(f"  - B.Tech: {data.get('btech', 'NA')} students")
                lines.append(f"  - M.Tech (CSP): {data.get('mtech_csp', 'NA')} students")
                lines.append(f"  - M.Tech (VLSI): {data.get('mtech_vlsi', 'NA')} students")
                lines.append(f"  - PhD: {data.get('phd', 'NA')} students")

        return "\n".join(lines)

    def _find_supervisors_by_research_area(self, query: str) -> List[Dict]:
        """Find supervisors of PhD students whose research area matches query keywords.

        Uses word-boundary-aware matching via _topic_matches_text() to prevent
        false positives from short keywords appearing as substrings in unrelated
        words (e.g. 'ai' matching 'uncertainty').
        """
        q_lower = query.lower()

        # Extract research-area keywords from query
        area_keywords = []
        stop_words = {
            'which', 'faculty', 'superise', 'supervise', 'supervises', 'supervising',
            'students', 'research', 'area', 'working', 'work', 'under', 'who', 'what',
            'list', 'with', 'from', 'the', 'their', 'have', 'does', 'professor',
            'members', 'domain', 'related', 'field', 'expert', 'works', 'iit', 'jammu',
            'department', 'and', 'for', 'are', 'but', 'not', 'you', 'your', 'our', 'out',
            'about', 'any', 'get', 'has', 'her', 'his', 'him', 'its', 'she', 'they', 'them',
            'these', 'those', 'this', 'that', 'than', 'then', 'into', 'only', 'some', 'such',
            'too', 'very', 'was', 'were', 'will', 'would', 'should', 'could', 'can', 'may',
            'might', 'must', 'shall', 'does', 'do', 'did', 'has', 'have', 'had', 'device', 'devices'
        }
        for word in q_lower.replace('?', '').replace(',', ' ').split():
            cleaned = word.strip()
            if len(cleaned) > 2 and cleaned not in stop_words:
                area_keywords.append(cleaned)

        if not area_keywords:
            return []

        # Find matching PhD students using word-boundary-aware matching
        matches = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("label") != "PhDStudent":
                continue
            student_area = data.get("research_area", "")
            if not student_area:
                continue
            # Check if any query keyword matches the research area
            if any(_topic_matches_text(kw, student_area) for kw in area_keywords):
                supervisors = []
                for _, target, edge_data in self.graph.out_edges(nid, data=True):
                    if edge_data.get("type") == "SUPERVISED_BY":
                        sup_data = self.graph.nodes.get(target, {})
                        supervisors.append(sup_data.get("name", target))
                matches.append({
                    "student": data.get("name", nid),
                    "area": data.get("research_area", ""),
                    "supervisors": supervisors
                })

        # Find matching Faculty nodes whose research interests or academic interests match keywords
        faculty_matches = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("label") != "Faculty":
                continue
            interests = data.get("research_interests", "")
            acad_interests = data.get("academic_interests", "")
            
            matched = False
            for kw in area_keywords:
                if (interests and _topic_matches_text(kw, interests)) or (acad_interests and _topic_matches_text(kw, acad_interests)):
                    matched = True
                    break
            
            if matched:
                faculty_matches.append(data.get("name", nid))

        if not matches and not faculty_matches:
            return []

        # Build context entries
        results = []
        if matches:
            lines = [f"**Research Area Supervisor Lookup**"]
            lines.append(f"Students with research areas matching '{' '.join(area_keywords)}':")
            supervisor_set = set()
            for m in matches:
                sup_str = ', '.join(m['supervisors']) if m['supervisors'] else 'Unknown'
                lines.append(f"  - {m['student']} — Area: {m['area']} — Supervisor(s): {sup_str}")
                supervisor_set.update(m['supervisors'])
            lines.append(f"\nFaculty supervising in this area: {', '.join(sorted(supervisor_set))}")

            results.append({
                "type": "entity", "score": 0.95, "label": "ResearchAreaLookup",
                "display": "\n".join(lines), "relationships": "",
            })

        if faculty_matches:
            lines_fac = [f"**Faculty Research Interests Lookup**"]
            lines_fac.append(f"Faculty members with interests matching '{' '.join(area_keywords)}':")
            for f in sorted(faculty_matches):
                lines_fac.append(f"  - {f}")
            results.append({
                "type": "entity", "score": 0.98, "label": "ResearchAreaFacultyLookup",
                "display": "\n".join(lines_fac), "relationships": "",
            })

        return results

    def _local_search(self, query: str, top_k: int = 6) -> List[Dict]:
        """Entity search: name matching first, then embedding fallback."""
        results = []
        seen_ids = set()
        
        # Phase 1: Direct name matching (highest priority)
        name_matches = self._name_match(query)
        for node_id in name_matches:
            if not self.graph.has_node(node_id):
                continue
            display = self._get_node_display(node_id)
            rels = self._get_relationships_display(node_id)
            label = self.graph.nodes[node_id].get("label", "")
            results.append({
                "type": "entity", "score": 1.0, "label": label,
                "display": display, "relationships": rels,
            })
            seen_ids.add(node_id)
        
        # Also check for keyword triggers
        ql = query.lower()
        if any(kw in ql for kw in ["hod", "head of department", "department head"]):
            for node_id in self._entity_label_index.get("Faculty", []):
                if self.graph.nodes[node_id].get("is_hod"):
                    if node_id not in seen_ids:
                        display = self._get_node_display(node_id)
                        rels = self._get_relationships_display(node_id)
                        results.insert(0, {
                            "type": "entity", "score": 1.0, "label": "Faculty",
                            "display": display, "relationships": rels,
                        })
                        seen_ids.add(node_id)

        # Phase 1.5: Research-area based lookup for faculty and supervisors
        if any(kw in ql for kw in ["supervis", "faculty", "professor", "who work", "research", "expert", "specialist"]):
            area_results = self._find_supervisors_by_research_area(query)
            results.extend(area_results)
        
        # Phase 2: Embedding-based entity search (fills remaining slots)
        remaining = top_k - len(results)
        if remaining > 0:
            entity_matches = self.embeddings.search(query, top_k=remaining, type_filter="entity", department_filter=self.dept_code, min_score=0.35)
            for item, score in entity_matches:
                node_id = item["id"]
                if node_id in seen_ids or not self.graph.has_node(node_id):
                    continue
                display = self._get_node_display(node_id)
                rels = self._get_relationships_display(node_id)
                label = self.graph.nodes[node_id].get("label", "")
                results.append({
                    "type": "entity", "score": score, "label": label,
                    "display": display, "relationships": rels,
                })
                seen_ids.add(node_id)
        
        return results[:top_k]

    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic chunk search with minimum score threshold."""
        results = []
        chunk_matches = self.embeddings.search(query, top_k=top_k, type_filter="chunk", department_filter=self.dept_code, min_score=0.35)
        
        for item, score in chunk_matches:
            text = item["text"][:1200]
            meta = item.get("metadata", {})
            results.append({
                "type": "chunk",
                "score": score,
                "text": text,
                "source": meta.get("title", "Unknown"),
                "url": meta.get("url", ""),
            })
        return results

    def _global_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Community summary search."""
        results = []
        matches = self.embeddings.search(query, top_k=top_k, type_filter="community", department_filter=self.dept_code)
        
        for item, score in matches:
            comm_id = item["id"]
            report = next((r for r in self.community_reports if r["id"] == comm_id), None)
            summary = report["summary"] if report else item.get("text", "")
            
            # Get member names
            members_str = ""
            if report:
                for label, member_ids in report.get("members_by_type", {}).items():
                    names = [self.graph.nodes.get(m, {}).get("name", m) for m in member_ids[:8]]
                    members_str += f"\n  - {label}: {', '.join(names)}"
            
            results.append({
                "type": "community",
                "score": score,
                "summary": summary,
                "members": members_str,
            })
        return results

    def retrieve(self, query: str, local_top_k: int = 6, vector_top_k: int = 4,
                 global_top_k: int = 2, max_context_words: int = 3000) -> str:
        """Run full hybrid retrieval and return clean formatted context."""
        bundle = self.retrieve_bundle(
            query,
            local_top_k=local_top_k,
            vector_top_k=vector_top_k,
            global_top_k=global_top_k,
            max_context_words=max_context_words,
        )
        return bundle["context"]

    def retrieve_bundle(
        self,
        query: str,
        local_top_k: int = 6,
        vector_top_k: int = 4,
        global_top_k: int = 2,
        max_context_words: int = 3000,
    ) -> Dict[str, Any]:
        """Run full hybrid retrieval and return context plus provenance metadata.

        Deterministic graph context (admin info, rosters, topic experts, etc.) is
        computed first and injected as high-priority context. The LLM always
        generates the final response — deterministic data is never returned raw.
        For enumeration queries (roster listings), vector search is skipped to
        avoid noise; only the authoritative graph data + local search are used.
        """
        # --- Phase 0: Compute deterministic context from graph ---
        deterministic_context = ""
        is_enumeration = False  # If True, skip vector search

        # A. Main deterministic dispatch (admin, hod, labs, topics, emails, etc.)
        det_ctx = self.get_deterministic_context(query)
        if det_ctx:
            deterministic_context = det_ctx
            logger.info("Retrieved deterministic graph context for query.")

        # B. Roster/enumeration queries — authoritative listings, skip vector search
        if not deterministic_context:
            if self._is_faculty_roster_query(query):
                deterministic_context = self._faculty_roster_context()
                is_enumeration = True
                logger.info("Retrieved authoritative faculty roster (enumeration).")
            elif self._is_phd_roster_query(query):
                deterministic_context = self._phd_roster_context()
                is_enumeration = True
                logger.info("Retrieved authoritative PhD roster (enumeration).")
            elif self._is_mtech_roster_query(query):
                deterministic_context = self._mtech_roster_context()
                is_enumeration = True
                logger.info("Retrieved authoritative M.Tech roster (enumeration).")

        if self._is_exact_count_query(query):
            global_top_k = 0

        # For placement queries, inject structured placement data
        placement_context = ""
        if self._is_placement_query(query):
            placement_context = self._placement_context()
            logger.info("Injected structured placement data context.")

        local_results = self._local_search(query, top_k=local_top_k)
        # Skip vector search for enumeration queries (authoritative graph data suffices)
        vector_results = [] if is_enumeration else self._vector_search(query, top_k=vector_top_k)
        global_results = self._global_search(query, top_k=global_top_k) if global_top_k > 0 else []

        sections = []
        word_count = 0
        section_word_counts = {"graph": 0, "vector": 0, "community": 0}

        # Section 0: Deterministic context from graph (highest priority)
        if deterministic_context:
            sections.append("## Authoritative Department Data\n\n" + deterministic_context)
            dc_words = len(deterministic_context.split())
            word_count += dc_words
            section_word_counts["graph"] += dc_words

        # Section 0b: Structured placement data
        if placement_context:
            sections.append(placement_context)
            word_count += len(placement_context.split())
            section_word_counts["graph"] += len(placement_context.split())

        # Section 1: Matched Entities
        if local_results:
            entity_lines = ["## Matched Entities from Knowledge Graph\n"]
            for r in local_results:
                if word_count > max_context_words:
                    break
                entry = f"### {r['label']}\n{r['display']}"
                if r['relationships']:
                    entry += f"\n{r['relationships']}"
                entity_lines.append(entry)
                entry_words = len(entry.split())
                word_count += entry_words
                section_word_counts["graph"] += entry_words
            sections.append("\n\n".join(entity_lines))

        # Section 2: Relevant Text
        if vector_results and word_count < max_context_words:
            chunk_lines = ["## Relevant Department Information\n"]
            for r in vector_results:
                if word_count > max_context_words:
                    break
                remaining_words = max_context_words - word_count
                text = r["text"]
                words = text.split()
                if len(words) > remaining_words:
                    text = " ".join(words[:remaining_words]) + "..."
                
                source_info = f"[Source: {r['source']}]"
                if r['url']:
                    source_info += f" ({r['url']})"
                chunk_lines.append(f"{source_info}\n{text}")
                chunk_words = len(text.split()) + 5
                word_count += chunk_words
                section_word_counts["vector"] += chunk_words
            sections.append("\n\n".join(chunk_lines))

        # Section 3: Department Overview
        if global_results and word_count < max_context_words:
            comm_lines = ["## Department Overview\n"]
            for r in global_results:
                if word_count > max_context_words:
                    break
                entry = r["summary"]
                if r["members"]:
                    entry += f"\nMembers:{r['members']}"
                comm_lines.append(entry)
                entry_words = len(entry.split())
                word_count += entry_words
                section_word_counts["community"] += entry_words
            sections.append("\n\n".join(comm_lines))

        context = "\n\n---\n\n".join(sections)
        if not context.strip():
            context = "No relevant information found in the knowledge graph for this query."

        # Minimum context gate: if context is too thin, force unanswerable
        # to prevent the LLM from hallucinating on near-empty evidence
        context_word_count = len(context.split())
        if context_word_count < 20 and context.strip() != "No relevant information found in the knowledge graph for this query.":
            logger.info(f"Context too thin ({context_word_count} words) — forcing unanswerable.")
            context = "No relevant information found in the knowledge graph for this query."
            local_results = []
            vector_results = []
            global_results = []

        answerability = self._assess_answerability(
            query=query,
            local_results=local_results,
            vector_results=vector_results,
            global_results=global_results,
            context=context,
        )
        fallback_response = None
        if not answerability["answerable"]:
            fallback_response = self._build_unavailable_response(query, answerability["reason"])

        provenance = self._build_provenance(
            direct=False,
            local_results=local_results,
            vector_results=vector_results,
            global_results=global_results,
            section_word_counts=section_word_counts,
        )

        logger.info(f"Retrieved: ~{word_count} words, {len(local_results)} entities, "
                    f"{len(vector_results)} chunks, {len(global_results)} communities")
        return {
            "context": context,
            "provenance": provenance,
            "answerability": answerability,
            "fallback_response": fallback_response,
        }


def load_retriever(dept_code: str = "ee", data_dir: str = None) -> HybridRetriever:
    """Load all components and create a HybridRetriever."""
    from graphrag.kg_builder import KnowledgeGraphBuilder
    from graphrag.embeddings import EmbeddingEngine
    from graphrag.community import load_communities
    from departments import get_data_dir

    if data_dir is None:
        data_dir = get_data_dir(dept_code)

    logger.info(f"Loading knowledge graph for department '{dept_code}'...")
    graph, chunks = KnowledgeGraphBuilder.load(data_dir)
    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    logger.info("Loading embeddings...")
    engine = EmbeddingEngine()
    engine.load(data_dir)
    logger.info(f"FAISS index: {engine.index.ntotal} vectors")

    logger.info("Loading communities...")
    partition, reports = load_communities(data_dir)
    logger.info(f"Communities: {len(set(partition.values()))}")

    retriever = HybridRetriever(graph, engine, reports, dept_code=dept_code)
    logger.info(f"Hybrid retriever for '{dept_code}' ready!")
    return retriever
