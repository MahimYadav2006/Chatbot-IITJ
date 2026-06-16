"""
Smart Department Router for IIT Jammu Multi-Department Chatbot.

Detects which department(s) a query targets using department-name aliases.
Does NOT use subject-based matching (e.g., "machine learning") since subjects
can belong to multiple departments.

Routes:
  - Single department:  "Who is the CSE HOD?" → [computer_science_engineering]
  - Multi department:   "Compare EE and CSE faculty" → [ee, computer_science_engineering]
  - Broadcast:          "How many departments?" → all ingested departments
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from departments import DEPARTMENTS, resolve_department_code

logger = logging.getLogger(__name__)


# ─── Department-Name Aliases ────────────────────────────────────────────────
# ONLY department names, abbreviations, and common short forms.
# NOT subject keywords like "machine learning", "antenna", etc.

DEPT_NAME_ALIASES = {
    "administration": [
        "administration", "admin", "senate", "board of governors", "bog",
        "finance committee", "finance committe", "building and works", "bwc",
        "dean", "deans", "associate deans", "associate dean", "ad", "director", "registrar",
        "governance", "committee", "committees"
    ],
    "ee": [
        "electrical engineering", "electrical", "ee", "ee department",
        "dept of ee", "department of electrical",
    ],
    "computer_science_engineering": [
        "computer science and engineering", "computer science & engineering",
        "computer science engineering", "computer science", "comp sci",
        "computer engineering", "cse", "cs", "cse department",
        "dept of cse", "department of computer science",
    ],
    "mechanical_engineering": [
        "mechanical engineering", "mechanical", "mech", "me department",
        "dept of mechanical", "department of mechanical",
    ],
    "civil_engineering": [
        "civil engineering", "civil", "civil eng",
        "dept of civil", "department of civil",
    ],
    "chemical-engineering": [
        "chemical engineering", "chemical", "chem eng", "chemical dept",
        "dept of chemical", "department of chemical engineering",
    ],
    "bsbe": [
        "biosciences and bioengineering", "biosciences & bioengineering",
        "bioscience", "bioengineering", "bsbe", "bio science",
        "bio engineering", "biosciences",
    ],
    "chemistry": [
        "chemistry", "chemistry department", "chem department",
        "dept of chemistry", "department of chemistry",
    ],
    "hss": [
        "humanities and social sciences", "humanities & social sciences",
        "humanities", "social sciences", "hss",
        "dept of hss", "department of humanities",
    ],
    "idp": [
        "interdisciplinary programmes", "interdisciplinary programs",
        "interdisciplinary", "idp",
    ],
    "materials-engineering": [
        "materials engineering", "materials", "metallurgy",
        "materials dept", "dept of materials",
        "department of materials engineering",
    ],
    "mathematics": [
        "mathematics", "maths", "math", "math department",
        "dept of mathematics", "department of mathematics",
    ],
    "physics": [
        "physics", "physics department", "phy",
        "dept of physics", "department of physics",
    ],
}
SECTION_NAME_ALIASES = {
    "academics": [
        "academics", "academic affairs", "academic section", 
        "ug admission", "pg admission", "course registration",
        "examination", "grading", "convocation", "scholarship",
        "academic programs", "curriculum", "specialisation",
        "specialization", "course", "syllabus", "admission",
        "minor", "minors", "elective", "electives"
    ],
    "accounts": [
        "accounts", "accounts section", "finance", "billing",
        "fee payment", "financial aid", "payroll", "finance & accounts"
    ],
    "counselling": [
        "counselling", "counseling", "mental health", 
        "counselor", "counsellor", "psychological",
        "stress management", "therapy"
    ],
    "di": [
        "digital infrastructure", "di", "c3i", "it services", 
        "network services", "data center", "software development",
        "campus network", "internet"
    ],
    "e2": [
        "establishment ii", "establishment 2", "e2", "e-2",
        "hr matters", "service book", "leave management",
        "recruitment process"
    ],
    "alumni-affairs": [
        "alumni", "alumni affairs", "alumni office", "alumni medalist",
        "alumni award", "alumni directory", "alumni contact", "convocation medalist",
        "gold medal", "silver medal", "institute medalist"
    ],
    "cds": [
        "cds", "career development services", "career development",
        "placement", "placements", "campus placement", "campus recruitment",
        "recruiter", "recruiters", "past recruiters", "placement statistics",
        "placement report", "placement policy", "rise-up", "riseup",
        "campus internship", "campus internships", "placement internship",
        "placement internships", "internship placement", "internship offer",
        "campus hiring",
        "placement highlights", "highest package", "average package",
        "offer", "lpa", "ctc", "salary", "company", "eligible"
    ],
    "ir": [
        "international relations", "ir office", "office of international relations",
        "mou", "mous", "memorandum of understanding", "exchange program",
        "student exchange", "faculty exchange", "erasmus", "daad",
        "club", "clubs", "student clubs", "sports clubs", "sports fest",
        "ebsb", "coding club", "robotics", "robo-sapiens", "fintech",
        "beatstreet", "dramatizers", "malang", "wellness club", "nac",
        "abhivyakta", "anisoul", "astria", "kritash", "mesh club",
        "sae club", "re4m", "foot tinkerers",
        "hostel", "hostels", "residential life", "dining", "mess",
        "cultural events", "festivals", "fest", "fests",
        "anhad", "pravaah", "convoquer", "nexus", "pragyaan", "udyamitsav",
        "pahal", "inter-gen"
    ],
    "medical-centre": [
        "medical centre", "medical center", "health centre", "health center",
        "health and wellness", "doctor", "doctors", "hospital",
        "ambulance", "pharmacy", "dental", "physiotherapy", "ecg",
        "laboratory services", "ward facility", "dressing room",
        "empaneled hospital", "empaneled hospitals", "cghs",
        "medical facility", "medical facilities"
    ],
    "osd": [
        "osd", "outreach", "outreach and skill development",
        "skill development", "unnat bharat abhiyan", "uba",
        "ces", "centre for essential skills", "center for essential skills",
        "raise", "outreach division", "school visit",
        "capacity building", "training program"
    ],
}


@dataclass
class RouteResult:
    """Result of department routing."""
    departments: List[str]            # Canonical department codes
    confidence: str                   # "exact" | "broadcast"
    reason: str                       # Human-readable routing explanation
    query: str = ""                   # Original query
    sections: List[str] = field(default_factory=list)  # Canonical section codes


class DepartmentRouter:
    """Routes queries to the correct department(s) or section(s) based on aliases."""

    def __init__(self):
        # Build sorted alias → code lookup (longest aliases first for greedy matching)
        self._alias_map: List[tuple] = []
        for dept_code, aliases in DEPT_NAME_ALIASES.items():
            for alias in aliases:
                self._alias_map.append((alias.lower().strip(), dept_code))

        # Sort by alias length descending
        self._alias_map.sort(key=lambda x: len(x[0]), reverse=True)

        self._section_alias_map: List[tuple] = []
        for sec_code, aliases in SECTION_NAME_ALIASES.items():
            for alias in aliases:
                self._section_alias_map.append((alias.lower().strip(), sec_code))
        self._section_alias_map.sort(key=lambda x: len(x[0]), reverse=True)

        self._valid_codes = set(DEPARTMENTS.keys())

    def route(self, query: str) -> RouteResult:
        """
        Analyze a query and determine which department(s) and/or section(s) it targets.
        """
        from graphrag.intent_utils import is_academic_rules_query
        if is_academic_rules_query(query):
            return RouteResult(
                departments=[],
                sections=["academics"],
                confidence="exact",
                reason="Query identified as academic rules and regulations request",
                query=query,
            )

        detected_depts = self._detect_departments(query)
        detected_secs = self._detect_sections(query)

        # Inject academics for all department queries
        if detected_depts:
            if "academics" not in detected_secs:
                detected_secs.append("academics")

        if detected_depts or detected_secs:
            reasons = []
            if detected_depts:
                dept_names = [DEPARTMENTS[d]["name"] for d in detected_depts]
                reasons.append(f"Matched departments: {', '.join(dept_names)}")
            if detected_secs:
                from departments import SECTIONS
                sec_names = [SECTIONS[s]["name"] for s in detected_secs]
                reasons.append(f"Matched sections: {', '.join(sec_names)}")

            return RouteResult(
                departments=detected_depts,
                sections=detected_secs,
                confidence="exact",
                reason="; ".join(reasons),
                query=query,
            )

        # No specific department/section signal detected → broadcast
        return RouteResult(
            departments=[],
            sections=[],
            confidence="broadcast",
            reason="No specific department/section detected — will search all",
            query=query,
        )

    def _detect_departments(self, query: str) -> List[str]:
        """Detect department references in query text using greedy alias matching."""
        q_lower = re.sub(r"\s+", " ", query.lower()).strip()
        q_clean = re.sub(r"[?!.,;:'\"-]", " ", q_lower)
        q_clean = re.sub(r"\s+", " ", q_clean).strip()

        detected = []
        seen = set()
        matched_spans = []

        for alias, dept_code in self._alias_map:
            if dept_code in seen:
                continue

            pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
            match = re.search(pattern, q_clean)
            if match:
                start, end = match.start(), match.end()
                if any(s <= start < e or s < end <= e for s, e in matched_spans):
                    continue

                detected.append(dept_code)
                seen.add(dept_code)
                matched_spans.append((start, end))

        return detected

    def _detect_sections(self, query: str) -> List[str]:
        """Detect section references in query text using greedy alias matching."""
        q_lower = re.sub(r"\s+", " ", query.lower()).strip()
        q_clean = re.sub(r"[?!.,;:'\"-]", " ", q_lower)
        q_clean = re.sub(r"\s+", " ", q_clean).strip()

        detected = []
        seen = set()
        matched_spans = []

        for alias, sec_code in self._section_alias_map:
            if sec_code in seen:
                continue

            pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
            match = re.search(pattern, q_clean)
            if match:
                start, end = match.start(), match.end()
                if any(s <= start < e or s < end <= e for s, e in matched_spans):
                    continue

                detected.append(sec_code)
                seen.add(sec_code)
                matched_spans.append((start, end))

        return detected

    def get_ingested_departments(self) -> List[str]:
        """Return list of department codes that have been ingested."""
        from departments import get_data_dir
        ingested = []
        for code in DEPARTMENTS:
            data_dir = get_data_dir(code)
            if os.path.exists(os.path.join(data_dir, "graph.pkl")):
                ingested.append(code)
        return ingested

    def get_ingested_sections(self) -> List[str]:
        """Return list of section codes that have been ingested."""
        from departments import get_section_data_dir, SECTIONS
        ingested = []
        for code in SECTIONS:
            data_dir = get_section_data_dir(code)
            if os.path.exists(os.path.join(data_dir, "graph.pkl")):
                ingested.append(code)
        return ingested
