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


@dataclass
class RouteResult:
    """Result of department routing."""
    departments: List[str]            # Canonical department codes
    confidence: str                   # "exact" | "broadcast"
    reason: str                       # Human-readable routing explanation
    query: str = ""                   # Original query


class DepartmentRouter:
    """Routes queries to the correct department(s) based on department-name aliases."""

    def __init__(self):
        # Build sorted alias → code lookup (longest aliases first for greedy matching)
        self._alias_map: List[tuple] = []
        for dept_code, aliases in DEPT_NAME_ALIASES.items():
            for alias in aliases:
                self._alias_map.append((alias.lower().strip(), dept_code))

        # Sort by alias length descending so "computer science engineering"
        # matches before "computer science" or "computer"
        self._alias_map.sort(key=lambda x: len(x[0]), reverse=True)

        # Also build a set of all canonical codes for validation
        self._valid_codes = set(DEPARTMENTS.keys())

    def route(self, query: str) -> RouteResult:
        """
        Analyze a query and determine which department(s) it targets.

        Returns a RouteResult with:
          - departments: list of canonical department codes
          - confidence: "exact" if department was detected, "broadcast" otherwise
          - reason: human-readable explanation
        """
        detected = self._detect_departments(query)

        if detected:
            dept_names = [DEPARTMENTS[d]["name"] for d in detected]
            if len(detected) == 1:
                reason = f"Matched department: {dept_names[0]}"
            else:
                reason = f"Matched departments: {', '.join(dept_names)}"

            return RouteResult(
                departments=detected,
                confidence="exact",
                reason=reason,
                query=query,
            )

        # No department signal detected → broadcast
        return RouteResult(
            departments=[],  # Empty = broadcast to all
            confidence="broadcast",
            reason="No specific department detected — will search all departments",
            query=query,
        )

    def _detect_departments(self, query: str) -> List[str]:
        """
        Detect department references in query text using greedy alias matching.
        Returns deduplicated list of canonical department codes, preserving order.
        """
        q_lower = re.sub(r"\s+", " ", query.lower()).strip()
        # Remove punctuation for matching but keep the text
        q_clean = re.sub(r"[?!.,;:'\"-]", " ", q_lower)
        q_clean = re.sub(r"\s+", " ", q_clean).strip()

        detected = []
        seen = set()
        matched_spans = []  # Track matched character ranges to avoid overlaps

        for alias, dept_code in self._alias_map:
            if dept_code in seen:
                continue

            # Use word-boundary matching to avoid partial matches
            # e.g., "physics" shouldn't match inside "astrophysics"
            pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"

            match = re.search(pattern, q_clean)
            if match:
                # Check for overlap with already-matched spans
                start, end = match.start(), match.end()
                if any(s <= start < e or s < end <= e for s, e in matched_spans):
                    continue

                detected.append(dept_code)
                seen.add(dept_code)
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
