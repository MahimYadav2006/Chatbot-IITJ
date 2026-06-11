"""
Global Person Index for IIT Jammu Unified Chatbot.
Cross-references and resolves individuals across all loaded department and section graphs
to ensure factual consistency and complete role mapping.
"""

import logging
from collections import defaultdict
from graphrag.kg_builder import EntityResolver, normalize_name

logger = logging.getLogger(__name__)

class GlobalPersonIndex:
    def __init__(self):
        self.resolver = EntityResolver()
        # Canonical name -> list of role dicts
        self.person_roles = defaultdict(list)

    # Patterns that indicate a node name is a committee-role placeholder,
    # not an actual person (e.g. "Dean Student Affairs", "All Deans",
    # "Upto Five Student Representative Nominated By The Chairman").
    _ROLE_PLACEHOLDER_PREFIXES = (
        "dean ", "all dean", "upto ", "nominated ",
        "one representative", "two representative", "three representative",
        "four representative", "five representative",
    )

    @classmethod
    def _is_role_placeholder(cls, name: str) -> bool:
        """Return True if *name* looks like a committee-role slot, not a person."""
        nl = name.strip().lower()
        if any(nl.startswith(prefix) for prefix in cls._ROLE_PLACEHOLDER_PREFIXES):
            return True
        # Very long "names" with generic committee phrasing
        if len(nl.split()) > 8:
            return True
        return False

    def index_graph(self, graph, source_name: str, is_section: bool = False):
        """Extract person entities from a graph and index them."""
        for node_id, data in graph.nodes(data=True):
            label = data.get("label")
            if label in ("Faculty", "AdminOfficial", "SectionPerson", "SectionHead", "Counselor", "MedicalDoctor"):
                name = data.get("name") or str(node_id)
                # Strip prefix if any (e.g. ee:Dr. Alok Kumar Saxena -> Dr. Alok Kumar Saxena)
                if ":" in name and not name.startswith("http"):
                    name = name.split(":", 1)[1]

                # Skip committee-role placeholder names (not actual people)
                if self._is_role_placeholder(name):
                    continue
                    
                resolved_name = self.resolver.resolve(name)
                
                role = {
                    "source": source_name,
                    "is_section": is_section,
                    "label": label,
                    "designation": data.get("designation") or data.get("admin_role") or "",
                    "email": data.get("email") or data.get("contact_email") or "",
                    "phone": data.get("phone") or "",
                    "profile_url": data.get("profile_url") or "",
                    "office": data.get("office") or "",
                    "qualifications": data.get("qualifications") or "",
                    "experience": data.get("experience") or ""
                }
                
                # Check for duplicate roles from the same source to keep it clean
                existing = self.person_roles[resolved_name]
                duplicate = False
                for r in existing:
                    if r["source"] == role["source"] and r["designation"] == role["designation"]:
                        duplicate = True
                        break
                if not duplicate:
                    self.person_roles[resolved_name].append(role)

    def lookup(self, name_query: str) -> dict:
        """Look up all roles for a given name query, using entity resolver logic."""
        resolved = self.resolver.resolve(name_query)
        if resolved in self.person_roles:
            return {
                "name": resolved,
                "roles": self.person_roles[resolved]
            }
        
        # Try direct fuzzy matches on keys
        normalized_query = normalize_name(name_query)
        for name, roles in self.person_roles.items():
            if normalized_query.lower() in name.lower() or name.lower() in normalized_query.lower():
                return {
                    "name": name,
                    "roles": roles
                }
        return {}
