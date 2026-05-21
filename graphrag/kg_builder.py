"""
Knowledge Graph Builder for IIT Jammu EE Department.
Parses markdown files and constructs a NetworkX DiGraph with entities,
relationships, and clean text chunks.
"""

import os
import re
import json
import pickle
import logging
from collections import defaultdict
from difflib import SequenceMatcher

import networkx as nx

logger = logging.getLogger(__name__)

MARKDOWN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "iitjammu_ee_markdown")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

TITLE_PREFIXES = re.compile(
    r'\b(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?|Shri\.?|Professor|Assistant\s+Professor|Associate\s+Professor)\b',
    re.IGNORECASE
)

# Boilerplate patterns to strip from chunks
BOILERPLATE_PATTERNS = [
    r'^# Source URL:.*$',
    r'^html\s*$',
    r'^Electrical Engineering \| IIT Jammu\s*$',
    r'^\!\[.*?\]\(.*?\)\s*$',
    r'^- \[Home\].*$',
    r'^- \[Brief Info\].*$',
    r'^- \[Profile\].*$',
    r'^- \[Current Openings\].*$',
    r'^- \[Other Info\].*$',
    r'^- \[Personal Website\].*$',
    r'^\!\[Breadcrumbs.*$',
    r'^\!\[Image\]\(https://iitjammu\.ac\.in/ee/assets/images/electrical/logo.*$',
    r'^- Faculty\s*$',
    r'^- PhD Students\s*$',
    r'^- HoD Message\s*$',
    r'^- Research Areas\s*$',
    r'^### People\s*$',
    r'^- \[Faculty\]\(.*$',
    r'^- \[PhD Students\]\(.*$',
    r'^- \[Staff\]\(.*$',
    r'^- \[Project Staff\]\(.*$',
    r'^- \[MTech Students\]\(.*$',
    r'^### Research\s*$',
    r'^- \[Research Areas\]\(.*$',
    r'^- \[Startups\]\(.*$',
    r'^- \[Funded Projects\]\(.*$',
    r'^- \[Patents\]\(.*$',
    r'^- \[Awards and Honors\]\(.*$',
    r'^- \[Publications\]\(.*$',
    r'^\[Read more\]\(.*?\)\s*$',
    r'^\[Read\s*$',
    r'^More\]\(.*?\)\s*$',
    r'^\[Discover More\]\(.*?\)\s*$',
    r'^Welcome to\s*$',
    r'^ducation\s*$',
]


def clean_content_for_chunks(content: str) -> str:
    """Remove boilerplate HTML/navigation noise from content before chunking."""
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        skip = False
        for pat in BOILERPLATE_PATTERNS:
            if re.match(pat, line.strip(), re.IGNORECASE):
                skip = True
                break
        if not skip:
            cleaned.append(line)
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip markdown link syntax but keep link text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Strip leftover image references
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'Read\s*more', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Professor|Lecturer)(Research Experience)', r'\1\nResearch Experience', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_name(name: str) -> str:
    name = TITLE_PREFIXES.sub('', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.strip('., ')
    name = ' '.join(w.capitalize() for w in name.split())
    return name


def _initials_match(short: str, full: str) -> bool:
    """Check if an abbreviated name like 'B. N Subudhi' matches 'Badri Narayan Subudhi'."""
    short_parts = short.lower().replace('.', '').split()
    full_parts = full.lower().replace('.', '').split()
    if len(short_parts) < 2 or len(full_parts) < 2:
        return False
    # Last name must match
    if SequenceMatcher(None, short_parts[-1], full_parts[-1]).ratio() < 0.85:
        return False
    # All preceding parts in short must be initials matching full name parts
    short_prefixes = short_parts[:-1]
    full_prefixes = full_parts[:-1]
    if len(short_prefixes) > len(full_prefixes):
        return False
    for sp, fp in zip(short_prefixes, full_prefixes):
        if len(sp) <= 2:  # Initial like "b" or "bn"
            if not fp.startswith(sp[0]):
                return False
        else:
            if SequenceMatcher(None, sp, fp).ratio() < 0.80:
                return False
    return True


def _token_subset_match(name1: str, name2: str) -> bool:
    """
    Match variants like 'Anup Kumar Shukla' and 'Anup Shukla'.

    Requires strong agreement on first/last name while allowing optional
    middle tokens in either variant.
    """
    parts1 = name1.lower().replace('.', '').split()
    parts2 = name2.lower().replace('.', '').split()
    if len(parts1) < 2 or len(parts2) < 2:
        return False

    shorter, longer = (parts1, parts2) if len(parts1) <= len(parts2) else (parts2, parts1)
    if SequenceMatcher(None, shorter[0], longer[0]).ratio() < 0.85:
        return False
    if SequenceMatcher(None, shorter[-1], longer[-1]).ratio() < 0.85:
        return False

    search_start = 1
    for token in shorter[1:-1]:
        matched = False
        for idx in range(search_start, len(longer) - 1):
            if SequenceMatcher(None, token, longer[idx]).ratio() >= 0.85:
                search_start = idx + 1
                matched = True
                break
        if not matched:
            return False

    return True


def fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio() >= threshold


def _chunk_words(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _chunk_repeated_records(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Chunk roster-style markdown by repeated `####` records instead of raw word windows.

    This preserves complete student/faculty entries for retrieval and avoids
    partial-count failures on list pages.
    """
    if text.count("\n#### ") < 4:
        return []

    parts = re.split(r'(?=^####\s+)', text, flags=re.MULTILINE)
    if not parts:
        return []

    prefix = ""
    records = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("#### "):
            records.append(part)
        elif not prefix:
            prefix = part
        else:
            prefix = f"{prefix}\n\n{part}"

    if len(records) < 4:
        return []

    chunks = []
    current_parts = [prefix] if prefix else []
    current_words = len(prefix.split()) if prefix else 0

    for record in records:
        record_words = len(record.split())
        if record_words > chunk_size:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_words = 0
            chunks.extend(_chunk_words(record, chunk_size, overlap=0))
            continue

        if current_parts and current_words + record_words > chunk_size:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = [record]
            current_words = record_words
        else:
            current_parts.append(record)
            current_words += record_words

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Chunk markdown without splitting structured heading sections when possible."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    record_chunks = _chunk_repeated_records(text, chunk_size=chunk_size)
    if record_chunks:
        return record_chunks

    sections = [s.strip() for s in re.split(r'(?=^#{2,6}\s+)', text, flags=re.MULTILINE) if s.strip()]
    if len(sections) <= 1:
        return _chunk_words(text, chunk_size, overlap)

    chunks = []
    current_sections = []
    current_words = 0

    for section in sections:
        section_words = len(section.split())
        if section_words > chunk_size:
            if current_sections:
                chunks.append("\n\n".join(current_sections))
                current_sections = []
                current_words = 0
            chunks.extend(_chunk_words(section, chunk_size, overlap))
            continue

        if current_sections and current_words + section_words > chunk_size:
            chunks.append("\n\n".join(current_sections))
            current_sections = [section]
            current_words = section_words
        else:
            current_sections.append(section)
            current_words += section_words

    if current_sections:
        chunks.append("\n\n".join(current_sections))

    return chunks


def _extract_canonical_faculty(markdown_dir: str) -> set:
    """Extract the 24 canonical faculty names from ee_faculty-list.html.md."""
    flist_path = os.path.join(markdown_dir, "ee_faculty-list.html.md")
    if not os.path.exists(flist_path):
        logger.warning("ee_faculty-list.html.md not found; cannot build canonical faculty set.")
        return set()
    with open(flist_path, "r", encoding="utf-8") as f:
        content = f.read()
    names = re.findall(r'####\s*\[([^\]]+)\]', content)
    canonical = set()
    for n in names:
        canonical.add(normalize_name(n.strip()))
    logger.info(f"Canonical faculty registry: {len(canonical)} names")
    return canonical


class EntityResolver:
    def __init__(self, canonical_faculty: set = None):
        self.canonical_names = {}
        self.name_variants = defaultdict(set)
        self._canonical_faculty = canonical_faculty or set()
        # Pre-register canonical faculty
        for name in self._canonical_faculty:
            self.canonical_names[name] = name
            self.name_variants[name].add(name)

    def is_canonical_faculty(self, raw_name: str) -> bool:
        """Check if a name matches any canonical faculty member."""
        normalized = normalize_name(raw_name)
        if not normalized:
            return False
        if normalized in self._canonical_faculty:
            return True
        for canon in self._canonical_faculty:
            if fuzzy_match(normalized, canon):
                return True
            if _initials_match(normalized, canon):
                return True
            if _token_subset_match(normalized, canon):
                return True
        return False

    def resolve(self, raw_name: str) -> str:
        normalized = normalize_name(raw_name)
        if not normalized:
            return raw_name.strip()
        if normalized in self.canonical_names:
            return self.canonical_names[normalized]
        # Check canonical faculty with initials matching
        for canonical in self._canonical_faculty:
            if fuzzy_match(normalized, canonical):
                self.canonical_names[normalized] = canonical
                self.name_variants[canonical].add(normalized)
                return canonical
            if _initials_match(normalized, canonical):
                self.canonical_names[normalized] = canonical
                self.name_variants[canonical].add(normalized)
                return canonical
            if _token_subset_match(normalized, canonical):
                self.canonical_names[normalized] = canonical
                self.name_variants[canonical].add(normalized)
                return canonical
        # Check existing resolved names
        for canonical in list(self.name_variants.keys()):
            if fuzzy_match(normalized, canonical):
                self.canonical_names[normalized] = canonical
                self.name_variants[canonical].add(normalized)
                return canonical
            if _token_subset_match(normalized, canonical):
                self.canonical_names[normalized] = canonical
                self.name_variants[canonical].add(normalized)
                return canonical
        self.canonical_names[normalized] = normalized
        self.name_variants[normalized].add(normalized)
        return normalized


class KnowledgeGraphBuilder:
    def __init__(self, markdown_dir: str = MARKDOWN_DIR):
        self.markdown_dir = markdown_dir
        self._canonical_faculty = _extract_canonical_faculty(markdown_dir)
        self.graph = nx.DiGraph()
        self.resolver = EntityResolver(self._canonical_faculty)
        self.chunks = []

    def _add_node(self, node_id: str, label: str, **properties):
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(properties)
        else:
            self.graph.add_node(node_id, label=label, **properties)

    def _add_edge(self, source: str, target: str, rel_type: str, **properties):
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return
        if self.graph.has_edge(source, target):
            if self.graph.edges[source, target].get('type', '') == rel_type:
                return
        self.graph.add_edge(source, target, type=rel_type, **properties)

    def _create_document_node(self, filename: str, content: str) -> str:
        source_url = "https://iitjammu.ac.in/ee"
        url_match = re.search(r'# Source URL:\s*([^\n]+)', content)
        if url_match:
            source_url = url_match.group(1).strip()

        clean_title = (filename.replace(".html.md", "").replace(".md", "")
            .replace("ee_", "").replace("_", " ").title())

        doc_id = f"doc:{filename}"
        self._add_node(doc_id, "Document", title=clean_title,
            filename=filename, source_url=source_url)

        # Clean content before chunking — remove boilerplate
        clean_content = clean_content_for_chunks(content)
        text_chunks = chunk_text(clean_content)
        
        for idx, chunk_text_str in enumerate(text_chunks):
            if len(chunk_text_str.strip()) < 30:
                continue
            chunk_id = f"chunk_{filename}_{idx}"
            self._add_node(chunk_id, "TextChunk", text=chunk_text_str,
                doc_filename=filename, chunk_index=idx, source_url=source_url)
            self._add_edge(doc_id, chunk_id, "HAS_CHUNK")
            self.chunks.append((chunk_id, chunk_text_str, {
                "doc": filename, "url": source_url,
                "title": clean_title, "chunk_idx": idx
            }))
        return doc_id

    def _parse_faculty_profile(self, filename: str, content: str, doc_id: str):
        lines = [l.strip() for l in content.splitlines()]
        faculty_key = filename.split("__")[-1].replace(".md", "")
        faculty_name = normalize_name(faculty_key)

        for line in lines:
            if (line.startswith("- ") and len(line) > 2 and "@" not in line
                    and "Professor" not in line and "Home" not in line
                    and "Faculty" not in line and "http" not in line
                    and len(line) < 60):
                candidate = line[2:].strip()
                if candidate and not candidate.startswith("[") and not candidate.startswith("#"):
                    faculty_name = normalize_name(candidate)
                    break

        faculty_name = self.resolver.resolve(faculty_name)

        email = ""
        for line in lines:
            if "@" in line and "." in line:
                m = re.search(r'[\w\.\-]+@[\w\.\-]+\.\w+', line)
                if m:
                    email = m.group(0)
                    break

        designation = "Faculty Member"
        for line in lines:
            if any(kw in line for kw in ["Professor", "Lecturer"]):
                desg = line.replace("-", "").strip()
                if len(desg) < 80:
                    designation = desg
                    break

        edu_match = re.search(r'##### Education Qualification\n+(.*?)(?=\n+#####|$)', content, re.DOTALL)
        education = edu_match.group(1).strip()[:500] if edu_match else ""

        # Extract research experience (postdocs, international positions)
        research_experience = ""
        exp_match = re.search(
            r'(?:Research Experience|Research\s*experience|Post[- ]?[Dd]octoral|Work Experience)'
            r'\s*\n+(.*?)(?=\n+#####|\n+Teaching|\n+Research Interests|$)',
            content, re.DOTALL | re.IGNORECASE
        )
        if exp_match:
            research_experience = exp_match.group(1).strip()[:600]

        interests_match = re.search(
            r'Research Interests:\s*\n*(.*?)(?=\n+Teaching Interests:|\n+#####|$)', content, re.DOTALL)
        research_interests = []
        if interests_match:
            interests_str = interests_match.group(1).strip()
            research_interests = [i.strip() for i in re.split(r'[,;\n]', interests_str)
                                  if i.strip() and len(i.strip()) > 3]

        self._add_node(faculty_name, "Faculty", name=faculty_name, email=email,
            designation=designation, education=education,
            research_experience=research_experience, source_file=filename)
        self._add_edge(faculty_name, doc_id, "PROFILE_DOCUMENT")

        for interest in research_interests[:10]:
            interest_clean = interest.strip().rstrip('.')
            if len(interest_clean) > 5:
                self._add_node(interest_clean, "ResearchArea", name=interest_clean)
                self._add_edge(faculty_name, interest_clean, "RESEARCHES_IN")

    def _parse_phd_list(self, filename: str, content: str, doc_id: str):
        student_blocks = re.findall(
            r'#### ([^\n]+)\n+(.*?)(?=#### |Source URL:|$)', content, re.DOTALL)

        for name, block in student_blocks:
            student_name = name.strip()
            if not student_name or len(student_name) < 2:
                continue

            sup_match = re.search(r'\*\*Supervisor\*\*\n+([^\n]+)', block, re.IGNORECASE)
            supervisors = []
            if sup_match:
                sup_text = sup_match.group(1).strip()
                sup_text = TITLE_PREFIXES.sub('', sup_text)
                raw_sups = re.split(r'\s*(?:,\s*|\s+[Aa]nd\s+|\s+\&\s+)\s*', sup_text)
                supervisors = [self.resolver.resolve(s) for s in raw_sups
                              if s.strip() and len(s.strip()) > 2]

            area_match = re.search(r'Research Area\n+([^\n]+)', block, re.IGNORECASE)
            research_area = area_match.group(1).strip() if area_match else ""

            self._add_node(student_name, "PhDStudent", name=student_name,
                research_area=research_area, source_file=filename)
            self._add_edge(student_name, doc_id, "SOURCE_DOCUMENT")

            for sup in supervisors:
                if sup:
                    # Only tag as Faculty if they are in the canonical list
                    if self.resolver.is_canonical_faculty(sup):
                        self._add_node(sup, "Faculty", name=sup)
                    else:
                        self._add_node(sup, "ExternalPerson", name=sup)
                    self._add_edge(student_name, sup, "SUPERVISED_BY")

            if research_area and research_area != "Unknown":
                self._add_node(research_area, "ResearchArea", name=research_area)
                self._add_edge(student_name, research_area, "STUDIES")

    def _parse_funded_projects(self, filename: str, content: str, doc_id: str):
        # First, join multi-line wrapped entries into single logical lines.
        # Each entry starts with "- [N]" or "[N]". Continuation lines that
        # don't start with "- [" or "#" or are blank are joined.
        raw_lines = content.splitlines()
        logical_lines = []
        current = ""
        for line in raw_lines:
            stripped = line.strip()
            if re.match(r'^- \[\d+\]', stripped) or re.match(r'^\[\d+\]', stripped):
                if current:
                    logical_lines.append(current)
                current = stripped
            elif current and stripped and not stripped.startswith('#') and not stripped.startswith('- ['):
                # Continuation of the current entry
                current += " " + stripped
            else:
                if current:
                    logical_lines.append(current)
                    current = ""
        if current:
            logical_lines.append(current)

        for line in logical_lines:
            # Extract project serial number [N]
            num_m = re.match(r'(?:- )?\[(\d+)\]', line)
            project_number = num_m.group(1) if num_m else ""

            # Match: - [N] Title: ..., Funding Agency: ...
            m = re.search(
                r'(?:- \[\d+\]|\[\d+\])\s*(?:Title:\s*)?(.+?)\s*,\s*(?:Funding Agency:\s*)(.+?)\.?\s*$',
                line, re.IGNORECASE
            )
            if not m:
                # Also handle entries like "[7]NRB DRDO Project : ..."
                m2 = re.search(r'(?:- \[\d+\]|\[\d+\])\s*(.+?)$', line)
                if m2:
                    text = m2.group(1).strip().rstrip('.')
                    # Try to split on "Funding Agency:" if present
                    fa_split = re.split(r',?\s*Funding Agency:\s*', text, maxsplit=1, flags=re.IGNORECASE)
                    if len(fa_split) == 2:
                        title = fa_split[0].replace('Title:', '').strip().rstrip(',')
                        agency = fa_split[1].strip().rstrip('.')
                    else:
                        # Entry without explicit Funding Agency label
                        title = text.replace('Title:', '').strip()
                        agency = ""
                    if title:
                        proj_id = f"project:{title[:60]}"
                        self._add_node(proj_id, "Project", title=title, agency=agency,
                                       project_number=project_number, source_file=filename)
                        self._add_edge(proj_id, doc_id, "SOURCE_DOCUMENT")
                        if agency:
                            agency_id = f"agency:{agency}"
                            self._add_node(agency_id, "FundingAgency", name=agency)
                            self._add_edge(proj_id, agency_id, "FUNDED_BY")
                continue

            title, agency = m.group(1).strip(), m.group(2).strip()
            title = title.rstrip(',').strip()
            agency = agency.rstrip('.').strip()
            proj_id = f"project:{title[:60]}"
            self._add_node(proj_id, "Project", title=title, agency=agency,
                           project_number=project_number, source_file=filename)
            self._add_edge(proj_id, doc_id, "SOURCE_DOCUMENT")
            agency_id = f"agency:{agency}"
            self._add_node(agency_id, "FundingAgency", name=agency)
            self._add_edge(proj_id, agency_id, "FUNDED_BY")

    def _parse_patents(self, filename: str, content: str, doc_id: str):
        patents = re.findall(
            r'\*\*Title\*\*\s*:\s*(.*?)\n+\*\*Inventors?\*\*\s*:\s*(.*?)\n+'
            r'\*\*Application No\*\*.*?:\s*(.*?)(?=\*\*Title\*\*|\Z)', content, re.DOTALL)
        for title, inventors_str, app_no in patents:
            patent_title = title.strip().replace('\n', ' ')
            patent_id = f"patent:{patent_title[:60]}"
            self._add_node(patent_id, "Patent", title=patent_title,
                application_no=app_no.strip().split('\n')[0], source_file=filename)
            self._add_edge(patent_id, doc_id, "SOURCE_DOCUMENT")
            for inv in re.split(r'[,;]|\band\b', inventors_str, flags=re.IGNORECASE):
                inv_name = self.resolver.resolve(inv)
                if inv_name and len(inv_name) > 2:
                    if self.resolver.is_canonical_faculty(inv_name):
                        self._add_node(inv_name, "Faculty", name=inv_name)
                    else:
                        self._add_node(inv_name, "ExternalPerson", name=inv_name)
                    self._add_edge(inv_name, patent_id, "INVENTED")

    def _parse_startups(self, filename: str, content: str, doc_id: str):
        startup_patterns = [
            {"name": "Data Sailors", "mentor": "Ankit Dubey",
             "desc": "Focuses on resource monitoring, data analytics, and forecasting using AI/ML. Incubated at IIT Jammu."},
            {"name": "Servotech Private Limited", "mentor": "Sudhakar Modem",
             "desc": "Commercialization of oxygen concentrators and other clean energy projects."},
        ]
        for s in startup_patterns:
            mentor = self.resolver.resolve(s["mentor"])
            self._add_node(s["name"], "Startup", name=s["name"],
                description=s["desc"], source_file=filename)
            self._add_edge(s["name"], doc_id, "SOURCE_DOCUMENT")
            self._add_node(mentor, "Faculty", name=mentor)
            self._add_edge(mentor, s["name"], "MENTORED_STARTUP")

    def _parse_research_areas(self, filename: str, content: str, doc_id: str):
        current_category = None
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('## ') and not line.startswith('## Research'):
                current_category = line[3:].strip()
                cat_id = f"category:{current_category}"
                self._add_node(cat_id, "ResearchCategory",
                    name=current_category, source_file=filename)
                self._add_edge(cat_id, doc_id, "SOURCE_DOCUMENT")
            elif line.startswith('- ') and current_category:
                sub_area = line[2:].strip()
                if sub_area and len(sub_area) > 5:
                    self._add_node(sub_area, "ResearchArea",
                        name=sub_area, category=current_category)
                    cat_id = f"category:{current_category}"
                    self._add_edge(sub_area, cat_id, "BELONGS_TO_CATEGORY")

    def _parse_faculty_list(self, filename: str, content: str, doc_id: str):
        faculty_entries = re.findall(
            r'####\s*\[([^\]]+)\]\(([^\)]+)\)\s*\n+\s*(.*?)(?=####|\Z)', content, re.DOTALL)
        for idx, (name, profile_url, snippet) in enumerate(faculty_entries, start=1):
            faculty_name = self.resolver.resolve(name.strip())
            designation = ""
            for kw in ["Professor", "Lecturer"]:
                if kw in snippet:
                    desg_match = re.search(rf'({kw}[^\n]*)', snippet)
                    if desg_match:
                        designation = desg_match.group(1).strip()[:60]
                        break
            self._add_node(faculty_name, "Faculty", name=faculty_name,
                profile_url=profile_url, faculty_order=idx)
            if designation:
                self.graph.nodes[faculty_name]['designation'] = designation
            self._add_edge(faculty_name, doc_id, "SOURCE_DOCUMENT")

    def _parse_hod(self, filename: str, content: str, doc_id: str):
        hod_match = re.search(r'###\s*(?:Dr\.?|Prof\.?)\s*([\w\s]+?)\n+\s*Head of Department', content)
        if not hod_match:
            hod_match = re.search(r'(?:Dr\.?|Prof\.?)\s*([\w\s]+?)\n+\s*Head', content)
        if not hod_match:
            hod_match = re.search(r'##\s*(?:Dr\.?|Prof\.?)\s*([\w\s]+?)\n', content)
        if hod_match:
            hod_name = self.resolver.resolve(hod_match.group(1).strip())
            self._add_node(hod_name, "Faculty", name=hod_name, is_hod=True,
                          designation="Head of Department (HoD), Associate Professor")
            self._add_edge(hod_name, doc_id, "SOURCE_DOCUMENT")
            dept_id = "IIT Jammu EE Department"
            self._add_node(dept_id, "Department",
                name="Department of Electrical Engineering", institution="IIT Jammu")
            self._add_edge(hod_name, dept_id, "HOD_OF")

            # Extract official HoD email and store on department node
            hod_email_match = re.search(r'hod\.ee@iitjammu\.ac\.in', content)
            if hod_email_match:
                self.graph.nodes[dept_id]['hod_official_email'] = 'hod.ee@iitjammu.ac.in'

    def _parse_placement_data(self, filename: str, content: str, doc_id: str):
        """Parse placement industry data into structured PlacementData nodes."""
        # The table format:
        # | UG | 37.5 | 14.46 | 41.5 | 5 | 87.88 | 20.22 | 53 | 8 |
        # Columns: Program | %_23-24 | Mean_23-24 | Max_23-24 | Min_23-24 | %_22-23 | Mean_22-23 | Max_22-23 | Min_22-23
        for line in content.splitlines():
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) < 8:
                continue
            program = cells[0].strip()
            if program in ('UG', 'M.Tech (CSP)', 'M.Tech (VLSI)'):
                try:
                    # 2023-24 data: columns 1-4
                    pct_2324 = cells[1].strip()
                    mean_2324 = cells[2].strip()
                    max_2324 = cells[3].strip()
                    min_2324 = cells[4].strip()
                    # 2022-23 data: columns 5-8
                    pct_2223 = cells[5].strip()
                    mean_2223 = cells[6].strip()
                    max_2223 = cells[7].strip() if len(cells) > 7 else 'NA'
                    min_2223 = cells[8].strip() if len(cells) > 8 else 'NA'

                    # Create structured node for 2023-24
                    node_id_2324 = f"placement:{program}:2023-24"
                    self._add_node(node_id_2324, "PlacementData",
                        name=f"{program} Placement 2023-24",
                        program=program, year="2023-24",
                        percentage=pct_2324,
                        mean_salary=mean_2324,
                        max_salary=max_2324,
                        min_salary=min_2324,
                        source_file=filename)
                    self._add_edge(node_id_2324, doc_id, "SOURCE_DOCUMENT")

                    # Create structured node for 2022-23
                    if pct_2223 != 'NA':
                        node_id_2223 = f"placement:{program}:2022-23"
                        self._add_node(node_id_2223, "PlacementData",
                            name=f"{program} Placement 2022-23",
                            program=program, year="2022-23",
                            percentage=pct_2223,
                            mean_salary=mean_2223,
                            max_salary=max_2223,
                            min_salary=min_2223,
                            source_file=filename)
                        self._add_edge(node_id_2223, doc_id, "SOURCE_DOCUMENT")
                except (IndexError, ValueError):
                    continue

    def _parse_higher_studies(self, filename: str, content: str, doc_id: str):
        """Parse higher studies (placement-academia) data into structured nodes."""
        for line in content.splitlines():
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) < 8:
                continue
            if cells[0] == 'Higher Studies':
                try:
                    # 2023-24: B.Tech, M.Tech(CSP), M.Tech(VLSI), PhD
                    # 2022-23: B.Tech EE, M.Tech CSP, M.Tech VLSI, PhD
                    hs_2324 = {
                        'B.Tech': cells[1],
                        'M.Tech (CSP)': cells[2],
                        'M.Tech (VLSI)': cells[3],
                        'PhD': cells[4]
                    }
                    hs_2223 = {
                        'B.Tech': cells[5],
                        'M.Tech (CSP)': cells[6],
                        'M.Tech (VLSI)': cells[7],
                        'PhD': cells[8] if len(cells) > 8 else '0'
                    }

                    node_id_2324 = "higher_studies:2023-24"
                    self._add_node(node_id_2324, "HigherStudiesData",
                        name="Higher Studies 2023-24", year="2023-24",
                        btech=hs_2324['B.Tech'],
                        mtech_csp=hs_2324['M.Tech (CSP)'],
                        mtech_vlsi=hs_2324['M.Tech (VLSI)'],
                        phd=hs_2324['PhD'],
                        source_file=filename)
                    self._add_edge(node_id_2324, doc_id, "SOURCE_DOCUMENT")

                    node_id_2223 = "higher_studies:2022-23"
                    self._add_node(node_id_2223, "HigherStudiesData",
                        name="Higher Studies 2022-23", year="2022-23",
                        btech=hs_2223['B.Tech'],
                        mtech_csp=hs_2223['M.Tech (CSP)'],
                        mtech_vlsi=hs_2223['M.Tech (VLSI)'],
                        phd=hs_2223['PhD'],
                        source_file=filename)
                    self._add_edge(node_id_2223, doc_id, "SOURCE_DOCUMENT")
                except (IndexError, ValueError):
                    continue

    def _parse_labs(self, content: str, doc_id: str):
        lab_patterns = re.findall(r'####\s+(.*?Lab.*?)\n', content, re.IGNORECASE)
        seen = set()
        for lab_name in lab_patterns:
            lab_name = lab_name.strip()
            # Remove markdown link syntax from lab names
            lab_name = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', lab_name)
            if lab_name in seen or len(lab_name) < 5:
                continue
            seen.add(lab_name)
            lab_id = f"lab:{lab_name}"
            self._add_node(lab_id, "Lab", name=lab_name)
            self._add_edge(lab_id, doc_id, "SOURCE_DOCUMENT")

    def build(self) -> nx.DiGraph:
        if not os.path.exists(self.markdown_dir):
            raise FileNotFoundError(f"Markdown directory not found: {self.markdown_dir}")

        filenames = [f for f in os.listdir(self.markdown_dir)
                     if f.endswith(".md") and f != "00_combined_ee_site.md"
                     and not f.endswith(".json")]

        logger.info(f"Processing {len(filenames)} markdown files...")

        doc_map = {}
        for filename in sorted(filenames):
            filepath = os.path.join(self.markdown_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            doc_id = self._create_document_node(filename, content)
            doc_map[filename] = (doc_id, content)

        logger.info(f"Phase 1: Created {len(doc_map)} document nodes with text chunks.")

        # Parse faculty list FIRST to register canonical names
        if "ee_faculty-list.html.md" in doc_map:
            doc_id, content = doc_map["ee_faculty-list.html.md"]
            self._parse_faculty_list("ee_faculty-list.html.md", content, doc_id)
            logger.info("Phase 1.5: Parsed faculty list (canonical registry populated).")

        for filename, (doc_id, content) in doc_map.items():
            if "faculty.html_faculty__" in filename:
                self._parse_faculty_profile(filename, content, doc_id)
            elif filename == "ee_phd-list.html.md":
                self._parse_phd_list(filename, content, doc_id)
            elif filename == "ee_funded-projects.html.md":
                self._parse_funded_projects(filename, content, doc_id)
            elif filename == "ee_patent.html.md":
                self._parse_patents(filename, content, doc_id)
            elif filename == "ee_startups.html.md":
                self._parse_startups(filename, content, doc_id)
            elif filename == "ee_research-areas.html.md":
                self._parse_research_areas(filename, content, doc_id)
            elif filename == "ee_faculty-list.html.md":
                pass  # Already parsed above
            elif filename == "ee_hod.html.md":
                self._parse_hod(filename, content, doc_id)
            elif filename == "ee_placement-industry.html.md":
                self._parse_placement_data(filename, content, doc_id)
            elif filename == "ee_placement-academia.html.md":
                self._parse_higher_studies(filename, content, doc_id)
            if filename in ("ee.md", "ee_index.html.md"):
                self._parse_labs(content, doc_id)

        logger.info(f"Phase 2: Entity extraction complete.")

        # Cross-link faculty to research areas from PhD supervisions
        faculty_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('label') == 'Faculty']
        for faculty in faculty_nodes:
            students = [e[0] for e in self.graph.in_edges(faculty)
                       if self.graph.edges[e[0], faculty].get('type') == 'SUPERVISED_BY']
            for student in students:
                for _, target in self.graph.out_edges(student):
                    if self.graph.nodes[target].get('label') == 'ResearchArea':
                        if not self.graph.has_edge(faculty, target):
                            self._add_edge(faculty, target, "RESEARCHES_IN")

        dept_id = "IIT Jammu EE Department"
        if not self.graph.has_node(dept_id):
            self._add_node(dept_id, "Department",
                name="Department of Electrical Engineering",
                institution="IIT Jammu", website="https://iitjammu.ac.in/ee")

        # Store official faculty count on department node
        self.graph.nodes[dept_id]['faculty_count'] = len(faculty_nodes)
        self.graph.nodes[dept_id]['phd_student_count'] = sum(
            1 for _, data in self.graph.nodes(data=True) if data.get('label') == 'PhDStudent'
        )

        # Only link actual Faculty (not ExternalPerson) to department
        for faculty in faculty_nodes:
            self._add_edge(faculty, dept_id, "MEMBER_OF")

        label_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            label_counts[data.get('label', 'Unknown')] += 1

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        logger.info(f"Node types: {dict(label_counts)}")
        return self.graph

    def save(self, output_dir: str = DATA_DIR):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "graph.pkl"), "wb") as f:
            pickle.dump(self.graph, f)
        chunks_data = [{"id": c[0], "text": c[1], "metadata": c[2]} for c in self.chunks]
        with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)
        with open(os.path.join(output_dir, "resolver.pkl"), "wb") as f:
            pickle.dump(self.resolver, f)
        logger.info(f"Graph saved to {output_dir}")

    @staticmethod
    def load(data_dir: str = DATA_DIR):
        with open(os.path.join(data_dir, "graph.pkl"), "rb") as f:
            graph = pickle.load(f)
        with open(os.path.join(data_dir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return graph, chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    builder = KnowledgeGraphBuilder()
    G = builder.build()
    builder.save()
    print(f"\n✅ Knowledge Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Chunks generated: {len(builder.chunks)}")
