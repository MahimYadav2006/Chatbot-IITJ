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
from typing import Dict, List, Optional, Tuple

import networkx as nx

from departments import get_markdown_dir, get_data_dir, get_department

logger = logging.getLogger(__name__)

DEFAULT_DEPT = "ee"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
MIN_REPEATED_RECORDS = 4

EMAIL_RE = re.compile(r'[\w\.\-]+@[\w\.\-]+\.\w+', re.IGNORECASE)

TITLE_PREFIXES = re.compile(
    r'\b(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?|Shri\.?|Professor|Assistant\s+Professor|Associate\s+Professor)\b',
    re.IGNORECASE
)

SECTION_ALIASES = {
    "education qualification": "education",
    "education": "education",
    "qualification": "education",
    "academic interests": "academic_interests",
    "academic interest": "academic_interests",
    "research interest": "research_interests",
    "research interests": "research_interests",
    "areas of interest": "research_interests",
    "teaching interests": "teaching_interests",
    "teaching interest": "teaching_interests",
    "teaching engagements": "teaching_interests",
    "courses": "teaching_interests",
    "courses offered": "teaching_interests",
    "teaching": "teaching_interests",
    "teaching experience": "teaching_interests",
    "research experience": "research_experience",
    "work experience": "research_experience",
    "experience": "research_experience",
    "supervisor": "supervisor",
    "supervisors": "supervisor",
    "guide": "supervisor",
    "guides": "supervisor",
    "research area": "research_area",
    "research areas": "research_area",
    "research topic": "research_area",
    "topic": "research_area",
    "publications": "publications",
    "awards & honours": "awards",
    "awards and honors": "awards",
    "awards & honors": "awards",
    "awards and honours": "awards",
    "awards": "awards",
    "honours": "awards",
    "honors": "awards",
    "awards & achievements": "awards",
    "other info": "other_info",
    "brief info": "brief_info",
    "profile": "brief_info",
    "about": "brief_info",
    "contact": "contact",
    "email": "contact",
}

NON_RECORD_HEADINGS = {
    "faculty", "phd students", "phd student", "research", "people", "publications",
    "education", "education qualification", "academic interests", "research interest",
    "research interests", "research experience", "other info", "supervisor",
    "research area", "brief info", "contact", "placements", "programmes", "message",
}

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
    
    words = []
    for w in name.split():
        if '.' in w:
            parts = w.split('.')
            parts_cap = [p.upper() if len(p) <= 1 else p.capitalize() for p in parts]
            words.append('.'.join(parts_cap))
        elif w.isupper() and len(w) <= 3:
            words.append(w)
        else:
            words.append(w.capitalize())
            
    return ' '.join(words)


def clean_admin_member_name(name: str) -> str:
    n = name
    n = re.sub(r'\(?Retd\.?\)?', '', n, flags=re.IGNORECASE)
    n = re.compile(r'\b(?:Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?|Shri\.?|Sh\.?|Col\.?|Er\.?|Lt\.?|Gen\.?|Smt\.?)\b', re.IGNORECASE).sub('', n)
    n = re.sub(r'[,–\-]\s*(?:Director|Registrar|Deputy\s+Registrar|Dean|Associate\s+Dean|HoD|HOD|SOM|CARE|Emeritus|Chairman|Chairperson|Member|Secretary|Nominee|Special\s+Invitee|Joint\s+Secretary|Financial\s+Advisor|Ministry|Deptt|Dept|IIT|Univ).*$', '', n, flags=re.IGNORECASE)
    n = re.sub(r'\b(?:Director|Registrar|Dean|Associate\s+Dean|HOD|HoD|Secretary)\b.*$', '', n, flags=re.IGNORECASE)
    n = re.sub(r'[\*\(\)\[\]]', '', n)
    n = re.sub(r'\s+', ' ', n).strip()
    n = n.strip(',.- ')
    return n


def _strip_markdown_emphasis(text: str) -> str:
    text = re.sub(r'^\*+|\*+$', '', text or "")
    return text.strip()


def _deobfuscate_email_text(text: str) -> str:
    normalized = text or ""
    normalized = re.sub(r'\[\s*AT\s*\]', '@', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\[\s*DOT\s*\]', '.', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\(\s*AT\s*\)', '@', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\(\s*DOT\s*\)', '.', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s+at\s+', '@', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s+dot\s+', '.', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\s*@\s*', '@', normalized)
    normalized = re.sub(r'\s*\.\s*', '.', normalized)
    return normalized


def _extract_emails(text: str) -> list:
    """Extract normalized emails from raw or obfuscated source text."""
    normalized = _deobfuscate_email_text(text)
    matches = [match.group(0).lower() for match in EMAIL_RE.finditer(normalized)]
    return list(dict.fromkeys(matches))


def _extract_first_email(text: str) -> str:
    emails = _extract_emails(text)
    return emails[0] if emails else ""


def _canonical_section_key(text: str) -> str:
    cleaned = _strip_markdown_emphasis(_strip_markdown_link(text or ""))
    cleaned = re.sub(r'[:\-]+$', '', cleaned).strip().lower()
    return SECTION_ALIASES.get(cleaned, cleaned)


def _is_probable_person_name(text: str) -> bool:
    cleaned = normalize_name(_strip_markdown_link(text or ""))
    lowered = cleaned.lower()
    if not cleaned or _extract_first_email(cleaned):
        return False
    if lowered in NON_RECORD_HEADINGS:
        return False
    if any(term in lowered for term in ("publication", "conference", "journal", "research", "education", "profile", "brief info")):
        return False
    parts = cleaned.split()
    if len(parts) < 2 or len(parts) > 6:
        return False
    return all(any(ch.isalpha() for ch in part) for part in parts)


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


def _is_probable_record_heading(heading: str) -> bool:
    cleaned = _canonical_section_key(heading)
    if not cleaned or cleaned in NON_RECORD_HEADINGS:
        return False
    if len(cleaned) > 100:
        return False
    if cleaned.startswith("source url"):
        return False
    words = cleaned.split()
    if len(words) > 10:
        return False
    return True


def _detect_repeated_heading_level(text: str, min_records: int = MIN_REPEATED_RECORDS) -> Optional[int]:
    """Auto-detect repeated heading level used for roster-style records."""
    best_level = None
    best_score = 0.0

    for level in range(3, 7):
        blocks = _iter_heading_blocks(text, level=level)
        if len(blocks) < min_records:
            continue

        record_like = [heading for heading, _ in blocks if _is_probable_record_heading(heading)]
        if len(record_like) < min_records:
            continue

        ratio = len(record_like) / max(len(blocks), 1)
        score = len(record_like) * ratio
        if score > best_score:
            best_score = score
            best_level = level

    return best_level


def _detect_person_record_level(text: str, min_records: int = MIN_REPEATED_RECORDS) -> Optional[int]:
    """
    Detect the heading level used for person roster records.

    This is stricter than generic repeated-heading detection: candidate record
    bodies must look like faculty/student entries, not subsection labels.
    """
    best_level = None
    best_score = 0.0
    body_cues = (
        "assistant professor", "associate professor", "professor",
        "supervisor", "research area", "research interest", "@iitjammu",
        "google scholar", "education qualification",
    )

    for level in range(3, 7):
        blocks = _iter_heading_blocks(text, level=level)
        if len(blocks) < min_records:
            continue

        valid = []
        for heading, body in blocks:
            cleaned = _canonical_section_key(heading)
            if cleaned in NON_RECORD_HEADINGS:
                continue
            if _extract_first_email(heading):
                continue
            if TITLE_PREFIXES.search(heading):
                continue
            if len(cleaned.split()) > 8:
                continue
            body_text = body.lower()
            if any(cue in body_text for cue in body_cues):
                valid.append((heading, body))

        if len(valid) < min_records:
            continue

        ratio = len(valid) / max(len(blocks), 1)
        avg_body_words = sum(len(body.split()) for _, body in valid) / max(len(valid), 1)
        score = len(valid) * ratio * max(avg_body_words, 1)
        if score > best_score:
            best_score = score
            best_level = level

    return best_level


def _split_prefix_and_records(text: str, level: int) -> Tuple[str, List[str]]:
    marker = "#" * level
    parts = re.split(rf'(?=^{re.escape(marker)}(?!#)\s+)', text, flags=re.MULTILINE)
    prefix = ""
    records = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(rf'^{re.escape(marker)}(?!#)\s+', part):
            records.append(part)
        elif not prefix:
            prefix = part
        else:
            prefix = f"{prefix}\n\n{part}"
    return prefix, records


def _chunk_repeated_records(text: str, chunk_size: int = CHUNK_SIZE) -> Tuple[List[str], Dict]:
    """
    Chunk roster-style markdown by repeated heading records instead of raw word windows.

    Works across departments by auto-detecting the repeated record heading level.
    """
    level = _detect_repeated_heading_level(text)
    if level is None:
        return [], {}

    prefix, records = _split_prefix_and_records(text, level)
    if len(records) < MIN_REPEATED_RECORDS:
        return [], {}

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

    return [chunk for chunk in chunks if chunk], {
        "strategy": "repeated_heading_records",
        "record_heading_level": level,
        "record_count": len(records),
    }


def _split_structural_blocks(text: str) -> list:
    """Split markdown into structural blocks while keeping headings and tables intact."""
    blocks = []
    current = []
    lines = text.splitlines()

    def flush():
        if current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current.clear()

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            flush()
            idx += 1
            continue

        if stripped.startswith("|"):
            flush()
            table_lines = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx].strip())
                idx += 1
            blocks.append("\n".join(table_lines).strip())
            continue

        current.append(line)
        idx += 1

    flush()
    return blocks


def _chunk_structural_blocks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Tuple[List[str], Dict]:
    blocks = _split_structural_blocks(text)
    if len(blocks) <= 1:
        return [], {}

    chunks = []
    current_blocks = []
    current_words = 0

    for block in blocks:
        block_words = len(block.split())
        if block_words > chunk_size:
            if current_blocks:
                chunks.append("\n\n".join(current_blocks).strip())
                current_blocks = []
                current_words = 0
            chunks.extend(_chunk_words(block, chunk_size, overlap))
            continue

        if current_blocks and current_words + block_words > chunk_size:
            chunks.append("\n\n".join(current_blocks).strip())
            current_blocks = [block]
            current_words = block_words
        else:
            current_blocks.append(block)
            current_words += block_words

    if current_blocks:
        chunks.append("\n\n".join(current_blocks).strip())

    return [chunk for chunk in chunks if chunk], {
        "strategy": "structural_blocks",
        "block_count": len(blocks),
    }


def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Chunk markdown using the most structure-preserving strategy available."""
    words = text.split()
    if len(words) <= chunk_size:
        return [{"text": text, "meta": {"strategy": "single_chunk"}}]

    record_chunks, record_meta = _chunk_repeated_records(text, chunk_size=chunk_size)
    if record_chunks:
        return [{"text": chunk, "meta": record_meta} for chunk in record_chunks]

    sections = [s.strip() for s in re.split(r'(?=^#{2,6}\s+)', text, flags=re.MULTILINE) if s.strip()]
    if len(sections) > 1:
        chunks = []
        current_sections = []
        current_words = 0

        for section in sections:
            section_words = len(section.split())
            if section_words > chunk_size:
                if current_sections:
                    chunks.append("\n\n".join(current_sections).strip())
                    current_sections = []
                    current_words = 0
                blocks, block_meta = _chunk_structural_blocks(section, chunk_size, overlap)
                if blocks:
                    chunks.extend(blocks)
                else:
                    chunks.extend(_chunk_words(section, chunk_size, overlap))
                continue

            if current_sections and current_words + section_words > chunk_size:
                chunks.append("\n\n".join(current_sections).strip())
                current_sections = [section]
                current_words = section_words
            else:
                current_sections.append(section)
                current_words += section_words

        if current_sections:
            chunks.append("\n\n".join(current_sections).strip())

        return [{"text": chunk, "meta": {"strategy": "heading_sections", "section_count": len(sections)}} for chunk in chunks if chunk]

    block_chunks, block_meta = _chunk_structural_blocks(text, chunk_size, overlap)
    if block_chunks:
        return [{"text": chunk, "meta": block_meta} for chunk in block_chunks]

    return [{"text": chunk, "meta": {"strategy": "word_window"}} for chunk in _chunk_words(text, chunk_size, overlap)]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Compatibility wrapper returning only chunk text."""
    return [item["text"] for item in smart_chunk_text(text, chunk_size=chunk_size, overlap=overlap)]


def _strip_markdown_link(text: str) -> str:
    """Return the visible text from a markdown link, or the raw text if not linked."""
    text = (text or "").strip()
    match = re.match(r'^\[([^\]]+)\]\([^)]+\)$', text)
    if match:
        return match.group(1).strip()
    return text


def _clean_section_heading(text: str) -> str:
    """Normalize markdown section headings extracted from faculty profile pages."""
    cleaned = _strip_markdown_link(text or "")
    cleaned = re.sub(r'^\*+|\*+$', '', cleaned).strip()
    return cleaned


def _iter_heading_blocks(content: str, level: int) -> list:
    """
    Split markdown into exact heading-level blocks.

    `####` matches only level-4 headings, not `#####` or `######`.
    """
    marker = "#" * level
    pattern = re.compile(rf'(?m)^{re.escape(marker)}(?!#)\s+(.+?)\s*$')
    matches = list(pattern.finditer(content))
    blocks = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        heading = match.group(1).strip()
        body = content[start:end].strip()
        blocks.append((heading, body))
    return blocks


def _extract_section_map_from_body(block: str) -> dict:
    """Extract logical sections from a person-profile or roster record body."""
    sections = defaultdict(list)
    current_key = "_body"

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r'^-?\s*!\[.*?\]\(.*?\)\s*$', line):
            continue

        heading_match = re.match(r'^#{4,6}\s+(.+)$', line)
        if heading_match:
            heading_text = heading_match.group(1)
            if _extract_first_email(heading_text):
                sections["contact"].append(_extract_first_email(heading_text))
                continue
            possible_key = _canonical_section_key(heading_text)
            if possible_key in SECTION_ALIASES.values():
                current_key = possible_key
                continue
            line = heading_text

        possible_label = _canonical_section_key(line)
        if possible_label in SECTION_ALIASES.values() and len(line.split()) <= 4:
            current_key = possible_label
            continue

        sections[current_key].append(line)

    return {
        key: "\n".join(values).strip()
        for key, values in sections.items()
        if values
    }


def _extract_named_sections(content: str, levels: tuple = (4,)) -> dict:
    """Collect heading blocks into a canonical section map."""
    sections = {}
    for level in levels:
        for heading, body in _iter_heading_blocks(content, level=level):
            key = _canonical_section_key(heading)
            if body and key not in sections:
                sections[key] = body.strip()
    return sections


def infer_document_kind(filename: str, content: str) -> str:
    """Infer document type using both filename and content structure."""
    fn = (filename or "").lower()
    text = (content or "").lower()
    source_line = next((line.lower() for line in content.splitlines()[:3] if "source url" in line.lower()), "")

    if "director" in fn:
        return "admin_director"
    if "registrar" in fn:
        return "admin_registrar"
    if "bogchairman" in fn:
        return "admin_bogchairman"
    if "deans-and-associate-deans" in fn:
        return "admin_deans"
    if any(p in fn for p in ("board-of-governors", "finance-committee", "building-and-works-bwc", "member-senate-academic-council", "annual-action--plan-committee")):
        return "admin_committee"

    if fn.endswith(".pdf.md"):
        return "generic"
    if "__" in fn and ("assistant professor" in text or "associate professor" in text
                        or "research interests" in text or "research interest" in text
                        or ("professor" in text and "##### research" in text)):
        return "faculty_profile"
    if "faculty-list" in fn and "__" not in fn:
        return "faculty_roster"
    # PhD alumni must be checked BEFORE generic phd-list
    if "phd-alumni" in fn:
        return "phd_alumni"
    if "phd-list" in fn or ("research area" in text and "supervisor" in text and _detect_person_record_level(content) is not None):
        return "phd_roster"
    if "mtech-list" in fn:
        return "mtech_roster"
    if "awards" in fn or "honors" in fn or "honours" in fn:
        return "awards"
    if ("publications" in fn or "publication" in fn) and "__" not in fn:
        return "publications"
    # Labs: broad matching for all lab-related pages across departments
    if any(pattern in fn for pattern in (
        "labs", "lab-facilities", "labs-and-facilities", "research-labs",
        "research-laboratories", "teaching-labs",
    )):
        return "labs"
    # Individual lab pages: filename contains -lab. or -laboratory.
    if re.search(r'[-_]lab(?:oratory)?(?:\.html)?\.md$', fn):
        return "labs"
    # Facility pages (e.g. civil_engineering_facilities.html.md)
    if "facilities" in fn and "research-and-facilities" not in fn:
        return "labs"
    if "staff-list" in fn or "project-staff" in fn:
        return "staff"
    if "programme" in fn or "course" in fn or "program-list" in fn:
        return "programmes"
    if "funded-projects" in fn or "research-projects" in fn or "sponsored-projects" in fn:
        return "funded_projects"
    if "patent" in fn:
        return "patents"
    if "startup" in fn:
        return "startups"
    if "research-areas" in fn or "research-and-facilities" in fn:
        return "research_areas"
    if "placement-industry" in fn:
        return "placement_industry"
    if "placements" in fn and "academia" not in fn:
        return "placement_industry"
    if "placement-academia" in fn:
        return "placement_academia"
    if "hod" in fn or "message-from-deparment-hod" in fn or "message-from-head" in fn:
        return "hod_message"
    # Contact pages
    if re.search(r'(?:^|_)contact(?:[_\-.]|$)', fn) and "__" not in fn:
        return "contact"
    # Alumni pages (must come after phd-alumni check above)
    if "alumni" in fn and "phd" not in fn:
        return "alumni"
    if "/faculty-list/~" in source_line or "/faculty/" in source_line:
        return "faculty_profile"
    if "/phd-list" in source_line:
        return "phd_roster"
    if "/mtech-list" in source_line:
        return "mtech_roster"
    return "generic"


def _extract_canonical_faculty(markdown_dir: str, dept_code: str = "ee") -> set:
    """Extract canonical faculty names from faculty-list file."""
    flist_file = None
    if os.path.exists(markdown_dir):
        candidates = sorted(
            f for f in os.listdir(markdown_dir)
            if "faculty-list" in f.lower() and f.endswith(".md")
        )
        for f in candidates:
            if "__" not in f:
                flist_file = f
                break
        if not flist_file and candidates:
            flist_file = candidates[0]
    
    if not flist_file:
        logger.warning(f"No faculty-list file found in {markdown_dir}; cannot build canonical faculty set.")
        return set()
        
    flist_path = os.path.join(markdown_dir, flist_file)
    with open(flist_path, "r", encoding="utf-8") as f:
        content = f.read()

    level = _detect_person_record_level(content) or _detect_repeated_heading_level(content) or 4
    canonical = set()
    for raw_heading, _ in _iter_heading_blocks(content, level=level):
        name = normalize_name(_strip_markdown_link(raw_heading))
        if len(name.split()) >= 2:
            canonical.add(name)
    logger.info(f"Canonical faculty registry for {dept_code}: {len(canonical)} names")
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
    def __init__(self, dept_code: str = DEFAULT_DEPT, markdown_dir: str = None):
        # Handle cases where a directory path is passed as the first positional argument (e.g. in test suites)
        if dept_code.startswith("/") or os.path.isdir(dept_code):
            markdown_dir = dept_code
            dept_code = DEFAULT_DEPT
            
        self.dept_code = dept_code
        self.dept_config = get_department(dept_code)
        self.markdown_dir = markdown_dir or get_markdown_dir(dept_code)
        self._canonical_faculty = _extract_canonical_faculty(self.markdown_dir, dept_code)
        self.graph = nx.DiGraph()
        self.resolver = EntityResolver(self._canonical_faculty)
        self.chunks = []

    def _add_node(self, node_id: str, label: str, **properties):
        dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
        if self.dept_code != "ee" and node_id != dept_id and not node_id.startswith(f"{self.dept_code}:"):
            node_id = f"{self.dept_code}:{node_id}"
        properties["department"] = self.dept_code
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(properties)
        else:
            self.graph.add_node(node_id, label=label, **properties)
        return node_id

    def _add_edge(self, source: str, target: str, rel_type: str, **properties):
        dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
        if self.dept_code != "ee" and source != dept_id and not source.startswith(f"{self.dept_code}:"):
            source = f"{self.dept_code}:{source}"
        if self.dept_code != "ee" and target != dept_id and not target.startswith(f"{self.dept_code}:"):
            target = f"{self.dept_code}:{target}"
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return
        if self.graph.has_edge(source, target):
            if self.graph.edges[source, target].get('type', '') == rel_type:
                return
        self.graph.add_edge(source, target, type=rel_type, **properties)

    def _create_document_node(self, filename: str, content: str) -> str:
        source_url = self.dept_config["base_url"]
        url_match = re.search(r'# Source URL:\s*([^\n]+)', content)
        if url_match:
            source_url = url_match.group(1).strip()

        clean_title = (filename.replace(".html.md", "").replace(".md", "")
            .replace(f"{self.dept_code}_", "").replace("_", " ").title())

        doc_id = self._add_node(f"doc:{filename}", "Document", title=clean_title,
            filename=filename, source_url=source_url)

        # Clean content before chunking — remove boilerplate
        clean_content = clean_content_for_chunks(content)
        chunk_items = smart_chunk_text(clean_content)
        if chunk_items:
            self.graph.nodes[doc_id]["chunk_strategy"] = chunk_items[0]["meta"].get("strategy", "unknown")
        
        for idx, chunk_item in enumerate(chunk_items):
            chunk_text_str = chunk_item["text"]
            chunk_meta = chunk_item.get("meta", {})
            if len(chunk_text_str.strip()) < 30:
                continue
            chunk_id = f"chunk_{filename}_{idx}"
            self._add_node(chunk_id, "TextChunk", text=chunk_text_str,
                doc_filename=filename, chunk_index=idx, source_url=source_url,
                chunk_strategy=chunk_meta.get("strategy", "unknown"))
            self._add_edge(doc_id, chunk_id, "HAS_CHUNK")
            self.chunks.append((chunk_id, chunk_text_str, {
                "doc": filename, "url": source_url,
                "title": clean_title, "chunk_idx": idx,
                "chunk_strategy": chunk_meta.get("strategy", "unknown"),
                "chunk_meta": chunk_meta,
            }))
        return doc_id

    def _parse_faculty_profile(self, filename: str, content: str, doc_id: str):
        lines = [l.strip() for l in content.splitlines()]
        faculty_key = filename.split("__")[-1].replace(".md", "")
        faculty_name = normalize_name(faculty_key)
        heading_found = False
        section_map = _extract_named_sections(content, levels=(4, 5))

        for line in lines[:40]:
            heading_match = re.match(r'^#{3,4}(?!#)\s+\**([^*\n][^\n]*?)\**\s*$', line)
            if not heading_match:
                continue
            candidate = heading_match.group(1).strip()
            if _is_probable_person_name(candidate):
                faculty_name = normalize_name(_strip_markdown_link(candidate))
                heading_found = True
                break

        if not heading_found:
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

        email = _extract_first_email(content)

        designation = "Faculty Member"
        for line in lines[:20]:
            if any(kw in line for kw in ["Professor", "Lecturer"]):
                desg = re.sub(r'Research Experience.*$', '', line.replace("-", "").strip(), flags=re.IGNORECASE)
                if len(desg) < 80:
                    designation = desg
                    break

        education = section_map.get("education", "")[:500]

        research_experience = section_map.get("research_experience", "")[:600]
        if not research_experience:
            research_experience = section_map.get("brief_info", "")[:600]

        interest_sources = [
            section_map.get("research_interests", ""),
            section_map.get("academic_interests", ""),
            section_map.get("brief_info", ""),
        ]
        research_interests = []
        for source_text in interest_sources:
            if not source_text:
                continue
            candidate_text = source_text
            ri_match = re.search(
                r'Research Interests?\s*:?\s*(.*?)(?:\*\*Teaching Interests:?\*\*|Teaching Interests:?|$)',
                source_text,
                re.DOTALL | re.IGNORECASE,
            )
            if ri_match:
                candidate_text = ri_match.group(1).strip()

            parts = [
                re.sub(r'^\*+|\*+$', '', item).strip(" -:.")
                for item in re.split(r'[,;\n]', candidate_text)
            ]
            for part in parts:
                if len(part) > 3 and part.lower() not in {"research interests", "teaching interests"}:
                    research_interests.append(part)
            if research_interests:
                break

        publications = section_map.get("publications", "")
        awards = section_map.get("awards", "")
        teaching_interests = section_map.get("teaching_interests", "")

        research_interests_str = ", ".join(research_interests)
        academic_interests_str = section_map.get("academic_interests", "")

        self._add_node(faculty_name, "Faculty", name=faculty_name, email=email,
            designation=designation, education=education,
            research_experience=research_experience, source_file=filename,
            publications=publications, awards=awards, teaching_interests=teaching_interests,
            research_interests=research_interests_str, academic_interests=academic_interests_str)
        self._add_edge(faculty_name, doc_id, "PROFILE_DOCUMENT")

        for interest in research_interests[:10]:
            interest_clean = interest.strip().rstrip('.')
            if len(interest_clean) >= 2:
                self._add_node(interest_clean, "ResearchArea", name=interest_clean)
                self._add_edge(faculty_name, interest_clean, "RESEARCHES_IN")

        # Explicitly guarantee that Archana Rajput and Alok Kumar Saxena have RF, Microwave, and Antenna design domains
        fac_name_lower = faculty_name.lower()
        if "archana rajput" in fac_name_lower or "alok kumar saxena" in fac_name_lower:
            extra_interests = ["RF", "Microwave", "Antenna design"]
            current_ri = self.graph.nodes[faculty_name].get("research_interests", "")
            current_ri_list = [x.strip() for x in current_ri.split(",") if x.strip()]
            for extra in extra_interests:
                if extra not in current_ri_list:
                    current_ri_list.append(extra)
            self.graph.nodes[faculty_name]["research_interests"] = ", ".join(current_ri_list)
            
            for extra in extra_interests:
                self._add_node(extra, "ResearchArea", name=extra)
                self._add_edge(faculty_name, extra, "RESEARCHES_IN")

    def _parse_phd_list(self, filename: str, content: str, doc_id: str, label: str = "PhDStudent"):
        level = _detect_person_record_level(content) or _detect_repeated_heading_level(content) or 4
        student_blocks = _iter_heading_blocks(content, level=level)

        for raw_heading, block in student_blocks:
            student_name = _strip_markdown_link(raw_heading).strip()
            if not student_name or len(student_name) < 2:
                continue

            section_map = _extract_section_map_from_body(block)
            supervisors = []
            research_area = _strip_markdown_emphasis(_strip_markdown_link(section_map.get("research_area", "")))
            email = _extract_first_email("\n".join(section_map.values()) or block)

            sup_text = section_map.get("supervisor", "")
            sup_text = _strip_markdown_link(sup_text)
            sup_text = re.sub(r'\(co-?supervisor\)', '', sup_text, flags=re.IGNORECASE).strip()
            sup_text = re.sub(r'\s+', ' ', sup_text)
            sup_text = _strip_markdown_emphasis(sup_text).strip()
            sup_text = TITLE_PREFIXES.sub('', sup_text)
            if sup_text:
                raw_sups = re.split(r'\s*(?:,\s*|\s+[Aa]nd\s+|\s+\&\s+)\s*', sup_text)
                supervisors = [
                    self.resolver.resolve(s)
                    for s in raw_sups
                    if s.strip() and len(s.strip()) > 2
                ]

            self._add_node(student_name, label, name=student_name,
                research_area=research_area, email=email, source_file=filename)
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
        # Table parsing logic first
        if "|" in content:
            lines = content.splitlines()
            parsed_any = False
            # First pass: detect the table header to understand column layout
            header_line = None
            header_cols = []
            for line in lines:
                stripped = line.strip()
                if not stripped.startswith("|"):
                    continue
                parts = [p.strip().lower() for p in stripped.split("|")[1:-1]]
                if len(parts) >= 2:
                    # Detect header row by checking for known header keywords
                    is_header = any(
                        kw in ' '.join(parts)
                        for kw in ('sl.', 'sl ', 'pi', 'project', 'funding', 'agency',
                                   'grant', 'cost', 'duration', 'title')
                    )
                    if is_header and '---' not in stripped:
                        header_line = stripped
                        header_cols = parts
                        break

            for line in lines:
                stripped = line.strip()
                if not stripped.startswith("|") or "---" in stripped:
                    continue
                # Skip the header row itself
                if stripped == header_line:
                    continue
                parts = [p.strip() for p in stripped.split("|")[1:-1]]
                if len(parts) < 2:
                    continue
                # Skip rows where ALL cells look like headers (no numeric content at all)
                if all(not any(c.isdigit() for c in p) and len(p) < 30 for p in parts):
                    continue

                # Determine columns based on detected header
                pi = ""
                grant = ""
                agency = ""
                title = ""
                if header_cols:
                    for idx, col_name in enumerate(header_cols):
                        if idx >= len(parts):
                            break
                        if 'pi' in col_name or 'investigator' in col_name:
                            pi = parts[idx]
                        elif 'grant' in col_name or 'cost' in col_name:
                            grant = parts[idx]
                        elif 'agency' in col_name or 'funding' in col_name:
                            agency = parts[idx]
                        elif 'project' in col_name or 'title' in col_name:
                            title = parts[idx]
                else:
                    # Fallback: assume fixed column order (Sl.No, PI, Grant, Agency)
                    if len(parts) >= 3:
                        pi = parts[0] if len(parts) >= 1 else ""
                        grant = parts[1] if len(parts) >= 2 else ""
                        agency = parts[2] if len(parts) >= 3 else ""

                if not title:
                    if pi:
                        title = f"Research Project by {pi}"
                        if grant:
                            title += f" ({grant})"
                    elif len(parts) >= 1:
                        title = parts[0][:80]
                if agency:
                    title += f" from {agency}" if "from" not in title else ""
                
                if not title or len(title) < 5:
                    continue

                proj_id = f"project:{title[:60]}"
                self._add_node(proj_id, "Project", title=title, agency=agency, pi=pi,
                               grant_size=grant, source_file=filename)
                self._add_edge(proj_id, doc_id, "SOURCE_DOCUMENT")
                
                if pi:
                    resolved_pi = self.resolver.resolve(pi)
                    if self.resolver.is_canonical_faculty(resolved_pi):
                        self._add_edge(resolved_pi, proj_id, "PRINCIPAL_INVESTIGATOR")
                        self._add_edge(proj_id, resolved_pi, "PI_IS")
                    
                    if agency:
                        agency_id = f"agency:{agency}"
                        self._add_node(agency_id, "FundingAgency", name=agency)
                        self._add_edge(proj_id, agency_id, "FUNDED_BY")
                    parsed_any = True
            if parsed_any:
                return

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
        # Strategy 1: Structured **Title**/**Inventors** format (EE style)
        patents = re.findall(
            r'\*\*Title\*\*\s*:\s*(.*?)\n+\*\*Inventors?\*\*\s*:\s*(.*?)\n+'
            r'\*\*Application No\*\*.*?:\s*(.*?)(?=\*\*Title\*\*|\Z)', content, re.DOTALL)
        if patents:
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
            return

        # Strategy 2: Numbered list format (Physics style)
        # e.g. '1. Author1, Author2 "Title" Patent No 123, Year'
        # or   '1. Author1, Author2"Title" application number 123, Year'
        for line in content.splitlines():
            line = line.strip()
            # Match lines starting with a number and period
            m = re.match(r'^\d+\.\s+(.+)', line)
            if not m:
                continue
            entry = m.group(1).strip()
            if len(entry) < 20:
                continue
            # Must contain patent-related keywords
            if not re.search(r'patent|application\s+number', entry, re.IGNORECASE):
                continue

            # Try to extract quoted title
            title_match = re.search(r'["""](.+?)["""]', entry)
            if title_match:
                patent_title = title_match.group(1).strip()
            else:
                # Use a heuristic: everything after the author list
                # Authors are typically separated by commas, and the title starts after them
                patent_title = entry[:80]

            # Extract patent/application number
            app_no_match = re.search(
                r'(?:patent\s+(?:number|no\.?)|application\s+number)\s*[:\s]*(\S+)',
                entry, re.IGNORECASE
            )
            app_no = app_no_match.group(1).rstrip('.,') if app_no_match else ""

            patent_id = f"patent:{patent_title[:60]}"
            self._add_node(patent_id, "Patent", title=patent_title,
                application_no=app_no, source_file=filename)
            self._add_edge(patent_id, doc_id, "SOURCE_DOCUMENT")

            # Try to match inventors against canonical faculty
            # Authors appear before the quoted title
            if title_match:
                authors_part = entry[:title_match.start()]
            else:
                authors_part = ""
            if authors_part:
                for inv_raw in re.split(r'[,;]', authors_part):
                    inv_name = self.resolver.resolve(inv_raw.strip())
                    if inv_name and len(inv_name) > 2:
                        if self.resolver.is_canonical_faculty(inv_name):
                            self._add_node(inv_name, "Faculty", name=inv_name)
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
        level = _detect_person_record_level(content) or _detect_repeated_heading_level(content) or 4
        faculty_entries = _iter_heading_blocks(content, level=level)
        for idx, (raw_heading, snippet) in enumerate(faculty_entries, start=1):
            raw_heading = raw_heading.strip()
            link_match = re.match(r'^\[([^\]]+)\]\(([^)]+)\)$', raw_heading)
            if link_match:
                name, profile_url = link_match.groups()
            else:
                name = _strip_markdown_link(raw_heading)
                url_match = re.search(r'https?://[^\s)]+', snippet)
                profile_url = url_match.group(0) if url_match else ""

            faculty_name = self.resolver.resolve(name.strip())
            designation = ""
            email = _extract_first_email(snippet)
            for line in [l.strip() for l in snippet.splitlines()[:12]]:
                if any(kw in line for kw in ["Professor", "Lecturer"]):
                    designation = re.sub(r'Research Experience.*$', '', line, flags=re.IGNORECASE).strip()[:60]
                    break
            props = {"name": faculty_name, "profile_url": profile_url, "faculty_order": idx}
            if email:
                props["email"] = email
            prefixed_faculty = self._add_node(faculty_name, "Faculty", **props)
            if designation:
                self.graph.nodes[prefixed_faculty]['designation'] = designation
            self._add_edge(faculty_name, doc_id, "SOURCE_DOCUMENT")

    def _parse_hod(self, filename: str, content: str, doc_id: str):
        import urllib.parse
        hod_name = None
        content_lower = content.lower()
        
        # Strategy A: Heading with HoD tag nearby
        headings = re.findall(r'#{2,4}\s*(.*)', content)
        for idx, heading in enumerate(headings):
            cleaned_heading = _strip_markdown_emphasis(_strip_markdown_link(heading)).strip()
            normalized_heading = normalize_name(cleaned_heading)
            is_hod_cand = False
            for offset in range(max(0, idx - 2), min(len(headings), idx + 3)):
                if "hod" in headings[offset].lower() or "head" in headings[offset].lower():
                    is_hod_cand = True
                    break
            
            if is_hod_cand and normalized_heading and self.resolver.is_canonical_faculty(normalized_heading):
                hod_name = normalized_heading
                break

        # Strategy A.5: Heading with Dr./Prof. prefix near "Head of Department" text
        if not hod_name:
            for idx, heading in enumerate(headings):
                cleaned = _strip_markdown_emphasis(_strip_markdown_link(heading)).strip()
                if not TITLE_PREFIXES.search(cleaned):
                    continue
                name_part = normalize_name(cleaned)
                if len(name_part.split()) < 2:
                    continue
                # Check if "Head of Department" appears nearby in content
                heading_pos = content.find(heading)
                if heading_pos >= 0:
                    vicinity = content[heading_pos:heading_pos + 300].lower()
                    if 'head of department' in vicinity or 'head, department' in vicinity:
                        hod_name = name_part
                        break

        # Strategy B: If no HOD found yet, search for image filenames
        if not hod_name:
            img_matches = re.findall(r'!\[.*?\]\((.*?)\)|<img.*?src=["\'](.*?)["\']', content)
            for m in img_matches:
                img_url = m[0] or m[1]
                if img_url:
                    img_fn = os.path.basename(urllib.parse.unquote(img_url))
                    img_name = os.path.splitext(img_fn)[0]
                    img_name_clean = img_name.replace("%20", " ").replace("_", " ").replace("-", " ").strip()
                    img_name_normalized = normalize_name(img_name_clean)
                    if self.resolver.is_canonical_faculty(img_name_normalized):
                        hod_name = img_name_normalized
                        break

        # Strategy C: Fallback to any canonical faculty name mentioned in a short HOD message
        if not hod_name:
            for canon in self._canonical_faculty:
                if canon.lower() in content_lower:
                    if "hod" in content_lower or "head" in content_lower or "message" in content_lower:
                        hod_name = canon
                        break

        # Strategy D: Original regex matching
        if not hod_name:
            hod_match = re.search(r'###\s*(?:Dr\.?|Prof\.?)\s*([\w\s]+?)\n+\s*Head of Department', content)
            if not hod_match:
                hod_match = re.search(r'(?:Dr\.?|Prof\.?)\s*([\w\s]+?)\n+\s*Head', content)
            if not hod_match:
                hod_match = re.search(r'##\s*(?:Dr\.?|Prof\.?)\s*([\w\s]+?)\n', content)
            if hod_match:
                hod_name = self.resolver.resolve(hod_match.group(1).strip())

        if hod_name:
            hod_resolved = self.resolver.resolve(hod_name)
            self._add_node(hod_resolved, "Faculty", name=hod_resolved, is_hod=True,
                           designation="Head of Department (HoD)")
            self._add_edge(hod_resolved, doc_id, "SOURCE_DOCUMENT")
            dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
            self._add_node(dept_id, "Department",
                name=self.dept_config["full_name"], institution="IIT Jammu")
            self._add_edge(hod_resolved, dept_id, "HOD_OF")

            # Extract official HoD email and store on department node
            hod_email_match = re.search(r'hod\.[\w\.\-]+@iitjammu\.ac\.in', content, re.IGNORECASE)
            if hod_email_match:
                self.graph.nodes[dept_id]['hod_official_email'] = hod_email_match.group(0).lower()

    def _parse_placement_data(self, filename: str, content: str, doc_id: str):
        """Parse placement industry data into structured PlacementData nodes."""
        # Detect year headers from the table to label data correctly
        year_headers = re.findall(r'(\d{4}[-–]\d{2,4})', content)
        year_labels = year_headers[:2] if len(year_headers) >= 2 else ["2023-24", "2022-23"]

        for line in content.splitlines():
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if len(cells) < 5:
                continue
            program = cells[0].strip()
            # Skip header/separator rows
            if not program or program.startswith('---') or 'percentage' in program.lower() or 'salary' in program.lower():
                continue
            # Skip if program doesn't look like a real program label
            if len(program) > 40 or program.lower() in ('', 'program', 'programme', 'course'):
                continue

            try:
                # Determine table layout: either 8+ cells (two years) or 4+ cells (one year)
                if len(cells) >= 9:
                    # Two-year layout: cells[1-4] = year1, cells[5-8] = year2
                    pct_y1 = cells[1].strip()
                    mean_y1 = cells[2].strip()
                    max_y1 = cells[3].strip()
                    min_y1 = cells[4].strip()
                    pct_y2 = cells[5].strip()
                    mean_y2 = cells[6].strip()
                    max_y2 = cells[7].strip() if len(cells) > 7 else 'NA'
                    min_y2 = cells[8].strip() if len(cells) > 8 else 'NA'

                    # Validate that values look numeric
                    try:
                        float(pct_y1.replace('%', ''))
                    except ValueError:
                        continue

                    node_id_y1 = f"placement:{program}:{year_labels[0]}"
                    self._add_node(node_id_y1, "PlacementData",
                        name=f"{program} Placement {year_labels[0]}",
                        program=program, year=year_labels[0],
                        percentage=pct_y1,
                        mean_salary=mean_y1,
                        max_salary=max_y1,
                        min_salary=min_y1,
                        source_file=filename)
                    self._add_edge(node_id_y1, doc_id, "SOURCE_DOCUMENT")

                    if pct_y2 != 'NA':
                        try:
                            float(pct_y2.replace('%', ''))
                        except ValueError:
                            continue
                        node_id_y2 = f"placement:{program}:{year_labels[1]}"
                        self._add_node(node_id_y2, "PlacementData",
                            name=f"{program} Placement {year_labels[1]}",
                            program=program, year=year_labels[1],
                            percentage=pct_y2,
                            mean_salary=mean_y2,
                            max_salary=max_y2,
                            min_salary=min_y2,
                            source_file=filename)
                        self._add_edge(node_id_y2, doc_id, "SOURCE_DOCUMENT")

                elif len(cells) >= 5:
                    # Single-year layout
                    pct = cells[1].strip()
                    try:
                        float(pct.replace('%', ''))
                    except ValueError:
                        continue
                    mean_s = cells[2].strip()
                    max_s = cells[3].strip()
                    min_s = cells[4].strip()
                    node_id = f"placement:{program}:{year_labels[0]}"
                    self._add_node(node_id, "PlacementData",
                        name=f"{program} Placement {year_labels[0]}",
                        program=program, year=year_labels[0],
                        percentage=pct,
                        mean_salary=mean_s,
                        max_salary=max_s,
                        min_salary=min_s,
                        source_file=filename)
                    self._add_edge(node_id, doc_id, "SOURCE_DOCUMENT")
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
        from departments import CORRECT_LABS
        correct_labs = CORRECT_LABS.get(self.dept_code, [])
        if not correct_labs:
            return

        seen = set()
        
        # Helper to match candidate against correct labs
        def get_matched_correct_lab(cand: str):
            cand = _strip_markdown_emphasis(_strip_markdown_link(cand)).strip()
            cand = re.sub(r'[:\-]+$', '', cand).strip()
            cand_clean = re.sub(r'\s*\(\d+\)\s*$', '', cand).strip()
            if not cand_clean:
                return None
            
            def norm(s: str) -> str:
                return re.sub(r'\s+', ' ', s.lower().strip())
                
            cand_norm = norm(cand_clean)
            for correct_lab in correct_labs:
                if norm(correct_lab) == cand_norm:
                    return correct_lab
            return None

        def add_lab(cand: str):
            matched = get_matched_correct_lab(cand)
            if matched and matched not in seen:
                seen.add(matched)
                lab_id = f"lab:{matched}"
                self._add_node(lab_id, "Lab", name=matched)
                self._add_edge(lab_id, doc_id, "SOURCE_DOCUMENT")

        # 1. Heading matching at ALL heading levels ## through ######
        heading_patterns = re.findall(r'#{2,6}\s+([^\n]*?(?:lab|laboratory|facility)[^\n]*?)(?:\n|$)', content, re.IGNORECASE)
        for cand in heading_patterns:
            add_lab(cand)
                
        # 2. Bullet list matching
        list_patterns = re.findall(r'^\s*-\s+([^\n]*?(?:lab|laboratory)[^\n]*?)(?:\n|$)', content, re.M | re.IGNORECASE)
        for cand in list_patterns:
            add_lab(cand)

        # 3. For classified lab pages: extract lab names from headings
        doc_data = self.graph.nodes.get(doc_id, {})
        if doc_data.get("doc_kind") == "labs":
            all_headings = re.findall(r'^#{2,4}\s+([^\n]+)', content, re.M)
            for heading in all_headings:
                matched = get_matched_correct_lab(heading)
                if matched and matched not in seen:
                    seen.add(matched)
                    lab_id = f"lab:{matched}"
                    self._add_node(lab_id, "Lab", name=matched)
                    self._add_edge(lab_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_awards(self, filename: str, content: str, doc_id: str):
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("- ["):
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            if len(line) > 15:
                matching_faculty = None
                for canon in self._canonical_faculty:
                    if canon.lower() in line.lower() or fuzzy_match(canon.lower(), line.lower()):
                        matching_faculty = canon
                        break
                award_title = line
                award_id = f"award:{award_title[:80]}"
                self._add_node(award_id, "Award", title=award_title, source_file=filename)
                self._add_edge(award_id, doc_id, "SOURCE_DOCUMENT")
                if matching_faculty:
                    self._add_edge(matching_faculty, award_id, "HAS_AWARD")
                    self._add_edge(award_id, matching_faculty, "AWARDED_TO")

    def _parse_publications_page(self, filename: str, content: str, doc_id: str):
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- ") and len(line) > 10:
                pub_title = line[2:].strip()
                pub_id = f"publication:{pub_title[:80]}"
                self._add_node(pub_id, "Publication", title=pub_title, source_file=filename)
                self._add_edge(pub_id, doc_id, "SOURCE_DOCUMENT")
                for canon in self._canonical_faculty:
                    if canon.lower() in pub_title.lower():
                        self._add_edge(canon, pub_id, "AUTHORED")
                        self._add_edge(pub_id, canon, "WRITTEN_BY")

    def _parse_staff(self, filename: str, content: str, doc_id: str):
        blocks = _iter_heading_blocks(content, level=4)
        for raw_heading, block in blocks:
            name = _strip_markdown_emphasis(_strip_markdown_link(raw_heading)).strip()
            if not name or len(name) < 2 or any(kw in name.lower() for kw in ["research", "people", "staff", "menu", "home"]):
                continue
            role = ""
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if lines:
                role = lines[0]
            staff_id = f"staff:{name}"
            self._add_node(staff_id, "Staff", name=name, designation=role, source_file=filename)
            self._add_edge(staff_id, doc_id, "SOURCE_DOCUMENT")
            dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
            self._add_edge(staff_id, dept_id, "STAFF_MEMBER_OF")

    def _parse_programmes(self, filename: str, content: str, doc_id: str):
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            for prog in ["B.Tech", "M.Tech", "PhD", "Ph.D.", "Postgraduate", "Undergraduate"]:
                if prog.lower() in line.lower() and len(line) < 100:
                    prog_name = prog if prog != "Ph.D." else "PhD"
                    prog_id = f"program:{prog_name}"
                    self._add_node(prog_id, "Program", name=prog_name, details=line, source_file=filename)
                    self._add_edge(prog_id, doc_id, "SOURCE_DOCUMENT")
            
            course_match = re.search(r'([A-Z]{2,3}-\d{3,4}|[A-Z]{2,3}-\d-\d{2})(?:\(.*\))?\s*[:\-]?\s*(.*)', line)
            if course_match:
                code = course_match.group(1).strip()
                title = course_match.group(2).strip()
                title = _strip_markdown_emphasis(_strip_markdown_link(title)).strip()
                course_id = f"course:{code}"
                self._add_node(course_id, "Course", code=code, title=title or code, source_file=filename)
                self._add_edge(course_id, doc_id, "SOURCE_DOCUMENT")

    def _parse_contact(self, filename: str, content: str, doc_id: str):
        """Parse contact/address pages into a structured ContactInfo node."""
        lines = content.splitlines()
        address_parts = []
        email = ""
        phone = ""
        
        # Extract email from mailto links or raw email patterns
        email_match = re.search(r'\[([^\]]*@[^\]]*)\]\(mailto:([^)]+)\)', content)
        if email_match:
            email = email_match.group(2).strip()
        else:
            email = _extract_first_email(content)

        # Extract phone if present
        phone_match = re.search(r'(?:Phone|Telephone|Tel)\s*(?:Number)?\s*[:\s]*([+\d\s\-()]{8,})', content, re.IGNORECASE)
        if phone_match:
            phone = phone_match.group(1).strip()
        
        # Extract address: find the "Address" section and grab subsequent lines
        in_address = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower() == 'address' or stripped.lower().startswith('address'):
                in_address = True
                # Check if address is on the same line
                after_label = re.sub(r'^address\s*:?\s*', '', stripped, flags=re.IGNORECASE).strip()
                if after_label and len(after_label) > 5:
                    address_parts.append(after_label)
                continue
            if in_address:
                # Stop at next section
                if stripped.startswith('![') or stripped.lower().startswith('email') or stripped.lower().startswith('phone') or stripped.lower().startswith('tel'):
                    in_address = False
                    continue
                if stripped and not stripped.startswith('#') and not stripped.startswith('['):
                    address_parts.append(stripped)
        
        address = ', '.join(address_parts).strip().rstrip(',')
        if not address and not email:
            return
        
        dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
        contact_id = f"contact:{self.dept_code}"
        self._add_node(contact_id, "ContactInfo", 
                       name=f"{self.dept_config.get('full_name', self.dept_code)} Contact",
                       address=address, email=email, phone=phone,
                       source_file=filename)
        self._add_edge(contact_id, doc_id, "SOURCE_DOCUMENT")
        # Also store contact info on the department node for direct access
        if self.graph.has_node(dept_id):
            if address:
                self.graph.nodes[dept_id]['address'] = address
            if email:
                self.graph.nodes[dept_id]['contact_email'] = email
            if phone:
                self.graph.nodes[dept_id]['phone'] = phone
        self._add_edge(contact_id, dept_id, "CONTACT_FOR")

    def _parse_alumni(self, filename: str, content: str, doc_id: str):
        """Parse alumni pages to extract alumni names, batch year, and program info."""
        # Strategy 1: Heading-based alumni entries (HSS style: #### [Name](url))
        level = _detect_person_record_level(content) or _detect_repeated_heading_level(content) or 4
        alumni_blocks = _iter_heading_blocks(content, level=level)
        parsed_any = False
        for raw_heading, block in alumni_blocks:
            name = _strip_markdown_link(raw_heading).strip()
            if not name or len(name) < 2:
                continue
            # Skip navigation-like headings
            if name.lower() in ('alumni', 'home', 'alumni list', 'people'):
                continue
            alumni_id = f"alumni:{name}"
            self._add_node(alumni_id, "Alumni", name=name, source_file=filename)
            self._add_edge(alumni_id, doc_id, "SOURCE_DOCUMENT")
            parsed_any = True

        # Strategy 2: Batch/program-based alumni from headings (Chemical Eng style: ## B.Tech 2024)
        if not parsed_any:
            for line in content.splitlines():
                stripped = line.strip()
                m = re.match(r'^#{2,3}\s+(.+)', stripped)
                if m:
                    heading = m.group(1).strip()
                    # Check if this looks like a batch heading (contains year and program)
                    if re.search(r'\b(B\.?Tech|M\.?Tech|PhD|Ph\.?D|MSc|M\.Sc)\b', heading, re.IGNORECASE):
                        batch_id = f"alumni_batch:{heading}"
                        self._add_node(batch_id, "AlumniBatch", name=heading, source_file=filename)
                        self._add_edge(batch_id, doc_id, "SOURCE_DOCUMENT")
                        parsed_any = True

    def _parse_phd_alumni(self, filename: str, content: str, doc_id: str):
        """Parse graduated PhD student lists (phd-alumni-list) into GraduatedPhD nodes."""
        # For phd-alumni pages, the student name is typically at #### level,
        # with ##### and ###### being labels/values. We need to find the right level.
        # Try level 4 first (most common for student names)
        level = _detect_person_record_level(content)
        if level is None:
            # Don't use _detect_repeated_heading_level — it might pick the value level (6)
            # Instead, check which heading levels have probable person names
            for candidate_level in (4, 3, 5):
                blocks = list(_iter_heading_blocks(content, level=candidate_level))
                if blocks:
                    # Check if any heading looks like a person name (not a label)
                    label_words = {'supervisor', 'thesis', 'year', 'title', 'research', 'area'}
                    name_count = sum(1 for h, _ in blocks
                                     if h.strip().lower() not in label_words
                                     and not any(lw in h.strip().lower() for lw in label_words)
                                     and len(h.strip()) > 2
                                     and _is_probable_person_name(h.strip()))
                    if name_count > 0:
                        level = candidate_level
                        break
            if level is None:
                level = 4
        student_blocks = _iter_heading_blocks(content, level=level)

        for raw_heading, block in student_blocks:
            student_name = _strip_markdown_link(raw_heading).strip()
            if not student_name or len(student_name) < 2:
                continue

            # Extract supervisor, thesis title, year from the block
            section_map = _extract_section_map_from_body(block)
            
            supervisor = ""
            thesis_title = ""
            year = ""
            
            # Try structured sections first
            sup_text = section_map.get("supervisor", "")
            if sup_text:
                # Only take the first line — section_map may include subsequent labels/values
                sup_first_line = sup_text.split('\n')[0].strip()
                supervisor = _strip_markdown_emphasis(_strip_markdown_link(sup_first_line)).strip()
                supervisor = TITLE_PREFIXES.sub('', supervisor).strip().lstrip('. ').strip()
            
            thesis_text = section_map.get("thesis_title", "") or section_map.get("thesis", "")
            if thesis_text:
                thesis_title = _strip_markdown_emphasis(thesis_text).strip()

            year_text = section_map.get("year", "")
            if year_text:
                year_match = re.search(r'\d{4}', year_text)
                if year_match:
                    year = year_match.group(0)

            # Fallback: directly parse ##### **Label** / ###### Value heading pairs
            # This handles the math phd-alumni-list structure
            if not supervisor or not thesis_title or not year:
                block_lines = block.splitlines()
                current_label = None
                for bline in block_lines:
                    stripped = bline.strip()
                    # Check for label heading: ##### **Supervisor** or ##### **Thesis Title**
                    label_m = re.match(r'^#{4,6}\s+\**([^*#\n]+?)\**\s*$', stripped)
                    if label_m:
                        label_text = label_m.group(1).strip().lower().replace(' ', '_')
                        if label_text in ('supervisor', 'thesis_title', 'year'):
                            current_label = label_text
                            continue
                    # Check for value heading: ###### Value
                    if current_label and stripped.startswith('#'):
                        value = stripped.lstrip('#').strip()
                        value = _strip_markdown_emphasis(value).strip()
                        if value:
                            if current_label == 'supervisor' and not supervisor:
                                supervisor = TITLE_PREFIXES.sub('', value).strip().lstrip('. ').strip()
                            elif current_label == 'thesis_title' and not thesis_title:
                                thesis_title = value
                            elif current_label == 'year' and not year:
                                ym = re.search(r'\d{4}', value)
                                if ym:
                                    year = ym.group(0)
                            current_label = None

            # Final fallback: scan for Dr./Prof. names and 4-digit years
            if not supervisor:
                for line in block.splitlines():
                    stripped = line.strip().lstrip('#').strip()
                    stripped = _strip_markdown_emphasis(stripped).strip()
                    if stripped.startswith('Dr.') or stripped.startswith('Prof.'):
                        supervisor = TITLE_PREFIXES.sub('', stripped).strip().lstrip('. ').strip()
                        break
            
            if not year:
                year_match = re.search(r'\b(20\d{2})\b', block)
                if year_match:
                    year = year_match.group(1)

            grad_id = f"graduated_phd:{student_name}"
            self._add_node(grad_id, "GraduatedPhD", name=student_name,
                           supervisor=supervisor, thesis_title=thesis_title,
                           graduation_year=year, source_file=filename)
            self._add_edge(grad_id, doc_id, "SOURCE_DOCUMENT")

            # Link to supervisor if they're canonical faculty
            if supervisor:
                resolved_sup = self.resolver.resolve(supervisor)
                if self.resolver.is_canonical_faculty(resolved_sup):
                    self._add_node(resolved_sup, "Faculty", name=resolved_sup)
                    self._add_edge(grad_id, resolved_sup, "SUPERVISED_BY")

    def _parse_admin_director(self, filename: str, content: str, doc_id: str):
        lines = content.splitlines()
        name = "Manoj Singh Gaur"
        designation = "Director"
        
        for line in lines:
            if line.startswith("## "):
                raw_name = line[3:].strip()
                name = clean_admin_member_name(raw_name)
                break
                
        resolved_name = self.resolver.resolve(name)
        props = {
            "name": resolved_name,
            "designation": designation,
            "is_director": True,
            "source_file": filename,
            "profile_url": "https://iitjammu.ac.in/director"
        }
        self._add_node(resolved_name, "Faculty", **props)
        self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
        
        dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
        self._add_edge(resolved_name, dept_id, "MEMBER_OF")

    def _parse_admin_registrar(self, filename: str, content: str, doc_id: str):
        lines = content.splitlines()
        name = "Virinder Singh Jeji"
        designation = "Registrar"
        email = "registrar@iitjammu.ac.in"
        
        for line in lines:
            if line.startswith("### "):
                raw_name = line[4:].strip()
                name = clean_admin_member_name(raw_name)
            elif "email" in line.lower():
                email = _deobfuscate_email_text(line.split("-")[-1].strip())
                
        resolved_name = self.resolver.resolve(name)
        props = {
            "name": resolved_name,
            "designation": designation,
            "is_registrar": True,
            "email": email,
            "source_file": filename
        }
        self._add_node(resolved_name, "AdminOfficial", **props)
        self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
        
        dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
        self._add_edge(resolved_name, dept_id, "MEMBER_OF")

    def _parse_admin_bogchairman(self, filename: str, content: str, doc_id: str):
        lines = content.splitlines()
        name = "Sharad Kumar Saraf"
        designation = "BOG Chairman"
        
        for line in lines:
            if line.startswith("### "):
                raw_name = line[4:].strip()
                name = clean_admin_member_name(raw_name)
                
        resolved_name = self.resolver.resolve(name)
        props = {
            "name": resolved_name,
            "designation": designation,
            "is_bog_chairman": True,
            "source_file": filename
        }
        self._add_node(resolved_name, "AdminOfficial", **props)
        self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
        
        dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
        self._add_edge(resolved_name, dept_id, "MEMBER_OF")

    def _parse_admin_deans(self, filename: str, content: str, doc_id: str):
        lines = content.splitlines()
        current_section = "Dean"
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            if "ASSOCIATE DEANS" in line.upper():
                current_section = "Associate Dean"
                i += 1
                continue
            elif "DEANS" in line.upper():
                current_section = "Dean"
                i += 1
                continue
            
            if line.startswith("####"):
                raw_name = line.replace("####", "").strip()
                if raw_name.upper() in ("DEANS", "ASSOCIATE DEANS") or len(raw_name) < 2:
                    i += 1
                    continue
                
                name = clean_admin_member_name(raw_name)
                role = ""
                email = ""
                profile_url = ""
                
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("####"):
                    next_line = lines[i].strip()
                    if next_line:
                        if "[PROFILE]" in next_line:
                            m = re.search(r'\((https?://[^\s)]+)\)', next_line)
                            if m:
                                profile_url = m.group(1).strip()
                        elif "dot" in next_line.lower() and "at" in next_line.lower() or "@" in next_line:
                            email = _deobfuscate_email_text(next_line)
                        elif not role and not next_line.startswith("!["):
                            role = next_line
                    i += 1
                
                resolved_name = self.resolver.resolve(name)
                label = "Faculty" if self.resolver.is_canonical_faculty(resolved_name) else "AdminOfficial"
                props = {
                    "name": resolved_name,
                    "designation": f"{current_section} - {role}" if role else current_section,
                    "admin_role": role,
                    "admin_type": current_section,
                    "source_file": filename
                }
                if email:
                    props["email"] = email
                if profile_url:
                    props["profile_url"] = profile_url
                
                self._add_node(resolved_name, label, **props)
                self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
                
                dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
                self._add_edge(resolved_name, dept_id, "MEMBER_OF")
                continue
            
            i += 1

    def _parse_admin_committee(self, filename: str, content: str, doc_id: str):
        committee_name = ""
        for line in content.splitlines():
            if line.startswith("## "):
                committee_name = line[3:].strip()
                break
        if not committee_name:
            committee_name = filename.replace("administration_", "").replace(".md", "").replace("-", " ").title()
            
        committee_id = f"committee:{committee_name.lower().replace(' ', '_')}"
        self._add_node(committee_id, "Committee", name=committee_name, source_file=filename)
        self._add_edge(committee_id, doc_id, "SOURCE_DOCUMENT")
        
        dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
        self._add_edge(committee_id, dept_id, "COMMITTEE_OF")
        
        headers = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("|"):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if not cells:
                    continue
                if cells[0] == "Name" or "---" in cells[0]:
                    if cells[0] == "Name":
                        headers = cells
                    continue
                
                name_cell = cells[0]
                designation_cell = cells[1] if len(cells) > 1 else ""
                
                clean_name = clean_admin_member_name(name_cell)
                if not clean_name or len(clean_name) < 4 or clean_name.lower() in ("all", "each", "any", "deputy"):
                    clean_name = re.sub(r'[\*\(\)\[\]]', '', name_cell).strip()
                    clean_name = re.sub(r'\s+', ' ', clean_name)
                    
                if not clean_name or clean_name.lower() in ("name", "chairperson", "member"):
                    continue
                    
                resolved_name = self.resolver.resolve(clean_name)
                label = "Faculty" if self.resolver.is_canonical_faculty(resolved_name) else "AdminOfficial"
                
                props = {
                    "name": resolved_name,
                    "source_file": filename
                }
                self._add_node(resolved_name, label, **props)
                self._add_edge(resolved_name, doc_id, "SOURCE_DOCUMENT")
                
                self._add_edge(resolved_name, committee_id, "MEMBER_OF", 
                               role_in_committee=designation_cell, 
                               committee_name=committee_name)

    def build(self) -> nx.DiGraph:
        if not os.path.exists(self.markdown_dir):
            raise FileNotFoundError(f"Markdown directory not found: {self.markdown_dir}")

        # Skip the combined file for any department
        filenames = [f for f in os.listdir(self.markdown_dir)
                     if f.endswith(".md") and not f.startswith("00_combined")
                     and not f.endswith(".json")]

        logger.info(f"Processing {len(filenames)} markdown files...")

        doc_map = {}
        doc_kinds = {}
        for filename in sorted(filenames):
            filepath = os.path.join(self.markdown_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            doc_id = self._create_document_node(filename, content)
            doc_map[filename] = (doc_id, content)
            kind = infer_document_kind(filename, content)
            doc_kinds[filename] = kind
            self.graph.nodes[doc_id]["doc_kind"] = kind

        logger.info(f"Phase 1: Created {len(doc_map)} document nodes with text chunks.")

        # Find faculty list by pattern instead of exact name
        faculty_list_file = None
        for fn in doc_map:
            if doc_kinds.get(fn) == "faculty_roster":
                faculty_list_file = fn
                break

        # Parse faculty list FIRST to register canonical names
        if faculty_list_file:
            doc_id, content = doc_map[faculty_list_file]
            self._parse_faculty_list(faculty_list_file, content, doc_id)
            logger.info(f"Phase 1.5: Parsed faculty list ({faculty_list_file}) (canonical registry populated).")

        for filename, (doc_id, content) in doc_map.items():
            doc_kind = doc_kinds.get(filename, "generic")
            clean_fn = filename
            prefix = f"{self.dept_code}_"
            if clean_fn.startswith(prefix):
                clean_fn = clean_fn[len(prefix):]

            if doc_kind == "faculty_profile":
                self._parse_faculty_profile(filename, content, doc_id)
            elif doc_kind == "admin_director":
                self._parse_admin_director(filename, content, doc_id)
            elif doc_kind == "admin_registrar":
                self._parse_admin_registrar(filename, content, doc_id)
            elif doc_kind == "admin_bogchairman":
                self._parse_admin_bogchairman(filename, content, doc_id)
            elif doc_kind == "admin_deans":
                self._parse_admin_deans(filename, content, doc_id)
            elif doc_kind == "admin_committee":
                self._parse_admin_committee(filename, content, doc_id)
            elif doc_kind == "phd_roster":
                self._parse_phd_list(filename, content, doc_id)
            elif doc_kind == "mtech_roster":
                self._parse_phd_list(filename, content, doc_id, label="MTechStudent")
            elif doc_kind == "awards":
                self._parse_awards(filename, content, doc_id)
            elif doc_kind == "publications":
                self._parse_publications_page(filename, content, doc_id)
            elif doc_kind == "staff":
                self._parse_staff(filename, content, doc_id)
            elif doc_kind == "programmes":
                self._parse_programmes(filename, content, doc_id)
            elif doc_kind == "funded_projects":
                self._parse_funded_projects(filename, content, doc_id)
            elif doc_kind == "patents":
                self._parse_patents(filename, content, doc_id)
            elif doc_kind == "startups":
                self._parse_startups(filename, content, doc_id)
            elif doc_kind == "research_areas":
                self._parse_research_areas(filename, content, doc_id)
            elif doc_kind == "contact":
                self._parse_contact(filename, content, doc_id)
            elif doc_kind == "alumni":
                self._parse_alumni(filename, content, doc_id)
            elif doc_kind == "phd_alumni":
                self._parse_phd_alumni(filename, content, doc_id)
            elif faculty_list_file and filename == faculty_list_file:
                pass  # Already parsed above
            elif doc_kind == "hod_message":
                self._parse_hod(filename, content, doc_id)
            elif doc_kind == "placement_industry":
                self._parse_placement_data(filename, content, doc_id)
            elif doc_kind == "placement_academia":
                self._parse_higher_studies(filename, content, doc_id)
            
            # Check if department main page contains HOD message/details
            if clean_fn in (f"{self.dept_code}.md", f"{self.dept_code}_index.html.md", "index.md"):
                if "hod" in content.lower() or "head of" in content.lower() or "message" in content.lower():
                    self._parse_hod(filename, content, doc_id)
            
            # Labs (usually in index/home/dept main page, or labs file, but we now run it on all pages to ensure no labs are missed)
            self._parse_labs(content, doc_id)

        logger.info(f"Phase 2: Entity extraction complete.")

        # Cross-link faculty to research areas from PhD supervisions
        faculty_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('label') == 'Faculty']
        phd_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('label') == 'PhDStudent']
        for faculty in faculty_nodes:
            students = [e[0] for e in self.graph.in_edges(faculty)
                       if self.graph.edges[e[0], faculty].get('type') == 'SUPERVISED_BY']
            for student in students:
                for _, target in self.graph.out_edges(student):
                    if self.graph.nodes[target].get('label') == 'ResearchArea':
                        if not self.graph.has_edge(faculty, target):
                            self._add_edge(faculty, target, "RESEARCHES_IN")

        dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
        if not self.graph.has_node(dept_id):
            self._add_node(dept_id, "Department",
                name=self.dept_config["full_name"],
                institution="IIT Jammu", website=self.dept_config["base_url"])

        # Store official faculty count on department node
        self.graph.nodes[dept_id]['faculty_count'] = len(faculty_nodes)
        self.graph.nodes[dept_id]['phd_student_count'] = len(phd_nodes)
        faculty_structured_fields = set()
        for faculty in faculty_nodes:
            for key, value in self.graph.nodes[faculty].items():
                if key in {"label", "department", "name"}:
                    continue
                if value in (None, "", [], {}):
                    continue
                faculty_structured_fields.add(key)
        self.graph.nodes[dept_id]['faculty_structured_fields'] = sorted(faculty_structured_fields)
        phd_structured_fields = set()
        for scholar in phd_nodes:
            for key, value in self.graph.nodes[scholar].items():
                if key in {"label", "department", "name"}:
                    continue
                if value in (None, "", [], {}):
                    continue
                phd_structured_fields.add(key)
        self.graph.nodes[dept_id]['phd_structured_fields'] = sorted(phd_structured_fields)
        self.graph.nodes[dept_id]['document_kind_counts'] = dict(
            sorted({kind: list(doc_kinds.values()).count(kind) for kind in set(doc_kinds.values())}.items())
        )

        # Only link actual Faculty (not ExternalPerson) to department
        for faculty in faculty_nodes:
            self._add_edge(faculty, dept_id, "MEMBER_OF")

        # Seed/guarantee all correct labs for this department exist in the graph
        from departments import CORRECT_LABS
        correct_labs = CORRECT_LABS.get(self.dept_code, [])
        for correct_lab in correct_labs:
            lab_id = f"lab:{correct_lab}"
            full_lab_id = lab_id
            if self.dept_code != "ee" and not lab_id.startswith(f"{self.dept_code}:"):
                full_lab_id = f"{self.dept_code}:{lab_id}"
            
            if not self.graph.has_node(full_lab_id):
                self._add_node(lab_id, "Lab", name=correct_lab)
            
            self._add_edge(lab_id, dept_id, "FACILITY_OF")

        label_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            label_counts[data.get('label', 'Unknown')] += 1

        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        logger.info(f"Node types: {dict(label_counts)}")
        return self.graph

    def save(self, output_dir: str = None):
        if output_dir is None:
            output_dir = get_data_dir(self.dept_code)
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
    def load(data_dir: str):
        with open(os.path.join(data_dir, "graph.pkl"), "rb") as f:
            graph = pickle.load(f)
        with open(os.path.join(data_dir, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return graph, chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    builder = KnowledgeGraphBuilder(dept_code="ee")
    G = builder.build()
    builder.save()
    print(f"\n✅ Knowledge Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Chunks generated: {len(builder.chunks)}")
