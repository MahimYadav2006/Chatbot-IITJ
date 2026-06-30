# Part 5 — Entity Resolution, Identity Management & Maintenance

> *The cross-cutting systems that keep the knowledge graph coherent, deduplicated, and production-ready.*

---

## 1. The Entity Resolution Problem

Consider a single faculty member, "Alok Kumar Saxena", who appears across scraped pages in these variants:

```
Faculty Roster:    "Alok Kumar Saxena"
PhD Student Page:  "Dr. A. K. Saxena" (supervisor field)
Patent Document:   "Prof. Alok Saxena"
Funded Project:    "A.K. Saxena" (PI column)
Committee Roster:  "Dr. Alok Kumar Saxena, Associate Professor"
HoD Message:       "Prof. A. K. Saxena" (heading)
```

Without entity resolution, the knowledge graph would create **6 separate Faculty nodes** — each with partial information, none connected to the others. A query "Who supervises PhD students under Dr. Saxena?" would return incomplete results because the PhD supervision edges point to "Dr. A. K. Saxena" while the faculty profile is under "Alok Kumar Saxena".

The `EntityResolver` class (kg_builder.py, line 784) solves this by maintaining a **canonical name registry** and resolving every name variant to its canonical form.

---

## 2. EntityResolver Architecture

### 2.1 Data Structures

```python
class EntityResolver:
    def __init__(self, canonical_faculty: set = None):
        self.canonical_names = {}           # normalized → canonical
        self.name_variants = defaultdict(set)  # canonical → {variants}
        self._canonical_faculty = canonical_faculty or set()
```

- **`canonical_names`**: Forward lookup. Maps every normalized name ever seen to its resolved canonical form. Once a mapping is established, future lookups are O(1).
- **`name_variants`**: Reverse index. Maps each canonical name to all known variants. Used for debugging and for the `GlobalPersonIndex`.
- **`_canonical_faculty`**: The ground-truth set of faculty names extracted from the faculty roster file. These names have the highest trust level and are never overwritten.

### 2.2 Canonical Faculty Seeding

Before any entity parsing begins, `_extract_canonical_faculty()` (line 751) populates the registry:

```python
def _extract_canonical_faculty(self) -> set:
    faculty_file = self._find_faculty_list_file()
    if not faculty_file:
        return set()
    
    content = self._read_file(faculty_file)
    level = _detect_person_record_level(content)
    if level is None:
        return set()
    
    names = set()
    for heading, body in _iter_heading_blocks(content, level):
        normalized = normalize_name(heading)
        if _is_probable_person_name(normalized):
            names.add(normalized)
    return names
```

**Why faculty roster first?** The roster file has the most reliable name formats — they come from the official university directory, not from free-text mentions in papers or committee notifications. By seeding these first, every subsequent name variant resolves against the authoritative form.

### 2.3 The Resolution Cascade: `resolve()`

```python
def resolve(self, raw_name: str) -> str:
    normalized = normalize_name(raw_name)
    if not normalized:
        return raw_name
    
    # Step 1: Cache hit
    if normalized in self.canonical_names:
        return self.canonical_names[normalized]
    
    # Step 2: Check canonical faculty (3 matchers)
    for canonical in self._canonical_faculty:
        if fuzzy_match(normalized, canonical):          # Matcher 1
            self._register(normalized, canonical)
            return canonical
        if _initials_match(normalized, canonical):      # Matcher 2
            self._register(normalized, canonical)
            return canonical
        if _token_subset_match(normalized, canonical):  # Matcher 3
            self._register(normalized, canonical)
            return canonical
    
    # Step 3: Check previously resolved names
    for existing_canonical in list(self.canonical_names.values()):
        if fuzzy_match(normalized, existing_canonical):
            self._register(normalized, existing_canonical)
            return existing_canonical
        if _initials_match(normalized, existing_canonical):
            self._register(normalized, existing_canonical)
            return existing_canonical
        if _token_subset_match(normalized, existing_canonical):
            self._register(normalized, existing_canonical)
            return existing_canonical
    
    # Step 4: No match — register as new canonical
    self._register(normalized, normalized)
    return normalized
```

### 2.4 The Three Matchers — Detailed Analysis

#### Matcher 1: `fuzzy_match()` — SequenceMatcher

```python
def fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio() >= threshold
```

Uses Python's `difflib.SequenceMatcher` which computes the ratio of matching characters. The 0.85 threshold was tuned empirically:

| Comparison | Ratio | Match? |
|---|---|---|
| "alok kumar saxena" vs "alok kumar saxena" | 1.00 | ✓ |
| "alok kumar saxena" vs "alok k saxena" | 0.87 | ✓ |
| "arvind gupta" vs "aurbind gupta" | 0.86 | ✓ |
| "alok saxena" vs "amit sharma" | 0.42 | ✗ |
| "ankit dubey" vs "ankit kumar" | 0.55 | ✗ |

**Edge case**: Short names like "Ram Kumar" vs "Ram Kaur" have ratio 0.82 — correctly rejected. But "Ram Kumar" vs "Ram Kumari" has ratio 0.91 — a false positive. The subsequent matchers and the canonical faculty priority mitigate most such cases.

#### Matcher 2: `_initials_match()` — Abbreviated Name Resolution

```python
def _initials_match(short: str, full: str) -> bool:
    short_parts = short.lower().replace('.', '').split()
    full_parts = full.lower().replace('.', '').split()
    
    # Last name must match (ratio >= 0.85)
    if SequenceMatcher(None, short_parts[-1], full_parts[-1]).ratio() < 0.85:
        return False
    
    # Preceding tokens in short must be initials matching full
    for sp, fp in zip(short_parts[:-1], full_parts[:-1]):
        if len(sp) <= 2:  # Initial like "a" or "ak"
            if not fp.startswith(sp[0]):
                return False
        else:
            if SequenceMatcher(None, sp, fp).ratio() < 0.80:
                return False
    return True
```

**Resolution examples:**
- "B. N. Subudhi" → matches "Badri Narayan Subudhi" ✓
- "A. K. Saxena" → matches "Alok Kumar Saxena" ✓
- "S. Singh" → matches "Sukhdev Singh" ✓ but also "Satish Singh" ✓ (ambiguous!)

**Ambiguity handling**: When multiple canonical faculty share initials (e.g., "S. Singh" could match "Sukhdev Singh" or "Satish Singh"), the resolver takes the **first match** in the canonical set. This is a known limitation — the system relies on the fact that within a single department, initial collisions are rare. Cross-department resolution doesn't happen during ingestion (each department builds its own resolver).

#### Matcher 3: `_token_subset_match()` — Missing Middle Names

```python
def _token_subset_match(name1: str, name2: str) -> bool:
    parts1 = name1.lower().replace('.', '').split()
    parts2 = name2.lower().replace('.', '').split()
    
    shorter, longer = (parts1, parts2) if len(parts1) <= len(parts2) else (parts2, parts1)
    
    # First and last tokens must match
    if SequenceMatcher(None, shorter[0], longer[0]).ratio() < 0.85:
        return False
    if SequenceMatcher(None, shorter[-1], longer[-1]).ratio() < 0.85:
        return False
    
    # Middle tokens of shorter must appear in order in longer
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
```

**Resolution examples:**
- "Anup Shukla" → matches "Anup Kumar Shukla" ✓ (middle name dropped)
- "Sartaj Hasan" → matches "Sartaj Ul Hasan" ✓ (middle particle dropped)
- "Alok Saxena" → matches "Alok Kumar Saxena" ✓

### 2.5 The `normalize_name()` Pre-Processor

Before any matching, all names pass through normalization (line 150):

```python
def normalize_name(name: str) -> str:
    # Step 1: Strip academic titles
    name = TITLE_PREFIXES.sub('', name)  # Remove Dr., Prof., Mr., etc.
    
    # Step 2: Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.strip('., ')
    
    # Step 3: Smart capitalization
    words = []
    for w in name.split():
        if '.' in w:
            # "a.k." → "A.K."
            parts = w.split('.')
            parts_cap = [p.upper() if len(p) <= 1 else p.capitalize() for p in parts]
            words.append('.'.join(parts_cap))
        elif w.isupper() and len(w) <= 3:
            words.append(w)  # Keep "AK", "BN" as-is
        else:
            words.append(w.capitalize())
    return ' '.join(words)
```

**Why strip titles?** "Dr. A. K. Saxena" and "Prof. Alok Kumar Saxena" should resolve to the same person. Titles are noise for matching purposes. They're preserved in the node's `designation` property, not in the canonical name.

### 2.6 The `clean_admin_member_name()` Function

Committee rosters have even noisier name formats. This specialized cleaner (line 169) handles:

```python
def clean_admin_member_name(name: str) -> str:
    n = re.sub(r'\(?Retd\.?\)?', '', name)           # Remove "(Retd.)"
    n = TITLE_RE.sub('', n)                            # Remove Dr./Prof./Sh./Col./Er./etc.
    n = re.sub(r'[,–-]\s*(?:Director|Registrar|Dean|...').*$', '', n)  # Remove trailing role
    n = re.sub(r'\b(?:Director|Registrar|Dean|...).*$', '', n)         # Remove inline role
    n = re.sub(r'[\*\(\)\[\]]', '', n)                 # Remove markdown formatting
    return n.strip(',.- ')
```

**Examples:**
- `"Dr. Alok Kumar Saxena, Associate Professor"` → `"Alok Kumar Saxena"`
- `"Sh. Manoj Dhar (Retd.)"` → `"Manoj Dhar"`
- `"Col. R.K. Sharma, Director IIIT"` → `"R.K. Sharma"`

---

## 3. The GlobalPersonIndex

The `GlobalPersonIndex` (`graphrag/person_index.py`) operates at **runtime**, not during ingestion. It aggregates person entities across all loaded department and section graphs into a single lookup table.

### 3.1 Construction

```python
class GlobalPersonIndex:
    def __init__(self):
        self.person_roles = defaultdict(list)  # name → [{source, designation, dept, ...}]
        self.resolver = EntityResolver()
    
    def load_from_graphs(self, dept_graphs: dict, section_graphs: dict):
        PERSON_LABELS = {"Faculty", "AdminOfficial", "SectionPerson",
                         "SectionHead", "Counselor", "MedicalDoctor", "PhDStudent"}
        
        for dept_code, graph in dept_graphs.items():
            for node, data in graph.nodes(data=True):
                if data.get("label") in PERSON_LABELS:
                    name = data.get("name", node)
                    if self._is_role_placeholder(name):
                        continue
                    resolved = self.resolver.resolve(name)
                    role = {
                        "source": dept_code,
                        "designation": data.get("designation", ""),
                        "department": data.get("department", dept_code),
                        "email": data.get("email", ""),
                        "label": data.get("label", ""),
                    }
                    self._add_role(resolved, role)
```

### 3.2 The Role Placeholder Filter: `_is_role_placeholder()`

This is a critical anti-hallucination mechanism. Committee rosters contain entries that look like names but are actually position descriptions:

```
"Dean Student Affairs"           → role placeholder, not a person
"All Deans"                      → role placeholder
"Upto Five Student Representative Nominated By The Chairman"  → role placeholder
"One SC/ST Representative"       → role placeholder
"Nominee of Vice Chancellor"     → role placeholder
```

If these were indexed as people, a query like "Who is on the anti-ragging committee?" might return "Dean Student Affairs" as if it were a person's name, causing the LLM to hallucinate a non-existent person.

```python
_ROLE_PLACEHOLDER_PREFIXES = (
    "dean ", "all dean", "upto ", "nominated ", "one representative",
    "two representative", "three representative", "nominee ",
    "special invitee", "any ", "respective ", "concerned ",
)

_ROLE_DESCRIPTOR_WORDS = {
    "officer", "affairs", "liaison", "coordinator", "convener",
    "chairperson", "chairman", "representative", "nominee",
    "warden", "provost", "controller", "superintendent",
    "registrar", "director", "secretary", "advisor",
    "incharge", "in-charge", "head", "member",
}

def _is_role_placeholder(self, name: str) -> bool:
    lowered = name.lower().strip()
    
    # Check 1: Known prefixes
    if any(lowered.startswith(p) for p in _ROLE_PLACEHOLDER_PREFIXES):
        return True
    
    # Check 2: Too many role-descriptor words (≥2)
    words = set(lowered.split())
    descriptor_count = len(words & _ROLE_DESCRIPTOR_WORDS)
    if descriptor_count >= 2:
        return True
    
    # Check 3: Too long to be a personal name (≥8 words)
    if len(lowered.split()) >= 8:
        return True
    
    # Check 4: Entirely composed of role words
    if words.issubset(_ROLE_DESCRIPTOR_WORDS | {"of", "the", "and", "or", "a", "an"}):
        return True
    
    return False
```

**Test cases:**
| Input | Result | Reason |
|---|---|---|
| "Sartaj Ul Hasan" | person ✓ | No role words |
| "Dean Student Affairs" | placeholder ✗ | Starts with "dean " |
| "Registrar" | placeholder ✗ | Single role word, subset check |
| "Dr. Amit Sharma" | person ✓ | Title stripped, 2 normal words |
| "Upto Five Student Representatives" | placeholder ✗ | Starts with "upto " |
| "Joint Secretary Ministry of Education" | placeholder ✗ | 2+ role descriptors: secretary, ministry |

### 3.3 Cross-Source Role Deduplication

When the same person appears in multiple graphs:

```python
def _add_role(self, name: str, role: dict):
    existing = self.person_roles[name]
    for r in existing:
        if r["source"] == role["source"] and r["designation"] == role["designation"]:
            return  # Skip duplicate
    self.person_roles[name].append(role)
```

Result for "Sartaj Ul Hasan":
```python
person_roles["Sartaj Ul Hasan"] = [
    {"source": "cse", "designation": "Professor", "label": "Faculty"},
    {"source": "academics", "designation": "Dean, Academic Programs", "label": "SectionHead"},
]
```

This enables the chatbot to answer "Who is Sartaj Ul Hasan?" with both his departmental and administrative roles.

---

## 4. Helper Functions: The Utility Layer

### 4.1 `_strip_markdown_link()`
```python
def _strip_markdown_link(text: str) -> str:
    match = re.match(r'^\[([^\]]+)\]\([^)]+\)$', text)
    return match.group(1).strip() if match else text
```
Converts `[Dr. Smith](https://...)` → `Dr. Smith`. Used throughout entity extraction.

### 4.2 `_deobfuscate_email_text()`
```python
def _deobfuscate_email_text(text: str) -> str:
    text = re.sub(r'\[\s*AT\s*\]', '@', text)
    text = re.sub(r'\[\s*DOT\s*\]', '.', text)
    text = re.sub(r'\s+at\s+', '@', text)
    text = re.sub(r'\s+dot\s+', '.', text)
    return text
```
Handles obfuscated emails like `saxena [AT] iitjammu [DOT] ac [DOT] in` → `saxena@iitjammu.ac.in`. University pages often obfuscate emails to prevent scraping — ironic given that we *are* scraping them.

### 4.3 `_is_probable_person_name()`
```python
def _is_probable_person_name(text: str) -> bool:
    if text.lower() in NON_RECORD_HEADINGS:  # "faculty", "phd students", etc.
        return False
    if any(term in text.lower() for term in ("publication", "conference", "journal")):
        return False
    parts = text.split()
    if len(parts) < 2 or len(parts) > 6:
        return False
    return all(any(ch.isalpha() for ch in part) for part in parts)
```
Guards against extracting section headings ("Publications", "Research") as person names. Requires 2–6 words, all containing alphabetic characters.

### 4.4 `_detect_person_record_level()`

A stricter variant of `_detect_repeated_heading_level()` that requires body content to contain faculty/student cues:

```python
body_cues = (
    "assistant professor", "associate professor", "professor",
    "supervisor", "research area", "research interest", "@iitjammu",
    "google scholar", "education qualification",
)
```

This prevents the system from treating a page of research area headings as a person roster.

---

## 5. Maintenance Guide

### 5.1 Adding a New Department

1. **Register** in `departments.py`: Add to `DEPARTMENTS` dict with code, name, full_name, base_url.
2. **Add aliases** to `DEPT_ALIASES` if the department has alternative names.
3. **Scrape** the department pages into `scraped_data/{code}/`.
4. **Add labs** to `CORRECT_LABS` if the department has known labs.
5. **Run ingestion**: `python ingest.py --dept {code}`.
6. **Verify**: Check `data/{code}/` for all 6 output files.

### 5.2 Adding a New Section

1. **Register** in `departments.py`: Add to `SECTIONS` dict.
2. **Create scraper** if the section has a non-standard page layout.
3. **Add parsers** to `SectionKGBuilder` if the section has unique content formats.
4. **Scrape** into `scraped_data/sections/{code}/`.
5. **Run**: `python ingest.py --section {code}`.

### 5.3 When Parsers Break

Parsers break when the university redesigns their website. Signs:

- **Empty graph**: 0 entity nodes → the heading structure changed.
- **Missing entities**: Faculty nodes exist but have no research interests → subsection heading format changed.
- **Duplicate entities**: Same person appears twice → `EntityResolver` failed to match a new name variant.

**Diagnosis**: Run ingestion with `LOG_LEVEL=DEBUG` and check:
```
DEBUG: infer_document_kind('ee_faculty-list__xyz.html.md') = 'faculty_profile'
DEBUG: _parse_faculty_profile: extracted name='...' from heading
DEBUG: EntityResolver: resolved 'A. K. Saxena' → 'Alok Kumar Saxena'
```

### 5.4 Updating the EntityResolver

If a new name variant isn't being resolved:
1. Check if the canonical name exists in the faculty roster file.
2. If not, add the person to the roster or add a hard-coded mapping in the resolver.
3. If the matcher threshold is too strict, lower it (but risk false positives).

### 5.5 FAISS Index Staleness

The FAISS index must be rebuilt whenever:
- Graph content changes (new entities, updated properties)
- Community assignments change
- The embedding model is updated
- The entity description templates change

There is no automatic staleness detection — the operator must track when the last rebuild occurred relative to the last scrape.

### 5.6 Performance Monitoring

Key metrics to track per department:

| Metric | Healthy Range | Action if Out of Range |
|---|---|---|
| Entity nodes | 50–500 | <50: parser may be broken. >500: check for duplicates |
| Text chunks | 100–1500 | <100: scraper may have failed. >1500: review chunk size |
| Communities | 3–20 | <3: graph too sparse. >20: check resolution parameter |
| FAISS vectors | 200–2000 | Should be sum of chunks + entities + communities |
| Resolution ratio | >80% of names resolved to canonical | <80%: review canonical faculty list |
| Ingestion time (no LLM) | 10–60s | >120s: review file count and chunk sizes |

---

## 6. Known Limitations and Future Improvements

### 6.1 Current Limitations

1. **Parser brittleness**: Heavily coupled to specific HTML/markdown structures. Website redesigns require parser updates.
2. **No incremental updates**: Full rebuild required for any change.
3. **Single-department resolver scope**: Entity resolution operates within one department at a time. Cross-department resolution only happens at runtime via `GlobalPersonIndex`.
4. **Initials ambiguity**: "S. Singh" could match multiple faculty members. First-match wins.
5. **No validation gate**: No automated check that entity counts match expectations after ingestion.

### 6.2 Potential Improvements

1. **Schema validation**: Post-ingestion checks comparing entity counts against expected baselines.
2. **Incremental embedding**: Add new vectors without rebuilding the full FAISS index using `index.add()`.
3. **Cross-department resolution during ingestion**: Share a global resolver across all department builds.
4. **LLM-based entity extraction**: For pages that don't follow any known template, use an LLM to extract entities. Currently these fall through to `generic` doc_kind with chunk-only indexing.
5. **Automated parser health checks**: Compare entity counts across consecutive ingestion runs and alert on significant drops.

---

*This concludes the 5-part technical documentation of the IIT Jammu chatbot data ingestion pipeline. Together, these documents cover: (1) pipeline orchestration, (2) knowledge graph construction with 25+ parsers, (3) community detection and LLM summarization, (4) embedding generation and FAISS indexing, and (5) entity resolution, identity management, and maintenance guidance.*
