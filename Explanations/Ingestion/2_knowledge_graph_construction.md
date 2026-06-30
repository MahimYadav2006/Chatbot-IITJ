# Part 2 — Knowledge Graph Construction (Phase 1)

> *From raw markdown to a typed, deduplicated knowledge graph: every parser, heuristic, and design decision explained.*

---

## 1. Phase 1 Entry Point: `KnowledgeGraphBuilder.build()`

The `build()` method in `kg_builder.py` (line 2116) is the single entry point for Phase 1. It orchestrates a 5-step process:

```
Step 1: File Discovery & Document Node Creation
Step 2: Document Kind Classification (infer_document_kind)
Step 3: Faculty Roster Pre-Parse (canonical name seeding)
Step 4: Per-File Entity Extraction (25+ parsers)
Step 5: Cross-Linking & Department Node Assembly
```

### 1.1 File Discovery

```python
filenames = [f for f in os.listdir(self.markdown_dir)
             if f.endswith(".md") and not f.startswith("00_combined")
             and not f.endswith(".json")]
```

The `00_combined` exclusion is critical — some scraping runs produce a concatenated mega-file of all pages. Including it would double-count every entity. The `.json` exclusion prevents metadata files from being parsed as content.

### 1.2 The Two-Pass Architecture

Phase 1 uses a **two-pass** approach over the file list:

**Pass 1** (lines 2129–2137): Creates `Document` nodes and text chunks for *every* file. This pass also runs `infer_document_kind()` to classify each file.

**Pass 2** (lines 2148–2214): Runs the appropriate entity-specific parser based on the `doc_kind` classification. The faculty roster is parsed *between* the passes (line 2149–2152) to seed the `EntityResolver`.

This separation is intentional — all document/chunk nodes must exist before entity parsers can create `SOURCE_DOCUMENT` edges back to them.

---

## 2. Document Node Creation & Smart Chunking

For every markdown file, `_create_document_node()` (line 884) performs:

1. **Source URL extraction** — Parses the `# Source URL:` header line injected by the scraper.
2. **Title generation** — Converts the filename to a human-readable title: `ee_faculty-list.html.md` → `"Faculty List"`.
3. **Boilerplate stripping** — Calls `clean_content_for_chunks()` to remove navigation noise.
4. **Smart chunking** — Calls `smart_chunk_text()` to split the cleaned content.
5. **Chunk node creation** — Each chunk becomes a `TextChunk` node connected to the document via a `HAS_CHUNK` edge.

### 2.1 Boilerplate Stripping: `clean_content_for_chunks()`

The scraped markdown contains extensive navigation boilerplate — sidebar menus, breadcrumbs, "Read more" links, department logos. The `BOILERPLATE_PATTERNS` list (line 87) defines 30+ regex patterns to strip:

```python
BOILERPLATE_PATTERNS = [
    r'^# Source URL:.*$',                    # Scraper metadata
    r'^html\s*$',                            # Bare HTML artifacts
    r'^\!\[.*?\]\(.*?\)\s*$',                # Inline images
    r'^- \[Home\].*$',                       # Navigation links
    r'^- \[Faculty\]\(.*$',                  # Sidebar menu items
    r'^\[Read more\]\(.*?\)\s*$',            # Truncated teasers
    r'^### People\s*$',                      # Section headers that are navigation
    r'^ducation\s*$',                        # Truncated text artifacts
    ...
]
```

After pattern removal, the function also:
- Collapses triple+ newlines to double newlines
- Strips markdown link syntax but keeps link text: `[Dr. Smith](url)` → `Dr. Smith`
- Removes leftover image references
- Fixes concatenated text like `"ProfessorResearch Experience"` → `"Professor\nResearch Experience"`

### 2.2 The 4-Tier Smart Chunking Cascade

`smart_chunk_text()` (line 529) implements a **cascading strategy selector** — it tries the most structure-preserving strategy first and falls back to simpler ones:

```
Tier 1: Single Chunk (≤400 words) → return as-is
    ↓ (text too long)
Tier 2: Repeated Heading Records → chunk by roster entries
    ↓ (no repeated records detected)
Tier 3: Heading Sections → chunk by ## / ### / #### boundaries
    ↓ (no heading structure)
Tier 4: Structural Blocks → chunk by paragraphs + tables
    ↓ (single block)
Tier 5: Word Window → fixed 400-word sliding window with 80-word overlap
```

#### Tier 1: Single Chunk
If the entire cleaned text is ≤400 words, it's returned as a single chunk. No splitting needed.

#### Tier 2: Repeated Heading Records (`_chunk_repeated_records`)
This is the most important strategy for roster-style pages (faculty lists, PhD student lists, staff lists). The function:

1. **Auto-detects the heading level** used for records by scanning levels 3–6 and scoring each by how many headings look like roster entries (via `_detect_repeated_heading_level`).
2. **Splits** the text into a prefix (content before the first record) and individual records.
3. **Groups** records into chunks that fit within the 400-word budget, never splitting a single record across chunks.

The detection uses `_is_probable_record_heading()` which filters out known non-record headings like "Education", "Research Interests", "Publications" — these are subsection labels *within* a record, not record boundaries.

**Why this matters**: Without record-aware chunking, a word-window chunk might split a faculty member's name from their research interests, destroying the semantic relationship. Record-aware chunking guarantees each chunk contains complete person entries.

#### Tier 3: Heading Sections
If no repeated records are detected, the text is split at heading boundaries (`## `, `### `, etc.). Sections that exceed 400 words are further split using Tier 4.

#### Tier 4: Structural Blocks (`_chunk_structural_blocks`)
Splits by paragraph boundaries and table boundaries. Tables are kept intact — a markdown table is never split mid-row. This preserves tabular data like fee structures and placement statistics.

#### Tier 5: Word Window
The fallback for unstructured text. A sliding window of 400 words with 80-word overlap ensures no information is lost at boundaries.

### 2.3 Chunk Metadata

Every chunk carries metadata about *how* it was created:

```python
{"text": "...", "meta": {"strategy": "repeated_heading_records", "record_heading_level": 4, "record_count": 23}}
{"text": "...", "meta": {"strategy": "heading_sections", "section_count": 8}}
{"text": "...", "meta": {"strategy": "structural_blocks", "block_count": 12}}
{"text": "...", "meta": {"strategy": "word_window"}}
```

This metadata is stored on both the chunk node (`chunk_strategy` property) and the document node, enabling downstream analysis of chunking effectiveness.

---

## 3. Document Kind Classification: `infer_document_kind()`

The `infer_document_kind()` function (line 668) is the **routing brain** of Phase 1. It examines each file's name and content to assign one of 25+ document kinds. The classifier uses a **priority-ordered cascade** of pattern matches:

### 3.1 The Classification Priority Chain

```python
# Priority 1: Administrative pages (checked by filename only)
if "director" in fn:          return "admin_director"
if "registrar" in fn:         return "admin_registrar"
if "bogchairman" in fn:       return "admin_bogchairman"
if "deans-and-associate-deans" in fn: return "admin_deans"
if any(p in fn for p in ("board-of-governors", "finance-committee", ...)):
                              return "admin_committee"

# Priority 2: PDF-converted files (always generic — too unpredictable)
if fn.endswith(".pdf.md"):    return "generic"

# Priority 3: Faculty profiles (filename + content heuristics)
if "__" in fn and ("assistant professor" in text or "research interests" in text):
                              return "faculty_profile"

# Priority 4: Faculty roster (filename only, no __ in name)
if "faculty-list" in fn and "__" not in fn:
                              return "faculty_roster"

# Priority 5: PhD alumni (must come BEFORE generic phd-list)
if "phd-alumni" in fn:        return "phd_alumni"

# Priority 6: PhD roster (filename + content structure)
if "phd-list" in fn or ("research area" in text and "supervisor" in text
                         and _detect_person_record_level(content)):
                              return "phd_roster"

# ... 15+ more rules ...

# Final fallback: Source URL heuristics
if "/faculty-list/~" in source_line: return "faculty_profile"
if "/phd-list" in source_line:       return "phd_roster"

# Ultimate fallback
return "generic"
```

### 3.2 Why Priority Order Matters

The ordering is carefully designed to avoid misclassification:

- **`phd_alumni` before `phd_roster`**: Both contain "phd" in the filename, but alumni pages have different structure (graduated students with thesis titles) vs. active student rosters (with supervisors and ongoing research areas).
- **`faculty_profile` requires `__`**: The double-underscore convention (`ee_faculty-list__dr-alok-kumar-saxena.html.md`) distinguishes individual profile pages from the master roster.
- **`admin_*` types are checked first**: Administrative pages have unique parsers and must not fall through to generic handling.
- **PDF files always go to `generic`**: PDF-to-markdown conversion produces unpredictable layouts that no specialized parser can handle reliably.

### 3.3 The Source URL Fallback

Some files don't follow naming conventions. The classifier extracts the `# Source URL:` header from the first 3 lines and uses URL path patterns as a last resort:

```python
source_line = next((line.lower() for line in content.splitlines()[:3]
                    if "source url" in line.lower()), "")
if "/faculty-list/~" in source_line: return "faculty_profile"
```

The `~` in the URL is a telltale sign of an individual faculty page on the IIT Jammu website.

### 3.4 Complete Document Kind → Parser Mapping

| doc_kind | Parser Method | Entity Types Created |
|---|---|---|
| `faculty_profile` | `_parse_faculty_profile` | Faculty, ResearchArea |
| `faculty_roster` | `_parse_faculty_list` | Faculty |
| `phd_roster` | `_parse_phd_list` | PhDStudent, Faculty (supervisors) |
| `mtech_roster` | `_parse_phd_list(label="MTechStudent")` | MTechStudent |
| `phd_alumni` | `_parse_phd_alumni` | GraduatedPhD |
| `hod_message` | `_parse_hod` | Faculty (with `is_hod=True`) |
| `funded_projects` | `_parse_funded_projects` | Project, FundingAgency |
| `patents` | `_parse_patents` | Patent, Faculty/ExternalPerson |
| `startups` | `_parse_startups` | Startup, Faculty |
| `research_areas` | `_parse_research_areas` | ResearchCategory, ResearchArea |
| `awards` | `_parse_awards` | Award |
| `publications` | `_parse_publications_page` | Publication |
| `staff` | `_parse_staff` | Staff |
| `programmes` | `_parse_programmes` | Program, Course |
| `labs` | `_parse_labs` | Lab |
| `contact` | `_parse_contact` | ContactInfo |
| `alumni` | `_parse_alumni` | Alumni, AlumniBatch |
| `placement_industry` | `_parse_placement_data` | PlacementData |
| `placement_academia` | `_parse_higher_studies` | HigherStudiesData |
| `admin_director` | `_parse_admin_director` | Faculty (with `is_director=True`) |
| `admin_registrar` | `_parse_admin_registrar` | AdminOfficial |
| `admin_bogchairman` | `_parse_admin_bogchairman` | AdminOfficial |
| `admin_deans` | `_parse_admin_deans` | Faculty/AdminOfficial |
| `admin_committee` | `_parse_admin_committee` | Committee, Faculty/AdminOfficial |
| `generic` | (no entity parser — chunks only) | — |

Note: `_parse_labs` runs on **every** file regardless of `doc_kind` (line 2214), because lab mentions can appear anywhere — in department index pages, research areas pages, or dedicated lab pages.

---

## 4. Entity Resolution: The `EntityResolver` Class

The `EntityResolver` (line 784) is the **deduplication engine** that prevents the knowledge graph from fragmenting into disconnected name variants. Without it, "Dr. A. K. Saxena", "Alok Kumar Saxena", "A.K. Saxena", and "Prof. Alok Saxena" would each create separate Faculty nodes with no shared edges.

### 4.1 Architecture

```python
class EntityResolver:
    def __init__(self, canonical_faculty: set = None):
        self.canonical_names = {}        # normalized_name → canonical_name
        self.name_variants = defaultdict(set)  # canonical_name → {variant1, variant2, ...}
        self._canonical_faculty = canonical_faculty or set()
        # Pre-register canonical faculty as self-mappings
        for name in self._canonical_faculty:
            self.canonical_names[name] = name
            self.name_variants[name].add(name)
```

The resolver maintains two data structures:
- `canonical_names`: A flat lookup mapping every seen normalized name to its canonical form.
- `name_variants`: A reverse index mapping each canonical name to all its known variants.

### 4.2 The Resolution Cascade

When `resolve(raw_name)` is called, it executes a 3-tier matching cascade:

```
Input: "Dr. A. K. Saxena"
    │
    ▼
Step 1: normalize_name() → "A K Saxena"
    │
    ▼
Step 2: Check canonical_names cache → miss
    │
    ▼
Step 3: Check canonical faculty (3 matchers in sequence):
    ├── fuzzy_match("a k saxena", "alok kumar saxena") → miss (ratio < 0.85)
    ├── _initials_match("a k saxena", "alok kumar saxena") → HIT ✓
    │       Last name "saxena" == "saxena" ✓
    │       "a" matches start of "alok" ✓
    │       "k" matches start of "kumar" ✓
    └── Result: "Alok Kumar Saxena"
    │
    ▼
Step 4: Cache the mapping: canonical_names["a k saxena"] = "Alok Kumar Saxena"
Step 5: Record variant: name_variants["Alok Kumar Saxena"].add("a k saxena")
    │
    ▼
Return: "Alok Kumar Saxena"
```

### 4.3 The Three Matchers

#### Matcher 1: `fuzzy_match()` (SequenceMatcher, threshold ≥ 0.85)
Catches minor spelling variations: "Aurbind Gupta" vs "Arvind Gupta" (ratio = 0.87).

#### Matcher 2: `_initials_match()`
Specialized for abbreviated names. Rules:
- Last names must match (ratio ≥ 0.85)
- All preceding tokens in the short form must be initials (≤2 chars) matching the start of corresponding tokens in the full form
- Example: "B. N. Subudhi" matches "Badri Narayan Subudhi"

#### Matcher 3: `_token_subset_match()`
Handles missing middle names: "Anup Shukla" matches "Anup Kumar Shukla". Rules:
- First and last tokens must match (ratio ≥ 0.85)
- Middle tokens in the shorter name must appear in order in the longer name

### 4.4 Priority: Canonical Faculty First

The cascade checks canonical faculty names **before** checking previously resolved names. This ensures that if a canonical faculty member's name appears in any variant, it always resolves to the official registry entry — not to some earlier, potentially incorrect resolution.

### 4.5 The `normalize_name()` Function

Before any matching, names go through normalization (line 150):

```python
def normalize_name(name: str) -> str:
    name = TITLE_PREFIXES.sub('', name)  # Remove Dr., Prof., Mr., etc.
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.strip('., ')
    # Capitalize: initials stay uppercase, words get title-cased
    # "a.k. saxena" → "A.K. Saxena"
    # "SHARMA" → "SHARMA" (≤3 chars stays uppercase)
```

### 4.6 Canonical Faculty Seeding

The `_extract_canonical_faculty()` function (line 751) builds the ground-truth faculty registry by:

1. Finding the `faculty-list` file (prioritizing non-profile files without `__`).
2. Auto-detecting the heading level for person records.
3. Extracting and normalizing all heading names with ≥2 words.

This registry typically contains 15–30 names per department and serves as the "anchor set" for all subsequent entity resolution.

---

## 5. Deep Dive: Individual Parsers

### 5.1 `_parse_faculty_profile` — The Most Complex Parser

This parser (line 920) handles individual faculty profile pages. It extracts:

**Name extraction** (4 strategies):
1. Heading match: `### **Dr. Alok Kumar Saxena**` → extract from `###`/`####` headings in first 40 lines.
2. The heading must pass `_is_probable_person_name()` to avoid matching section headings like "Publications".
3. Bullet-point fallback: `- Alok Kumar Saxena` → extract from list items (excluding emails, URLs, and role titles).
4. Filename fallback: `ee_faculty-list__dr-alok-kumar-saxena.html.md` → extract from the `__` suffix.

**Section map extraction**: Uses `_extract_named_sections()` to parse `####`/`#####` headings into a canonical section map via `SECTION_ALIASES`:
```
"Education Qualification" → "education"
"Research Interest"       → "research_interests"
"Awards & Honours"        → "awards"
```

**Research interest parsing**: Extracts from multiple sources in priority order:
1. `research_interests` section
2. `academic_interests` section
3. `brief_info` section

Each source is further parsed for comma/semicolon/newline-separated items, with markdown emphasis (`**bold**`) stripped.

**Hard-coded domain guarantees** (line 1014): For specific faculty members (Archana Rajput, Alok Kumar Saxena), the parser explicitly injects RF, Microwave, and Antenna design research areas. This compensates for cases where the scraper can't extract these from dynamic page elements.

### 5.2 `_parse_hod` — The 4-Strategy HoD Detector

The HoD parser (line 1371) is the most complex detection problem because HoD pages have wildly inconsistent formats across departments:

**Strategy A: Heading with HoD tag nearby**
Scans all headings, checks if any nearby heading (within ±2 positions) contains "hod" or "head", then verifies the candidate name against the canonical faculty registry.

**Strategy A.5: Dr./Prof. prefix near "Head of Department" text**
For headings with academic prefixes, checks if "Head of Department" appears within 300 characters in the content.

**Strategy B: Image filename extraction**
Some HoD pages only show the name in the profile photo's filename: `![ ](Prof_Alok_Kumar_Saxena.jpg)`. The parser URL-decodes the filename, normalizes it, and checks against canonical faculty.

**Strategy C: Canonical name in content**
Last-resort scan: checks if any canonical faculty name appears in the content alongside "hod", "head", or "message" keywords.

**Strategy D: Regex patterns**
Traditional regex: `### Dr. Name\n\nHead of Department`.

### 5.3 `_parse_funded_projects` — Dual-Format Handler

This parser (line 1074) handles two distinct markdown formats:

**Format 1: Markdown tables** (detected by `|` in content)
- Auto-detects header columns by scanning for keywords: "pi", "investigator", "grant", "cost", "agency", "project", "title"
- Skips header rows and separator rows (`---`)
- Skips rows where all cells are non-numeric (likely sub-headers)
- Creates `Project` + `FundingAgency` nodes with `FUNDED_BY` edges
- Links PIs to projects via `PRINCIPAL_INVESTIGATOR` edges (only if PI is canonical faculty)

**Format 2: Numbered lists** (detected by `- [N]` pattern)
- Joins multi-line wrapped entries into single logical lines
- Parses `[N] Title: ..., Funding Agency: ...` format
- Handles entries without explicit "Funding Agency:" labels

### 5.4 `_parse_patents` — Dual-Strategy Extraction

**Strategy 1: Structured `**Title**`/`**Inventors**` format** (EE-style)
Uses regex to capture Title, Inventors, and Application No blocks.

**Strategy 2: Numbered list format** (Physics-style)
Parses `1. Author1, Author2 "Title" Patent No 123, Year` entries.
Extracts quoted titles, patent numbers, and matches inventors against canonical faculty.

### 5.5 `_parse_labs` — The Multi-Source Lab Detector

Labs are mentioned inconsistently across pages, so this parser (line 1587) uses three extraction strategies:

1. **Heading matching**: Regex for headings containing "lab", "laboratory", or "facility" at any level (##–######).
2. **Bullet list matching**: Regex for `- Lab Name` items.
3. **Lab page headings**: For files classified as `doc_kind == "labs"`, extracts all `##`–`####` headings.

Every candidate is validated against `CORRECT_LABS` from the registry using case-insensitive, whitespace-normalized matching. Only confirmed lab names become `Lab` nodes.

After all files are parsed, the `build()` method **seeds any missing labs** from `CORRECT_LABS` (line 2265–2277), ensuring completeness even if the parser missed some.

---

## 6. Node & Edge Management

### 6.1 `_add_node()` — Namespace-Aware Node Creation

```python
def _add_node(self, node_id: str, label: str, **properties):
    dept_id = f"IIT Jammu {self.dept_code.upper()} Department"
    if self.dept_code != "ee" and node_id != dept_id and not node_id.startswith(f"{self.dept_code}:"):
        node_id = f"{self.dept_code}:{node_id}"
    properties["department"] = self.dept_code
    if self.graph.has_node(node_id):
        self.graph.nodes[node_id].update(properties)  # Merge, don't overwrite
    else:
        self.graph.add_node(node_id, label=label, **properties)
    return node_id
```

Key behaviors:
- **Namespace prefixing**: Non-EE departments get their node IDs prefixed with `{dept_code}:` to prevent cross-department collisions (e.g., both CSE and EE might have a "Machine Learning" research area).
- **EE exception**: The EE department (the first one built) uses unprefixed IDs for backward compatibility.
- **Upsert semantics**: If a node already exists, its properties are *merged* (updated), not replaced. This allows multiple parsers to enrich the same entity.

### 6.2 `_add_edge()` — Safe Edge Creation

```python
def _add_edge(self, source: str, target: str, rel_type: str, **properties):
    # Apply namespace prefixing (same logic as _add_node)
    if not self.graph.has_node(source) or not self.graph.has_node(target):
        return  # Silent skip — both endpoints must exist
    if self.graph.has_edge(source, target):
        if self.graph.edges[source, target].get('type', '') == rel_type:
            return  # Deduplicate identical edges
    self.graph.add_edge(source, target, type=rel_type, **properties)
```

The existence check (`has_node`) prevents dangling edges. The duplicate check prevents the same relationship from being created twice.

---

## 7. Cross-Linking: The Post-Parse Enrichment

After all individual parsers run, `build()` performs cross-linking (line 2218):

### 7.1 Faculty → ResearchArea via PhD Supervisions

```python
for faculty in faculty_nodes:
    students = [e[0] for e in self.graph.in_edges(faculty)
               if self.graph.edges[e[0], faculty].get('type') == 'SUPERVISED_BY']
    for student in students:
        for _, target in self.graph.out_edges(student):
            if self.graph.nodes[target].get('label') == 'ResearchArea':
                if not self.graph.has_edge(faculty, target):
                    self._add_edge(faculty, target, "RESEARCHES_IN")
```

This creates **inferred** `RESEARCHES_IN` edges: if Dr. Smith supervises a PhD student working on "Machine Learning", then Dr. Smith is inferred to research "Machine Learning". This enriches faculty nodes that might not have explicit research interest listings on their profile pages.

### 7.2 Department Node Assembly

The department node aggregates statistics computed from the graph:

```python
self.graph.nodes[dept_id]['faculty_count'] = len(faculty_nodes)
self.graph.nodes[dept_id]['phd_student_count'] = len(phd_nodes)
self.graph.nodes[dept_id]['faculty_structured_fields'] = sorted(faculty_structured_fields)
self.graph.nodes[dept_id]['phd_structured_fields'] = sorted(phd_structured_fields)
self.graph.nodes[dept_id]['document_kind_counts'] = dict(...)
```

These statistics are used by the retrieval layer to generate department overview responses and by validation tools to check ingestion completeness.

### 7.3 Lab Seeding

```python
from departments import CORRECT_LABS
for correct_lab in CORRECT_LABS.get(self.dept_code, []):
    lab_id = f"lab:{correct_lab}"
    if not self.graph.has_node(full_lab_id):
        self._add_node(lab_id, "Lab", name=correct_lab)
    self._add_edge(lab_id, dept_id, "FACILITY_OF")
```

Every lab from the canonical registry gets a `FACILITY_OF` edge to the department node, ensuring complete lab listings.

---

## 8. The Section KG Builder: `SectionKGBuilder`

The `SectionKGBuilder` (`section_kg_builder.py`) mirrors `KnowledgeGraphBuilder` in structure but has completely different parsers optimized for institutional content:

### 8.1 Key Differences from Department Builder

| Aspect | KnowledgeGraphBuilder | SectionKGBuilder |
|---|---|---|
| Canonical faculty seeding | Yes (`_extract_canonical_faculty`) | No (cold-start resolver) |
| Node ID prefix | `{dept_code}:` | `{section_code}:` |
| `doc_kind` classification | `infer_document_kind()` | Filename pattern matching in `build()` |
| Notification dates | Not extracted | Extracted from `**Date:**` headers |

### 8.2 Section-Specific Parsers

- **`_parse_people_list`**: Handles `#### Name` / `###### Designation` / `##### email` format.
- **`_parse_counselling_team`**: Parses coordinator tables with phone, email, and office columns.
- **`_parse_counselor_profiles`**: Splits `**Name**` blocks to extract bios.
- **`_parse_di_team`**: Parses Digital Infrastructure team hierarchy.
- **`_parse_committee_document`**: Handles DPGC/DUGC committee rosters partitioned by department headings.
- **`_parse_advisor_document`**: Parses faculty advisor assignment tables by programme and batch year.
- **`_parse_fee_structure_document`**: Extracts fee tables segmented by programme level and income category.
- **`_parse_notification_policy_document`**: The richest parser — extracts notification number, date, category, applies_to, keywords, summary, eligibility criteria, procedure steps, and hard-coded key facts for known policy types.
- **`_parse_curriculum_document`**: Parses course tables with code, name, L-T-P structure, and credits. Handles semester grouping, elective buckets, and superseded curriculum versions.

### 8.3 The Policy Notification Parser: Structured Knowledge Extraction

The `_parse_notification_policy_document()` method (line 586) is perhaps the most sophisticated parser in the entire codebase. It extracts 10 dimensions from policy documents:

1. **Notification number**: Regex for `IITJMU/...` format
2. **Date**: Multi-format parsing (3 cascading regex patterns)
3. **Title**: From H1 heading or filename
4. **Category**: 15+ filename-based rules (internship_policy, financial_policy, phd_policy, grading_policy, etc.)
5. **Applies to**: Content-based detection of B.Tech, PG, PhD, International Students
6. **Keywords**: 20+ topic keywords matched against content
7. **Summary**: First 3 non-boilerplate paragraphs
8. **Eligibility criteria**: Extracted from sections with headings containing "eligibility", "condition", "requirement"
9. **Procedure steps**: Extracted from sections with headings containing "procedure", "step", "process"
10. **Key facts**: Hard-coded slabs and limits for known policy types (fee waiver percentages, internship durations, transfer CGPA requirements, etc.)

The hard-coded key facts are a deliberate design choice: these are high-value, frequently-queried data points (e.g., "What is the CGPA requirement for PhD transfer?") that are difficult to extract reliably from free-form text. By encoding them directly, the system guarantees precise answers.

---

## Next: Part 3 — Community Detection & Summarization

The next document covers how the flat knowledge graph is partitioned into thematic clusters using Louvain community detection, and how LLM-generated summaries create a semantic bridge between structured entities and free-text retrieval.
