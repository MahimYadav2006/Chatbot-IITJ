# Part 1 — Pipeline Overview & Orchestration

> *How raw markdown becomes a queryable knowledge base: the definitive technical reference.*

---

## 1. Design Philosophy: "Structure Over Fuzzy"

Before diving into code, it's critical to understand the *why* behind this pipeline. Most chatbot RAG systems follow a simple pattern: **chunk text → embed → vector search**. This works for generic Q&A but fails catastrophically for an institutional knowledge base where users ask questions like:

- *"Who is the HoD of the EE department?"*
- *"What projects is Dr. Ankit Dubey working on?"*
- *"List the PhD students supervised by Prof. Saxena."*
- *"What is the fee structure for M.Tech 2025 batch?"*

These queries demand **structured, relational answers** — not a fuzzy similarity match against text blobs. The IIT Jammu chatbot solves this with a philosophy called **"Structure Over Fuzzy"**:

| Traditional RAG | This Pipeline (GraphRAG) |
|---|---|
| Chunk raw text into ~500-word windows | Parse markdown into **typed entities** (Faculty, Lab, Course, etc.) |
| Embed chunks, hope similarity catches the answer | Build a **directed knowledge graph** with explicit relationships |
| Flat retrieval — every chunk is equal | **Hierarchical retrieval** — entities, communities, and chunks each serve different query types |
| No deduplication — "Dr. A. K. Saxena" and "Alok Kumar Saxena" are separate | **Fuzzy entity resolution** unifies name variants into canonical nodes |
| Generic chunking loses table structure | **Structure-aware chunking** preserves tables, headings, and records as atomic units |

The result is a system that can answer both factual/relational queries (graph traversal) *and* open-ended questions (vector search) from the same underlying data.

### 1.1 Why Not Just Use an LLM with Raw Text?

A natural question: why not feed all the markdown directly to a large language model and let it answer? Three reasons:

1. **Context window limits.** Even with 128K-token models, the combined scraped data for all departments exceeds any single context window. You *must* have a retrieval layer.

2. **Hallucination risk.** Without structured grounding, LLMs confidently fabricate faculty names, research areas, and policy details. The knowledge graph provides a verifiable source of truth.

3. **Deterministic queries.** When a user asks "Who is the HoD?", the answer should come from a graph lookup (`is_hod=True`), not from the LLM's probabilistic interpretation of a markdown page. Deterministic paths eliminate an entire class of wrong answers.

### 1.2 The Three Retrieval Tiers

The pipeline produces three retrieval tiers, each optimized for a different query type:

```
┌──────────────────────────────────────────────────────────┐
│  TIER 1: Structured Graph Queries                        │
│  ──────────────────────────────────                      │
│  "Who is the HoD?" → graph.nodes[x]['is_hod'] == True   │
│  "List PhD students of Prof. X" → in_edges(SUPERVISED_BY)│
│  Zero hallucination risk. Deterministic.                 │
├──────────────────────────────────────────────────────────┤
│  TIER 2: Entity Description Search                       │
│  ──────────────────────────────────                      │
│  "Tell me about Dr. Ankit Dubey" → entity embedding match│
│  Rich natural-language summaries of graph entities.       │
│  Bridges structured data with semantic similarity.        │
├──────────────────────────────────────────────────────────┤
│  TIER 3: Raw Chunk Search                                │
│  ──────────────────────────────                          │
│  "What is the department's vision?" → chunk similarity   │
│  Falls back to raw text when no entity matches.          │
│  Handles open-ended, exploratory queries.                │
└──────────────────────────────────────────────────────────┘
```

The retrieval layer at query time walks down these tiers. The ingestion pipeline's job is to ensure all three tiers are populated with high-quality, deduplicated data.

---

## 2. The Dual-Track Ingestion Model

The pipeline handles two fundamentally different data sources, each with its own builder class but sharing the same 4-phase pipeline:

### Track A: Academic Departments

These are the 11+ department websites (EE, CSE, Physics, Chemistry, Mathematics, Civil Engineering, BSBE, HSS, Materials Engineering, Mechanical Engineering, Chemical Engineering, etc.), each following a similar SPA structure with pages for faculty profiles, PhD students, research areas, funded projects, and HoD messages.

- **Builder**: `graphrag/kg_builder.py` → `KnowledgeGraphBuilder`
- **Data Source**: `scraped_data/{dept_code}/` (e.g., `scraped_data/ee/`)
- **Output**: `data/{dept_code}/` (e.g., `data/ee/`)
- **Entity Types**: `Faculty`, `PhDStudent`, `MTechStudent`, `GraduatedPhD`, `ResearchArea`, `ResearchCategory`, `Lab`, `Project`, `Patent`, `Startup`, `Course`, `Program`, `Award`, `Publication`, `Staff`, `PlacementData`, `HigherStudiesData`, `ContactInfo`, `Alumni`, `AlumniBatch`, `AdminOfficial`, `Committee`, `FundingAgency`, `Department`, `ExternalPerson`

### Track B: Institutional Sections

These are cross-cutting institutional pages — Academics, Administration, Student Life, Medical Centre, Placements, PMRF, Counselling, Digital Infrastructure, Quick Links (committees, adjunct faculty, anti-ragging, etc.), Central Instruments Facility, and more.

- **Builder**: `graphrag/section_kg_builder.py` → `SectionKGBuilder`
- **Data Source**: `scraped_data/sections/{section_code}/` (e.g., `scraped_data/sections/academics/`)
- **Output**: `data/sections/{section_code}/`
- **Entity Types**: `SectionHead`, `SectionPerson`, `Counselor`, `MedicalDoctor`, `CommitteeMember`, `PolicyNotification`, `FeeStructure`, `AcademicProgram`, `Specialization`, `Course`, `ElectiveBucket`, `SectionContact`, `Hostel`, `Club`, `Holiday`, `AdjunctFaculty`, and many more

### 2.1 Why Two Tracks?

Department pages and section pages have fundamentally different content structures:

| Aspect | Department Pages | Section Pages |
|---|---|---|
| **Primary entity** | Faculty member (profile-centric) | Policy, committee, or service |
| **Content layout** | Heading-based profiles with subsections | Tables, notification documents, rosters |
| **Name resolution** | Seeded by canonical faculty registry | No pre-existing registry; resolver starts cold |
| **Relationship model** | Faculty → ResearchArea, Faculty → Student | Committee → Members, Policy → Programme |
| **Parser count** | 25+ specialized parsers | 15+ specialized parsers |

Trying to handle both with a single parser would produce brittle, unmaintainable code. The dual-track design lets each builder specialize in its domain's markdown structure while sharing the downstream community detection and embedding infrastructure.

### 2.2 How Routing Works

The routing decision happens in `ingest.py` based on CLI arguments:

```python
# Department track
if args.dept:
    canonical_dept = resolve_department_code(dept)  # "computer_science_engineering" → "cse"
    ingest_department(canonical_dept, ...)

# Section track
if args.section:
    ingest_section(section_code, ...)
```

The `resolve_department_code()` function handles aliases — users can pass `"computer_science_engineering"`, `"cse"`, or `"computer_science"` and they all resolve to the canonical `"cse"` code. This alias resolution is defined in `departments.py` via the `DEPT_ALIASES` mapping.

### 2.3 The Registry: `departments.py`

This module is the **single source of truth** for all department and section configurations. It defines:

| Registry Element | Purpose | Example |
|---|---|---|
| `DEPARTMENTS` dict | Maps dept codes to config (name, base_url, full_name) | `"ee" → {"name": "Electrical Engineering", ...}` |
| `SECTIONS` dict | Maps section codes to config | `"academics" → {"name": "Academic Office", ...}` |
| `DEPT_ALIASES` dict | Maps human-friendly names to canonical codes | `"computer_science_engineering" → "cse"` |
| `CORRECT_LABS` dict | Canonical lab name lists per department | `"ee" → ["Power Electronics Lab", ...]` |
| `get_markdown_dir(code)` | Returns input path for a department | `scraped_data/ee/` |
| `get_data_dir(code)` | Returns output path for a department | `data/ee/` |
| `get_section_markdown_dir(code)` | Returns input path for a section | `scraped_data/sections/academics/` |
| `get_section_data_dir(code)` | Returns output path for a section | `data/sections/academics/` |

The `CORRECT_LABS` dictionary deserves special attention. Some department pages list labs in inconsistent formats or miss them entirely. The registry provides a ground-truth list of lab names per department. During KG construction, the builder **seeds** these labs into the graph regardless of whether the parser extracted them — this guarantees that queries like "What labs does the EE department have?" always return complete results.

---

## 3. The 4-Phase Pipeline

Every ingestion run — whether for a department or a section — follows the same 4-phase progression. The orchestrator is `ingest.py`:

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW MARKDOWN FILES                          │
│              scraped_data/{dept_code}/*.md                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Knowledge Graph Construction                          │
│  ─────────────────────────────────────                          │
│  • Read every .md file in the directory                          │
│  • Smart-chunk text into contextual pieces                       │
│  • Classify each file by "doc_kind" (faculty_profile,            │
│    phd_roster, funded_projects, hod_message, etc.)               │
│  • Run the appropriate entity-specific parser                    │
│  • Build a NetworkX DiGraph with typed nodes + edges             │
│  • Cross-link entities (Faculty→ResearchArea via PhD students)   │
│                                                                  │
│  Output: graph.pkl, chunks.json, resolver.pkl                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Community Detection                                    │
│  ────────────────────────────                                    │
│  • Convert DiGraph → undirected for Louvain                      │
│  • Filter out TextChunk/Document nodes (structural noise)        │
│  • Remove isolated nodes (degree 0)                              │
│  • Run python-louvain with resolution=1.0, seed=42               │
│  • Group entity nodes into thematic clusters                     │
│                                                                  │
│  Output: partition dict (node → community_id)                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: Community Summarization                                │
│  ────────────────────────────────                                │
│  • Build structured text reports for each community              │
│  • Feed reports to an LLM (Gemini/Ollama) for summarization      │
│  • Fallback to rule-based summary if LLM unavailable             │
│                                                                  │
│  Output: communities.json                                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: Embedding Generation & FAISS Indexing                  │
│  ──────────────────────────────────────────────                  │
│  • Generate entity descriptions (natural language summaries)     │
│  • Encode chunks + entity descriptions + community summaries     │
│    using SentenceTransformers (all-mpnet-base-v2)                │
│  • L2-normalize all vectors                                      │
│  • Build a FAISS FlatIP index (exact inner-product search)       │
│                                                                  │
│  Output: embeddings.faiss, embeddings_meta.json                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Why This Exact Order?

The phases are strictly sequential because each depends on the previous:

1. **Graph must exist** before community detection can partition its nodes. You can't cluster entities that haven't been extracted yet.
2. **Communities must be detected** before the LLM can summarize them. The Louvain partition determines which entities belong to which cluster.
3. **All textual artifacts** (chunks, entity descriptions, community summaries) must exist before they can be embedded into vectors. The FAISS index is the *last* artifact produced.

This is intentionally **not** parallelizable — correctness trumps speed for an offline ingestion process. The entire pipeline for one department takes 30-90 seconds without LLM summarization, or 5-10 minutes with it.

### 3.2 Phase Ordering: Faculty List First

Within Phase 1, there's a critical sub-ordering: the **faculty roster file is parsed before all other files**. This is because the faculty roster seeds the `EntityResolver` with canonical names. Without this seeding, subsequent parsers (PhD student lists, funded projects, patents) would create duplicate nodes for the same faculty member under different name variants.

```python
# From KnowledgeGraphBuilder.build():
# Parse faculty list FIRST to register canonical names
if faculty_list_file:
    doc_id, content = doc_map[faculty_list_file]
    self._parse_faculty_list(faculty_list_file, content, doc_id)
    logger.info(f"Phase 1.5: Parsed faculty list — canonical registry populated.")
```

This "Phase 1.5" ensures that when the PhD roster mentions "Dr. A. K. Saxena" as a supervisor, the resolver can match it against the canonical "Alok Kumar Saxena" already registered from the faculty list.

---

## 4. The Orchestrator: `ingest.py` — Line-by-Line

The entry point is clean and procedural. Here's the actual control flow for departments:

### 4.1 Department Ingestion (`ingest_department`)

```python
def ingest_department(dept_code: str, skip_summaries: bool = False):
    # 1. Resolve canonical code and load config
    canonical_code = resolve_department_code(dept_code)
    dept_config = get_department(canonical_code)
    markdown_dir = get_markdown_dir(canonical_code)
    data_dir = get_data_dir(canonical_code)
    os.makedirs(data_dir, exist_ok=True)

    # 2. Phase 1: Build KG
    builder = KnowledgeGraphBuilder(dept_code=canonical_code)
    graph = builder.build()
    builder.save(data_dir)  # → graph.pkl, chunks.json, resolver.pkl

    # 3. Phase 2: Community Detection
    partition = detect_communities(graph, resolution=1.0)
    reports = build_community_reports(graph, partition)

    # 4. Phase 3: LLM Summarization
    if skip_summaries:
        reports = summarize_communities(reports, llm_fn=None)  # rule-based fallback
    else:
        llm = create_llm_from_env()  # Gemini or Ollama
        reports = summarize_communities(reports, llm_fn=llm)
    save_communities(reports, partition, data_dir)  # → communities.json

    # 5. Phase 4: Embeddings + FAISS
    engine = EmbeddingEngine()
    entity_descriptions = create_entity_descriptions(graph)
    community_items = [{"id": r["id"], "text": r.get("summary", r["text"]), ...} for r in reports]
    engine.build_index(chunks, entity_descriptions, community_items, dept_code=canonical_code)
    engine.save(data_dir)  # → embeddings.faiss, embeddings_meta.json
```

### 4.2 Section Ingestion (`ingest_section`)

The section flow is nearly identical but uses `SectionKGBuilder` and `create_section_entity_descriptions`:

```python
def ingest_section(section_code: str, skip_summaries: bool = False):
    builder = SectionKGBuilder(section_code=section_code)
    graph = builder.build()
    builder.save(data_dir)

    # Same Phase 2-4 as departments, but with a safety check:
    if n_nodes < 5:
        # Skip LLM summarization for tiny sections (not worth the API call)
        reports = summarize_communities(reports, llm_fn=None)
```

The `n_nodes < 5` guard prevents wasting LLM API calls on sections with too few entities to form meaningful communities.

### 4.3 Batch Modes

```bash
# Single department
python ingest.py --dept ee

# All 11+ academic departments (sequential loop)
python ingest.py --all

# Single institutional section
python ingest.py --section academics

# All sections (Quick Links, Placements, Medical, etc.)
python ingest.py --all-sections

# Skip expensive LLM summarization (faster iteration)
python ingest.py --dept ee --skip-summaries
```

The `--all` mode iterates through `DEPARTMENTS` sequentially. Each department is wrapped in a `try/except` so that a failure in one department doesn't abort the entire batch:

```python
for dept in DEPARTMENTS:
    try:
        ingest_department(dept, skip_summaries=args.skip_summaries)
    except Exception as e:
        logger.error(f"❌ Failed to ingest {dept.upper()}: {e}")
```

### 4.4 LLM Provider Selection

The `create_llm_from_env()` function selects the LLM provider based on environment variables:

- **Gemini** (default for production): Uses the Google Generative AI API. Subject to rate limits (429 errors) — the system implements exponential backoff with automatic model-switching fallback.
- **Ollama** (recommended for bulk ingestion): Uses a local Ollama instance, avoiding rate limits entirely. Configured via `OLLAMA_HOST` and `OLLAMA_MODEL` environment variables.

---

## 5. Output Artifacts

Each domain (department or section) produces exactly **6 files** in its `data/` directory:

| File | Format | Size (typical) | Purpose |
|---|---|---|---|
| `graph.pkl` | Python pickle (NetworkX DiGraph) | 200KB–2MB | The full knowledge graph with all entities, relationships, and node properties |
| `chunks.json` | JSON array | 100KB–500KB | Text chunks with metadata (source file, URL, chunk strategy, index) |
| `resolver.pkl` | Python pickle (EntityResolver) | 5KB–50KB | The entity resolution state — maps name variants to canonical forms |
| `communities.json` | JSON | 10KB–100KB | Community reports with LLM summaries, member lists, partition mappings |
| `embeddings.faiss` | FAISS binary | 500KB–5MB | The vector index (FlatIP) containing all embedded text |
| `embeddings_meta.json` | JSON array | 50KB–500KB | Metadata for each vector — maps IDs back to source chunks/entities/communities |

### 5.1 Why Pickle for the Graph?

NetworkX graphs don't serialize cleanly to JSON — they contain complex Python objects as node/edge attributes (sets, defaultdicts, nested dicts). Pickle is the pragmatic choice for an offline pipeline where the producer and consumer are the same Python version. The trade-off is non-portability, which is acceptable for a single-machine deployment.

### 5.2 Why FAISS FlatIP (not IVF or HNSW)?

The dataset is small enough (typically ~500–2000 vectors per department) that exact inner-product search is both fast and accurate. Approximate indices like IVF or HNSW add complexity and tuning parameters (nlist, ef_search) without meaningful speedup at this scale. FlatIP guarantees deterministic, exact results.

### 5.3 Why L2-Normalized Vectors with Inner Product?

The `EmbeddingEngine.encode()` method passes `normalize_embeddings=True` to the SentenceTransformer. This L2-normalizes every vector to unit length. When all vectors have unit norm, inner product (`IP`) becomes mathematically equivalent to cosine similarity — but FAISS's `IndexFlatIP` is faster than its cosine implementation because it avoids a per-query normalization step.

---

## 6. The GlobalPersonIndex: Cross-Domain Identity Resolution

After all departments and sections are ingested independently, the runtime builds a **GlobalPersonIndex** that cross-references individuals across all loaded graphs. This is critical because the same person can appear in multiple graphs:

- **Dr. Sartaj Ul Hasan** appears in the CSE department graph (as Faculty) *and* the Academics section graph (as Dean, Academic Programs).
- **Dr. Badri Narayan Subudhi** appears in the EE department graph *and* the Digital Infrastructure section graph (as Dean, DI).

The `GlobalPersonIndex` (`graphrag/person_index.py`) solves this by:

1. **Iterating** over every loaded graph's nodes.
2. **Filtering** to person-type labels: `Faculty`, `AdminOfficial`, `SectionPerson`, `SectionHead`, `Counselor`, `MedicalDoctor`, `PhDStudent`.
3. **Resolving** each name through a global `EntityResolver`.
4. **Aggregating** all roles for each resolved identity into a single lookup table.

### 6.1 The Role Placeholder Filter

A critical subtlety: committee rosters often contain entries like "Dean Student Affairs" or "All Deans" or "Upto Five Student Representative Nominated By The Chairman". These are **position descriptions**, not people. If indexed as person entities, they would pollute lookup results and cause the chatbot to return administrative jargon instead of actual names.

The `_is_role_placeholder()` method filters these out using:

```python
_ROLE_PLACEHOLDER_PREFIXES = (
    "dean ", "all dean", "upto ", "nominated ",
    "one representative", "two representative", ...
)

_ROLE_DESCRIPTOR_WORDS = {
    "officer", "affairs", "liaison", "coordinator", "convener",
    "chairperson", "chairman", "representative", "nominee",
    "warden", "provost", "controller", "superintendent", ...
}
```

The filter triggers when:
- The name starts with a known prefix ("dean ", "upto ", etc.)
- The name contains 2+ role-descriptor words
- The name has 8+ words (too long to be a personal name)
- The name consists entirely of role title words

### 6.2 Deduplication Across Sources

When the same person appears from two sources, the index stores both roles but deduplicates by `(source, designation)` pair:

```python
for r in existing:
    if r["source"] == role["source"] and r["designation"] == role["designation"]:
        duplicate = True
        break
if not duplicate:
    self.person_roles[resolved_name].append(role)
```

This ensures that a faculty member appearing in both the department graph and a committee roster gets both entries — but not two identical entries from the same source.

---

## 7. Idempotency and Rebuild Strategy

The pipeline is **idempotent by design**: running ingestion for the same department twice produces identical output. There's no "delta" or "update" mode — each run:

1. **Rebuilds the entire KG** from scratch by reading all markdown files.
2. **Overwrites** the existing `graph.pkl`, `chunks.json`, etc.

### 7.1 Why Not Incremental?

For this scale (~30–50 markdown files per department), a full rebuild takes seconds for Phase 1 and ~1 minute for Phase 4 (embedding). The bottleneck is Phase 3 (LLM summarization), which takes ~5 minutes per department with API calls. Since scraping happens infrequently (weekly or after content updates), the rebuild-from-scratch approach is pragmatic and eliminates an entire class of stale-data bugs.

### 7.2 When to Trigger a Rebuild

Rebuilds should be triggered when:

- The scraper produces new or updated markdown files
- A parser is modified (e.g., new regex for a changed page layout)
- The `CORRECT_LABS` registry is updated
- The entity resolution logic changes
- A new department or section is added to `departments.py`

### 7.3 The Skip-Summaries Optimization

During development, use `--skip-summaries` to bypass the expensive LLM step. This produces a functional pipeline where community summaries are rule-based concatenations instead of LLM-generated prose. The rule-based fallback:

```python
def _rule_based_summary(members_by_type):
    parts = []
    faculty = members_by_type.get("Faculty", [])
    students = members_by_type.get("PhDStudent", [])
    areas = members_by_type.get("ResearchArea", [])
    if faculty:
        parts.append(f"This group includes {len(faculty)} faculty member(s): {', '.join(faculty[:5])}")
    if students:
        parts.append(f"{len(students)} PhD student(s)")
    if areas:
        parts.append(f"working in areas like {', '.join(areas[:3])}")
    return ". ".join(parts) + "." if parts else "A cluster of related entities."
```

This is adequate for testing retrieval accuracy without burning API credits.

---

## 8. Error Handling and Resilience

### 8.1 Per-Department Isolation

In `--all` mode, each department is wrapped in a try/except. A parsing failure in one department (e.g., due to malformed markdown) does not abort the entire batch.

### 8.2 LLM Failover in Sections

The section ingestion has an additional safety layer — if the LLM client fails to initialize (missing API key, network error), it silently falls back to rule-based summaries:

```python
try:
    llm = create_llm_from_env()
    reports = summarize_communities(reports, llm_fn=llm)
except Exception as e:
    logger.warning(f"Failed to initialize LLM: {e}. Falling back to skip-summaries.")
    reports = summarize_communities(reports, llm_fn=None)
```

### 8.3 The `env_config.load_env_file()` Guard

The very first line of `ingest.py` after imports calls `load_env_file()`. This loads environment variables from `.env` files, ensuring that API keys (Gemini, Ollama endpoints) are available before any module tries to use them. Without this, lazy imports of `graphrag.llm` would fail with cryptic key-not-found errors.

---

## 9. Data Flow Diagram: End-to-End

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  University  │────▶│   Scraper    │────▶│  scraped_data/   │
│   Website    │     │ (Playwright) │     │  {code}/*.md     │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │   ingest.py      │
                                          │   (Orchestrator) │
                                          └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
           ┌────────▼────────┐           ┌────────▼────────┐           ┌────────▼────────┐
           │  KG Builder     │           │  Community       │           │  Embedding      │
           │  (Phase 1)      │           │  (Phases 2+3)    │           │  (Phase 4)      │
           │                 │           │                  │           │                 │
           │ • doc_kind      │           │ • Louvain        │           │ • mpnet-base-v2 │
           │ • 25+ parsers   │           │ • LLM summaries  │           │ • FAISS FlatIP  │
           │ • EntityResolver│           │ • Rule fallback  │           │ • 3-source index│
           └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
                    │                              │                              │
                    ▼                              ▼                              ▼
           graph.pkl                      communities.json              embeddings.faiss
           chunks.json                                                  embeddings_meta.json
           resolver.pkl
                    │                              │                              │
                    └──────────────────────────────┼──────────────────────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  Runtime Server  │
                                          │  (Retriever +    │
                                          │   LLM Reasoning) │
                                          └─────────────────┘
```

---

## Next: Part 2 — Knowledge Graph Construction

In the next document, we dive deep into Phase 1: how raw markdown files are classified by `infer_document_kind()`, chunked by the 4-tier `smart_chunk_text()` cascade, and parsed into a typed knowledge graph with 25+ specialized parsers and fuzzy entity resolution.
