# IIT Jammu Unified GraphRAG Chatbot

> A production-grade, multi-tenant university assistant powered by Graph-based Retrieval-Augmented Generation (GraphRAG). Covers **13 academic departments**, **19 institutional sections**, **11 student data domains**, **6 quick-link directories**, and **1 media section** — totalling **50+ independent knowledge bases** served through a single unified API.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Topology](#system-topology)
3. [Project Structure](#project-structure)
4. [Data Acquisition — Crawler](#data-acquisition--crawler)
5. [Knowledge Graph Construction](#knowledge-graph-construction)
6. [Ingestion Pipeline](#ingestion-pipeline)
7. [Retrieval Engine](#retrieval-engine)
8. [Multi-Department Routing](#multi-department-routing)
9. [LLM Integration](#llm-integration)
10. [Anti-Hallucination Defense](#anti-hallucination-defense)
11. [Academic Rules Engine](#academic-rules-engine)
12. [Data Taxonomy](#data-taxonomy)
13. [Evaluation Framework](#evaluation-framework)
14. [Configuration & Environment](#configuration--environment)
15. [Quick Start](#quick-start)
16. [Scalability](#scalability)

---

## Architecture Overview

The system follows a **"Structure Over Fuzzy Parsing"** philosophy — deterministic SQL/graph lookups handle factual queries (fees, staff, committees), while RAG handles open-ended reasoning. This hybrid approach eliminates hallucinations on structured data while preserving natural language flexibility.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Flask REST API (:5050)                       │
│                            app.py                                   │
├──────────┬──────────┬───────────┬──────────────┬───────────────────-─┤
│ Identity │  Intent  │  Dept     │  Retrieval   │  Faithfulness      │
│  Check   │  Class.  │  Router   │  Engine      │  Verifier          │
│ (Person  │ (Intent  │ (dept_    │ (multi_      │ (verifier.py)      │
│  Index)  │  Utils)  │  router)  │  retriever)  │                    │
├──────────┴──────────┴───────────┴──────────────┴────────────────────┤
│                    GraphRAG Core (graphrag/)                        │
│  ┌────────────┐ ┌────────────┐ ┌─────────────┐ ┌────────────────┐  │
│  │ KG Builder │ │ Hybrid     │ │ Section     │ │ Rules          │  │
│  │ + Section  │ │ Retriever  │ │ Retriever   │ │ Retriever +    │  │
│  │ KG Builder │ │ (FAISS+KG) │ │ (Determ.)   │ │ SQLite DB      │  │
│  └────────────┘ └────────────┘ └─────────────┘ └────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    Data Layer                                       │
│  scraped_data/   →  data/{dept}/  →  FAISS indices + KG pickles    │
│  (Markdown)         (Processed)      (Runtime)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## System Topology

The request lifecycle has **5 stages**:

### Stage 1 — Identity Resolution
Every query passes through `GlobalPersonIndex`, which cross-references individuals across all loaded graphs. If a person is mentioned, their roles across departments/sections are injected as priority context.

### Stage 2 — Intent Classification
`intent_utils.py` classifies academic rules queries (CGPA, attendance, course withdrawal) and short-circuits them to the `RulesRetriever` backed by a structured SQLite database.

### Stage 3 — Department Routing
`DepartmentRouter.route()` uses greedy alias matching against 200+ aliases to map queries to 1–N department/section codes. Cross-routing rules inject related knowledge bases (e.g., "admissions" triggers both `students-ug-admissions` and `academics`).

### Stage 4 — Retrieval
Based on routing results, one of four scenarios executes:
| Scenario | Trigger | Strategy |
|----------|---------|----------|
| **A** — Single Department | 1 dept matched | Direct `HybridRetriever.retrieve_bundle()` |
| **B** — Single Section | 1 section matched | Direct `SectionRetriever.retrieve_bundle()` |
| **C** — Multi-target | 2+ targets matched | `MultiDepartmentRetriever.retrieve_multi()` with merged contexts |
| **D** — Broadcast | No clear match | `retrieve_broadcast()` — searches ALL loaded retrievers, ranks by relevance score |

### Stage 5 — Verification & Response
The LLM generates a response from the retrieved context. `ResponseVerifier` then runs a faithfulness check — if >50% of factual claims (or >30% in multi-dept mode) lack grounding, the response is replaced with a safe fallback.

---

## Project Structure

```
chatbot/
├── app.py                    # Flask API server (933 lines)
├── crawler.py                # Playwright-based BFS web crawler (1211 lines)
├── departments.py            # Central department & section registry
├── dept_router.py            # Intent-based multi-department router
├── ingest.py                 # Ingestion pipeline orchestrator
├── env_config.py             # Environment variable loader
├── utils.py                  # URL canonicalization & content filters
├── requirements.txt          # Python dependencies
│
├── graphrag/                 # Core retrieval engine
│   ├── kg_builder.py         # Department knowledge graph builder (2316 lines)
│   ├── section_kg_builder.py # Section-specific KG builder (3515 lines)
│   ├── retriever.py          # Hybrid retriever (FAISS + KG) (2493 lines)
│   ├── section_retriever.py  # Section retriever with deterministic answers (1924 lines)
│   ├── multi_retriever.py    # Cross-department orchestrator
│   ├── embeddings.py         # SentenceTransformer + FAISS engine
│   ├── community.py          # Louvain community detection
│   ├── llm.py                # Multi-provider LLM client (Ollama/Gemini/Bedrock)
│   ├── verifier.py           # Post-generation faithfulness verifier
│   ├── person_index.py       # Global cross-department person index
│   ├── intent_utils.py       # Academic rules intent classifier
│   ├── rules_retriever.py    # Structured rules retrieval engine
│   ├── rules_parser.py       # Academic PDF → SQLite parser
│   ├── rules_db.py           # SQLite schema for academic regulations
│   └── rules.db              # Pre-built SQLite database
│
├── scraped_data/             # Raw crawled markdown (multi-tenant)
│   ├── {department}/         # 13 department directories
│   ├── sections/             # 19 institutional sections
│   ├── students/             # 11 student data categories
│   ├── Quick/                # 6 quick-link directories
│   └── Media/                # Media & communications
│
├── data/                     # Processed runtime data
│   ├── {department}/         # KG pickles + FAISS indices per dept
│   └── sections/{section}/   # KG pickles + FAISS indices per section
│
├── scripts/                  # Automation & tooling
│   ├── crawl_students.py     # Student section crawler
│   ├── crawl_media.py        # Media section crawler
│   ├── crawl_quick.py        # Quick-links crawler
│   ├── convert_pdfs_to_md.py # PDF → Markdown converter
│   ├── download_academics_pdfs.py  # Academic PDF downloader
│   ├── ingest_rules.py       # Academic rules SQLite ingestion
│   └── flatten_curriculum_docs.py  # Curriculum doc flattener
│
├── evaluation/               # QA evaluation framework
│   ├── run_eval.py           # Main evaluation runner
│   ├── pipeline.py           # Evaluation pipeline logic
│   ├── generate_qna.py       # Golden QA dataset generator
│   └── qna_dataset.json      # Golden question-answer pairs
│
├── static/                   # Frontend assets
├── templates/                # Flask HTML templates
│   └── index.html            # Chat UI
└── tests/                    # Test suite
```

---

## Data Acquisition — Crawler

**File:** `crawler.py` (1211 lines)

The crawler uses **stateful Playwright rendering** with BFS discovery to handle JavaScript-heavy IIT Jammu pages.

### Crawling Strategy
1. **Discovery Phase** — BFS from `base_url` with seed paths (`/`, `/index.html`). Each page is rendered via Playwright with progressive fallback: `networkidle` → `load` → `domcontentloaded` → `commit + 10s wait`.
2. **Evaluation** — Every page is scored against generic-fallback detection and minimum content thresholds (`MIN_CONTENT_LEN`). Duplicate final URLs are rejected.
3. **Link Extraction** — Anchors, `data-href`, `onclick` handlers, and JavaScript `location.href` assignments are all harvested.
4. **Conversion** — HTML → clean Markdown via recursive DOM traversal preserving tables, lists, headings, links, and images. Binary files (PDF, XLSX, DOCX, CSV, images) are downloaded and parsed via `pdfplumber`, `openpyxl`, `python-docx`, or `pytesseract`.
5. **Manifest** — Every accept/reject decision is persisted to `crawl_manifest.json` for auditability.

### Specialized Crawlers
| Script | Target | Output |
|--------|--------|--------|
| `crawler.py` | Department websites | `scraped_data/{dept}/` |
| `scripts/crawl_students.py` | Student data pages | `scraped_data/students/` |
| `scripts/crawl_media.py` | Media & events | `scraped_data/Media/` |
| `scripts/crawl_quick.py` | Quick-link pages | `scraped_data/Quick/` |

---

## Knowledge Graph Construction

### Department KG Builder (`kg_builder.py` — 2316 lines)

Parses crawled markdown into a NetworkX `DiGraph` with typed entities and relationships.

**Document Type Detection** — `infer_document_kind()` classifies each file by filename patterns and content heuristics into one of 20+ types:

| Document Kind | Entity Types Extracted |
|--------------|----------------------|
| `faculty_profile` | Faculty, ResearchArea, ResearchCategory |
| `faculty_roster` | Faculty (batch) |
| `phd_roster` | PhDStudent, supervisor edges |
| `funded_projects` | Project, FundingAgency |
| `patents` | Patent |
| `startups` | Startup |
| `placement_industry` | PlacementData (salary, company) |
| `placement_academia` | HigherStudiesData |
| `admin_committee` | Committee, AdminOfficial, MEMBER_OF edges |
| `admin_director` | AdminOfficial (is_director=True) |
| `admin_deans` | AdminOfficial (Dean/Associate Dean) |
| `labs` | Lab |
| `staff` | Staff |
| `programmes` | Programme |
| `alumni` / `phd_alumni` | Alumni, AlumniBatch, GraduatedPhD |
| `contact` | ContactInfo |

**Entity Resolution** — The `EntityResolver` class handles name variants across documents using:
- Fuzzy matching (SequenceMatcher ≥ 0.85)
- Initials expansion (`B. N Subudhi` → `Badri Narayan Subudhi`)
- Token-subset matching (`Anup Kumar Shukla` ↔ `Anup Shukla`)
- Canonical faculty registry pre-seeded from `faculty-list` files

**Smart Chunking** — `smart_chunk_text()` selects the best chunking strategy:
1. **Repeated heading records** — Auto-detects roster-style repeated headings (e.g., faculty profiles at `####` level)
2. **Heading sections** — Splits on `##`–`######` boundaries
3. **Structural blocks** — Preserves tables and paragraphs as atomic units
4. **Word window fallback** — 400-word chunks with 80-word overlap

### Section KG Builder (`section_kg_builder.py` — 3515 lines)

Specialized parsers for institutional sections with domain-specific entity extraction:

| Section | Entities Parsed |
|---------|----------------|
| `academics` | CommitteeMember (DPGC/DUGC), FacultyAdvisor, ProgramCoordinator, FeeStructure, PolicyNotification, Specialization, AcademicProgram, Course |
| `counselling` | Counselor, SectionHead |
| `cds` | PlacementRecord, CompanyVisit |
| `medical-centre` | MedicalDoctor, EmpaneledHospital |
| `students-faq` | FAQ (question/answer pairs) |
| `students-schedule` | ScheduleEvent, Holiday |
| `students-*-admissions` | AdmissionInfo |
| `quick-committees` | CommitteeMember (SC/ST Cell, Ethics, ICC) |
| `quick-staff` | StaffMember |
| `quick-contacts` | ContactEntry (VoIP directory) |
| `quick-anti-ragging` | AntiRaggingSquad member |
| `media` | NewsEvent, Achievement |

---

## Ingestion Pipeline

**File:** `ingest.py` (337 lines)

The `ingest_department()` function orchestrates the full pipeline:

```
Markdown Files → KG Builder → Community Detection → Embedding Index → Disk
```

### Steps

1. **Graph Construction** — `KGBuilder.build()` or `SectionKGBuilder.build()` parses all `.md` files in the department/section directory.
2. **Community Detection** — Louvain algorithm (`resolution=1.0`) on the entity subgraph (excluding TextChunk/Document nodes). Produces community reports with member listings and relationship summaries.
3. **Community Summarization** — Each community gets an LLM-generated or rule-based summary describing its research cluster.
4. **Embedding Generation** — `all-mpnet-base-v2` (768-dim) encodes all chunks, entity descriptions, and community summaries.
5. **FAISS Indexing** — `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity).
6. **Persistence** — Graph pickle, community JSON, FAISS index, and metadata JSON are saved to `data/{dept}/` or `data/sections/{section}/`.

### Running Ingestion

```bash
# Single department
python ingest.py --dept ee

# Single section
python ingest.py --section academics

# All departments
python ingest.py --all-depts

# All sections
python ingest.py --all-sections

# Everything
python ingest.py --all
```

---

## Retrieval Engine

### HybridRetriever (`retriever.py` — 2493 lines)

Each department has its own `HybridRetriever` instance combining three retrieval channels:

| Channel | Method | Purpose |
|---------|--------|---------|
| **Graph (Local)** | Entity name index + label index traversal | Precise entity lookups, relationship traversal |
| **Vector (Global)** | FAISS cosine similarity search | Semantic matching for open-ended queries |
| **Community** | Community report matching | High-level synthesis and topic summaries |

**Deterministic Fast-Paths** — Before RAG, the retriever checks for query types that can be answered deterministically:

| Query Type | Detection Method | Response Source |
|-----------|-----------------|----------------|
| Department contact / HoD | Trigger phrase matching | `_build_department_contact_answer()` |
| Labs & facilities | Keyword detection | `CORRECT_LABS` registry in `departments.py` |
| Address & phone | Keyword detection | ContactInfo graph nodes |
| Graduated PhDs | Trigger phrases | GraduatedPhD graph nodes |
| Alumni | Keyword matching | Alumni/AlumniBatch nodes |
| Administration (Director, Registrar, Deans, Committees) | Role/committee matching | Admin graph traversal |
| Research topic experts | `_topic_matches_text()` with abbreviation expansion | Faculty → RESEARCHES_IN edges |

**Answerability Assessment** — Before sending context to the LLM, `_assess_answerability()` checks:
1. Are the query's inferred concepts (startup, patent, lab, placement, etc.) structurally present in the graph?
2. Do retrieved results contain sufficient keyword overlap with the query?
3. Is the vector similarity score above threshold (≥0.45)?

If evidence is insufficient, a safe "I don't have that information" response is returned directly, bypassing the LLM entirely.

### SectionRetriever (`section_retriever.py` — 1924 lines)

Handles institutional sections with extensive deterministic lookup logic:

- **Academics:** DPGC/DUGC committees, faculty advisors, program coordinators, fee structures, policy notifications, specializations, courses
- **Student FAQ:** Word-overlap scoring against FAQ question nodes
- **Student Schedule:** Event filtering by academic level and keyword
- **Certificate Programs:** Program listing and detail lookup
- **CDS:** Placement statistics and company visit data
- **Counselling:** Counselor profiles and contact info
- **Medical Centre:** Doctor directory and hospital listings

### MultiDepartmentRetriever (`multi_retriever.py` — 567 lines)

Orchestrates cross-department retrieval with three modes:

1. **`retrieve_single()`** — Direct delegation to one department
2. **`retrieve_multi()`** — Parallel retrieval from N departments with relevance filtering and merged context
3. **`retrieve_broadcast()`** — Searches ALL loaded retrievers, scores by `(graph_avg × graph_items + vector_avg × vector_items) / total_items`, returns top-N relevant results

**Topic Query Detection** — `_is_topic_query()` identifies broad research queries (e.g., "Who works on deep learning?") using structural verb patterns and 30+ known research subject keywords. Topic queries trigger full broadcast across all departments.

**Relevance Filtering** — `_is_bundle_relevant()` prevents noise by checking:
- Direct graph match (highest confidence)
- Focus term presence in context text (compound topics like "computer vision" handled atomically)
- Vector similarity ≥ 0.45

---

## Multi-Department Routing

**File:** `dept_router.py` (613 lines)

The `DepartmentRouter` maps natural language queries to department/section codes.

### Alias Matching
200+ aliases cover:
- Canonical codes: `ee`, `computer_science_engineering`, `civil_engineering`
- Short forms: `cse`, `mech`, `chem eng`
- Full names: `electrical engineering`, `biosciences and bioengineering`
- Colloquial: `cs`, `bio`, `hss`
- Section aliases: `placement` → `cds`, `library` → `library`, `hostel` → `sw`

### Cross-Routing Rules
When a query matches certain intents, additional knowledge bases are injected:
- **Student/academic terms** → inject `academics`, `students-faq`, relevant admission sections
- **Person names with honorifics** → inject `administration` for dean/committee lookups
- **Placement/salary** → inject `cds`
- **Fee/scholarship** → inject `academics`
- **NIRF/ranking** → inject `administration`, `students-why-iitjammu`

---

## LLM Integration

**File:** `graphrag/llm.py` (680 lines)

Three provider backends, hot-swappable via `LLM_PROVIDER` env var:

| Provider | Class | Default Model | Auth |
|----------|-------|---------------|------|
| **Ollama** (local) | `OllamaLLM` | `llama3.1` (8B) | None |
| **Gemini** (cloud) | `GeminiLLM` | `gemini-2.5-flash-lite` | `GEMINI_API_KEY` |
| **AWS Bedrock** | `BedrockLLM` | `qwen.qwen3-32b-v1:0` | `AWS_BEARER_TOKEN_BEDROCK` |

### Prompt Engineering
- **Single-department prompt** — `get_system_prompt()` generates department-scoped system prompts with security rules, privacy guardrails, and anti-hallucination instructions.
- **Multi-department prompt** — `get_unified_system_prompt()` adds cross-department attribution rules.
- **Response sanitization** — `sanitize_response()` strips leaked HTML, converts raw anchors to markdown links, and removes kramdown-style attributes.

### Resilience
- Ollama: 3 retries with 2s backoff
- Gemini: 2 retries, transient 5xx retried after 2s, 429 not retried
- Bedrock: Automatic inference profile detection and fallback model switching

---

## Anti-Hallucination Defense

A **4-layer defense** prevents fabricated responses:

### Layer 1 — Scope Guard
`intent_utils.is_academic_rules_query()` uses 100+ keyword patterns with negative guards (CDS, counselling, medical, sports) to prevent misrouting.

### Layer 2 — Confidence Gate
`_assess_answerability()` checks concept support in the graph before LLM generation. Unsupported concepts → immediate safe fallback.

### Layer 3 — Grounding Enforcement
System prompts explicitly instruct: "Answer ONLY from the information provided above. NEVER fabricate names, emails, phone numbers, designations, or statistics."

### Layer 4 — Post-Generation Verification (`verifier.py`)
After LLM generation, `ResponseVerifier` uses the LLM itself to extract and verify factual claims against the retrieved context:
- Extracts claims (names, numbers, emails, designations)
- Checks each claim against context for explicit support
- **Single-dept threshold:** >50% claims supported → faithful
- **Multi-dept threshold:** >30% claims supported → faithful (context truncation compensation)
- Failed responses are replaced with safe fallback messages

Verification is enabled via `VERIFY_RESPONSES=true` (disabled by default for Gemini/Bedrock to conserve API quota).

---

## Academic Rules Engine

A specialized retrieval pipeline for academic regulations, built from parsed PDF documents.

### Components

| File | Purpose |
|------|---------|
| `rules_parser.py` | Parses academic regulation PDFs into structured sections |
| `rules_db.py` | SQLite schema: `sections`, `grade_scale`, `milestones`, `credit_requirements`, `rule_facts` |
| `rules_retriever.py` | Hybrid retrieval: structured lookup + evidence-ranked section search |
| `rules.db` | Pre-built SQLite database |

### Retrieval Strategy
1. **Intent Classification** — Detects target program (UG/MTech/PhD) and intent (grades, milestones, credits)
2. **Structured Lookups** — Grade scale table, program milestones, credit requirements, key facts (CGPA thresholds, etc.)
3. **Evidence-Ranked Sections** — Token-based scoring with phrase expansions, course code detection, and domain-specific boosts (e.g., attendance policy +150, PhD duration +150)

---

## Data Taxonomy

### Departments (13)

| Code | Department | Template |
|------|-----------|----------|
| `administration` | Administration of IIT Jammu | A |
| `ee` | Electrical Engineering | A |
| `computer_science_engineering` | Computer Science & Engineering | B |
| `mechanical_engineering` | Mechanical Engineering | A |
| `civil_engineering` | Civil Engineering | B |
| `chemical-engineering` | Chemical Engineering | B |
| `bsbe` | Biosciences & Bioengineering | A |
| `chemistry` | Chemistry | B |
| `hss` | Humanities & Social Sciences | B |
| `idp` | Interdisciplinary Programmes | B |
| `materials-engineering` | Materials Engineering | B |
| `mathematics` | Mathematics | B |
| `physics` | Physics | B |

### Institutional Sections (19)

`academics`, `alumni-affairs`, `cds`, `counselling`, `di`, `e2`, `saral`, `accounts`, `hindicell`, `ir`, `library`, `medical-centre`, `osd`, `sp`, `rc`, `sw`, `security`, `tlu`, `tinkerers-lab`

### Student Data Sections (11)

`students-faq`, `students-schedule`, `students-phd-admissions`, `students-pg-admissions`, `students-ug-admissions`, `students-certificate-programs`, `students-online-education`, `students-pmrf`, `students-visvesvaraya`, `students-why-iitjammu`, `students-academic-downloads`

### Quick-Link Sections (6)

`quick-adjunct-faculty`, `quick-anti-ragging`, `quick-committees`, `quick-staff`, `quick-contacts`, `quick-rti`

### Media Section (1)

`media` — News events, achievements, press coverage

---

## Evaluation Framework

**Directory:** `evaluation/`

### Golden QA Dataset
`qna_dataset.json` contains curated question-answer pairs across domains for regression testing.

### Evaluation Pipeline
```bash
# Run full evaluation
python evaluation/run_eval.py

# Department-specific evaluation
python evaluation/run_department_eval.py --dept ee
```

The pipeline measures:
- **Routing accuracy** — Did the query reach the correct department(s)?
- **Retrieval relevance** — Does the context contain the answer?
- **Response faithfulness** — Is the LLM response grounded in context?
- **Factual correctness** — Does the response match the golden answer?

---

## Configuration & Environment

Copy `.env.example` to `.env` and configure:

```bash
# LLM Provider: ollama | gemini | bedrock
LLM_PROVIDER=ollama

# Ollama (local)
OLLAMA_API_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=llama3.1

# Gemini (cloud)
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash-lite

# AWS Bedrock
AWS_BEARER_TOKEN_BEDROCK=your-token-here
BEDROCK_MODEL=qwen.qwen3-32b-v1:0
BEDROCK_REGION=us-east-1

# Runtime
VERIFY_RESPONSES=true          # Enable L4 faithfulness verification
EMBEDDING_DEVICE=cpu           # cpu | cuda
SCRAPED_DATA_ROOT=./scraped_data
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama running locally (or Gemini/Bedrock API keys)
- ~4GB RAM for embedding model + FAISS indices

### Installation

```bash
# Clone and install
git clone https://github.com/MahimYadav2006/Chatbot-IITJ.git && cd chatbot
pip install -r requirements.txt

# Install Playwright browsers (for crawling only)
playwright install chromium

# Copy environment config
cp .env.example .env
# Edit .env with your LLM provider settings
```

### Crawl & Ingest

```bash
# Crawl a department website
python crawler.py --dept ee

# Ingest into knowledge graph + vector index
python ingest.py --dept ee

# Or ingest everything
python ingest.py --all
```

### Run the Server

```bash
python app.py
# Server starts on http://0.0.0.0:5050
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main chat endpoint. Body: `{"query": "..."}` |
| `/api/health` | GET | Health check with loaded departments and component status |

### Chat Response Format

```json
{
  "response": "...",
  "routed_departments": ["ee"],
  "routed_sections": [],
  "routing_reason": "Matched alias 'electrical engineering'",
  "retrieval_time": 1.23,
  "total_time": 3.45,
  "information_available": true
}
```

---

## Scalability

The system is optimized for **single-node GPU inference** with Ollama. For scaling considerations, see [SCALING.md](SCALING.md).

### Current Performance Profile
- **Cold start:** ~30s (loads all FAISS indices + embedding model)
- **Query latency:** 2–5s (routing + retrieval + LLM generation)
- **Memory:** ~4GB base + ~200MB per department index
- **Embedding model:** `all-mpnet-base-v2` on CPU (default) or GPU

### Key Design Decisions
1. **Department-scoped isolation** — Each department has its own KG + FAISS index. No cross-contamination of entity namespaces.
2. **Deterministic-first retrieval** — Structured lookups (fees, committees, contacts) bypass RAG entirely, eliminating hallucination risk on factual data.
3. **Broadcast with relevance gating** — Cross-department queries search all knowledge bases but only include departments with genuinely relevant context.
4. **Local-first inference** — Ollama with quantized models (Llama 3.1 8B) for zero-cost, privacy-preserving deployment.

---

## Dependencies

```
networkx>=2.6          # Knowledge graph representation
numpy>=1.21            # Numerical operations
scipy>=1.7             # Scientific computing
sentence-transformers>=2.2.0  # Embedding model (all-mpnet-base-v2)
faiss-cpu>=1.7         # Vector similarity search
python-louvain>=0.16   # Community detection
requests>=2.25         # HTTP client for LLM APIs
flask>=2.0             # Web framework
beautifulsoup4         # HTML parsing
Pillow                 # Image processing
playwright>=1.48.0     # Browser automation for crawling
```

---

## License

Internal project — IIT Jammu.
