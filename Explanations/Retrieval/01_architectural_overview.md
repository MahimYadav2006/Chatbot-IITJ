# 01 — Retrieval Pipeline: Architectural Overview

## Table of Contents

1. [Design Philosophy: "Structure Over Fuzzy"](#design-philosophy-structure-over-fuzzy)
2. [Why Not a Pure Vector-Based Pipeline?](#why-not-a-pure-vector-based-pipeline)
3. [The Four Retrieval Layers](#the-four-retrieval-layers)
4. [Component Map](#component-map)
5. [Execution Flow: From Query to Context](#execution-flow-from-query-to-context)
6. [Multi-Tenant Architecture](#multi-tenant-architecture)
7. [Design Trade-offs and Justifications](#design-trade-offs-and-justifications)

---

## Design Philosophy: "Structure Over Fuzzy"

The IIT Jammu chatbot's retrieval pipeline is engineered around a single guiding principle: **deterministic graph-backed answers must always take priority over probabilistic vector-search answers.** This is the "Structure Over Fuzzy" philosophy.

### What This Means in Practice

When a user asks *"Who is the HoD of Electrical Engineering?"*, the system does **not** embed this query, search a FAISS index, retrieve the top-5 chunks, and hope the LLM can extract the answer from a scraped HTML table. Instead, it:

1. **Detects the intent** — a department contact / HoD query.
2. **Traverses the knowledge graph** — finds the `Faculty` node with `is_hod=True`.
3. **Constructs a deterministic answer** — formats the name, email, profile URL.
4. **Injects this as authoritative context** — the LLM generates its final phrasing using this ground-truth data as its primary source.

The LLM always generates the user-facing response (even for deterministic data), but the *evidence* fed to it is pre-verified graph data, not a lossy semantic search result.

### Why This Matters for an Institutional Chatbot

Unlike a general-purpose RAG system where approximate answers are acceptable ("here are some relevant paragraphs"), an institutional chatbot has a **zero-tolerance policy for factual errors**. Telling a student the wrong HoD, the wrong fee structure, or the wrong supervisor creates real-world consequences. The architecture therefore treats the knowledge graph as the single source of truth for structured institutional data, and only falls back to fuzzy retrieval for unstructured content that cannot be pre-indexed.

---

## Why Not a Pure Vector-Based Pipeline?

A common RAG architecture is: `Query → Embed → FAISS Search → Top-K Chunks → LLM`. This is simple and works well for open-domain QA. However, it has critical failure modes for institutional data:

| Failure Mode | Example | What Goes Wrong |
|---|---|---|
| **Count hallucination** | "How many faculty in EE?" | Vector search retrieves 3 chunks mentioning 3 faculty. LLM says "3" when the real count is 14. |
| **Entity confusion** | "Email of Dr. Sharma?" | Two faculty named Sharma exist in different departments. Vector search returns both; LLM picks the wrong one. |
| **Stale data mixing** | "What is the fee for M.Tech?" | Old and new fee notifications both indexed. Vector search can't distinguish currency. |
| **Enumeration failure** | "List all PhD students" | Vector search caps at top-K. If K < total PhD students, the list is incomplete. |
| **Substring collision** | "AI research" → retrieves chunks about "uncertainty" (contains "ai" substring) | Short keywords match unrelated text in vector space. |

The hybrid architecture solves each of these by routing enumeration, contact, roster, and structured data queries to the graph layer, which returns **complete, authoritative, provenance-backed** answers.

---

## The Four Retrieval Layers

The retrieval pipeline executes in a strict priority order. Each layer either satisfies the query fully or passes control to the next:

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: Department Router (dept_router.py)            │
│  Determines WHICH retriever(s) to query                 │
│  ↓                                                      │
│  LAYER 2: Deterministic Graph Retrieval                 │
│  (retriever.py → get_deterministic_context)             │
│  Regex + graph traversal for known query patterns       │
│  ↓ (if no deterministic match)                          │
│  LAYER 3: Hybrid RAG Retrieval                          │
│  Entity Search (graph) + Vector Search (FAISS) +        │
│  Community Summaries (Louvain)                          │
│  ↓ (after context assembled)                            │
│  LAYER 4: Answerability Gate + Post-Generation Verify   │
│  (verifier.py → ResponseVerifier)                       │
│  Checks if evidence supports an answer;                 │
│  post-generation faithfulness check                     │
└─────────────────────────────────────────────────────────┘
```

### Layer 1: Department Router
**File:** `dept_router.py` → `DepartmentRouter`

Routes the query to the correct department(s) and/or section(s). Uses **greedy alias matching** — not semantic similarity — to detect department references. This is deterministic: "CSE" always routes to `computer_science_engineering`, never to a random department that has a semantically similar name.

### Layer 2: Deterministic Graph Retrieval
**File:** `graphrag/retriever.py` → `HybridRetriever.get_deterministic_context()`

Intercepts queries that have **known structural answers** in the graph: HoD lookups, lab listings, faculty rosters, PhD supervision chains, placement data, alumni records, administration committees. If a match is found, the deterministic context is injected as the highest-priority evidence block.

### Layer 3: Hybrid RAG Retrieval
**Files:** `graphrag/retriever.py` → `_local_search()`, `_vector_search()`, `_global_search()`

When the deterministic layer has no answer (or returns partial context), three sub-systems execute in parallel:

- **Entity Search (Local):** Name-matching against graph nodes, then embedding-based entity lookup.
- **Vector Search:** FAISS cosine-similarity search over text chunks.
- **Community Search:** Louvain-detected community summaries for broad topic context.

### Layer 4: Answerability Gate + Post-Generation Verification
**Files:** `graphrag/retriever.py` → `_assess_answerability()`, `graphrag/verifier.py` → `ResponseVerifier`

Before the LLM generates: checks if retrieved evidence actually supports an answer. After the LLM generates: verifies that factual claims in the response are grounded in the context (the "L4 verifier").

---

## Component Map

```
┌──────────────────────────────────────────────────────────────────┐
│                     QUERY ENTRY POINT (app.py)                   │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              dept_router.py → DepartmentRouter                   │
│                                                                  │
│  - DEPT_NAME_ALIASES: 12 departments, 150+ aliases               │
│  - SECTION_NAME_ALIASES: 30+ sections, 400+ aliases              │
│  - Cross-routing injection (media↔ir, complaints→committees)     │
│  - Academic rules intercept (intent_utils.py)                    │
│                                                                  │
│  Output: RouteResult(departments=[], sections=[], confidence)    │
└──────────────────┬───────────────────────────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌─────────────────┐ ┌─────────────────────┐
│  HybridRetriever│ │  SectionRetriever   │
│  (departments)  │ │  (sections)         │
│  retriever.py   │ │  section_retriever  │
│  2493 lines     │ │  .py — 1924 lines   │
└────────┬────────┘ └──────────┬──────────┘
         │                     │
         ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│           multi_retriever.py → MultiDepartmentRetriever          │
│                                                                  │
│  - retrieve_single(): delegate to one department                 │
│  - retrieve_multi(): merge N departments with headers            │
│  - retrieve_broadcast(): search ALL, rank by relevance           │
│  - _is_topic_query(): detect cross-department research topics    │
│  - _is_bundle_relevant(): filter noise from broadcast results    │
└──────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│              Shared Infrastructure                                │
│                                                                  │
│  embeddings.py    → FAISS index + all-mpnet-base-v2              │
│  community.py     → Louvain community detection + summarization  │
│  rules_retriever  → Academic rules SQLite FTS5                   │
│  rules_db.py      → Structured grade/milestone/credit tables     │
│  intent_utils.py  → Academic rules intent classifier             │
│  verifier.py      → Post-generation faithfulness checker (L4)    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Execution Flow: From Query to Context

Here is the complete execution path for a query like *"Who supervises Rahul Kumar in EE?"*:

### Step 1: Routing
`DepartmentRouter.route()` detects "EE" → routes to `ee` department.

### Step 2: Deterministic Dispatch
`HybridRetriever.get_deterministic_context()` is called. Inside:
- `_extract_supervisor_query_name()` regex matches "Rahul Kumar".
- `_find_entity_by_name("rahul kumar", allowed_labels=("PhDStudent", "MTechStudent"))` resolves to a graph node.
- Graph traversal follows `SUPERVISED_BY` edges to find supervisor(s).
- Deterministic answer: *"Rahul Kumar is a PhD scholar supervised by Dr. X."*

### Step 3: Context Assembly
`retrieve_bundle()` packages the deterministic context as `## Authoritative Department Data`, plus any supplementary entity/vector/community results into a structured context block.

### Step 4: Answerability Assessment
`_assess_answerability()` confirms that the query concepts ("supervisor") are supported by the graph data. Sets `answerable=True`.

### Step 5: LLM Generation
The assembled context is passed to the LLM with the original query. The LLM generates a natural-language response grounded in the authoritative context.

### Step 6: Post-Generation Verification
`ResponseVerifier.verify()` checks if the generated response's factual claims (names, emails, designations) are present in the context. If >50% of claims are unsupported, the response is flagged as unfaithful.

---

## Multi-Tenant Architecture

The system operates as a **multi-tenant retrieval engine** where each department and institutional section has its own isolated knowledge graph, FAISS index, and community structure:

```
data/
├── ee/                          # Electrical Engineering tenant
│   ├── graph.pkl                # NetworkX DiGraph
│   ├── chunks.json              # Text chunks
│   ├── embeddings.faiss         # FAISS vector index
│   ├── embeddings_meta.json     # FAISS metadata
│   └── communities.json         # Louvain partition + reports
├── computer_science_engineering/
│   ├── graph.pkl
│   ├── ...
├── sections/
│   ├── academics/
│   │   ├── graph.pkl
│   │   ├── chunks.json
│   │   ├── embeddings.faiss
│   │   └── ...
│   ├── cds/                     # Career Development Services
│   ├── medical-centre/
│   ├── ir/                      # International Relations
│   └── ...
```

### Why Per-Tenant Isolation?

1. **Precision:** A query about "Dr. Sharma in EE" should not retrieve Dr. Sharma from CSE. Department-scoped FAISS indices prevent cross-contamination.
2. **Scalability:** Adding a new department means adding a new data directory and retriever instance — no reindexing of existing departments.
3. **Independent updates:** Scraping and re-ingesting one department does not affect others.

### Cross-Tenant Queries

When a query doesn't target a specific department (e.g., "Who works on deep learning at IIT Jammu?"), the `MultiDepartmentRetriever` executes a **broadcast search**: it queries ALL loaded retrievers, scores each result bundle, filters out irrelevant ones, and merges the top-N results with department headers:

```
## Department of Electrical Engineering
[EE context about deep learning faculty]

---

## Department of Computer Science and Engineering
[CSE context about deep learning faculty]
```

---

## Design Trade-offs and Justifications

### Trade-off 1: Regex Intent Detection vs. LLM-Based Classification

**Choice:** Regex-based intent detection (100+ patterns in `get_deterministic_context()`).

**Why not LLM classification?**
- **Latency:** An LLM call adds 1-3 seconds. Regex classification is <1ms.
- **Determinism:** Regex always gives the same result for the same input. LLM classification can be non-deterministic.
- **Cost:** Every query would require an additional LLM API call just for routing.
- **Coverage:** The set of institutional query patterns is finite and enumerable. Regex can cover 95%+ of them.

**Downside:** New query patterns require manual regex additions. This is acceptable for an institutional system with a well-defined domain.

### Trade-off 2: NetworkX (In-Memory) vs. Neo4j (Database)

**Choice:** NetworkX in-memory graphs.

**Why not Neo4j?**
- **Zero external dependencies:** The system runs on a single machine with no database server.
- **Cold-start time:** Loading a NetworkX graph from pickle takes ~100ms. Neo4j connection setup takes seconds.
- **Memory footprint:** Department graphs are small (100-500 nodes). A full Neo4j instance is overkill.
- **Query patterns:** All graph queries are simple 1-2 hop traversals. Cypher's expressiveness is unnecessary.

**Downside:** No ACID transactions, no concurrent writes. Acceptable for a read-heavy chatbot workload.

### Trade-off 3: FAISS (Inner Product) vs. Other Vector Stores

**Choice:** FAISS `IndexFlatIP` (brute-force inner product search).

**Why not HNSW, Pinecone, or Qdrant?**
- **Index size:** Each department has <5000 vectors. Brute-force search on 5000 768-dim vectors takes <5ms.
- **No external service:** FAISS runs in-process. No network calls, no API keys, no cloud vendor lock-in.
- **Simplicity:** `IndexFlatIP` is exact — no approximation errors from HNSW.

**Downside:** Doesn't scale to millions of vectors. With current department sizes (100-500 entities + 1000-4000 chunks), this is not a concern.

### Trade-off 4: Per-Query Community Search vs. Pre-Computed Summaries

**Choice:** Louvain community detection at ingestion time; community summaries stored and searched at query time.

**Why not real-time community analysis?**
- **Stability:** Community structure doesn't change between ingestion runs.
- **Performance:** Louvain on 500 nodes takes 50ms. No need to recompute per query.
- **Context quality:** Pre-computed summaries provide high-level department overviews that raw entity lists cannot.

---

## Summary

The retrieval pipeline is a **4-layer system** that prioritizes deterministic graph data over probabilistic vector search:

| Layer | Component | Purpose | Latency |
|-------|-----------|---------|---------|
| 1 | DepartmentRouter | Route to correct tenant(s) | <1ms |
| 2 | Deterministic Graph | Authoritative graph lookups | <5ms |
| 3 | Hybrid RAG | Entity + Vector + Community search | 50-200ms |
| 4 | Answerability + Verifier | Evidence validation | 1-3s (LLM call) |

The architecture ensures that for **any query with a known structural answer**, the system returns provenance-backed evidence without relying on the LLM to "figure it out" from raw text chunks. For open-ended queries, it gracefully degrades to semantic search with answerability gating.
