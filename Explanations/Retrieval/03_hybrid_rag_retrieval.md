# 03 — Hybrid RAG Retrieval: Entity Search, Vector Search, and Community Summaries

## Table of Contents

1. [When Does Hybrid RAG Execute?](#when-does-hybrid-rag-execute)
2. [The Three Search Channels](#the-three-search-channels)
3. [Entity Search (Local Search)](#entity-search-local-search)
4. [Vector Search (Semantic Search)](#vector-search-semantic-search)
5. [Community Search (Global Search)](#community-search-global-search)
6. [Context Assembly and Word Budget](#context-assembly-and-word-budget)
7. [The FAISS Embedding Engine](#the-faiss-embedding-engine)
8. [Community Detection and Summarization](#community-detection-and-summarization)
9. [Section-Specific Hybrid Retrieval](#section-specific-hybrid-retrieval)
10. [The Rules Retriever Subsystem](#the-rules-retriever-subsystem)
11. [Why Three Channels Instead of One?](#why-three-channels-instead-of-one)

---

## When Does Hybrid RAG Execute?

Hybrid RAG retrieval activates **after** the deterministic layer returns `None` (no structural graph answer found), or **alongside** a deterministic answer to provide supplementary context. The `retrieve_bundle()` method orchestrates both layers:

```python
def retrieve_bundle(self, query, local_top_k=6, vector_top_k=4,
                    global_top_k=2, max_context_words=3000):
    # Phase 0: Deterministic context from graph (highest priority)
    det_ctx = self.get_deterministic_context(query)
    
    # Phase 1: Entity search (name matching + embedding fallback)
    local_results = self._local_search(query, top_k=local_top_k)
    
    # Phase 2: Vector search (FAISS semantic similarity)
    # SKIPPED for enumeration queries (roster listings)
    vector_results = [] if is_enumeration else self._vector_search(query, top_k=vector_top_k)
    
    # Phase 3: Community search (Louvain cluster summaries)
    global_results = self._global_search(query, top_k=global_top_k)
    
    # Assemble and return context bundle
    return {"context": ..., "provenance": ..., "answerability": ...}
```

**Key architectural decision:** For enumeration queries (faculty roster, PhD roster), vector search is **skipped entirely**. The deterministic graph data is authoritative for listings, and vector search would only introduce noise (e.g., text chunks mentioning faculty names out of context).

---

## The Three Search Channels

| Channel | Method | Data Source | Purpose | Default Top-K |
|---------|--------|-------------|---------|--------------|
| **Entity Search** | `_local_search()` | Graph nodes | Find specific entities (people, labs, projects) | 6 |
| **Vector Search** | `_vector_search()` | FAISS index over text chunks | Find semantically relevant passages | 4 |
| **Community Search** | `_global_search()` | Louvain community summaries | Provide departmental overview context | 2 |

---

## Entity Search (Local Search)

Entity search finds **specific graph entities** whose names, attributes, or embeddings match the query. It operates in three phases:

### Phase 1: Direct Name Matching (`_name_match`)

This is the highest-priority entity matching strategy. It uses a sophisticated token-based fuzzy matching algorithm:

```python
def _name_match(self, query: str) -> List[str]:
    # 1. Clean query: remove punctuation, strip title prefixes (Dr., Prof.)
    # 2. For each entity name in the index:
    #    a. Try exact match with full name
    #    b. Try substring match (>= 5 chars)
    #    c. Try fuzzy match using SequenceMatcher (ratio > 0.80)
    #    d. Try token-by-token comparison (exact or fuzzy)
    # 3. Deduplicate and return top 8 by score
```

**Why fuzzy matching?** Users misspell names frequently. "Anurag Misra" should match "Anurag Mishra". The `SequenceMatcher` with a 0.80 threshold catches these while avoiding false positives.

**Stop words filtering:** Common query words like "who", "is", "the", "under", "supervisor", "department" are excluded from name matching to prevent them from colliding with entity names.

**Score tiers:**
| Match Type | Score | Example |
|-----------|-------|---------|
| Exact full name | 1.0 | "anand mishra" → "Anand Mishra" |
| Substring (≥5 chars) | 0.95 | "anand" → "Anand Mishra" |
| Fuzzy full name (ratio > 0.80) | 0.72-0.90 | "anurag misra" → "Anurag Mishra" |
| Exact token match | 0.95 | "mishra" → "Anand Mishra" |
| Substring token (≥4 chars) | 0.70 | "mish" → "Anand Mishra" |
| Fuzzy token (ratio > 0.70) | 0.50-0.63 | "mishre" → "Mishra" |

### Phase 1.5: Research Area Supervisor Lookup

If the query contains faculty/supervisor/research keywords, this phase triggers:

```python
if any(kw in ql for kw in ["supervis", "faculty", "professor", "who work", "research", "expert"]):
    area_results = self._find_supervisors_by_research_area(query)
```

This performs a dedicated search through PhD students' research areas and faculty research interests to find experts in a given topic. It uses the same `_topic_matches_text()` word-boundary-aware matching as the deterministic layer.

### Phase 2: Embedding-Based Entity Search

For remaining slots (after name matches), FAISS cosine similarity search finds entities whose embedding descriptions match the query:

```python
remaining = top_k - len(results)
if remaining > 0:
    entity_matches = self.embeddings.search(
        query, top_k=remaining,
        type_filter="entity",           # Only entity embeddings
        department_filter=self.dept_code, # Only this department's entities
        min_score=0.35                   # Minimum similarity threshold
    )
```

**Why a 0.35 minimum score?** This filters out very low-similarity matches that would add noise to the context. The threshold was empirically tuned to balance recall (finding relevant entities) against precision (excluding irrelevant ones).

### Entity Display and Relationship Formatting

Once entities are found, they are formatted with rich context:

```python
def _get_node_display(self, node_id: str) -> str:
    # For Faculty: name, designation, email, HoD status, education, research
    # For PhDStudent: name, research area, email
    # For Project: title, project number, funding agency
    # For Patent: title, application number
    # For PlacementData: program, year, percentage, salary stats
    # For Lab: name only
    # ...

def _get_relationships_display(self, node_id: str) -> str:
    # Outgoing edges: SUPERVISED_BY → "Supervisor: X"
    #                 RESEARCHES_IN → "Research Area: X"
    #                 INVENTED → "Patent: X"
    # Incoming edges: SUPERVISED_BY → "PhD Student: X"
    #                 BELONGS_TO_CATEGORY → "Sub-area: X"
    # Special: co-inventor detection for patents
```

---

## Vector Search (Semantic Search)

Vector search finds **text chunks** whose embeddings are semantically similar to the query. This catches information that isn't captured by named entities — narrative paragraphs, policy descriptions, historical context, etc.

### Implementation

```python
def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
    chunk_matches = self.embeddings.search(
        query, top_k=top_k,
        type_filter="chunk",               # Only text chunks
        department_filter=self.dept_code,   # Department isolation
        min_score=0.35                      # Minimum similarity
    )
    for item, score in chunk_matches:
        text = item["text"][:1200]          # Truncate long chunks
        results.append({
            "type": "chunk",
            "score": score,
            "text": text,
            "source": meta.get("title", "Unknown"),
            "url": meta.get("url", ""),
        })
```

### Key Design Decisions

**Department filtering:** Each query only searches the FAISS index for vectors belonging to the current department. This prevents cross-department contamination (e.g., CSE chunks appearing in EE results).

**Chunk truncation at 1200 characters:** Long chunks eat into the LLM's context window. Truncation keeps the context compact while preserving the most relevant opening content.

**Min score threshold of 0.35:** Set conservatively. Below 0.35, the semantic match is too weak to be useful — these results would likely confuse the LLM rather than help it.

---

## Community Search (Global Search)

Community search retrieves **pre-computed community summaries** that provide high-level departmental context. These are generated during ingestion by the Louvain community detection algorithm.

### Implementation

```python
def _global_search(self, query: str, top_k: int = 3) -> List[Dict]:
    matches = self.embeddings.search(
        query, top_k=top_k,
        type_filter="community",           # Only community summaries
        department_filter=self.dept_code
    )
    for item, score in matches:
        # Find the original community report for richer data
        report = next((r for r in self.community_reports if r["id"] == comm_id), None)
        summary = report["summary"] if report else item.get("text", "")
        
        # Include member names for context
        for label, member_ids in report.get("members_by_type", {}).items():
            names = [self.graph.nodes.get(m, {}).get("name", m) for m in member_ids[:8]]
            members_str += f"\n  - {label}: {', '.join(names)}"
```

### When Is Community Search Skipped?

```python
if self._is_exact_count_query(query):
    global_top_k = 0  # Skip community search for exact count queries
```

**Why?** Community summaries contain approximate descriptions like "a group of 5 faculty members working on..." If the user asks "How many faculty?", the community summary might suggest an incorrect count. Exact counts must come from the graph, not from community descriptions.

---

## Context Assembly and Word Budget

After all three channels return results, `retrieve_bundle()` assembles them into a structured context string with a strict **word budget** (default 3000 words):

```
## Authoritative Department Data      ← Deterministic context (highest priority)
[deterministic graph answer]

---

## Matched Entities from Knowledge Graph  ← Entity search results
### Faculty
**Dr. Anand Mishra**
  - Designation: Assistant Professor
  - Email: anand@iitjammu.ac.in
  - PhD Student: Rahul Kumar (Research Area: Deep Learning)

---

## Relevant Department Information     ← Vector search results
[Source: ee_faculty_page.md] (url)
Chunk text about department activities...

---

## Department Overview                 ← Community search results
Community 3 in IIT Jammu EE Department:
  Faculty: Anand Mishra, Priya Singh, ...
  PhDStudent: Rahul Kumar, ...
  Relationships: SUPERVISED_BY: 5, RESEARCHES_IN: 3
```

### Word Budget Enforcement

Each section's word count is tracked incrementally. Once the budget is reached, remaining results are dropped:

```python
for r in vector_results:
    if word_count > max_context_words:
        break
    remaining_words = max_context_words - word_count
    words = text.split()
    if len(words) > remaining_words:
        text = " ".join(words[:remaining_words]) + "..."
    ...
```

### Minimum Context Gate

If the total assembled context is fewer than 20 words (excluding the "no information" message), the system forces the query to be unanswerable:

```python
if context_word_count < 20 and context.strip() != "No relevant information...":
    context = "No relevant information found..."
    local_results = vector_results = global_results = []
```

This prevents the LLM from hallucinating on near-empty evidence (e.g., a context that contains only a section header with no actual content).

---

## The FAISS Embedding Engine

### Architecture

**File:** `graphrag/embeddings.py` → `EmbeddingEngine`

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Model** | `all-mpnet-base-v2` | Best balance of quality vs. speed for academic text |
| **Dimension** | 768 | MPNet output dimension |
| **Index Type** | `IndexFlatIP` (brute-force inner product) | Exact search; fast for <10K vectors |
| **Normalization** | L2 normalized before indexing | Inner product on L2-normalized vectors = cosine similarity |
| **Device** | CPU by default (`EMBEDDING_DEVICE` env var) | Avoids GPU contention with Ollama |

### Why `all-mpnet-base-v2`?

This model was chosen over alternatives because:

- **Better than `all-MiniLM-L6-v2`:** Higher quality embeddings (768d vs 384d), critical for distinguishing between similar institutional terms.
- **Faster than `all-roberta-large-v1`:** 4x faster inference, important for real-time chat.
- **Academic text optimized:** Trained on diverse academic and professional text, relevant for IIT Jammu's content.

### What Gets Embedded?

Three types of items are indexed:

1. **Text Chunks:** Raw scraped text from department websites, truncated to 1000 characters.
2. **Entity Descriptions:** Rich text descriptions generated by `create_entity_descriptions()`:
   - Faculty: name, department, designation, email, research areas, supervised students, patents, startups
   - PhD Students: name, research topic, email, supervisors
   - Research Areas: name, category
   - Projects: title, funding agency
   - Patents: title, application number
   - Labs: name, department
3. **Community Summaries:** Louvain community text representations with member names and relationship counts.

Each item is tagged with `type` (chunk/entity/community) and `department` code for filtered search.

---

## Community Detection and Summarization

### The Louvain Algorithm

**File:** `graphrag/community.py`

The system uses the **Louvain method for community detection** on the entity graph:

```python
def detect_communities(graph: nx.DiGraph, resolution: float = 1.0):
    # 1. Filter to entity nodes only (exclude TextChunk, Document)
    entity_labels = {"Faculty", "PhDStudent", "ResearchArea", "ResearchCategory",
                     "Project", "Patent", "Startup", "Lab", ...}
    
    # 2. Convert to undirected subgraph
    subgraph = graph.subgraph(entity_nodes).to_undirected()
    
    # 3. Remove isolated nodes
    connected_nodes = [n for n in subgraph.nodes() if subgraph.degree(n) > 0]
    
    # 4. Run Louvain with fixed random seed for reproducibility
    partition = community_louvain.best_partition(subgraph, resolution=1.0, random_state=42)
```

### Why Louvain?

- **No parameter tuning needed:** The resolution parameter (1.0) works well for academic department graphs.
- **Deterministic with fixed seed:** `random_state=42` ensures reproducibility.
- **Handles varying sizes:** Works with graphs from 50 nodes (small departments) to 500+ nodes (CSE).
- **Meaningful clusters:** Naturally groups faculty with their students and research areas.

### Community Reports

Each detected community gets a structured report:

```python
report = {
    "id": "community_0",
    "community_id": 0,
    "members": [node_ids],
    "members_by_type": {"Faculty": [...], "PhDStudent": [...], "ResearchArea": [...]},
    "internal_edges": [{"source": ..., "target": ..., "type": "SUPERVISED_BY"}],
    "text": "Community 0 in IIT Jammu EE Department:\n  Faculty: Dr. A, Dr. B\n  ...",
    "summary": "",   # Populated by LLM or rule-based fallback
    "size": 12,
}
```

### Summarization: LLM vs. Rule-Based

If an LLM function is available, communities get rich natural-language summaries:

```
This group includes Dr. Anand Mishra and Dr. Priya Singh, who work on deep learning
and computer vision. They supervise 5 PhD scholars researching topics including
reinforcement learning and object detection.
```

If no LLM is available, a **rule-based fallback** generates structured summaries:

```
This group includes 2 faculty member(s): Anand Mishra, Priya Singh.
5 PhD student(s) working in areas like Deep Learning, Computer Vision, NLP.
```

---

## Section-Specific Hybrid Retrieval

The `SectionRetriever` has its own `retrieve_bundle()` that differs from `HybridRetriever`:

### Key Differences

| Aspect | HybridRetriever (Departments) | SectionRetriever (Sections) |
|--------|-------------------------------|----------------------------|
| **Community search** | Yes (Louvain communities) | No |
| **Entity embedding search** | Yes (FAISS entity index) | FAISS chunk index only |
| **Text chunk ranking** | Embedding-based | Word overlap scoring |
| **Default max words** | 3000 | 4500 |
| **Rules DB integration** | No | Yes (for academics) |

### Section Retrieval Flow

```python
def retrieve_bundle(self, query):
    # 1. Academic rules check (academics section only)
    if self.section_code == "academics":
        rr = RulesRetriever()
        if is_rules_query:
            rules_context = rr.generate_context(rr.retrieve(query))
    
    # 2. Deterministic context from section graph
    direct_ctx = self.get_deterministic_context(query)
    
    # 3. Semantic vector search (if FAISS index available)
    vector_results = self.embeddings.search(query, ...)
    
    # 4. Text chunk ranking (word overlap fallback)
    for chunk in self.chunks:
        overlap = len(q_words.intersection(chunk_words))
        if overlap > 0:
            local_results.append(...)
    
    # 5. Combine all context blocks
    context = "\n\n---\n\n".join(combined_blocks)
```

### Chunk Filtering for Academics

The academics section applies special filters to exclude noise from combined/aggregated documents:

```python
# Skip combined/aggregated docs that duplicate information
if "00_combined" in doc_name:
    continue
# For academic rules queries, only use parsed PDF documents
if is_academic_rules_request and "parsed_documents" not in doc_name:
    continue
```

---

## The Rules Retriever Subsystem

For the academics section, there is a dedicated **Rules Retriever** that handles academic regulations queries (grading, milestones, credit requirements, etc.):

### Architecture

**Files:** `graphrag/rules_retriever.py` → `RulesRetriever`, `graphrag/rules_db.py` → `RulesDB`

The rules system uses a **SQLite FTS5 (Full-Text Search)** database containing parsed sections from academic regulation PDFs. It provides:

1. **Intent Classification:** Detects the target program (UG/MTech/PhD) and specific intent (grades, milestones, credits, facts).
2. **Structured Lookups:** Direct table queries for grade scales, program milestones, credit requirements, and rule facts.
3. **Evidence-Ranked Section Retrieval:** Custom scoring of PDF sections using token overlap, phrase matching, and domain-specific boosts.

### Intent Classification

```python
def classify_intent(self, query: str) -> Dict[str, Any]:
    # Program Classification
    if "phd" in q: program = "PhD"
    elif "mtech" in q: program = "MTech"
    elif "btech" in q: program = "UG"
    
    # Fact Lookups (specific CGPA thresholds, credit requirements, etc.)
    if "minor" and "cgpa": facts = ["min_cgpa_minor", "min_credits_minor"]
    if "attendance": facts = ["attendance_policy"]
    
    # Structural Intents
    intents = {
        "grades": has grade/grading/scale terms,
        "milestones": has exam/seminar/timeline terms,
        "credits": has credit/requirement terms,
        "facts": list of fact keys,
        "program": "UG" | "MTech" | "PhD" | None
    }
```

### Evidence Scoring

The `_rank_sections()` method implements a sophisticated scoring system:

```python
# Token-level scoring
for term in terms:
    if term in title_tokens: score += 6.0   # Title match is strong
    if term in body_tokens:  score += 1.0   # Body match is weak

# Phrase-level scoring  
for phrase in phrases:
    if phrase in title_lower: score += 14.0  # Exact phrase in title
    if phrase in body_lower: score += 5.0    # Exact phrase in body

# Domain-specific boosts
if "attendance" in q and "attendance policy" in body: score += 150.0
if "phd duration" and "minimum period of registration" in body: score += 150.0
if course_code_match: score += 80.0         # Exact course code match
```

### Phrase Expansion System

The rules retriever expands user queries into related search terms:

```python
PHRASE_EXPANSIONS = {
    "btp": ["btp", "btech project", "project allotment"],
    "course withdrawal": ["withdrawal of course", "course withdrawal", "ww"],
    "change department": ["change of department", "department change", "branch change"],
    "ra category": ["ra category", "research assistant", "course structure for students under ra category"],
    ...
}
```

This handles the vocabulary gap between user queries (informal) and PDF content (formal/bureaucratic).

---

## Why Three Channels Instead of One?

Each channel captures a different type of information:

| Channel | Captures | Cannot Capture |
|---------|----------|----------------|
| **Entity Search** | Specific people, projects, labs by name | Narrative context, policy descriptions |
| **Vector Search** | Semantically relevant passages, policies | Exact entity attributes (email, designation) |
| **Community Search** | Department-level themes, member clusters | Specific facts about individuals |

### Example: "Tell me about Dr. Anand Mishra's research group"

- **Entity search** finds: Dr. Anand Mishra (node), his supervised students, his research areas.
- **Vector search** finds: Text chunks from his profile page describing his lab's focus and recent publications.
- **Community search** finds: The community that includes Dr. Mishra, showing other faculty and students in his research cluster.

Together, they provide a comprehensive, multi-faceted context that a single channel could never achieve.

### Provenance Tracking

Each channel's contribution is tracked in the provenance metadata:

```python
provenance = {
    "route": "graph+vector",
    "source_mode": "both",
    "graph": {
        "direct": False,
        "items": 3,
        "avg_score": 0.85,
        "labels": {"Faculty": 1, "PhDStudent": 2},
        "word_count": 150,
    },
    "vector": {
        "items": 2,
        "avg_score": 0.65,
        "sources": ["ee_faculty_page.md", "ee_research.md"],
        "word_count": 300,
    },
    "community": {
        "items": 1,
        "avg_score": 0.55,
        "word_count": 80,
    },
}
```

This provenance data is used downstream for relevance scoring in broadcast mode and for debugging retrieval quality.
