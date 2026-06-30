# Part 4 — Embedding Generation & FAISS Indexing (Phase 4)

> *How text chunks, entity descriptions, and community summaries are encoded into a unified vector index for semantic retrieval.*

---

## 1. Why Three Embedding Sources?

Phase 4 builds the FAISS index that powers the chatbot's semantic search. Unlike simple RAG systems that embed only text chunks, this pipeline embeds **three distinct text sources** into the same vector space:

```
┌─────────────────────────────────────────────────────────┐
│                    FAISS FlatIP Index                     │
│                                                          │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ Text Chunks  │  │ Entity           │  │ Community  │ │
│  │ (raw text)   │  │ Descriptions     │  │ Summaries  │ │
│  │              │  │ (graph → text)   │  │ (LLM prose)│ │
│  │ ~500-1500    │  │ ~100-400         │  │ ~5-15      │ │
│  │ vectors      │  │ vectors          │  │ vectors    │ │
│  └─────────────┘  └──────────────────┘  └────────────┘ │
│                                                          │
│  Total: ~600-2000 vectors per department                 │
└─────────────────────────────────────────────────────────┘
```

### Why not just embed chunks?

| Source | Answers queries like... | Why chunks alone fail |
|---|---|---|
| **Text Chunks** | "What is the department's vision?" | Works fine for open-ended questions |
| **Entity Descriptions** | "Tell me about Dr. Saxena" | Chunks fragment faculty info across 5+ pages; the entity description consolidates it |
| **Community Summaries** | "What research happens in EE?" | No single chunk covers a research group; the community summary synthesizes it |

The three sources form a **retrieval hierarchy**: entity descriptions for precision, community summaries for breadth, and chunks for fallback coverage.

---

## 2. Entity Description Generation: `create_entity_descriptions()`

The `create_entity_descriptions()` function in `embeddings.py` (line 20) transforms structured graph nodes into natural-language text suitable for embedding:

```python
def create_entity_descriptions(graph: nx.DiGraph) -> list:
    descriptions = []
    SKIP_LABELS = {"TextChunk", "Document"}
    
    for node, data in graph.nodes(data=True):
        label = data.get("label", "")
        if label in SKIP_LABELS:
            continue
        
        desc = _describe_entity(node, data, label)
        if desc and len(desc.strip()) > 20:
            descriptions.append({
                "id": f"entity_{node}",
                "text": desc,
                "metadata": {
                    "entity_id": node,
                    "label": label,
                    "name": data.get("name", node),
                    "source": "entity_description"
                }
            })
    return descriptions
```

### 2.1 The `_describe_entity()` Dispatch

Each entity type gets a specialized description template:

#### Faculty Description
```python
def _describe_faculty(data):
    parts = [f"{data['name']} is a {data.get('designation', 'faculty member')} "
             f"in the {data.get('department', '')} department at IIT Jammu."]
    
    if data.get("research_interests"):
        parts.append(f"Research interests: {data['research_interests']}.")
    if data.get("education"):
        parts.append(f"Education: {data['education']}.")
    if data.get("email"):
        parts.append(f"Contact: {data['email']}.")
    if data.get("is_hod"):
        parts.append("Currently serves as Head of Department.")
    if data.get("lab"):
        parts.append(f"Associated with {data['lab']}.")
    
    return " ".join(parts)
```

**Example output:**
```
"Alok Kumar Saxena is an Associate Professor in the EE department at IIT Jammu. 
Research interests: Power Electronics, Motor Drives, Renewable Energy. 
Education: PhD from IIT Delhi, M.Tech from IIT Bombay. 
Contact: alok.saxena@iitjammu.ac.in. Associated with Power Electronics Lab."
```

#### PhD Student Description
```python
def _describe_phd_student(data):
    parts = [f"{data['name']} is a PhD student at IIT Jammu."]
    if data.get("supervisor"):
        parts.append(f"Supervisor: {data['supervisor']}.")
    if data.get("research_area"):
        parts.append(f"Research area: {data['research_area']}.")
    if data.get("year"):
        parts.append(f"Batch: {data['year']}.")
    return " ".join(parts)
```

#### Other Entity Types
- **Lab**: `"{name} is a laboratory/research facility in the {dept} department."`
- **Project**: `"{title} is a funded research project by {PI}, funded by {agency} for Rs. {amount}."`
- **Patent**: `"{title} is a patent filed by {inventors}."`
- **ResearchArea**: `"{name} is a research area in the {dept} department at IIT Jammu."`
- **PlacementData**: `"{name} was placed at {company} with a package of {package} ({year})."`
- **PolicyNotification**: `"{title}. Category: {category}. Applies to: {applies_to}. Summary: {summary}."`
- **FeeStructure**: `"Fee structure for {programme}, entry year {year}: General/OBC/EWS: {fee_gen}, SC/ST/PwD: {fee_sc}."`

### 2.2 The Section Entity Description Generator

Sections use `create_section_entity_descriptions()` in `section_kg_builder.py`, which handles section-specific labels:

```python
def create_section_entity_descriptions(graph: nx.DiGraph, section_code: str) -> list:
    for node, data in graph.nodes(data=True):
        label = data.get("label", "")
        
        if label == "SectionHead":
            desc = f"{data['name']} is the {data.get('designation', 'head')} of the {section_name}."
        elif label == "CommitteeMember":
            desc = f"{data['name']} serves as {data.get('designation', 'member')} on the {data.get('committee_name', '')} committee."
        elif label == "PolicyNotification":
            # Rich multi-field description
            desc = _describe_policy_notification(data)
        elif label == "FeeStructure":
            desc = f"Fee structure for {data['programme']}..."
        elif label == "Course":
            desc = f"{data['name']} ({data.get('code', '')}) — {data.get('credits', '')} credits, L-T-P: {data.get('ltp', '')}."
        # ... more labels ...
```

### 2.3 Why Generate Descriptions Instead of Embedding Raw Properties?

Raw node properties are key-value pairs: `{"name": "Alok Kumar Saxena", "designation": "Associate Professor", "research_interests": "Power Electronics, Motor Drives"}`. These are great for programmatic lookup but terrible for embedding because:

1. **Embedding models understand sentences, not key-value pairs.** The sentence "Alok Kumar Saxena is an Associate Professor whose research interests include Power Electronics" embeds much more meaningfully than the concatenated string "name: Alok Kumar Saxena designation: Associate Professor".

2. **Contextual bridging.** The description adds context words like "at IIT Jammu", "in the EE department", "is a PhD student" that help the embedding model understand the *type* of information, not just the *content*.

3. **Query compatibility.** Users ask "Who works on power electronics?" — the embedding model needs to match this against text that contains "works on" or "research interests include", not against "research_interests: Power Electronics".

### 2.4 Length Filtering

Descriptions shorter than 20 characters are discarded. These typically come from entities with no meaningful properties (e.g., a ResearchArea node with only a name like "AI"). Such short texts produce low-quality embeddings that could pollute search results.

---

## 3. The Embedding Engine: `EmbeddingEngine`

The `EmbeddingEngine` class in `embeddings.py` (line 80) manages the encoding and indexing:

```python
class EmbeddingEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = os.environ.get("EMBEDDING_DEVICE", "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.metadata = []
```

### 3.1 Model Selection: `all-mpnet-base-v2`

The `all-mpnet-base-v2` model is chosen for specific reasons:

| Criterion | all-mpnet-base-v2 | Alternatives Considered |
|---|---|---|
| **Embedding dimension** | 768 | all-MiniLM-L6-v2: 384 (too low for entity precision) |
| **Performance** | #1 on SBERT benchmarks (at time of selection) | BGE-base: similar quality but larger |
| **Speed** | ~100 texts/sec on CPU | all-MiniLM: faster but less accurate |
| **Max sequence** | 384 tokens (~300 words) | Sufficient for entity descriptions |
| **License** | Apache 2.0 | Free for any use |

The 768-dimensional embeddings provide sufficient representational capacity to distinguish between similar entities (e.g., two faculty members in related fields).

### 3.2 The `EMBEDDING_DEVICE` Environment Variable

```python
self.device = os.environ.get("EMBEDDING_DEVICE", "cpu")
```

This is not just a convenience toggle — it's a **resource management** mechanism. The chatbot runtime may be running Ollama (LLM inference) on the GPU simultaneously. Loading the SentenceTransformer on GPU while Ollama is active can cause CUDA OOM errors. Setting `EMBEDDING_DEVICE=cpu` during ingestion prevents GPU contention.

In production, the device is typically:
- **`cpu`** during ingestion (Ollama uses GPU)
- **`cuda`** during query-time encoding (only SentenceTransformer needs GPU)

### 3.3 The `encode()` Method

```python
def encode(self, texts: list, batch_size: int = 64) -> np.ndarray:
    embeddings = self.model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # L2-normalize to unit vectors
    )
    return embeddings.astype(np.float32)
```

**`normalize_embeddings=True`**: This L2-normalizes every vector to unit length. When all vectors have ||v|| = 1, inner product equals cosine similarity:

```
cos(a, b) = (a · b) / (||a|| × ||b||)
           = (a · b) / (1 × 1)     [when L2-normalized]
           = a · b                   [= inner product]
```

This allows using FAISS's `IndexFlatIP` (inner product) as an exact cosine similarity search — IP is computationally cheaper than explicit cosine because it skips the normalization division.

---

## 4. Building the FAISS Index: `build_index()`

```python
def build_index(self, chunks, entity_descriptions, community_items, dept_code=None):
    all_texts = []
    all_metadata = []
    
    # Source 1: Text Chunks
    for chunk in chunks:
        all_texts.append(chunk["text"])
        all_metadata.append({
            "source": "chunk",
            "doc": chunk.get("doc", ""),
            "url": chunk.get("url", ""),
            "title": chunk.get("title", ""),
            "chunk_idx": chunk.get("chunk_idx", 0),
            "dept_code": dept_code,
        })
    
    # Source 2: Entity Descriptions
    for desc in entity_descriptions:
        all_texts.append(desc["text"])
        all_metadata.append({
            "source": "entity_description",
            "entity_id": desc["metadata"]["entity_id"],
            "label": desc["metadata"]["label"],
            "name": desc["metadata"]["name"],
            "dept_code": dept_code,
        })
    
    # Source 3: Community Summaries
    for item in community_items:
        all_texts.append(item["text"])
        all_metadata.append({
            "source": "community_summary",
            "community_id": item["metadata"]["community_id"],
            "size": item["metadata"]["size"],
            "dept_code": dept_code,
        })
    
    # Encode all texts into vectors
    embeddings = self.encode(all_texts)  # Shape: (N, 768)
    
    # Build FAISS index
    dim = embeddings.shape[1]  # 768
    self.index = faiss.IndexFlatIP(dim)
    self.index.add(embeddings)
    self.metadata = all_metadata
```

### 4.1 The Three-Source Metadata Schema

Each vector in the FAISS index has a corresponding metadata entry. The `source` field is the discriminator:

```json
// Chunk metadata
{"source": "chunk", "doc": "ee_faculty-list.html.md", "url": "https://...", "title": "Faculty List", "chunk_idx": 0, "dept_code": "ee"}

// Entity description metadata
{"source": "entity_description", "entity_id": "Alok Kumar Saxena", "label": "Faculty", "name": "Alok Kumar Saxena", "dept_code": "ee"}

// Community summary metadata
{"source": "community_summary", "community_id": 0, "size": 12, "dept_code": "ee"}
```

At retrieval time, the `source` field tells the retriever what kind of match it found:
- `"chunk"` → Return the raw text, cite the source URL
- `"entity_description"` → Fetch the full entity from the graph for structured attributes
- `"community_summary"` → Return the summary for broad thematic context

### 4.2 Why IndexFlatIP?

```python
self.index = faiss.IndexFlatIP(dim)
```

**IndexFlatIP** performs **exact** inner-product search — it compares the query vector against every vector in the index. For the dataset sizes involved (500–2000 vectors), this is:

| Metric | IndexFlatIP | IndexIVFFlat (approximate) |
|---|---|---|
| **Search time (1000 vectors)** | ~0.1ms | ~0.05ms |
| **Recall** | 100% (exact) | ~95-99% (approximate) |
| **Index build time** | O(1) | Requires training (nlist centroids) |
| **Memory** | 768 × N × 4 bytes | Same + centroid overhead |
| **Complexity** | Zero parameters | nlist, nprobe tuning required |

The 0.05ms speedup from approximate search is meaningless at this scale. Exact search guarantees deterministic, reproducible results with zero tuning.

### 4.3 Index Size Estimation

For a typical department (1000 vectors × 768 dimensions × 4 bytes/float):
- **FAISS index**: ~3MB
- **Metadata JSON**: ~200KB
- **Total**: ~3.2MB per department

For all 11+ departments + 10+ sections: **~70MB total**. Easily fits in memory on any modern machine.

---

## 5. Saving and Loading

### 5.1 Saving

```python
def save(self, data_dir: str):
    # Save FAISS index
    index_path = os.path.join(data_dir, "embeddings.faiss")
    faiss.write_index(self.index, index_path)
    
    # Save metadata
    meta_path = os.path.join(data_dir, "embeddings_meta.json")
    with open(meta_path, "w") as f:
        json.dump(self.metadata, f, indent=2)
```

### 5.2 Loading at Runtime

```python
def load(self, data_dir: str):
    index_path = os.path.join(data_dir, "embeddings.faiss")
    self.index = faiss.read_index(index_path)
    
    meta_path = os.path.join(data_dir, "embeddings_meta.json")
    with open(meta_path, "r") as f:
        self.metadata = json.load(f)
```

The FAISS binary format is compact and loads in milliseconds. The metadata JSON loads similarly fast. At startup, the chatbot loads all department + section indices into memory for instant retrieval.

---

## 6. Search at Query Time

While search is not part of the ingestion pipeline, understanding how the index is queried illuminates why the pipeline structures data the way it does:

```python
def search(self, query: str, top_k: int = 10) -> list:
    query_embedding = self.encode([query])  # Shape: (1, 768)
    scores, indices = self.index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:  # FAISS returns -1 for empty results
            continue
        meta = self.metadata[idx]
        results.append({
            "score": float(score),
            "text": self.texts[idx],  # Stored during build_index
            **meta
        })
    return results
```

A typical query result set might contain:
```
Score 0.89: entity_description — "Alok Kumar Saxena is an Associate Professor..."
Score 0.84: chunk — "...Power Electronics Lab conducts research on..."
Score 0.81: community_summary — "This group focuses on power electronics..."
Score 0.77: chunk — "...inverter topologies for renewable energy..."
```

The retrieval layer uses the `source` field to decide how to process each result, and the `dept_code` field to route multi-department queries.

---

## 7. The Embedding Pipeline in Numbers

| Metric | Typical Value | Range |
|---|---|---|
| Text chunks per department | 300–800 | 50–1500 |
| Entity descriptions per department | 80–200 | 20–400 |
| Community summaries per department | 5–15 | 3–25 |
| Total vectors per department | 500–1000 | 100–2000 |
| Encoding time (CPU, all vectors) | 15–60s | 5–120s |
| Encoding time (GPU, all vectors) | 2–10s | 1–30s |
| FAISS index size per department | 1–5MB | 0.5–10MB |
| Metadata JSON size | 100–500KB | 50KB–1MB |

### 7.1 Batch Size Optimization

The `encode()` method uses `batch_size=64` by default. This is tuned for CPU inference:
- **Too small** (batch_size=1): Python loop overhead dominates
- **Too large** (batch_size=256): Memory pressure on CPU, diminishing returns
- **Sweet spot** (batch_size=64): Balanced throughput on 8-16 core machines

On GPU, larger batch sizes (128–256) would be more efficient, but the current default works well for both.

---

## 8. Design Decisions and Trade-offs

### 8.1 Single Index vs. Separate Indices

**Decision**: All three sources (chunks, entities, communities) are embedded in a **single** FAISS index.

**Alternative**: Three separate indices, one per source type.

**Rationale**: A single index allows the retrieval layer to compare entity descriptions, chunks, and community summaries on the same scale. If a query is best answered by an entity description (score 0.89), it should rank above a mediocre chunk match (score 0.72), which is only possible if they're in the same vector space.

### 8.2 No Metadata Filtering in FAISS

FAISS doesn't support metadata-based filtering (e.g., "only search entity descriptions"). Filtering happens **post-search** in the retrieval layer. This is fine at the current scale because retrieving top-50 from 1000 vectors and filtering in Python is sub-millisecond.

### 8.3 No Index Updates

The FAISS index is **immutable** after creation. Adding new entities requires rebuilding the entire index. This aligns with the pipeline's idempotent rebuild-from-scratch philosophy.

### 8.4 L2 Normalization Trade-off

By L2-normalizing all vectors, we lose magnitude information. In theory, a long, detailed entity description should have a "stronger" embedding than a short, generic one. In practice, magnitude differences in SentenceTransformer outputs are dominated by text length rather than semantic content, so normalization **improves** retrieval quality by removing this length bias.

---

## Next: Part 5 — Entity Resolution, Identity Management & Maintenance

The final document provides a deep dive into the cross-cutting concerns: the EntityResolver's matching algorithms, the GlobalPersonIndex's role-placeholder filtering, the SectionKGBuilder's policy extraction pipeline, and long-term maintenance guidance for the entire system.
