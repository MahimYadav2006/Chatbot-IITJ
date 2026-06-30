# Part 3 — Community Detection & Summarization (Phases 2–3)

> *How the flat knowledge graph is partitioned into thematic clusters and enriched with LLM-generated semantic summaries.*

---

## 1. Why Community Detection?

The knowledge graph produced by Phase 1 contains hundreds of individual entities — faculty members, students, research areas, labs, projects, patents. Individually, these entities answer specific factual queries ("Who supervises X?"). But many user queries are **thematic**:

- *"What kind of research happens in the EE department?"*
- *"Tell me about the power electronics group."*
- *"Which faculty work on signal processing?"*

Answering these requires understanding **which entities naturally cluster together**. Community detection solves this by partitioning the graph into groups of tightly-connected entities — a faculty member, their students, their research areas, and their funded projects all end up in the same community.

### 1.1 The Retrieval Benefit

Community summaries serve as a **middle retrieval tier** between individual entities and raw text chunks:

```
Query: "What research happens in EE?"
    │
    ▼
Tier 1 (Entity Search): Too granular — returns individual research areas
Tier 2 (Community Search): ★ Returns thematic clusters like
    "Power Electronics group (Dr. X, Dr. Y, 5 PhD students,
     working on inverters, motor drives, and renewable energy)"
Tier 3 (Chunk Search): Too broad — returns random paragraphs
```

The community summary provides exactly the right level of abstraction for exploratory queries.

### 1.2 Graph → Communities → Summaries Pipeline

```
Phase 2: Community Detection
    Input:  graph.pkl (NetworkX DiGraph)
    Method: Louvain algorithm (python-louvain)
    Output: partition dict {node_id: community_id}

Phase 3: Community Summarization
    Input:  graph.pkl + partition
    Method: LLM summarization with rule-based fallback
    Output: communities.json
```

---

## 2. Phase 2: Community Detection — `detect_communities()`

The implementation lives in `community.py` (line 14):

```python
def detect_communities(graph: nx.DiGraph, resolution: float = 1.0) -> dict:
    entity_graph = nx.Graph()
    
    SKIP_LABELS = {"TextChunk", "Document"}
    for node, data in graph.nodes(data=True):
        if data.get("label") not in SKIP_LABELS:
            entity_graph.add_node(node, **data)
    
    for u, v, edata in graph.edges(data=True):
        if entity_graph.has_node(u) and entity_graph.has_node(v):
            entity_graph.add_edge(u, v, **edata)
    
    # Remove isolated nodes (they can't form meaningful communities)
    isolates = list(nx.isolates(entity_graph))
    entity_graph.remove_nodes_from(isolates)
    
    if entity_graph.number_of_nodes() == 0:
        return {}
    
    partition = community_louvain.best_partition(
        entity_graph, resolution=resolution, random_state=42
    )
    return partition
```

### 2.1 Step-by-Step Breakdown

#### Step 1: Label Filtering
The algorithm operates on **entity nodes only**. `TextChunk` and `Document` nodes are structural — they represent how information was stored, not what it means. Including them would create massive "super-communities" where every entity in a document clusters together purely because they share chunk edges.

```
Before filtering:
    Faculty_A ←[HAS_CHUNK]→ Chunk_1 ←[HAS_CHUNK]→ Faculty_B
    (A and B appear in the same chunk → would cluster together spuriously)

After filtering:
    Faculty_A ←[SUPERVISED_BY]→ Student_X ←[RESEARCHES]→ Area_Y
    (Meaningful semantic relationships drive clustering)
```

#### Step 2: DiGraph → Graph Conversion
Louvain requires an undirected graph. The directed knowledge graph is converted to undirected by treating every edge as bidirectional. This is semantically valid because relationship semantics are carried in edge `type` properties, not directionality — "Faculty A SUPERVISES Student B" and "Student B SUPERVISED_BY Faculty A" represent the same real-world relationship.

#### Step 3: Isolate Removal
Nodes with no edges (degree 0 in the undirected graph) are removed. These are typically entities extracted from text but never cross-linked to other entities — for example, a `ResearchArea` that no faculty member or student is connected to. Including them would create single-node communities that add noise without value.

#### Step 4: Louvain Partitioning
```python
partition = community_louvain.best_partition(
    entity_graph, resolution=1.0, random_state=42
)
```

**Why Louvain?**
- **Unsupervised**: No need to specify the number of communities (unlike K-means).
- **Scale-appropriate**: O(n log n) for the graph sizes we handle (100–500 entity nodes).
- **Modularity-based**: Maximizes intra-community edges and minimizes inter-community edges, naturally grouping research teams.

**Why `resolution=1.0`?**
The resolution parameter controls community granularity:
- `resolution < 1.0` → Fewer, larger communities (merges related groups)
- `resolution = 1.0` → Default modularity optimization
- `resolution > 1.0` → More, smaller communities (splits groups)

At 1.0, the algorithm typically produces 5–15 communities per department, each containing 5–30 entities — the right granularity for research group-level summaries.

**Why `random_state=42`?**
Louvain involves random node ordering during optimization. Fixing the seed ensures **reproducible** community assignments across runs, which is critical for:
- Debugging (same input → same communities)
- Testing (deterministic outputs for assertions)
- Idempotency (re-running ingestion doesn't shuffle communities)

### 2.2 Typical Community Structure

For the EE department, a typical partition might look like:

```
Community 0: [Dr. Saxena, Dr. Singh, PhD_Student_1, PhD_Student_2,
              "Power Electronics", "Motor Drives", "Renewable Energy",
              Lab: "Power Electronics Lab", Project: "SERB_123"]

Community 1: [Dr. Gupta, Dr. Dubey, PhD_Student_3,
              "Signal Processing", "Machine Learning", "IoT",
              Lab: "Signal Processing Lab"]

Community 2: [Dr. Sharma, PhD_Student_4, PhD_Student_5,
              "RF & Microwave", "Antenna Design",
              Patent: "Fractal Antenna", Startup: "AntennaTech"]
```

Each community captures a **research group** — the faculty, their students, their topics, their labs, and their outputs (projects, patents, startups).

### 2.3 What Happens to Filtered Nodes?

Isolated and chunk/document nodes are **not lost** — they still exist in `graph.pkl` and are still embedded in the FAISS index (via their text chunks). They simply don't participate in community-level retrieval. The retrieval layer can still find them through Tier 1 (entity) or Tier 3 (chunk) search.

---

## 3. Building Community Reports: `build_community_reports()`

After Louvain assigns each entity to a community, `build_community_reports()` (line 70) constructs structured text reports for each community:

```python
def build_community_reports(graph: nx.DiGraph, partition: dict) -> list:
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    reports = []
    for comm_id, members in sorted(communities.items()):
        report_text = _build_single_report(graph, members)
        reports.append({
            "id": f"community_{comm_id}",
            "community_id": comm_id,
            "members": members,
            "size": len(members),
            "text": report_text,
            "summary": None  # Populated in Phase 3
        })
    return reports
```

### 3.1 The `_build_single_report()` Function

This function (line 90) creates a structured text report from a community's members:

```python
def _build_single_report(graph: nx.DiGraph, members: list) -> str:
    parts = []
    members_by_type = defaultdict(list)
    
    for node in members:
        data = graph.nodes[node]
        label = data.get("label", "Entity")
        members_by_type[label].append((node, data))
    
    # Faculty section
    for node, data in members_by_type.get("Faculty", []):
        parts.append(f"Faculty: {data.get('name', node)}")
        if data.get("designation"):
            parts.append(f"  Designation: {data['designation']}")
        if data.get("research_interests"):
            parts.append(f"  Research Interests: {data['research_interests']}")
    
    # PhD Students section
    for node, data in members_by_type.get("PhDStudent", []):
        parts.append(f"PhD Student: {data.get('name', node)}")
        if data.get("research_area"):
            parts.append(f"  Research Area: {data['research_area']}")
        if data.get("supervisor"):
            parts.append(f"  Supervisor: {data['supervisor']}")
    
    # Research Areas, Labs, Projects, Patents, etc.
    for label in ["ResearchArea", "Lab", "Project", "Patent", "Startup", "Award"]:
        for node, data in members_by_type.get(label, []):
            parts.append(f"{label}: {data.get('name', data.get('title', node))}")
    
    return "\n".join(parts)
```

The report is a **structured, machine-readable text** — not prose. It lists entities by type with their key attributes. This serves as the input prompt for LLM summarization.

### 3.2 Example Raw Report

```
Faculty: Alok Kumar Saxena
  Designation: Associate Professor
  Research Interests: Power Electronics, Motor Drives, Renewable Energy
Faculty: Amitava Singh
  Designation: Assistant Professor
  Research Interests: Power Systems, Smart Grid
PhD Student: Ramesh Kumar
  Research Area: Inverter Topologies
  Supervisor: Alok Kumar Saxena
PhD Student: Priya Sharma
  Research Area: Solar Energy Systems
  Supervisor: Alok Kumar Saxena
ResearchArea: Power Electronics
ResearchArea: Motor Drives
ResearchArea: Renewable Energy
Lab: Power Electronics Lab
Project: SERB Project on Multi-Level Inverters
  PI: Alok Kumar Saxena
  Funding Agency: SERB
  Budget: Rs. 35 Lakhs
```

---

## 4. Phase 3: Community Summarization — `summarize_communities()`

The summarization function (line 130) transforms raw reports into natural-language summaries using an LLM:

```python
def summarize_communities(reports: list, llm_fn=None) -> list:
    for report in reports:
        if llm_fn and report["size"] >= 2:
            try:
                summary = _llm_summarize(report["text"], llm_fn)
                report["summary"] = summary
            except Exception as e:
                logger.warning(f"LLM summarization failed for {report['id']}: {e}")
                report["summary"] = _rule_based_summary(report)
        else:
            report["summary"] = _rule_based_summary(report)
    return reports
```

### 4.1 The LLM Summarization Prompt

```python
def _llm_summarize(report_text: str, llm_fn) -> str:
    prompt = f"""Summarize the following academic research group information into a concise 
paragraph (3-5 sentences). Focus on:
- The faculty members and their roles
- The research themes and areas
- The students working in the group
- Any notable projects, patents, or achievements

Group Information:
{report_text}

Write a natural, informative summary paragraph:"""
    
    return llm_fn(prompt)
```

The prompt is carefully designed to produce summaries that are:
- **Natural language** (not bullet points) — better for embedding and similarity search
- **Thematically focused** — mentions research themes, not administrative details
- **Appropriately sized** — 3–5 sentences, matching the typical context window a retrieval system would return

### 4.2 Example LLM Summary Output

```
"This research group focuses on power electronics and renewable energy systems, 
led by Associate Professor Dr. Alok Kumar Saxena and Assistant Professor Dr. 
Amitava Singh. The group's work spans inverter topologies, motor drives, smart 
grid technologies, and solar energy systems. Two PhD students, Ramesh Kumar 
(inverter topologies) and Priya Sharma (solar energy), are actively contributing 
to the research. The group has secured SERB funding for a project on multi-level 
inverters and operates from the Power Electronics Lab."
```

### 4.3 The Rule-Based Fallback

When LLM is unavailable (`llm_fn=None` or `--skip-summaries`), the system falls back to `_rule_based_summary()`:

```python
def _rule_based_summary(report: dict) -> str:
    members_by_type = _group_members_by_type(report)
    parts = []
    
    faculty = members_by_type.get("Faculty", [])
    students = members_by_type.get("PhDStudent", [])
    areas = members_by_type.get("ResearchArea", [])
    labs = members_by_type.get("Lab", [])
    
    if faculty:
        names = ", ".join(f[:5])  # Cap at 5 names
        parts.append(f"This group includes {len(faculty)} faculty member(s): {names}")
    if students:
        parts.append(f"{len(students)} PhD student(s)")
    if areas:
        area_names = ", ".join(areas[:3])
        parts.append(f"working in areas such as {area_names}")
    if labs:
        parts.append(f"with facilities including {', '.join(labs[:2])}")
    
    return ". ".join(parts) + "." if parts else "A cluster of related entities."
```

The rule-based summary is serviceable but lacks the narrative quality of LLM output. It's adequate for development and testing.

### 4.4 LLM Provider Selection and Error Handling

The `create_llm_from_env()` function in `graphrag/llm.py` selects the provider:

| Priority | Provider | Config | Best For |
|---|---|---|---|
| 1 | Gemini | `GEMINI_API_KEY` env var | Production (high quality) |
| 2 | Ollama | `OLLAMA_HOST`, `OLLAMA_MODEL` env vars | Bulk ingestion (no rate limits) |
| 3 | None (fallback) | No config | Development (rule-based) |

**Gemini rate limiting**: The Google Generative AI API enforces rate limits (429 errors). The system implements:
- Exponential backoff (1s → 2s → 4s → 8s, up to 5 retries)
- Automatic model switching (e.g., from `gemini-1.5-pro` to `gemini-1.5-flash` on repeated 429s)
- Graceful degradation (falls back to rule-based summary after max retries)

**Ollama advantages**: For bulk ingestion (`--all`), running a local Ollama instance with a 7B parameter model avoids all rate limits and network dependencies. The quality is slightly lower than Gemini Pro but sufficient for community summaries.

---

## 5. Saving Communities: `save_communities()`

```python
def save_communities(reports: list, partition: dict, data_dir: str):
    output = {
        "partition": partition,  # {node_id: community_id}
        "communities": []
    }
    for report in reports:
        output["communities"].append({
            "id": report["id"],
            "community_id": report["community_id"],
            "members": report["members"],
            "size": report["size"],
            "text": report["text"],       # Raw structured report
            "summary": report["summary"]  # LLM or rule-based summary
        })
    
    path = os.path.join(data_dir, "communities.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
```

### 5.1 Why Store Both `text` and `summary`?

The raw `text` field preserves the structured report for debugging and re-summarization. If you improve the LLM prompt or switch providers, you can re-run summarization on the existing reports without rebuilding the entire KG. The `summary` field is what gets embedded in Phase 4.

### 5.2 Why Store the Partition?

The partition mapping (`{node_id: community_id}`) is stored alongside the reports for two reasons:

1. **Runtime lookup**: The retrieval layer can map a matched entity back to its community to provide broader context.
2. **Debugging**: Developers can verify which community each entity belongs to without parsing the reports.

---

## 6. Edge Cases and Design Decisions

### 6.1 Single-Entity Communities

Louvain can produce communities with a single entity. These get rule-based summaries (the `report["size"] >= 2` check skips LLM for single-entity communities — it's not worth an API call for a trivial summary).

### 6.2 Empty Graphs

If Phase 1 produces a graph with no entity nodes (only Document and TextChunk nodes), `detect_communities()` returns an empty partition. Phase 3 produces zero reports. Phase 4 still embeds the text chunks — the system remains functional for chunk-based retrieval even without communities.

### 6.3 Very Large Communities

In dense departments, Louvain might produce a "mega-community" with 50+ members. The community report for such groups can be very long (2000+ words), which may exceed the LLM's effective summarization range. The current system handles this by trusting the LLM to extract the key themes, but a future improvement could split mega-communities into sub-communities using higher resolution values.

### 6.4 Cross-Type Community Coherence

The most valuable communities are **cross-type** — they contain Faculty + Students + ResearchAreas + Labs + Projects. These represent real research groups. Communities that are **mono-type** (all ResearchAreas, no Faculty) are less useful and typically result from sparse linking in the KG (missing SUPERVISED_BY or RESEARCHES_IN edges).

The quality of communities is directly proportional to the quality of entity resolution in Phase 1. If the resolver fails to link a PhD student to their supervisor, that student becomes an isolated node and is excluded from community detection entirely.

---

## 7. Performance Characteristics

| Operation | Typical Duration | Bottleneck |
|---|---|---|
| `detect_communities()` | 10–50ms | Negligible |
| `build_community_reports()` | 50–200ms | Graph traversal |
| `summarize_communities()` (LLM) | 2–10 minutes | API latency + rate limits |
| `summarize_communities()` (rule-based) | 5–20ms | Negligible |
| `save_communities()` | 10ms | Disk I/O |

Phase 3 with LLM is the **single biggest bottleneck** in the entire pipeline. For a department with 12 communities, each requiring an API call with ~500-token input and ~200-token output, the total time depends on:

- **Gemini**: ~5s per call × 12 communities = ~60s (with rate limit backoffs, can reach 5–10 minutes)
- **Ollama (local 7B)**: ~2s per call × 12 communities = ~24s

The `--skip-summaries` flag reduces this to near-zero, making it essential for development iteration.

---

## Next: Part 4 — Embedding Generation & FAISS Indexing

The next document covers the final transformation: how text chunks, entity descriptions, and community summaries are encoded into a unified FAISS vector index for semantic retrieval.
