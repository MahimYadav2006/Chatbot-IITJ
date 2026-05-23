# IIT Jammu EE Department GraphRAG Chatbot

A sophisticated Graph-based Retrieval-Augmented Generation (GraphRAG) chatbot for the IIT Jammu Electrical Engineering Department. This system intelligently extracts knowledge from departmental markdown documents, builds a structured knowledge graph, and provides accurate answers through hybrid retrieval and LLM-powered generation.

---

## 🎯 Project Overview

This chatbot system answers questions about:
- **Faculty**: Names, designations, emails, research areas, education
- **PhD Students**: Supervisors, research topics, department listings
- **Research**: Labs, funded projects, patents, startups, research areas
- **Placements**: Salary data, placement percentages, higher studies
- **Department**: Structure, programs, vision/mission

The system uses a **GraphRAG** architecture that combines:
1. **Knowledge Graph** - Structured entity relationships
2. **Vector Search** - Semantic similarity (FAISS)
3. **Community Detection** - Topic clustering (Louvain algorithm)
4. **LLM Generation** - Answer synthesis (Ollama/Llama3.1)

---

## 🏗️ Architecture Overview

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IIT Jammu EE ChatBot System                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐         ┌──────────────┐        ┌──────────────┐  │
│  │  Markdown   │         │   Web Flask  │        │  Evaluation  │  │
│  │  Documents  │         │  Application │        │   Framework  │  │
│  └──────┬──────┘         └──────┬───────┘        └──────┬───────┘  │
│         │                       │                       │            │
│         └───────────────────────┼───────────────────────┘            │
│                                 │                                     │
│         ┌───────────────────────▼───────────────────────┐            │
│         │         GraphRAG Pipeline & Engine             │            │
│         │  (Data Ingestion → Storage → Retrieval)       │            │
│         └───────────────────────────────────────────────┘            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Components

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Data Layer (data/)                               │
├────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │  graph.pkl       │  │  embeddings.     │  │  communities.    │     │
│  │  (NetworkX)      │  │  faiss + meta    │  │  json (reports)  │     │
│  │  - Entities      │  │  - Vectors       │  │  - Summaries     │     │
│  │  - Relations     │  │  - Metadata      │  │  - Partitions    │     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
│  ┌──────────────────┐  ┌──────────────────┐                            │
│  │  chunks.json     │  │  resolver.pkl    │                            │
│  │  (Text Chunks)   │  │  (Name Entity    │                            │
│  │  - Original text │  │   Resolution)    │                            │
│  │  - Metadata      │  │                  │                            │
│  └──────────────────┘  └──────────────────┘                            │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│               GraphRAG Modules (graphrag/)                              │
├────────────────────────────────────────────────────────────────────────┤
│  kg_builder.py      embeddings.py      community.py      llm.py        │
│  ├─ Parse markdown  ├─ SentenceTransf  ├─ Louvain detect ├─ Ollama    │
│  ├─ Extract entities│  ├─ FAISS index  │  ├─ Summarize   │  ├─ Chat   │
│  ├─ Build graph    │  └─ Vector search │  └─ Report build│  └─ Prompt│
│  └─ Save artifacts  └─ Entity desc.    └─ Save/Load     └─ Safety    │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Diagram

### Phase 1: Data Ingestion & Processing Pipeline

```
INGESTION PIPELINE (ingest.py)
─────────────────────────────────────────────────────────────────────

 ┌──────────────────┐
 │ Markdown Files   │
 │ (iitjammu_ee_*) │
 └────────┬─────────┘
          │
          ▼
 ┌──────────────────────────┐
 │ 1. KNOWLEDGE GRAPH BUILD │
 │   (kg_builder.py)        │
 └───────────┬──────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
 [Parse Docs]     [Extract Entities]
 - Faculty files  - Names (resolve)
 - PhD list      - Relationships
 - Projects      - Properties
 - Patents       - Text chunks
 - Startups        
 - Research areas  
 - Placement data  
             │
             ▼
    ┌────────────────────────┐
    │ NetworkX DiGraph       │
    │ - Nodes: Entities      │
    │ - Edges: Relations     │
    │ Example Relations:     │
    │   SUPERVISED_BY        │
    │   RESEARCHES_IN        │
    │   MEMBER_OF            │
    │   INVENTED             │
    │   FUNDED_BY            │
    └────────┬───────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ 2. COMMUNITY DETECTION   │
    │    (community.py)        │
    └───────────┬──────────────┘
                │
     ┌──────────┴──────────┐
     │                     │
     ▼                     ▼
  [Filter]            [Louvain]
  Entities only       Resolution: 1.0
  Remove chunks       Partition nodes
                │
                ▼
     ┌────────────────────────┐
     │ Community Reports      │
     │ - Members by type      │
     │ - Internal edges       │
     │ - Raw text repr.       │
     └────────┬───────────────┘
              │
              ▼
     ┌──────────────────────────┐
     │ 3. LLM SUMMARIZATION     │
     │    (community.py)        │
     └───────────┬──────────────┘
                 │
                 ▼
     ┌────────────────────────┐
     │ Community Summaries    │
     │ (Natural language)     │
     └────────┬───────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │ 4. EMBEDDING GENERATION  │
    │    (embeddings.py)       │
    └──────────┬───────────────┘
               │
    ┌──────────┼──────┬────────┐
    │          │      │        │
    ▼          ▼      ▼        ▼
  [Text]    [Entity] [Comm.]  [Desc.]
  Chunks    Desc.    Summary
            (graph)
    │        │       │        │
    └────────┼───────┴────────┘
             │
             ▼
    ┌─────────────────────────┐
    │ Batch Encode via        │
    │ Sentence-Transformers   │
    │ (all-mpnet-base-v2)     │
    │ 768-dim vectors         │
    └────────┬─────────────────┘
             │
             ▼
    ┌──────────────────────┐
    │ FAISS Index (IP)     │
    │ Inner Product search │
    └────────┬─────────────┘
             │
    ┌────────┴────────┬──────────┬──────────┐
    │                 │          │          │
    ▼                 ▼          ▼          ▼
 graph.pkl      embeddings.  communities.  chunks.json
               faiss + meta    json

[END OF INGESTION]
```

---

### Phase 2: Query Processing & Retrieval

```
QUERY EXECUTION (app.py + retriever.py)
─────────────────────────────────────────────────────────────────────

     ┌────────────────────┐
     │ User Query         │
     │ (via Web/API)      │
     └────────┬───────────┘
              │
              ▼
     ┌─────────────────────────────────┐
     │ Flask /api/chat Endpoint        │
     │ (app.py)                        │
     └────────┬────────────────────────┘
              │
              ▼
     ┌──────────────────────────────────┐
     │ retriever.get_direct_answer()    │
     │ (Check deterministic answers)    │
     └────────┬─────────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
[QnA Dataset]    [Pattern Matching]
Match exact      - Faculty roster?
High priority    - PhD roster?
                 - PhD supervisor?
                 - Research area?
    │                   │
    └─────────┬─────────┘
              │
     ┌────────▼─────────┐
     │ Direct Answer    │
     │ Found?           │
     └─────┬──────┬─────┘
           │ YES  │ NO
           │      │
           ▼      ▼
      [Return]  [Hybrid
       Answer]   Retrieval]
```

---

### Phase 2b: Hybrid Retrieval Process

```
HYBRID RETRIEVAL (retriever.retrieve())
─────────────────────────────────────────────────────────────────────

     ┌─────────────┐
     │ Query       │
     └────┬────────┘
          │
          ▼
     ┌────────────────────────┐
     │ Check Query Type       │
     └────┬───────┬──────┬────┘
          │       │      │
     ┌────▼─┐ ┌───▼──┐ ┌─▼─────┐
     │      │ │      │ │       │
     ▼      ▼ ▼      ▼ ▼       ▼
   [Faculty] [PhD] [Place] [Other]
    Roster  Roster   Query   Query
    Query   Query
     │       │       │       │
     └───────┴───────┴───────┘
             │
             ▼
     ┌──────────────────────────────────┐
     │ LOCAL SEARCH (Entity Matching)   │
     └────────┬─────────────────────────┘
              │
        ┌─────┴─────────┐
        │               │
        ▼               ▼
    [Phase 1]      [Phase 2]
    Name Match     Embedding
    (Direct)       (Fallback)
        │               │
        └───────┬───────┘
                │
                ▼
    ┌──────────────────────────┐
    │ Entity Results           │
    │ (Score: 0.8-1.0)         │
    └────────┬─────────────────┘
             │
             ▼
     ┌──────────────────────────────────┐
     │ VECTOR SEARCH (Chunk Matching)   │
     └────────┬─────────────────────────┘
              │
              ▼
     ┌────────────────────────────┐
     │ FAISS Inner Product Search │
     │ (top_k=5)                  │
     └────────┬────────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │ Chunk Results            │
    │ (Score: 0.5-0.9)         │
    └─────────┬────────────────┘
              │
              ▼
     ┌──────────────────────────────────┐
     │ GLOBAL SEARCH (Community Summary)│
     └────────┬─────────────────────────┘
              │
              ▼
     ┌────────────────────────────┐
     │ FAISS Community Search     │
     │ (top_k=2-3)                │
     └────────┬────────────────────┘
              │
              ▼
    ┌──────────────────────────┐
    │ Community Results        │
    │ (Score: 0.6-0.95)        │
    └─────────┬────────────────┘
              │
              ▼
     ┌───────────────────────────────────┐
     │ CONTEXT ASSEMBLY                  │
     │ (Sections by relevance)           │
     └────────┬──────────────────────────┘
              │
              ▼
     ┌──────────────────────────┐
     │ Final Context (~4500 words)
     │ Markdown-formatted       │
     └────────┬─────────────────┘
              │
         [Return to app.py]
```

---

### Phase 3: LLM-Based Answer Generation

```
GENERATION PHASE (llm.py + app.py)
─────────────────────────────────────────────────────────────────────

    ┌─────────────────────────┐
    │ Context + Query         │
    └────────┬────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ build_chat_prompt()      │
    │ (llm.py)                 │
    └────────┬─────────────────┘
             │
             ▼
    ┌────────────────────────────┐
    │ Prompt Structure:          │
    │ [System Prompt]            │
    │ [Context]                  │
    │ [Query]                    │
    └────────┬────────────────────┘
             │
             ▼
    ┌──────────────────────────┐
    │ OllamaLLM.generate()     │
    │ (model: llama3.1)        │
    │ (temp: 0.3)              │
    │ (max_tokens: 1400)       │
    └────────┬─────────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ Raw LLM Response       │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ sanitize_response()    │
    │ (Clean HTML/noise)     │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ Final Response         │
    │ (Markdown text)        │
    └────────┬───────────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │ JSON Response to User           │
    │ {                               │
    │   "response": "...",            │
    │   "retrieval_time": ms,         │
    │   "total_time": ms,             │
    │   "direct": bool                │
    │ }                               │
    └─────────────────────────────────┘
```

---

## 🔄 Detailed Processing Steps

### Step 1: Knowledge Graph Building (kg_builder.py)

**Input**: Markdown files from `iitjammu_ee_markdown/`

**Key Processes**:

1. **Content Cleaning**: Remove boilerplate patterns (navigation, breadcrumbs)
2. **Document Parsing**: Create document nodes with metadata (URL, title)
3. **Text Chunking**: Smart chunking respecting section boundaries (400-word chunks, 80-word overlap)
4. **Entity Extraction**: Type-specific parsing for faculty, PhD students, projects, patents, etc.
5. **Name Resolution**: Fuzzy matching handles name variants
6. **Relationship Creation**: Build edges between related entities

**Graph Statistics**:
- Nodes: ~500-1000+ (entities + chunks)
- Edges: ~1500-3000+ (relationships)
- Chunks: ~400-600 (text segments)

**Output**: `data/graph.pkl`, `data/chunks.json`, `data/resolver.pkl`

---

### Step 2: Community Detection (community.py)

**Process**:
1. Extract entity subgraph (filter out text chunks/documents)
2. Run Louvain algorithm (resolution=1.0)
3. Build community reports (members, relationships)
4. Generate natural language summaries via LLM

**Output**: `data/communities.json` (partition + reports with summaries)

---

### Step 3: Embedding Generation (embeddings.py)

**Input**: Text chunks, entity descriptions, community items

**Process**:
1. Create rich text descriptions for each entity
2. Batch encode using Sentence-Transformers (all-mpnet-base-v2)
3. Build FAISS index (Inner Product search)

**Output**: `data/embeddings.faiss`, `data/embeddings_meta.json`

**Statistics**:
- Total vectors: ~1200 (chunks: 500, entities: 400, communities: 300)
- Dimension: 768
- Index type: FAISS Flat IP

---

## 🎯 Query Processing Examples

### Example 1: "Who supervises Alice?"

```
1. Detect direct query pattern
2. Extract student name: "Alice"
3. Find PhDStudent node in graph
4. Get SUPERVISED_BY edges
5. Return: "Alice is supervised by Dr. Bob Smith."
   Time: ~50ms (no LLM)
```

### Example 2: "What are the placement statistics?"

```
1. Detect placement query
2. Local search: Entity matching → PlacementData nodes
3. Vector search: Find placement-related chunks
4. Inject structured placement data
5. LLM generates formatted response
   Time: ~800-1500ms (includes LLM)
```

### Example 3: "List all faculty"

```
1. Detect faculty roster query
2. Extract faculty roster from graph (24 faculty)
3. Build formatted roster context
4. Return direct answer (no LLM needed)
   Time: ~100ms
```

---

## 🔐 Safety & Security Features

- **System Prompt**: Security rules, privacy constraints, scope boundaries
- **Response Sanitization**: HTML tag removal, URL cleaning
- **Deterministic Queries**: Bypass LLM for exact counts (avoid hallucination)
- **Input Validation**: Empty query checks, context length limits

---

## 📈 Performance Characteristics

| Operation | Time |
|-----------|------|
| Knowledge Graph Build | 5-10s |
| Community Detection | 2-3s |
| Embedding Generation | 20-30s |
| Direct Query | 50-150ms |
| Hybrid Query | 800-1500ms |
| Vector Search | 50-100ms |
| LLM Generation | 500-1000ms |
| Total Ingestion | ~30-40s |

---

## 🚀 Running the System

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Ingestion

```bash
python ingest.py                  # Full pipeline
python ingest.py --skip-summaries # Skip LLM (faster)
```

### 3. Start Ollama

```bash
ollama serve
# In another terminal:
ollama pull llama3.1
```

### 4. Start ChatBot

```bash
python app.py
# Visit: http://localhost:5050
```

### 5. API Usage

```bash
curl -X POST http://localhost:5050/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Who supervises Alice?"}'
```

---

## 🧪 Testing & Evaluation

```bash
python evaluation/run_eval.py
```

Metrics: Exact match, partial match, BLEU score

---

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `ingest.py` | Main ingestion pipeline |
| `app.py` | Flask web application |
| `graphrag/kg_builder.py` | Knowledge graph construction |
| `graphrag/community.py` | Community detection & summarization |
| `graphrag/embeddings.py` | Vector embeddings & FAISS index |
| `graphrag/retriever.py` | Hybrid retrieval engine |
| `graphrag/llm.py` | LLM integration (Ollama) |
| `data/graph.pkl` | Serialized knowledge graph |
| `data/embeddings.faiss` | FAISS index |
| `data/communities.json` | Community reports |
| `data/chunks.json` | Text chunks |

---

## 🎓 Architecture Highlights

**Why GraphRAG?**
- Structured Knowledge: Captures relationships missed by traditional RAG
- Deterministic Answers: Entity graph enables exact queries without LLM hallucination
- Context Precision: Community detection improves relevance
- Scalability: FAISS enables fast similarity search

**Design Choices**:
- **Louvain Detection**: Fast, modular communities
- **Sentence-Transformers**: Domain-agnostic, pre-trained embeddings
- **FAISS Flat IP**: Simple, accurate, sufficient scale
- **Ollama/Llama3.1**: Open-source, local, good instruction following

---

## 📝 Summary

This GraphRAG system provides a complete pipeline for building intelligent QA systems over structured knowledge:

1. **Ingestion**: Markdown → Knowledge Graph → Communities → Embeddings
2. **Retrieval**: Entity search + Vector search + Community summaries  
3. **Generation**: LLM-powered synthesis with safety constraints
4. **Deterministic**: Direct answers for structural queries (no hallucination)

The architecture is modular, extensible, and suitable for academic/institutional knowledge bases.
