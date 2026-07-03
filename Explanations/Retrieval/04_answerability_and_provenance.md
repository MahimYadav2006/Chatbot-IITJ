# 04 — Answerability Gate, Provenance, and Post-Generation Verification

## Table of Contents

1. [The Hallucination Problem in Institutional RAG](#the-hallucination-problem-in-institutional-rag)
2. [Defense Layer 1: The Answerability Gate (Pre-Generation)](#defense-layer-1-the-answerability-gate-pre-generation)
3. [Defense Layer 2: Concept Support Validation](#defense-layer-2-concept-support-validation)
4. [Defense Layer 3: Provenance Tracking](#defense-layer-3-provenance-tracking)
5. [Defense Layer 4: Post-Generation Faithfulness Verification](#defense-layer-4-post-generation-faithfulness-verification)
6. [The Unavailable Response System](#the-unavailable-response-system)
7. [Broad Reasoning Query Exemption](#broad-reasoning-query-exemption)
8. [Why Not Just Trust the LLM?](#why-not-just-trust-the-llm)
9. [Complete Hallucination Defense Pipeline](#complete-hallucination-defense-pipeline)

---

## The Hallucination Problem in Institutional RAG

LLMs hallucinate. This is not a bug — it's an inherent property of autoregressive text generation. For a general-purpose chatbot, hallucination is acceptable (users can cross-check answers). For an **institutional chatbot**, hallucination is dangerous:

| Hallucination Type | Example | Consequence |
|---|---|---|
| **Fabricated entity** | LLM invents a faculty member "Dr. Rajesh" who doesn't exist | Student contacts non-existent person |
| **Attribute confusion** | LLM assigns Dr. A's email to Dr. B | Student sends confidential message to wrong person |
| **Count fabrication** | LLM says "15 PhD students" when there are 8 | Misleads prospective students |
| **Policy invention** | LLM invents a non-existent "summer internship credit" policy | Student makes incorrect academic decisions |
| **Stale data mixing** | LLM combines 2023 and 2025 placement data | Artificially inflated/deflated statistics |

The IIT Jammu chatbot implements a **4-layer defense system** to prevent these hallucinations:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Answerability Gate (pre-generation)                   │
│  → Checks if retrieved evidence supports an answer              │
│                                                                 │
│  LAYER 2: Concept Support Validation                            │
│  → Checks if specific query concepts exist in the knowledge     │
│    graph structurally                                           │
│                                                                 │
│  LAYER 3: Provenance Tracking                                   │
│  → Records which retrieval channels contributed evidence        │
│  → Enables downstream debugging and quality metrics             │
│                                                                 │
│  LAYER 4: Post-Generation Faithfulness Verification             │
│  → LLM-based verification of factual claims after generation    │
│  → Catches fabricated names, emails, numbers                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Defense Layer 1: The Answerability Gate (Pre-Generation)

**File:** `graphrag/retriever.py` → `_assess_answerability()`

This is the primary hallucination prevention mechanism. Before the LLM is even called, the system evaluates whether the retrieved evidence is sufficient to support a reliable answer.

### Logic Flow

```python
def _assess_answerability(self, query, local_results, vector_results, 
                          global_results, context) -> Dict:
    # Step 1: Identify concrete concepts the user is asking about
    concepts = self._infer_query_concepts(query)
    
    # Step 2: Exempt broad reasoning queries (they don't need specific evidence)
    if self._is_broad_reasoning_query(query):
        return {"answerable": True}
    
    # Step 3: Check if each concept is supported by the graph
    for concept in concepts:
        if not self._concept_supported_by_graph(concept):
            if not any(alias in context.lower() for alias in concept_aliases):
                missing_concepts.append(concept)
    
    # Step 4: If ANY concept is missing → unanswerable
    if missing_concepts:
        return {
            "answerable": False,
            "reason": f"The available data does not contain {concept_text} information",
            "missing_concepts": missing_concepts,
        }
    
    # Step 5: For factoid queries, check if ANY evidence was retrieved
    if not (local_results or vector_results or global_results):
        return {"answerable": False, "reason": "No relevant evidence found"}
    
    # Step 6: Evidence exists → answerable
    return {"answerable": True, "matched_terms": matched_terms[:10]}
```

### What Makes a Query "Unanswerable"?

A query is deemed unanswerable when:

1. **Concept not in graph:** The user asks about startups, but the department's knowledge graph has zero `Startup` nodes.
2. **Concept not in context:** Even after vector search, the retrieved text doesn't mention the concept.
3. **No evidence retrieved:** All three channels returned empty results.

### What Does NOT Make a Query Unanswerable

- **Low similarity scores:** If vector search finds matches above the 0.35 threshold, the query is answerable even if scores are mediocre. The LLM may still produce a useful answer.
- **Missing specific terms:** If the user asks about "VLSI" but the context mentions "Very Large Scale Integration" instead, this is fine — vector search handles synonyms.
- **Broad queries:** "Tell me about the department" is always answerable because it's a reasoning query, not a factoid.

---

## Defense Layer 2: Concept Support Validation

**File:** `graphrag/retriever.py` → `_infer_query_concepts()` + `_concept_supported_by_graph()`

This layer performs **structural validation** — checking whether the knowledge graph actually contains the type of data the user is asking about.

### Concept Inference

```python
def _infer_query_concepts(self, query: str) -> List[str]:
    concept_aliases = {
        "startup":        ("startup", "startups", "incubated", "venture", "company"),
        "patent":         ("patent", "patents", "invented", "invention"),
        "project":        ("project", "projects", "funded project", "funding", "grant"),
        "lab":            ("lab", "labs", "laboratory", "facilities"),
        "placement":      ("placement", "salary", "package", "higher studies"),
        "contact":        ("contact", "point of contact", "official contact"),
        "publication":    ("publication", "papers", "journal", "conference"),
        "award":          ("award", "awards", "honor", "recogni"),
        "course":         ("course", "programme", "curriculum", "btech", "mtech"),
        "staff":          ("staff", "laboratory assistant", "technician"),
        "alumni":         ("alumni", "former students", "passed out"),
        "graduated_phd":  ("phd alumni", "graduated phd", "phd graduates"),
        "address":        ("address", "location", "postal address"),
    }
```

### Graph-Level Concept Validation

```python
def _concept_supported_by_graph(self, concept: str) -> bool:
    # Each concept maps to specific graph node labels
    label_map = {
        "startup":       ("Startup",),
        "patent":        ("Patent",),
        "project":       ("Project",),
        "lab":           ("Lab",),
        "placement":     ("PlacementData", "HigherStudiesData"),
        "alumni":        ("Alumni", "AlumniBatch"),
        "graduated_phd": ("GraduatedPhD",),
        "address":       ("ContactInfo",),
    }
    
    # Special cases
    if concept == "contact":
        return self._get_hod_member() is not None or label_count("ContactInfo") > 0
    
    if concept == "publication":
        return label_count("Publication") > 0 or any_faculty_has("publications")
    
    # Generic case: check if any nodes of the required label exist
    return any(self._label_counts.get(label, 0) > 0 for label in labels)
```

### Why Structural Validation Matters

Consider this scenario:
- User asks: "What startups have been incubated in the Civil Engineering department?"
- The Civil Engineering knowledge graph has zero `Startup` nodes.
- Without concept validation, vector search might find a text chunk mentioning "startup" in a general context, and the LLM would fabricate startup names.
- With concept validation, the system detects that `Startup` nodes don't exist → returns "I don't have startup information for this department."

**This is not the same as answering "No startups."** The system explicitly says it *doesn't have the information*, not that none exist. This is a crucial epistemic distinction.

---

## Defense Layer 3: Provenance Tracking

**File:** `graphrag/retriever.py` → `_build_provenance()`

Every retrieval result includes provenance metadata documenting exactly which retrieval channels contributed evidence:

```python
def _build_provenance(self, direct, local_results, vector_results, 
                      global_results, section_word_counts) -> Dict:
    return {
        "route": "graph+vector",     # Which channels were used
        "source_mode": "both",       # Combined source mode
        "graph": {
            "direct": False,         # Was deterministic context used?
            "items": 3,              # How many graph entities found
            "avg_score": 0.85,       # Average match score
            "labels": {"Faculty": 1, "PhDStudent": 2},  # Entity type breakdown
            "word_count": 150,       # Words from graph sources
        },
        "vector": {
            "items": 2,              # How many text chunks found
            "avg_score": 0.65,       # Average semantic similarity
            "sources": ["ee_faculty_page.md"],  # Source documents
            "word_count": 300,       # Words from vector sources
        },
        "community": {
            "items": 1,              # How many community summaries
            "avg_score": 0.55,       # Average score
            "word_count": 80,        # Words from community summaries
        },
    }
```

### Source Mode Classification

| Route | Meaning |
|-------|---------|
| `direct_graph` | Answer came entirely from deterministic graph data |
| `graph` | Answer came from graph entity search only |
| `vector` | Answer came from FAISS vector search only |
| `graph+vector` | Both graph entities and vector chunks contributed |
| `community` | Only community summaries matched |
| `none` | No evidence found by any channel |

### How Provenance Is Used

1. **Relevance scoring in broadcast mode:** `MultiDepartmentRetriever._is_bundle_relevant()` uses provenance to filter out low-quality results during cross-department search.
2. **Debugging:** When the chatbot gives a wrong answer, provenance reveals whether the error was in graph data, vector retrieval, or LLM generation.
3. **Quality metrics:** Dashboard can track how often each channel is the primary evidence source, identifying gaps in the knowledge graph.

---

## Defense Layer 4: Post-Generation Faithfulness Verification

**File:** `graphrag/verifier.py` → `ResponseVerifier`

This is the final defense layer. After the LLM generates a response, the verifier checks whether the factual claims in the response are grounded in the retrieved context.

### Architecture

```python
class ResponseVerifier:
    def verify(self, query: str, context: str, response: str) -> VerificationResult:
        # Gate 1: Skip if verification is disabled via env var
        if not is_verification_enabled():
            return skip(reason="Disabled via env var")
        
        # Gate 2: Skip for non-factoid queries
        if not self._is_factoid_query(query):
            return skip(reason="Non-factoid query")
        
        # Gate 3: Skip for very short responses (likely refusals)
        if len(response.split()) < 10:
            return skip(reason="Response too short — likely a refusal")
        
        # Actual verification via LLM
        return self._run_verification(query, context, response)
```

### Factoid Query Detection

Only factoid queries (questions with specific, verifiable answers) are verified:

```python
def _is_factoid_query(self, query: str) -> bool:
    factoid_starters = (
        "who ", "what ", "which ", "where ", "when ",
        "name ", "list ", "give me ", "show me ",
        "tell me about ", "how many ", "how much ",
        "email ", "contact ",
    )
    return q.startswith(factoid_starters) or any(
        kw in q for kw in ("how many", "how much", "email of", "contact of")
    )
```

**Why not verify all queries?** Broad reasoning queries ("Summarize the department's research focus") don't have individually verifiable claims. Running verification on them would produce mostly false negatives (the LLM synthesizes, which looks like fabrication to the verifier).

### The Verification Prompt

The verifier uses a second LLM call with a strict, structured prompt:

```
You are a strict factual accuracy checker. Your job is to verify if a response 
is supported by the given context.

CONTEXT (source of truth):
{context}

RESPONSE TO VERIFY:
{response}

INSTRUCTIONS:
1. Extract each factual claim from the RESPONSE.
2. For each claim, check if it is EXPLICITLY supported by the CONTEXT.
3. A claim is SUPPORTED only if the exact fact appears in the CONTEXT.
4. A claim is NOT SUPPORTED if it is invented, inferred, or not present.
5. General statements like greetings are always SUPPORTED.
6. If the response attributes information to a specific department and 
   that department's section appears in the CONTEXT, the attribution is SUPPORTED.
7. If the CONTEXT contains multiple department sections, claims from 
   ANY section count as SUPPORTED.

Output ONLY valid JSON:
{"claims": [{"claim": "...", "supported": true}, ...], "overall_faithful": true/false}
```

### Decision Thresholds

The faithfulness decision uses different thresholds based on context type:

```python
# Single-department: >50% claims must be supported
# Multi-department: >30% claims must be supported (relaxed because context
#                   truncation causes false negatives for later departments)
threshold = 0.3 if is_multi_dept else 0.5
is_faithful = faithfulness_ratio > threshold
```

**Why the multi-department relaxation?** When the verifier truncates context to 3000 characters for a single department (or 10000 for multi-department), departments that appear later in the merged context may be cut off. Their claims would appear unsupported simply because the evidence was truncated, not because it doesn't exist.

### Verification Result Outcomes

| Outcome | Action |
|---------|--------|
| `faithful=True, claims all supported` | Response passes through unchanged |
| `faithful=True, some unsupported` | Response passes; unsupported claims logged as warnings |
| `faithful=False` | Response replaced with "I don't know" fallback |
| `skipped=True` | Verification skipped (non-factoid, disabled, or error) |

---

## The Unavailable Response System

When the answerability gate determines a query cannot be reliably answered, the system generates a structured "unavailable" response:

```python
def _build_unavailable_response(self, query, reason=None) -> str:
    # Special case: if it's a contact query, try to provide HoD info anyway
    if self._is_department_contact_query(query):
        contact_answer = self._build_department_contact_answer()
        if contact_answer:
            return contact_answer
    
    base = (
        f"I don't have that specific information for the "
        f"{self.dept_config['full_name']} at IIT Jammu."
    )
    if reason:
        base += f" {reason}"
    base += (
        f" You can check the IIT Jammu {self.dept_config['name']} website at "
        f"{self.dept_config['base_url']} for more details."
    )
    base += " If you're looking for information from a specific department, "
    base += "try mentioning the department name in your query."
    return base
```

### Design Decisions in Unavailable Responses

1. **Department-scoped:** "I don't have that for the Department of Electrical Engineering" — not "I don't know." This tells the user their query was understood but the specific data isn't available.

2. **Website redirect:** Always includes the official department website URL. This gives the user an actionable next step.

3. **Guidance:** Suggests mentioning the department name, helping users who may have been routed to the wrong department.

4. **Special case for contacts:** Even when the main query is unanswerable, if the user is asking for contact information, the system provides the HoD as a generic contact point. This ensures users always have *someone* to reach out to.

---

## Broad Reasoning Query Exemption

Not all queries need strict evidence gating. Broad reasoning queries are exempted:

```python
def _is_broad_reasoning_query(self, query: str) -> bool:
    q = re.sub(r"\s+", " ", query.lower()).strip()
    return any(term in q for term in (
        "summarize", "summarise", "overview", "analyze", "analyse", "insight",
        "trend", "compare", "comparison", "how does", "how do", "why",
        "based on", "synthesis", "relationship between"
    ))
```

**Why exempt these?**

Reasoning queries don't have single, verifiable answers. They require the LLM to **synthesize** information from multiple sources, draw connections, and provide interpretive analysis. Strict concept support validation would incorrectly block these queries.

Example:
- Query: "Analyze the research trends in the EE department"
- Concepts inferred: None specific (it's a broad analysis)
- Evidence: Community summaries + entity search results
- LLM behavior: Synthesizes a narrative about research focus areas

If the answerability gate blocked this because no single concept was structurally validated, the user would get "I don't have that information" — which is clearly wrong when the system has extensive department data.

---

## Why Not Just Trust the LLM?

A common objection: "Why build all this complexity? Can't the LLM just look at the context and decide for itself?"

### Problem 1: LLMs Don't Know What They Don't Know

When an LLM receives context that doesn't contain the answer, it doesn't say "I don't know." It **fabricates an answer** that sounds authoritative. This is the fundamental hallucination problem.

Example without answerability gate:
```
Context: [text about EE faculty research areas]
Query: "What startups have been incubated in EE?"
LLM response: "The EE department has incubated startups in areas of 
               IoT and signal processing."  ← FABRICATED
```

With answerability gate:
```
Concept validation: "startup" → 0 Startup nodes in graph → BLOCKED
Response: "I don't have startup information for the Department of 
           Electrical Engineering at IIT Jammu."
```

### Problem 2: LLMs Merge Entity Attributes

When multiple similar entities are in context, LLMs frequently **merge their attributes**:

```
Context: "Dr. Sharma (Assistant Prof, sharma@iitj) works on VLSI.
          Dr. Shukla (Associate Prof, shukla@iitj) works on control."
Query: "What is Dr. Sharma's designation?"
LLM response: "Dr. Sharma is an Associate Professor."  ← WRONG (confused with Shukla)
```

The post-generation verifier catches this: "Associate Professor" for Sharma is NOT supported by the context → claim flagged.

### Problem 3: LLMs Invent Counts

```
Context: [text chunk mentioning 3 faculty by name]
Query: "How many faculty are in EE?"
LLM response: "The EE department has 3 faculty members."  ← WRONG (real count is 14)
```

The deterministic layer intercepts this: faculty roster scan → count = 14 → deterministic answer injected. LLM never gets the chance to miscount.

---

## Complete Hallucination Defense Pipeline

Here's the complete end-to-end flow showing all defense layers:

```
USER QUERY: "List all startups from the EE department"
                │
                ▼
┌─────────────────────────────────────────────────┐
│  LAYER 2: Concept Support Validation            │
│  _infer_query_concepts → ["startup"]            │
│  _concept_supported_by_graph("startup")         │
│    → Startup label count = 0                    │
│    → NOT SUPPORTED                              │
│                                                 │
│  RESULT: answerable = False                     │
│  REASON: "startup information not available"    │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  UNAVAILABLE RESPONSE SYSTEM                    │
│                                                 │
│  "I don't have startup information for the      │
│   Department of Electrical Engineering at IIT   │
│   Jammu. You can check iitjammu.ac.in/ee for    │
│   more details."                                │
│                                                 │
│  ← LLM IS NEVER CALLED                         │
└─────────────────────────────────────────────────┘
```

Versus a query that passes all gates:

```
USER QUERY: "Who is Dr. Anand Mishra's supervisor?"
                │
                ▼
┌─────────────────────────────────────────────────┐
│  LAYER 1: Deterministic Graph Retrieval         │
│  _extract_supervisor_query_name → None          │
│  (Pattern doesn't match — "Anand Mishra" is a   │
│   faculty, not a student)                       │
│  → Falls through to hybrid RAG                  │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  HYBRID RAG RETRIEVAL                           │
│  Entity search: finds "Anand Mishra" Faculty    │
│  Vector search: finds profile page text chunks  │
│  Community search: finds research group cluster │
│                                                 │
│  LAYER 2: Concept Support Validation            │
│  _infer_query_concepts → [] (no special concept)│
│  → No concept check needed                     │
│                                                 │
│  LAYER 1: Answerability Gate                    │
│  local_results: 1, vector_results: 2            │
│  → Evidence exists → answerable = True          │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  LLM GENERATION                                 │
│  Context + Query → LLM → Response               │
│                                                 │
│  LAYER 4: Post-Generation Verification          │
│  _is_factoid_query("who is...") → True          │
│  Claims: [{claim: "Anand Mishra is a faculty",  │
│            supported: true}]                    │
│  Faithfulness ratio: 1.0 > 0.5 → PASS          │
└──────────────┬──────────────────────────────────┘
               │
               ▼
         RESPONSE TO USER
```

---

## Summary

The 4-layer hallucination defense system ensures that:

| Layer | Catches | Implementation | Latency Impact |
|-------|---------|----------------|----------------|
| **L1: Answerability Gate** | Queries with no retrieved evidence | Token matching + result count | <1ms |
| **L2: Concept Support** | Queries about data types not in the graph | Label count checks | <1ms |
| **L3: Provenance** | Debugging + downstream quality metrics | Metadata assembly | <1ms |
| **L4: Post-Gen Verify** | Fabricated claims in LLM output | Second LLM call | 1-3s |

The system is designed to be **conservatively defensive**: it would rather say "I don't know" than risk a fabricated answer. This is the correct posture for an institutional chatbot where factual accuracy is paramount.
