# 05 — Multi-Department Orchestration and Query Routing

## Table of Contents

1. [The Routing Problem](#the-routing-problem)
2. [Department Router: Alias-Based Routing](#department-router-alias-based-routing)
3. [Academic Rules Interception](#academic-rules-interception)
4. [Cross-Routing Injection Logic](#cross-routing-injection-logic)
5. [MultiDepartmentRetriever: The Orchestrator](#multidepartmentretriever-the-orchestrator)
6. [Single-Department Retrieval](#single-department-retrieval)
7. [Multi-Department Retrieval (Merge Mode)](#multi-department-retrieval-merge-mode)
8. [Broadcast Search (Discovery Mode)](#broadcast-search-discovery-mode)
9. [Relevance Filtering in Broadcast Mode](#relevance-filtering-in-broadcast-mode)
10. [Topic Query Detection for Cross-Department Search](#topic-query-detection-for-cross-department-search)
11. [Context Merging and Provenance Aggregation](#context-merging-and-provenance-aggregation)
12. [Why Not a Single Unified Index?](#why-not-a-single-unified-index)

---

## The Routing Problem

When a user asks "Who works on deep learning?", which department's retriever should handle the query? Multiple departments (CSE, EE, Mathematics) might have faculty working on deep learning. The system must:

1. **Detect department intent:** Is the user asking about a specific department?
2. **Route precisely:** If yes, send to that department's retriever.
3. **Broadcast intelligently:** If no, search ALL departments and filter by relevance.
4. **Merge results:** Combine cross-department results with clear department attribution.

This is the responsibility of two components working together:
- `DepartmentRouter` (in `dept_router.py`): Determines WHERE to route.
- `MultiDepartmentRetriever` (in `graphrag/multi_retriever.py`): Executes the routing decision.

---

## Department Router: Alias-Based Routing

**File:** `dept_router.py` → `DepartmentRouter`

### How Alias Matching Works

The router maintains two lookup tables:
1. `DEPT_NAME_ALIASES`: Maps department codes to name aliases (12 departments, 150+ aliases)
2. `SECTION_NAME_ALIASES`: Maps section codes to name aliases (30+ sections, 400+ aliases)

```python
DEPT_NAME_ALIASES = {
    "ee": [
        "electrical engineering", "electrical", "ee", "ee department",
        "dept of ee", "department of electrical",
    ],
    "computer_science_engineering": [
        "computer science and engineering", "computer science & engineering",
        "computer science", "comp sci", "cse", "cs", ...
    ],
    ...
}
```

### Greedy Matching Algorithm

The router uses **greedy alias matching** — aliases are sorted by length (longest first), and the first match wins:

```python
class DepartmentRouter:
    def __init__(self):
        # Sort aliases by length descending
        self._alias_map.sort(key=lambda x: len(x[0]), reverse=True)
    
    def _detect_departments(self, query: str) -> List[str]:
        # Clean query
        q_clean = re.sub(r"[?!.,;:'\"-]", " ", query.lower())
        
        for alias, dept_code in self._alias_map:
            if dept_code in seen:
                continue
            
            # Word-boundary matching to prevent partial matches
            pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
            match = re.search(pattern, q_clean)
            
            if match:
                # Check for span overlap (prevent double-matching)
                if not overlaps_with_previous(match.start(), match.end()):
                    detected.append(dept_code)
```

**Why greedy + longest-first?** Consider the alias "computer science and engineering" (30 chars) vs. "cs" (2 chars). Without longest-first ordering, a query mentioning "computer science and engineering" might match the "cs" alias first, losing the more precise department resolution. Longest-first ensures the most specific alias always wins.

**Why word-boundary matching?** Without it, "CSE" would match inside "traversed" (contains "se"), and "ee" would match inside "employee". The `(?<![a-z])` and `(?![a-z])` lookbehinds/lookaheads prevent this.

**Why span overlap checking?** In "Department of Computer Science and Engineering at EE", both CSE and EE should be detected. But in "electrical engineering" (which aliases to EE), we should only detect EE once, not also match "ee" within "electrical engineering". Span tracking prevents this double-counting.

### The RouteResult

```python
@dataclass
class RouteResult:
    departments: List[str]     # ["ee", "computer_science_engineering"]
    confidence: str            # "exact" or "broadcast"
    reason: str                # "Matched departments: EE, CSE"
    query: str                 # Original query
    sections: List[str]        # ["academics", "cds"]
```

| Confidence | Meaning |
|-----------|---------|
| `"exact"` | At least one department or section was explicitly matched |
| `"broadcast"` | No specific department/section detected → search everything |

---

## Academic Rules Interception

Before any department/section alias matching, the router checks if the query is about **academic rules and regulations**. This is handled by `is_academic_rules_query()` in `intent_utils.py`:

```python
def route(self, query: str) -> RouteResult:
    from graphrag.intent_utils import is_academic_rules_query
    if is_academic_rules_query(query):
        return RouteResult(
            departments=[],
            sections=["academics"],
            confidence="exact",
            reason="Query identified as academic rules and regulations request",
        )
```

### The Academic Rules Classifier

**File:** `graphrag/intent_utils.py` → `is_academic_rules_query()`

This is a 197-line function with extremely precise classification logic. It uses a **negative-guard-first** architecture:

```
┌─────────────────────────────────────────────────────┐
│  STEP 1: NEGATIVE GUARDS                            │
│  Check if query belongs to another section:          │
│  - CDS indicators (placement, recruiter, salary)     │
│  - Counselling indicators (stress, anxiety, therapy) │
│  - Medical indicators (doctor, hospital, ambulance)  │
│  - Alumni indicators (gold medal, convocation)       │
│  - Sports/hostel/fest indicators                     │
│  - Faculty research indicators (publication, patent) │
│  - Name honorific guard (Dr./Prof. + name → identity)│
│                                                      │
│  If ANY negative guard fires → return False           │
└──────────────┬──────────────────────────────────────┘
               │ (no guard fired)
               ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2: POSITIVE INDICATORS                         │
│  Check for academic rule keywords:                   │
│  - Notification keywords (DPGC, DUGC, fee structure) │
│  - JRF/SRF/fellowship keywords                      │
│  - Malpractice keywords (proxy, cheating, plagiarism)│
│  - Grade keywords (CGPA, SGPA, conversion, division) │
│  - Milestone keywords (comprehensive exam, thesis)   │
│  - Academic policy phrases (course withdrawal, BTP)  │
│  - Course code regex pattern                         │
│                                                      │
│  If ANY positive indicator fires → return True        │
└──────────────┬──────────────────────────────────────┘
               │ (no positive indicator)
               ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3: CONTEXTUAL COMBINATION                      │
│  Check if a rules keyword appears WITH an academic   │
│  context term:                                       │
│  - rules_keywords ∩ academic_context_terms ≠ ∅       │
│  → return True                                       │
│  Otherwise → return False                            │
└─────────────────────────────────────────────────────┘
```

### Why Negative Guards First?

Consider the query "What is the placement policy?":
- Contains "placement" → CDS section indicator → **should NOT route to academics**
- Also contains "policy" → academic rules keyword
- Without the negative guard, this would incorrectly route to academics

The CDS guard catches it first: `"placement" in q → return False`.

But consider "What is the gold medal criteria?":
- Contains "gold medal" → alumni indicator → would block
- BUT also contains "criteria" → asking about RULES for getting medals
- Special exception: `medal_rules_query = gold medal AND (criteria/eligibility/rule)`
- If this is true → **return True** (this IS an academic rules query about medal criteria)

### Special Case: Course Code Detection

Academic regulations PDFs contain course codes like `MAL055P4I`. Users might type these with spaces or slight variations. The regex catches them:

```python
if re.search(r"\b(?:[a-z]\s*)?[a-z]{2,3}\s*\d{3}\s*[up]\s*\d\s*[meix]\b", q):
    return True  # This is a course code → route to academics
```

---

## Cross-Routing Injection Logic

After initial department/section detection, the router performs **cross-routing injection** — adding related sections that should also be queried:

### Rule 1: Departments Always Include Academics
```python
if detected_depts:
    if "academics" not in detected_secs:
        detected_secs.append("academics")
```
**Why?** Department queries often overlap with academic information (courses, programs, faculty rosters).

### Rule 2: Student Sections Include Academics
```python
_student_academic_overlap = {
    "students-ug-admissions", "students-pg-admissions",
    "students-phd-admissions", "students-schedule", "students-faq",
}
if any(s in _student_academic_overlap for s in detected_secs):
    if "academics" not in detected_secs:
        detected_secs.append("academics")
```
**Why?** UG admissions and academic scheduling overlap heavily with the academics section's data.

### Rule 3: MoU Queries → Both Media and IR
```python
if any(term in q for term in ("mou", "memorandum of understanding", "partnerships")):
    if "ir" not in detected_secs: detected_secs.append("ir")
    if "media" not in detected_secs: detected_secs.append("media")
```
**Why?** MOUs appear on both the International Relations page and the Media page. Both should be searched.

### Rule 4: Holiday Queries → Both Media and Schedule
```python
if any(term in q for term in ("holiday", "holidays", "holiday list")):
    if "students-schedule" not in detected_secs: detected_secs.append("students-schedule")
    if "media" not in detected_secs: detected_secs.append("media")
```
**Why?** Holiday lists are published on the media page but are also part of the academic schedule.

### Rule 5: Complaints → Committees + Student Welfare
```python
if any(term in q for term in ("complaint", "grievance", "harassment", "discrimination")):
    for sec in ["quick-committees", "sw"]:
        if sec not in detected_secs: detected_secs.append(sec)
```
**Why?** Complaints can be handled by the ICC (Internal Complaints Committee) or the Student Welfare section.

### Rule 6: Admission Queries → Program-Specific Sections
```python
if "admission" in q:
    if "btech" in q: inject "students-ug-admissions"
    if "mtech" in q: inject "students-pg-admissions"
    if "phd" in q:   inject "students-phd-admissions"
```
**Why?** Each program level has its own dedicated admissions section with distinct data.

### Rule 7: Staff Queries → Quick-Staff + Department
```python
if "staff" in q:
    inject "quick-staff"
```
**Why?** Non-teaching staff are listed in a centralized staff directory, not on department pages.

### Rule 8: Contact Queries → Quick-Contacts
```python
if "phone number" in q or "voip" in q:
    inject "quick-contacts"
```
**Why?** VoIP/phone directories are maintained separately from department data.

---

## MultiDepartmentRetriever: The Orchestrator

**File:** `graphrag/multi_retriever.py` → `MultiDepartmentRetriever`

This class takes a `RouteResult` and executes the appropriate retrieval strategy:

```python
class MultiDepartmentRetriever:
    def __init__(self):
        self.dept_retrievers: Dict[str, HybridRetriever] = {}
        self.section_retrievers: Dict[str, SectionRetriever] = {}
    
    def retrieve(self, route: RouteResult, query: str) -> Dict:
        if len(route.departments) == 1 and not route.sections:
            return self.retrieve_single(route.departments[0], query)
        elif route.departments or route.sections:
            return self.retrieve_multi(route, query)
        else:
            return self.retrieve_broadcast(query)
```

### Three Execution Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Single** | Exactly 1 department, 0 sections | Direct delegation to that department's retriever |
| **Multi** | 1+ departments AND/OR 1+ sections | Query each, merge with department/section headers |
| **Broadcast** | No department, no section detected | Query ALL loaded retrievers, filter by relevance |

---

## Single-Department Retrieval

The simplest path. The query is delegated directly to one `HybridRetriever`:

```python
def retrieve_single(self, dept_code: str, query: str) -> Dict:
    retriever = self.dept_retrievers.get(dept_code)
    if not retriever:
        return {"context": "Department data not available.", ...}
    
    bundle = retriever.retrieve_bundle(query)
    return bundle
```

---

## Multi-Department Retrieval (Merge Mode)

When multiple departments and/or sections are detected, each is queried independently and results are merged:

```python
def retrieve_multi(self, route: RouteResult, query: str) -> Dict:
    bundles = []
    
    # Query each detected department
    for dept_code in route.departments:
        retriever = self.dept_retrievers.get(dept_code)
        if retriever:
            bundle = retriever.retrieve_bundle(query)
            bundles.append((dept_code, "department", bundle))
    
    # Query each detected section
    for section_code in route.sections:
        retriever = self.section_retrievers.get(section_code)
        if retriever:
            bundle = retriever.retrieve_bundle(query)
            bundles.append((section_code, "section", bundle))
    
    # Merge contexts with department/section headers
    merged_context = self._merge_bundles(bundles)
    return {"context": merged_context, ...}
```

### Context Merging Format

```
## Department of Electrical Engineering
[EE retrieval context]

---

## Department of Computer Science and Engineering
[CSE retrieval context]

---

## Academics Section
[Academics retrieval context]
```

The `---` separators and `## Department of...` headers are critical — they allow the LLM to attribute information to the correct source.

---

## Broadcast Search (Discovery Mode)

When no specific department or section is detected, the system performs a **broadcast search** across all loaded retrievers:

```python
def retrieve_broadcast(self, query: str) -> Dict:
    all_bundles = []
    
    # Phase 1: Topic query detection
    is_topic = self._is_topic_query(query)
    
    # Phase 2: Query ALL department retrievers
    for dept_code, retriever in self.dept_retrievers.items():
        bundle = retriever.retrieve_bundle(
            query,
            # In broadcast mode, suppress topic matching per-department
            # to prevent one department from short-circuiting the search
            suppress_topic_match=True if is_topic else False,
        )
        all_bundles.append((dept_code, "department", bundle))
    
    # Phase 3: Query ALL section retrievers
    for section_code, retriever in self.section_retrievers.items():
        bundle = retriever.retrieve_bundle(query)
        all_bundles.append((section_code, "section", bundle))
    
    # Phase 4: Filter by relevance
    relevant_bundles = [
        (code, kind, bundle) for code, kind, bundle in all_bundles
        if self._is_bundle_relevant(bundle, query)
    ]
    
    # Phase 5: Sort by relevance score
    relevant_bundles.sort(key=lambda x: self._score_bundle(x[2]), reverse=True)
    
    # Phase 6: Take top-N and merge
    top_bundles = relevant_bundles[:5]  # Cap at 5 to stay within LLM context
    merged_context = self._merge_bundles(top_bundles)
    
    return {"context": merged_context, ...}
```

### The `suppress_topic_match` Flag

This is a subtle but critical optimization. In broadcast mode, the system queries ALL departments for a topic like "deep learning." Without `suppress_topic_match=True`:

1. The EE retriever finds 3 faculty working on deep learning.
2. It returns a deterministic answer listing those 3 faculty.
3. The CSE retriever also finds 5 faculty working on deep learning.
4. The merged result shows EE's deterministic answer prominently, but CSE's list might be truncated.

With `suppress_topic_match=True`:
1. Both EE and CSE retrievers skip the deterministic topic handler.
2. Both go through full hybrid RAG retrieval.
3. Both return comparable context blocks.
4. Relevance scoring ranks them fairly.

---

## Relevance Filtering in Broadcast Mode

Not every department will have relevant results for every query. The system filters out noise:

```python
def _is_bundle_relevant(self, bundle: Dict, query: str) -> bool:
    # Check 1: Is the context meaningful? (not just the "no information" fallback)
    context = bundle.get("context", "")
    if "No relevant information" in context:
        return False
    
    # Check 2: Does the provenance show actual evidence?
    provenance = bundle.get("provenance", {})
    graph_items = provenance.get("graph", {}).get("items", 0)
    vector_items = provenance.get("vector", {}).get("items", 0)
    
    if graph_items == 0 and vector_items == 0:
        return False
    
    # Check 3: Is the average score above minimum threshold?
    avg_scores = [
        provenance.get("graph", {}).get("avg_score", 0),
        provenance.get("vector", {}).get("avg_score", 0),
    ]
    max_score = max(avg_scores)
    if max_score < 0.3:  # Below minimum relevance
        return False
    
    return True
```

### Bundle Scoring for Ranking

```python
def _score_bundle(self, bundle: Dict) -> float:
    provenance = bundle.get("provenance", {})
    
    # Direct graph answers get highest score
    if provenance.get("graph", {}).get("direct"):
        return 100.0
    
    # Weighted combination of channel scores
    graph_score = provenance.get("graph", {}).get("avg_score", 0) * 2.0
    vector_score = provenance.get("vector", {}).get("avg_score", 0) * 1.5
    community_score = provenance.get("community", {}).get("avg_score", 0) * 0.5
    
    item_bonus = min(
        provenance.get("graph", {}).get("items", 0) +
        provenance.get("vector", {}).get("items", 0),
        10
    ) * 0.1
    
    return graph_score + vector_score + community_score + item_bonus
```

**Why is graph score weighted 2x?** Graph entities are authoritative (structurally validated), while vector chunks are probabilistic. A high graph score indicates the query is directly relevant to entities in that department's knowledge graph.

---

## Topic Query Detection for Cross-Department Search

**File:** `graphrag/multi_retriever.py` → `_is_topic_query()`

The system detects whether a broadcast query is asking about a **research topic** (which should search across all departments) versus a **general institutional question** (which might have a single best source):

```python
def _is_topic_query(self, query: str) -> bool:
    q = query.lower()
    topic_indicators = (
        "work on", "working on", "research", "expert", "specialist",
        "who does", "find faculty", "people in", "researchers",
    )
    return any(indicator in q for indicator in topic_indicators)
```

When `_is_topic_query()` returns `True`, broadcast mode passes `suppress_topic_match=True` to each department's retriever, ensuring fair cross-department comparison.

---

## Context Merging and Provenance Aggregation

### Merging Strategy

Results from multiple departments/sections are merged with clear attribution:

```python
def _merge_bundles(self, bundles: List[Tuple]) -> str:
    sections = []
    for code, kind, bundle in bundles:
        if kind == "department":
            header = f"## Department of {dept_config['full_name']}"
        else:
            header = f"## {section_config['name']}"
        
        context = bundle.get("context", "")
        if context.strip():
            sections.append(f"{header}\n\n{context}")
    
    return "\n\n---\n\n".join(sections)
```

### Provenance Aggregation

For multi-department results, provenance is aggregated:

```python
merged_provenance = {
    "type": "multi" if len(bundles) > 1 else "single",
    "departments": [code for code, kind, _ in bundles if kind == "department"],
    "sections": [code for code, kind, _ in bundles if kind == "section"],
    "total_graph_items": sum(b["provenance"]["graph"]["items"] for _, _, b in bundles),
    "total_vector_items": sum(b["provenance"]["vector"]["items"] for _, _, b in bundles),
    "route": route_result.confidence,
}
```

---

## Why Not a Single Unified Index?

A common question: "Why not merge all departments into one giant FAISS index and search once?"

### Reason 1: Cross-Contamination

A single index means a query about "Dr. Sharma in EE" might retrieve Dr. Sharma from CSE too. The LLM would see both and might merge their attributes — assigning CSE Sharma's email to EE Sharma.

Per-department indices guarantee that EE queries only retrieve EE data.

### Reason 2: Enumeration Accuracy

If a student asks "How many faculty in EE?", the retriever needs to scan ALL Faculty nodes in the EE graph. A unified index would require filtering by department metadata, which adds complexity and failure modes.

### Reason 3: Independent Maintenance

When the EE department webpage is re-scraped and re-ingested, only the EE index needs rebuilding. Other departments are unaffected. A unified index would require complete reindexing.

### Reason 4: Provenance Clarity

With per-department indices, the provenance is unambiguous: "This fact came from the EE department's knowledge graph." With a unified index, provenance requires additional metadata tracking.

### Reason 5: Scalability

Adding a new department (e.g., a new center of excellence) is a self-contained operation:
1. Scrape the new department's website.
2. Ingest into a new graph + FAISS index.
3. Register the new retriever instance.
4. Update the alias map.

No existing departments are touched.

---

## Summary

The multi-department routing and orchestration system provides:

| Capability | Mechanism | Key File |
|-----------|-----------|----------|
| **Department detection** | Greedy alias matching with word boundaries | `dept_router.py` |
| **Section detection** | Same greedy matching, 400+ aliases | `dept_router.py` |
| **Academic rules intercept** | 197-line classifier with negative guards | `intent_utils.py` |
| **Cross-routing** | 8 injection rules for overlapping data | `dept_router.py` |
| **Single-dept retrieval** | Direct delegation | `multi_retriever.py` |
| **Multi-dept retrieval** | Independent queries + context merging | `multi_retriever.py` |
| **Broadcast search** | All-retriever scan + relevance filtering | `multi_retriever.py` |
| **Topic query optimization** | `suppress_topic_match` flag | `multi_retriever.py` |

The system ensures that every query reaches the right data source(s), and that cross-department queries receive fair, relevance-ranked results from all applicable knowledge bases.
