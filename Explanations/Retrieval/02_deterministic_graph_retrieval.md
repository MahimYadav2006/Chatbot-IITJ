# 02 — Deterministic Graph Retrieval: The "Regex-First" Engine

## Table of Contents

1. [What Is Deterministic Retrieval?](#what-is-deterministic-retrieval)
2. [The `get_deterministic_context()` Entry Point](#the-get_deterministic_context-entry-point)
3. [Intent Classification via Regex Patterns](#intent-classification-via-regex-patterns)
4. [Graph Traversal Strategies](#graph-traversal-strategies)
5. [The Complete Deterministic Dispatch Table](#the-complete-deterministic-dispatch-table)
6. [Topic/Expert Contact Matching](#topicexpert-contact-matching)
7. [Section-Specific Deterministic Retrieval](#section-specific-deterministic-retrieval)
8. [Why Regex Instead of an LLM Classifier?](#why-regex-instead-of-an-llm-classifier)
9. [Edge Cases and Guard Clauses](#edge-cases-and-guard-clauses)

---

## What Is Deterministic Retrieval?

Deterministic retrieval is the process of answering a query by **directly traversing pre-indexed graph data** rather than performing semantic similarity search. It produces answers that are:

- **Provenance-backed:** Every fact can be traced to a specific graph node and edge.
- **Complete:** Enumeration queries return ALL matching entities, not just the top-K.
- **Consistent:** The same query always produces the same answer (no LLM temperature variance).
- **Fast:** Graph traversal is O(nodes + edges), typically <5ms.

In the codebase, deterministic retrieval is handled by **two classes**:

| Class | File | Scope |
|-------|------|-------|
| `HybridRetriever` | `graphrag/retriever.py` | Academic departments (EE, CSE, ME, etc.) and Administration |
| `SectionRetriever` | `graphrag/section_retriever.py` | Institutional sections (Academics, CDS, Medical, IR, etc.) |

Both follow the same pattern: **regex-based intent detection → graph node lookup → formatted context string**.

---

## The `get_deterministic_context()` Entry Point

This is the single most important method in the entire retrieval system. It's the gateway that decides whether a query can be answered definitively from the graph.

### Signature and Behavior

```python
def get_deterministic_context(self, query: str, suppress_topic_match: bool = False) -> Optional[str]:
```

- **Returns `str`:** Deterministic context was found. This context is injected as the highest-priority evidence block for the LLM.
- **Returns `None`:** No deterministic match found. The system falls through to hybrid RAG retrieval (entity search + vector search + community search).
- **`suppress_topic_match`:** When `True`, skips the topic/expert contact matching block. Used during broadcast mode (cross-department search) to prevent a single department from short-circuiting the query.

### Execution Priority Order

The method processes intents in a strict priority order. The first match wins:

```python
# Priority 1: Administration (director, registrar, deans, committees)
if self.dept_code == "administration":
    admin_answer = self._build_administration_answer(query)

# Priority 2: Department contact / HoD queries
if self._is_department_contact_query(query):
    contact_answer = self._build_department_contact_answer()

# Priority 3: Lab/facility queries
if self._is_lab_query(query):
    labs_answer = self._build_labs_answer(query)

# Priority 4: Department address/location
if self._is_address_query(query):
    address_answer = self._build_address_answer()

# Priority 5: Graduated PhD alumni
if self._is_graduated_phd_query(query):
    grad_answer = self._build_graduated_phd_answer()

# Priority 6: General alumni
if self._is_alumni_query(query):
    alumni_answer = self._build_alumni_answer()

# Priority 7: Topic/expert contact matching (regex-extracted topic)
# Faculty and PhD scholars working on specific research areas

# Priority 8: Faculty analytics (gender breakdown, designation counts)
faculty_analytics_answer = self._build_faculty_analytics_answer(query)

# Priority 9: Email lookups by entity name
entity_email_name = self._extract_email_query_name(query)

# Priority 10: Supervisor lookups
student_name = self._extract_supervisor_query_name(query)

# Priority 11: Students under a specific supervisor
supervisor_query_name = self._extract_supervisor_from_students_query(query)

# Priority 12: Research area lookups
student_name_area = self._extract_research_area_query_name(query)

# Priority 13: M.Tech roster queries
if self._is_mtech_roster_query(query):
    ...

# Priority 14: PhD roster queries
if self._is_phd_roster_query(query):
    ...

# Priority 15: Faculty roster queries (count and list)
if self._is_faculty_roster_query(query):
    ...
```

---

## Intent Classification via Regex Patterns

Each intent category uses a combination of **keyword triggers** and **regex patterns** to classify queries. Here's how each works:

### Contact / HoD Detection (`_is_department_contact_query`)

```python
triggers = (
    "point of contact", "main contact", "official contact",
    "department contact", "contact person",
    "whom should i contact", "who should i contact", "who do i contact",
)
```

**Guard clause:** If the query contains "for" + a non-generic topic (e.g., "contact for VLSI research"), it's NOT a generic contact query — it's a topic query. This guard prevents the contact handler from intercepting research-topic queries.

```python
if any(re.search(rf"\b{term}\b", q) for term in ("for", "about", "regarding", "in")):
    m = re.search(r"\b(?:for|about|regarding|in)\b\s+(.+)", q)
    if m:
        target = m.group(1).strip()
        generic_terms = {"department", "dept", "admission", ...}
        if not all(w in generic_terms for w in words):
            return False  # This is a topic query, not generic contact
```

### Lab Detection (`_is_lab_query`)

Simple keyword matching:
```python
lab_triggers = (
    "lab", "labs", "laboratory", "laboratories", "research lab",
    "teaching lab", "facilities", "research facilities",
)
```

### Faculty Roster Detection (`_is_faculty_roster_query`)

This is more complex because it must distinguish between:
- ✅ "How many faculty in EE?" (roster query)
- ❌ "Which faculty work on VLSI?" (filtered topic query — NOT a roster query)

```python
# Step 1: Must mention faculty/professor
has_faculty_term = any(term in q for term in ("faculty", "professor", ...))

# Step 2: Must have count or list intent
count_intent = any(term in q for term in ("how many", "count", "total", ...))
roster_intent = any(term in q for term in ("faculty list", "list of faculty", ...))

# Step 3: Guard — not a filtered query
filtered_intent = any(term in q for term in ("work on", "research area", "patent", ...))
if filtered_intent and not explicit_roster_trigger:
    return False  # Don't hijack filtered questions
```

### Supervisor Extraction (Regex Named Groups)

For queries like *"Who is the supervisor of Rahul Kumar?"*, the system uses named-group regex:

```python
patterns = (
    r"^who supervises (?P<name>.+?)\??$",
    r"^who is (?P<name>.+?) supervised by\??$",
    r"^who is the supervisor of (?P<name>.+?)\??$",
    r"^who is (?P<name>.+?)'?s supervisor\??$",
    r"^who is (?P<name>.+?)'?s advisor\??$",
    r"^who advises (?P<name>.+?)\??$",
    r"^who is the advisor of (?P<name>.+?)\??$",
)
```

Each pattern captures the student name, which is then resolved against the graph.

---

## Graph Traversal Strategies

Once intent is classified and a name/concept is extracted, the system traverses the graph using specific edge types:

### Strategy 1: Node Attribute Lookup

For HoD queries: find any `Faculty` node where `is_hod=True`.

```python
for node_id in self._entity_label_index.get("Faculty", []):
    if self.graph.nodes[node_id].get("is_hod"):
        return self._build_department_contact_answer()
```

### Strategy 2: Edge-Following (1-hop)

For supervisor queries: from a `PhDStudent` node, follow `SUPERVISED_BY` edges to `Faculty` nodes.

```python
for _, target, edge_data in self.graph.out_edges(student_id, data=True):
    if edge_data.get("type") == "SUPERVISED_BY":
        supervisor_names.append(self.graph.nodes[target].get("name"))
```

### Strategy 3: Reverse Edge-Following

For "students under Dr. X": from a `Faculty` node, follow *incoming* `SUPERVISED_BY` edges from `PhDStudent` nodes.

```python
for source, _, edge_data in self.graph.in_edges(sup_id, data=True):
    if edge_data.get("type") == "SUPERVISED_BY":
        s_data = self.graph.nodes.get(source, {})
        if s_data.get("label") in ("PhDStudent", "MTechStudent"):
            students.append(s_data)
```

### Strategy 4: Multi-hop with Edge Type Filtering

For committee member lookups in administration: from a `Committee` node, follow incoming `MEMBER_OF` edges and include the `role_in_committee` edge attribute.

```python
for source, target, edge_data in self.graph.in_edges(committee_node, data=True):
    if edge_data.get("type") == "MEMBER_OF":
        member_data = self.graph.nodes.get(source, {})
        members.append({
            "name": member_data.get("name"),
            "role": edge_data.get("role_in_committee", ""),
        })
```

### Strategy 5: Full Node Scan with Label Filtering

For roster queries: scan ALL nodes of a specific label (e.g., `Faculty`, `PhDStudent`).

```python
for node_id, data in self.graph.nodes(data=True):
    if data.get("label") != "Faculty":
        continue
    roster.append({...})
```

---

## The Complete Deterministic Dispatch Table

| Intent | Detection Method | Graph Traversal | Output |
|--------|-----------------|-----------------|--------|
| **Director** | `"director" in q` | Node scan: `is_director=True` | Name, details |
| **Registrar** | `"registrar" in q` | Node scan: `is_registrar=True` | Name, email |
| **BoG Chairman** | `"bog chairman" in q` | Node scan: `is_bog_chairman=True` | Name |
| **Deans list** | `"list of deans" in q` | Node scan: `admin_type=="Dean"` | Sorted list with roles |
| **Associate Deans** | `"list of ad" in q` | Node scan: `admin_type=="Associate Dean"` | Sorted list with roles |
| **Specific Dean** | Stem-based word overlap | Word matching on `admin_role` | Name + role |
| **Committee members** | Alias matching against committee name | `MEMBER_OF` edge traversal | Sorted member list with roles |
| **HoD** | Contact query or "who is the hod" | `is_hod=True` on Faculty nodes | Name, email, profile |
| **Labs** | Lab trigger keywords | Lookup in `CORRECT_LABS` dictionary | Numbered lab list |
| **Address** | Address trigger keywords | `ContactInfo` node lookup | Address, email, phone |
| **Graduated PhDs** | "graduated phd" triggers | `GraduatedPhD` label scan | Year-sorted list |
| **Alumni** | "alumni" triggers | `Alumni`/`AlumniBatch` label scan | Batch and name lists |
| **Topic experts** | 11 regex patterns for topic extraction | Edge traversal: `RESEARCHES_IN`, `STUDIES` + text field matching | Faculty + PhD scholar lists |
| **Faculty analytics** | Analytic terms + attribute detection | Roster computation + Counter | Designation breakdown |
| **Email lookup** | Email regex patterns | Name resolution → node attribute | Direct email string |
| **Supervisor lookup** | Supervisor regex patterns | `SUPERVISED_BY` edge follow | Supervisor name(s) |
| **Students-under-X** | Reverse supervisor regex | Reverse `SUPERVISED_BY` edges | Student list with areas |
| **Research area** | Research area regex | Node attribute lookup | Area description |
| **M.Tech roster** | MTech + count/list intent | `MTechStudent` node scan | Count + full list |
| **PhD roster** | PhD + count/list intent | `PhDStudent` node scan | Count + full list |
| **Faculty roster** | Faculty + count/list intent | `Faculty` node scan | Count + full list with profiles |

---

## Topic/Expert Contact Matching

This is the most sophisticated deterministic matching block. It handles queries like:
- "Who works on deep learning?"
- "Which faculty are experts in VLSI?"
- "Find me someone who researches computer vision"
- "Machine learning researchers at IIT Jammu"

### The 11 Regex Patterns

```python
contact_patterns = [
    r"who\s+(?:should\s+i\s+|to\s+)?contact\s+(?:for|about|regarding)\s+(.+)",
    r"(?:who|which)\s+(?:faculty|professor|...)(?:working\s+on|expert\s+in|...)\s+(.+)",
    r"who\s+(?:to\s+)?(?:reach\s+out\s+to|write\s+to)\s+(?:for|about)\s+(.+)",
    r"(?:find|list|get|show)\s+(?:faculty|...)(?:working\s+on|expert\s+in|...)\s+(.+)",
    r"(?:experts?|specialists?)\s+(?:in|on|for)\s+(.+)",
    r"(?:faculty|professors?)\s+(?:for|in|on)\s+(.+)\s+(?:at\s+iit\s+jammu|...)",
    r"(.+?)\s+(?:researchers?|experts?)\s+(?:at\s+iit|in\s+iit)",
    r"(.+?)\s+research\s+(?:at|in)\s+(?:iit|the\s+department)",
    r"who\s+(?:is\s+)?(?:doing|working\s+in|involved\s+in)\s+(.+)",
    r"faculty\s+(?:in\s+the\s+area\s+of|specializing\s+in)\s+(.+)",
    r"(?:any|is\s+there\s+any)\s+(?:faculty|professor)\s+(?:working\s+on)\s+(.+)",
]
```

### The Two-Phase Matching Strategy

Once a topic is extracted (e.g., "deep learning"), matching happens in two phases:

**Phase A: Structural Edge Matching** — The strongest signal. Traverses `RESEARCHES_IN` and `STUDIES` edges from faculty/student nodes to `ResearchArea` nodes. Uses the `_topic_matches_text()` function with **word-boundary-aware matching** for short topics:

```python
def _topic_matches_text(topic: str, text: str) -> bool:
    # Short topics (<=4 chars like "ai", "nlp", "iot"):
    #   → Use \b regex to prevent "ai" matching "uncertainty"
    # Long topics (>=5 chars like "machine learning"):
    #   → Simple substring match is safe
    # Also expands abbreviations: "ai" → also searches "artificial intelligence"
```

**Why word-boundary matching?** Without it, a search for "ai" would match the word "uncertainty" (contains "ai" as a substring), "training" (contains "ai"), "domain" (contains "ai"), etc. Word-boundary regex (`\bai\b`) ensures only standalone "AI" matches.

**Phase B: Text Field Matching** — For faculty NOT already matched by structural edges, checks `research_interests` and `academic_interests` text fields. Note: it intentionally does NOT check `publications`, `education`, or `research_experience` fields because these contain incidental text that causes false positives.

### Output Formatting

The topic matching produces a structured answer with faculty listed first, then PhD scholars:

```
**Faculty members** working in areas related to **Deep Learning**:

- **Dr. Anand Mishra** (Assistant Professor) - Email: anand@iitjammu.ac.in
- **Dr. Priya Singh** (Associate Professor) - Email: priya@iitjammu.ac.in

**PhD/M.Tech scholars** working in areas related to **Deep Learning**:

- **Rahul Kumar** (PhD Scholar) — Research: Deep Reinforcement Learning — Supervisor(s): Dr. Anand Mishra

You can reach out to the faculty for tasks or queries related to deep learning.
```

---

## Section-Specific Deterministic Retrieval

The `SectionRetriever` class provides deterministic retrieval for institutional sections. Each section has its own set of domain-specific handlers:

### Academics Section
- **DPGC/DUGC committees:** Committee type → department → member list with roles
- **Faculty advisors:** Programme coordinator lookups with batch year
- **Fee structures:** Category (B.Tech/M.Tech/PhD) × Year × Gender × SC/ST filtering
- **Academic programs:** Minor/Honours/Micro specialization lookups with word-scoring
- **Course catalogs:** Semester-wise or category-wise course listings with L-T-P credits
- **Document links:** Scraped markdown file scanning for download links

### CDS (Career Development Services)
- **Recruiters:** Past recruiting company listings
- **Placement policies:** CTC categories, upgrade rules, eligibility criteria
- **Placement statistics:** Year-wise, degree-wise stats with salary data

### Medical Centre
- **Services:** Timings, descriptions for dental, physiotherapy, pharmacy, etc.
- **Doctors:** Name, qualification, experience, contact (with superlative guard)
- **Empaneled hospitals:** Name, location, rate type

### International Relations
- **MOUs:** Partner institutions with country filtering
- **Student clubs:** Matched or full-list display with reasoning query guard
- **Sports facilities:** Facility + event listings
- **Hostels/Fests:** Named entity matching or full listings

### Others
- **Counselling:** Counselor profiles, service descriptions
- **DI (Digital Infrastructure):** Division-specific chunk lookups
- **E2 (Establishment II):** HR function descriptions
- **Alumni Affairs:** Medalist lists by year, award winners
- **OSD (Outreach):** UBA programs, CES courses, events

---

## Why Regex Instead of an LLM Classifier?

This is a deliberate architectural decision. Let's compare the two approaches:

| Criterion | Regex Classification | LLM Classification |
|-----------|---------------------|---------------------|
| **Latency** | <1ms | 1-3 seconds (API call) |
| **Cost** | Zero | $0.001-0.01 per classification |
| **Determinism** | Same input → same output, always | Non-deterministic (temperature > 0) |
| **Debuggability** | Can print which pattern matched | Black box; need prompt engineering |
| **Coverage** | Handles 95%+ of institutional queries | Handles 99%+ of any query |
| **Maintenance** | Adding patterns is manual but trivial | Requires prompt tuning |
| **Failure mode** | Falls through to RAG (safe) | Misclassification → wrong data |

### The Key Insight

For an institutional chatbot, the **query domain is finite and enumerable**. People ask about faculty, students, labs, placements, fees, committees, and hostels. They don't ask about Shakespearean literature or quantum field theory. This bounded domain makes regex classification both feasible and superior to LLM classification.

When regex fails (no pattern matches), the system **gracefully degrades** to hybrid RAG retrieval — never to an error. This means the worst-case scenario for a missed regex pattern is a slightly slower response, not a broken interaction.

---

## Edge Cases and Guard Clauses

The codebase contains numerous guard clauses to prevent incorrect deterministic routing:

### Guard 1: Preventing Roster Hijacking
```python
# "Which faculty work on VLSI?" should NOT trigger the faculty roster
filtered_intent = any(term in q for term in (
    "work on", "research area", "patent", "project", "startup", "publication"
))
if filtered_intent and not explicit_roster_override:
    return False
```

### Guard 2: Preventing Admin Short-Circuit on Research Queries
```python
# "Compare Dr. A and Dr. B's research" should NOT return admin committee roles
non_admin_intents = (
    "compare", "research work", "research interest", "publication",
    "teaching", "specialization", ...
)
if any(term in q for term in non_admin_intents):
    return None  # Fall through to department-level retrieval
```

### Guard 3: Preventing Topic Contact from Intercepting Generic Contacts
```python
# "contact for department" → generic contact (return HoD)
# "contact for VLSI research" → topic query (find VLSI experts)
if target_phrase and not all(w in generic_terms for w in words):
    return False  # This is a topic query, not generic contact
```

### Guard 4: Superlative Queries Bypass Direct Answers
```python
# "Who is the most senior doctor?" needs LLM reasoning, not a dump of all doctors
superlative_indicators = ("highest", "most", "best", "senior most", ...)
if is_superlative:
    # Don't return raw doctor list; let LLM reason about seniority
    pass
```

### Guard 5: Reasoning Queries About Clubs
```python
# "Which club is better for technical skills?" needs LLM reasoning
reasoning_indicators = ("better", "best", "compare", "should i join", ...)
if is_reasoning and not is_list_query:
    pass  # Fall through to LLM
```

### Guard 6: Analysis/Summarization Bypass for Rosters
```python
# "Summarize the primary domains of PhD students" should NOT return the raw roster
if any(term in q for term in ("summarize", "analyze", "overview", "trend", ...)):
    return False
```

These guards ensure that deterministic retrieval only fires when the query genuinely warrants a structural graph answer, and defers to the LLM for queries requiring reasoning, comparison, or synthesis.

---

## Summary

The deterministic graph retrieval layer is the **backbone of factual accuracy** in the chatbot. Key statistics:

- **15 distinct intent categories** handled deterministically
- **11 topic-matching regex patterns** for research area queries
- **7 supervisor/email/research-area extraction patterns** for named-entity queries
- **6 guard clause categories** to prevent incorrect routing
- **2 matching strategies** for graph traversal (structural edges + text field matching)

The system processes approximately **60-70% of institutional queries** through the deterministic layer, meaning the majority of user interactions never touch the vector search system at all. This is by design: for an institutional chatbot, deterministic answers are always better than probabilistic ones.
