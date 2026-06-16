# IIT Jammu Chatbot Repository Map

This is a detailed repository map of the chatbot codebase, designed for LLM agents to quickly understand the structure, classes, and function signatures without having to read the full contents of every file.

## 📁 Repository Directory Structure
```text
chatbot
├── .env
├── .env.example
├── README.md
├── SCALING.md
├── app.py
├── crawler.py
├── debug_matching.py
├── debug_phd_headings.py
├── departments.py
├── dept_router.py
├── env_config.py
├── evaluation
│   ├── __init__.py
│   ├── evaluation_report.md
│   ├── evaluation_results.json
│   ├── generate_qna.py
│   ├── outputs
│   │   └── computer_science_engineering
│   │       ├── evaluation_report.md
│   │       ├── evaluation_results.json
│   │       └── question_set.json
│   ├── pipeline.py
│   ├── qna_dataset.json
│   ├── run_department_eval.py
│   ├── run_eval.py
│   └── test_evaluation.py
├── graphrag
│   ├── __init__.py
│   ├── community.py
│   ├── embeddings.py
│   ├── intent_utils.py
│   ├── kg_builder.py
│   ├── llm.py
│   ├── multi_retriever.py
│   ├── person_index.py
│   ├── retriever.py
│   ├── rules_db.py
│   ├── rules_parser.py
│   ├── rules_retriever.py
│   ├── section_kg_builder.py
│   ├── section_retriever.py
│   └── verifier.py
├── ingest.py
├── inspect_2_2.py
├── inspect_fact_section.py
├── inspect_fts.py
├── inspect_phd_sections.py
├── inspect_ug_2022.py
├── inspect_ug_2022_sections.py
├── requirements.txt
├── rules.db
├── scratch
│   ├── generate_repomap.py
│   ├── test_broadcast.py
│   ├── test_retrieval_queries.py
│   ├── test_verify.py
│   └── verify_academics_kg.py
├── scripts
│   ├── convert_pdfs_to_md.py
│   ├── download_academics_pdfs.py
│   ├── download_gdrive_pass2.py
│   ├── download_gdrive_playwright.py
│   ├── flatten_curriculum_docs.py
│   └── ingest_rules.py
├── test_crawl.py
├── test_parser.py
├── test_parser_class.py
├── test_query_db.py
├── test_retriever.py
├── test_start_detection.py
├── test_toc_extractor.py
├── test_toc_matcher.py
├── tests
│   ├── conftest.py
│   ├── test_chunk_quality.py
│   ├── test_crawler_architecture.py
│   ├── test_cse_faculty_profile_parsing.py
│   ├── test_cse_phd_email_retrieval.py
│   ├── test_data_integrity.py
│   ├── test_department_eval_pipeline.py
│   ├── test_entity_resolution.py
│   ├── test_response_quality.py
│   ├── test_retrieval.py
│   ├── test_rules_chatbot.py
│   └── test_section_chatbot.py
├── trace_ug_parsing.py
└── utils.py
```

---

## 🛠️ Modules & Signatures

### [app.py](file:///home/c3i/chatbot/app.py)
> Flask Web Application for IIT Jammu Unified GraphRAG Chatbot.
> 
> Provides a single, unified interface for querying all department knowledge
> graphs with automatic department detection and routing.

#### Top-level Functions
```python
def _log_retrieval_provenance(dept_codes, query, provenance, answerability=None):
    """Log which retrieval channels contributed to the answer."""
def get_retriever(dept_code: str):
    """Get or dynamically load retriever for the specified department."""
def get_section_retriever(sec_code: str):
    """Get or dynamically load retriever for the specified section."""
def _is_identity_query(query: str) -> bool:
    """
    Return True only when the query is asking about a person's identity/role/position.
    
    Returns False for queries about research areas, PhD scholars, publications,
    teaching, supervision, etc. — those need department-specific retrieval.
    Also returns False for admin-role queries (e.g. "who is the dean") that
    should be routed to the administration department's deterministic logic.
    """
def get_global_person_direct_answer(query: str) -> Optional[str]:
    """
    Check if query is about a person and return a deterministic direct answer from Global Person Index.
    
    Only returns a role-card for identity/role queries (e.g. "who is Dr. X?",
    "tell me about Prof. Y").  Queries about research areas, PhD scholars,
    publications, teaching, etc. are deliberately skipped so that the full
    retrieval pipeline (which has actual faculty data) can handle them.
    """
def init_app():
    """Initialize the LLM, router, verifier, and preload all ingested retrievers."""
def index():
    """Serve the main chat page."""
def chat():
    """
    Handle chat messages with automatic department and section routing.
    
    Flow:
      1. Global person index lookup to bypass retrieval for direct matches
      2. Router detects target department(s) and/or section(s) from query text
      3. MultiDepartmentRetriever/SectionRetriever fetches from the right scope
      4. LLM generates response with appropriate system prompt
      5. Verifier checks faithfulness before returning
    """
def health():
    """Health check endpoint."""
```


### [crawler.py](file:///home/c3i/chatbot/crawler.py)
> Dynamic Web Crawler for IIT Jammu Academic Departments.
> 
> Architecture:
>     1. Discover URLs from validated department pages only.
>     2. Normalize broken relative links against the department root.
>     3. Reject generic fallback pages before expanding them.
>     4. Persist a crawl manifest explaining every accept/reject decision.

#### `class LinkDecision:`

#### `class PageDecision:`

#### Top-level Functions
```python
def _compact_text(text: str) -> str:
def _score_content_node(node) -> float:
def _select_content_root(soup: BeautifulSoup):
def _clean_soup_for_markdown(html_content: str):
def _extract_click_target(raw_js: str) -> Optional[str]:
def _extract_candidate_links(soup: BeautifulSoup, base_url: str, current_url: str) -> List[str]:
def render_page_snapshot(url: str, context) -> Dict[str, str]:
def evaluate_page(url: str, snapshot: Dict[str, str]) -> PageDecision:
def discover_site(base_url: str, context, max_pages: Optional[int] = None):
def html_to_markdown(element, base_url):
    """Recursively convert BeautifulSoup HTML elements into clean, relation-preserving Markdown."""
def clean_markdown(text):
    """Normalize whitespace and remove excessive newlines and duplicate adjacent lines."""
def parse_binary_file(response, url, output_dir):
    """Download binary files to a secure temporary path and parse their content."""
def _build_output_filename(url: str, dept_code: str) -> str:
def _build_quality_flags(markdown_content: str, snapshot: Optional[Dict[str, str]] = None) -> List[str]:
def _build_fallback_markdown(snapshot: Optional[Dict[str, str]] = None) -> str:
def _html_snapshot_to_markdown(url: str, html_content: str, base_url: str, snapshot: Optional[Dict[str, str]] = None):
def save_markdown_document(
    url: str,
    markdown_content: str,
    output_dir: str,
    dept_code: str,
    page_title: Optional[str] = None,
    content_flags: Optional[List[str]] = None,
):
def download_and_convert(url, output_dir, playwright_context, dept_code, base_url, cached_snapshot=None):
    """Download a URL and convert it to Markdown."""
def _page_urls_to_export(page_decisions: List[PageDecision], page_snapshots: Dict[str, Dict[str, str]]) -> List[str]:
def _clear_output_dir(output_dir: str):
def crawl_department(dept_code: str, clean_output: bool = False, max_pages: Optional[int] = None):
def crawl_section(section_code: str, clean_output: bool = False, max_pages: Optional[int] = None):
def crawl_administration(clean_output: bool = False):
def main():
```


### [debug_matching.py](file:///home/c3i/chatbot/debug_matching.py)
*No classes or functions defined.*


### [debug_phd_headings.py](file:///home/c3i/chatbot/debug_phd_headings.py)
#### Top-level Functions
```python
def parse_sec_num(s):
```


### [departments.py](file:///home/c3i/chatbot/departments.py)
> Central department registry for IIT Jammu multi-department chatbot.
> All department-specific configuration lives here.

#### Top-level Functions
```python
def resolve_department_code(code: str) -> str:
    """Resolve a user-facing department code or alias to the canonical code."""
def get_department(code: str) -> dict:
    """Get department config by code. Raises KeyError if not found."""
def get_all_codes() -> list:
    """Get a list of all department codes."""
def get_scraped_data_root() -> str:
    """Return the root folder that stores crawled markdown for all departments."""
def get_scraped_markdown_dir(code: str) -> str:
    """Return the canonical crawl output directory under `scraped_data/`."""
def get_legacy_markdown_dir(code: str) -> str:
    """Return the original checked-in markdown directory path for a department."""
def get_markdown_dir(code: str) -> str:
    """
    Return the markdown directory path for a department.
    
    New crawls are stored under `scraped_data/<department>/`.
    For backward compatibility, if that folder does not exist yet but the legacy
    checked-in markdown directory exists, fall back to the legacy location.
    """
def get_data_dir(code: str) -> str:
    """Return the data directory path for a department."""
def get_section_markdown_dir(code: str) -> str:
    """Return the canonical crawl output directory for a section under `scraped_data/sections/`."""
def get_section_data_dir(code: str) -> str:
    """Return the data directory path for a section."""
```


### [dept_router.py](file:///home/c3i/chatbot/dept_router.py)
> Smart Department Router for IIT Jammu Multi-Department Chatbot.
> 
> Detects which department(s) a query targets using department-name aliases.
> Does NOT use subject-based matching (e.g., "machine learning") since subjects
> can belong to multiple departments.
> 
> Routes:
>   - Single department:  "Who is the CSE HOD?" → [computer_science_engineering]
>   - Multi department:   "Compare EE and CSE faculty" → [ee, computer_science_engineering]
>   - Broadcast:          "How many departments?" → all ingested departments

#### `class RouteResult:`
  > Result of department routing.

#### `class DepartmentRouter:`
  > Routes queries to the correct department(s) or section(s) based on aliases.
  ```python
      def __init__(self):
      def route(self, query: str) -> RouteResult:
      """Analyze a query and determine which department(s) and/or section(s) it targets."""
      def _detect_departments(self, query: str) -> List[str]:
      """Detect department references in query text using greedy alias matching."""
      def _detect_sections(self, query: str) -> List[str]:
      """Detect section references in query text using greedy alias matching."""
      def get_ingested_departments(self) -> List[str]:
      """Return list of department codes that have been ingested."""
      def get_ingested_sections(self) -> List[str]:
      """Return list of section codes that have been ingested."""
  ```


### [env_config.py](file:///home/c3i/chatbot/env_config.py)
#### Top-level Functions
```python
def load_env_file(path: str = ".env", override: bool = False) -> bool:
    """Load simple KEY=VALUE pairs from a local .env-style file."""
```


### [ingest.py](file:///home/c3i/chatbot/ingest.py)
> GraphRAG Ingestion Pipeline for IIT Jammu Departments.
> 
> Runs the full pipeline for a specific department or all departments:
>     1. Parse markdown files → extract entities & relationships
>     2. Build NetworkX knowledge graph
>     3. Run Louvain community detection
>     4. Generate embeddings (chunks + entities + community summaries)
>     5. Generate community summaries via LLM
>     6. Persist everything to data/{dept_code}/ directory
> 
> Usage:
>     python ingest.py --dept ee         # Ingest Electrical Engineering
>     python ingest.py --dept cse        # Ingest Computer Science
>     python ingest.py --all             # Ingest all 11+ departments
>     python ingest.py --skip-summaries  # Ingest default department skipping LLM community summaries

#### Top-level Functions
```python
def ingest_department(dept_code: str, skip_summaries: bool = False):
def ingest_section(section_code: str, skip_summaries: bool = False):
def main():
```


### [inspect_2_2.py](file:///home/c3i/chatbot/inspect_2_2.py)
*No classes or functions defined.*


### [inspect_fact_section.py](file:///home/c3i/chatbot/inspect_fact_section.py)
*No classes or functions defined.*


### [inspect_fts.py](file:///home/c3i/chatbot/inspect_fts.py)
*No classes or functions defined.*


### [inspect_phd_sections.py](file:///home/c3i/chatbot/inspect_phd_sections.py)
*No classes or functions defined.*


### [inspect_ug_2022.py](file:///home/c3i/chatbot/inspect_ug_2022.py)
*No classes or functions defined.*


### [inspect_ug_2022_sections.py](file:///home/c3i/chatbot/inspect_ug_2022_sections.py)
*No classes or functions defined.*


### [test_crawl.py](file:///home/c3i/chatbot/test_crawl.py)
#### Top-level Functions
```python
def debug_url(url):
def debug():
```


### [test_parser.py](file:///home/c3i/chatbot/test_parser.py)
*No classes or functions defined.*


### [test_parser_class.py](file:///home/c3i/chatbot/test_parser_class.py)
*No classes or functions defined.*


### [test_query_db.py](file:///home/c3i/chatbot/test_query_db.py)
*No classes or functions defined.*


### [test_retriever.py](file:///home/c3i/chatbot/test_retriever.py)
*No classes or functions defined.*


### [test_start_detection.py](file:///home/c3i/chatbot/test_start_detection.py)
*No classes or functions defined.*


### [test_toc_extractor.py](file:///home/c3i/chatbot/test_toc_extractor.py)
#### Top-level Functions
```python
def extract_toc_headings(filepath):
```


### [test_toc_matcher.py](file:///home/c3i/chatbot/test_toc_matcher.py)
#### Top-level Functions
```python
def is_valid_sec_num(s):
def extract_toc_headings(filepath):
def titles_match(toc_title, body_title):
```


### [trace_ug_parsing.py](file:///home/c3i/chatbot/trace_ug_parsing.py)
*No classes or functions defined.*


### [utils.py](file:///home/c3i/chatbot/utils.py)
#### Top-level Functions
```python
def _normalize_href(href: str) -> str:
    """Strip invisible whitespace and normalize HTML oddities from hrefs."""
def _collapse_duplicate_segments(path: str) -> str:
    """Collapse duplicate consecutive path segments and repeated department prefixes."""
def canonicalize_url(base_url: str, current_url: str, href: str) -> Optional[str]:
    """
    Resolve an href into a canonical absolute URL within the department site.
    
    IIT Jammu pages often emit broken relative links such as:
    - ``computer_science_engineering/about-us`` from nested pages
    - trailing non-breaking spaces in URLs
    - duplicated department prefixes
    
    This helper normalizes those cases so we do not invent URLs like
    ``.../program-list/computer_science_engineering/program-list/...``.
    """
def is_binary_url(url: str) -> bool:
def is_static_asset_url(url: str) -> bool:
def is_same_department_url(url: str, base_url: str) -> bool:
def classify_discovered_url(url: str, base_url: str, allowed_domain: str) -> Tuple[str, str]:
    """Classify a normalized URL for crawling."""
def is_generic_page(title: str, text: str) -> bool:
def is_generic_content(markdown: str) -> bool:
    """Return ``True`` if the extracted markdown looks like boilerplate only."""
```


### [evaluation/__init__.py](file:///home/c3i/chatbot/evaluation/__init__.py)
> Evaluation helpers for department-specific chatbot benchmarking.

*No classes or functions defined.*


### [evaluation/generate_qna.py](file:///home/c3i/chatbot/evaluation/generate_qna.py)
> Script to generate the QnA dataset for evaluation of the IIT Jammu EE Chatbot.

#### Top-level Functions
```python
def main():
```


### [evaluation/pipeline.py](file:///home/c3i/chatbot/evaluation/pipeline.py)
#### Top-level Functions
```python
def utc_now_iso() -> str:
def ensure_parent(path: Path) -> None:
def normalize_text(text: str) -> str:
def simplify_text(text: str) -> str:
def similarity_ratio(left: str, right: str) -> float:
def extract_json_payload(raw_text: str) -> Any:
    """Extract a JSON object or array from LLM output that may include fences/noise."""
def allocate_category_counts(total_questions: int) -> Dict[str, int]:
    """Spread the requested question count across varied evaluation categories."""
def default_output_paths(dept_code: str) -> Tuple[Path, Path, Path]:
def _sorted_nodes(graph, label: str, dept_code: str) -> List[Tuple[str, Dict[str, Any]]]:
def _related_names(
    graph,
    node_id: str,
    edge_type: str,
    target_labels: Iterable[str] | None = None,
    limit: int = 5,
) -> List[str]:
def _top_research_areas(graph, dept_code: str, limit: int = 20) -> List[Tuple[str, int]]:
def build_grounding_bundle(dept_code: str, max_chars: int = 10000) -> str:
    """Assemble a diverse grounding brief for question generation."""
def _category_generation_prompt(
    dept_code: str,
    category: str,
    question_count: int,
    grounding_bundle: str,
) -> str:
def _coerce_question_list(payload: Any) -> List[Dict[str, Any]]:
def _validate_generated_questions(
    payload: Any,
    dept_code: str,
    expected_category: str,
    question_count: int,
    seen_questions: set[str] | None = None,
) -> List[Dict[str, Any]]:
def _generate_validated_json(
    llm,
    prompt: str,
    validator,
    *,
    max_attempts: int = 3,
    max_tokens: int = 3200,
) -> Any:
def generate_question_set(
    dept_code: str,
    *,
    question_count: int = 24,
    model: str | None = None,
    output_path: Path | None = None,
    max_source_chars: int = 10000,
) -> Dict[str, Any]:
def _run_chatbot_query(
    retriever,
    llm,
    dept_code: str,
    question: str,
) -> Dict[str, Any]:
def _judge_prompt(item: Dict[str, Any], actual_answer: str) -> str:
def evaluate_single_answer(
    llm,
    item: Dict[str, Any],
    actual_answer: str,
) -> Dict[str, Any]:
def _validate_judgement(payload: Any) -> Dict[str, Any]:
def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
def evaluate_question_set(
    dataset: Dict[str, Any],
    *,
    dept_code: str | None = None,
    model: str | None = None,
    results_path: Path | None = None,
    report_path: Path | None = None,
) -> Dict[str, Any]:
def render_markdown_report(payload: Dict[str, Any]) -> str:
def load_dataset(path: Path) -> Dict[str, Any]:
```


### [evaluation/run_department_eval.py](file:///home/c3i/chatbot/evaluation/run_department_eval.py)
#### Top-level Functions
```python
def parse_args() -> argparse.Namespace:
def main() -> None:
```


### [evaluation/run_eval.py](file:///home/c3i/chatbot/evaluation/run_eval.py)
> Orchestrator script to run the full QnA generation and evaluation suite,
> and generate a beautiful markdown report.

#### Top-level Functions
```python
def run_command(cmd, desc):
def main():
```


### [evaluation/test_evaluation.py](file:///home/c3i/chatbot/evaluation/test_evaluation.py)
> Test and Evaluation script for IIT Jammu EE Chatbot.
> Queries the chatbot and uses LLM-as-a-judge to evaluate and classify responses.

#### Top-level Functions
```python
def evaluate_response(llm, question: str, expected: str, actual: str, category: str) -> dict:
    """Uses the LLM as an objective judge to classify and explain the chatbot response."""
def main():
```


### [graphrag/__init__.py](file:///home/c3i/chatbot/graphrag/__init__.py)
> GraphRAG - Graph-based Retrieval Augmented Generation
> for IIT Jammu EE Department Chatbot.
> 
> Components:
>     - kg_builder: Knowledge graph construction from markdown files
>     - embeddings: Embedding generation and FAISS indexing
>     - community: Community detection and summarization
>     - retriever: Hybrid retrieval engine
>     - llm: LLM integration via Ollama or Gemini

*No classes or functions defined.*


### [graphrag/community.py](file:///home/c3i/chatbot/graphrag/community.py)
> Community Detection and Summarization for GraphRAG.
> 
> Implements:
>     1. Louvain community detection on the entity graph
>     2. Hierarchical community structure at multiple resolutions
>     3. Community summarization via LLM

#### Top-level Functions
```python
def detect_communities(graph: nx.DiGraph, resolution: float = 1.0) -> Dict[str, int]:
    """
    Run Louvain community detection on the entity graph.
    
    We convert to undirected and filter to only entity nodes
    (excluding TextChunk and Document nodes) for meaningful communities.
    
    Args:
        graph: NetworkX directed graph
        resolution: Louvain resolution parameter (higher = more communities)
    
    Returns:
        Dict mapping node_id -> community_id
    """
def build_community_reports(graph: nx.DiGraph, partition: Dict[str, int]) -> List[Dict]:
    """
    Build structured reports for each community.
    
    Each report contains the community's member entities, their types,
    key relationships, and a text representation for embedding/summarization.
    
    Returns:
        List of community report dicts
    """
def summarize_communities(reports: List[Dict], llm_fn=None) -> List[Dict]:
    """
    Generate natural language summaries for each community using the LLM.
    
    Args:
        reports: Community reports from build_community_reports()
        llm_fn: Function that takes a prompt string and returns response text.
                 If None, uses a rule-based summary instead.
    
    Returns:
        Updated reports with 'summary' field populated
    """
def _rule_based_summary(members_by_type: Dict[str, List]) -> str:
    """Generate a simple rule-based summary when LLM is unavailable."""
def save_communities(reports: List[Dict], partition: Dict[str, int],
                     output_dir: str = DATA_DIR):
    """Save community data to disk."""
def load_communities(data_dir: str = DATA_DIR):
    """Load community data from disk."""
```


### [graphrag/embeddings.py](file:///home/c3i/chatbot/graphrag/embeddings.py)
> Embedding Engine for GraphRAG.
> Generates embeddings and FAISS index for chunks, entities, communities.

#### `class EmbeddingEngine:`
  ```python
      def __init__(self, model_name: str = EMBEDDING_MODEL):
      def _load_model(self):
      def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
      def encode_single(self, text: str) -> np.ndarray:
      def build_index(self, chunks: List[Dict], entity_descriptions: List[Dict],
                      community_summaries: List[Dict] = None, dept_code: str = "ee"):
      def search(self, query: str, top_k: int = 10,
                 type_filter: Optional[str] = None, department_filter: Optional[str] = None,
                 min_score: float = 0.0) -> List[Tuple[Dict, float]]:
      def save(self, output_dir: str = DATA_DIR):
      def load(self, data_dir: str = DATA_DIR):
  ```

#### Top-level Functions
```python
def create_entity_descriptions(graph) -> List[Dict]:
    """Create rich text descriptions for each entity for embedding search."""
```


### [graphrag/intent_utils.py](file:///home/c3i/chatbot/graphrag/intent_utils.py)
#### Top-level Functions
```python
def _normalized_query(query: str) -> str:
    """Normalize common OCR/user typos before lightweight intent checks."""
def is_academic_rules_query(query: str) -> bool:
    """
    Detect if a query is about academic rules, regulations, or policies.
    
    Uses precise keywords and patterns, and includes guards to prevent false matches
    on queries meant for other departments or sections (e.g. CDS, Counselling, specific faculty).
    """
```


### [graphrag/kg_builder.py](file:///home/c3i/chatbot/graphrag/kg_builder.py)
> Knowledge Graph Builder for IIT Jammu EE Department.
> Parses markdown files and constructs a NetworkX DiGraph with entities,
> relationships, and clean text chunks.

#### `class EntityResolver:`
  ```python
      def __init__(self, canonical_faculty: set = None):
      def is_canonical_faculty(self, raw_name: str) -> bool:
      """Check if a name matches any canonical faculty member."""
      def resolve(self, raw_name: str) -> str:
  ```

#### `class KnowledgeGraphBuilder:`
  ```python
      def __init__(self, dept_code: str = DEFAULT_DEPT, markdown_dir: str = None):
      def _add_node(self, node_id: str, label: str, **properties):
      def _add_edge(self, source: str, target: str, rel_type: str, **properties):
      def _create_document_node(self, filename: str, content: str) -> str:
      def _parse_faculty_profile(self, filename: str, content: str, doc_id: str):
      def _parse_phd_list(self, filename: str, content: str, doc_id: str, label: str = "PhDStudent"):
      def _parse_funded_projects(self, filename: str, content: str, doc_id: str):
      def _parse_patents(self, filename: str, content: str, doc_id: str):
      def _parse_startups(self, filename: str, content: str, doc_id: str):
      def _parse_research_areas(self, filename: str, content: str, doc_id: str):
      def _parse_faculty_list(self, filename: str, content: str, doc_id: str):
      def _parse_hod(self, filename: str, content: str, doc_id: str):
      def _parse_placement_data(self, filename: str, content: str, doc_id: str):
      """Parse placement industry data into structured PlacementData nodes."""
      def _parse_higher_studies(self, filename: str, content: str, doc_id: str):
      """Parse higher studies (placement-academia) data into structured nodes."""
      def _parse_labs(self, content: str, doc_id: str):
      def _parse_awards(self, filename: str, content: str, doc_id: str):
      def _parse_publications_page(self, filename: str, content: str, doc_id: str):
      def _parse_staff(self, filename: str, content: str, doc_id: str):
      def _parse_programmes(self, filename: str, content: str, doc_id: str):
      def _parse_contact(self, filename: str, content: str, doc_id: str):
      """Parse contact/address pages into a structured ContactInfo node."""
      def _parse_alumni(self, filename: str, content: str, doc_id: str):
      """Parse alumni pages to extract alumni names, batch year, and program info."""
      def _parse_phd_alumni(self, filename: str, content: str, doc_id: str):
      """Parse graduated PhD student lists (phd-alumni-list) into GraduatedPhD nodes."""
      def _parse_admin_director(self, filename: str, content: str, doc_id: str):
      def _parse_admin_registrar(self, filename: str, content: str, doc_id: str):
      def _parse_admin_bogchairman(self, filename: str, content: str, doc_id: str):
      def _parse_admin_deans(self, filename: str, content: str, doc_id: str):
      def _parse_admin_committee(self, filename: str, content: str, doc_id: str):
      def build(self) -> nx.DiGraph:
      def save(self, output_dir: str = None):
      def load(data_dir: str):
  ```

#### Top-level Functions
```python
def clean_content_for_chunks(content: str) -> str:
    """Remove boilerplate HTML/navigation noise from content before chunking."""
def normalize_name(name: str) -> str:
def clean_admin_member_name(name: str) -> str:
def _strip_markdown_emphasis(text: str) -> str:
def _deobfuscate_email_text(text: str) -> str:
def _extract_emails(text: str) -> list:
    """Extract normalized emails from raw or obfuscated source text."""
def _extract_first_email(text: str) -> str:
def _canonical_section_key(text: str) -> str:
def _is_probable_person_name(text: str) -> bool:
def _initials_match(short: str, full: str) -> bool:
    """Check if an abbreviated name like 'B. N Subudhi' matches 'Badri Narayan Subudhi'."""
def _token_subset_match(name1: str, name2: str) -> bool:
    """
    Match variants like 'Anup Kumar Shukla' and 'Anup Shukla'.
    
    Requires strong agreement on first/last name while allowing optional
    middle tokens in either variant.
    """
def fuzzy_match(name1: str, name2: str, threshold: float = 0.85) -> bool:
def _chunk_words(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
def _is_probable_record_heading(heading: str) -> bool:
def _detect_repeated_heading_level(text: str, min_records: int = MIN_REPEATED_RECORDS) -> Optional[int]:
    """Auto-detect repeated heading level used for roster-style records."""
def _detect_person_record_level(text: str, min_records: int = MIN_REPEATED_RECORDS) -> Optional[int]:
    """
    Detect the heading level used for person roster records.
    
    This is stricter than generic repeated-heading detection: candidate record
    bodies must look like faculty/student entries, not subsection labels.
    """
def _split_prefix_and_records(text: str, level: int) -> Tuple[str, List[str]]:
def _chunk_repeated_records(text: str, chunk_size: int = CHUNK_SIZE) -> Tuple[List[str], Dict]:
    """
    Chunk roster-style markdown by repeated heading records instead of raw word windows.
    
    Works across departments by auto-detecting the repeated record heading level.
    """
def _split_structural_blocks(text: str) -> list:
    """Split markdown into structural blocks while keeping headings and tables intact."""
def _chunk_structural_blocks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Tuple[List[str], Dict]:
def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Chunk markdown using the most structure-preserving strategy available."""
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Compatibility wrapper returning only chunk text."""
def _strip_markdown_link(text: str) -> str:
    """Return the visible text from a markdown link, or the raw text if not linked."""
def _clean_section_heading(text: str) -> str:
    """Normalize markdown section headings extracted from faculty profile pages."""
def _iter_heading_blocks(content: str, level: int) -> list:
    """
    Split markdown into exact heading-level blocks.
    
    `####` matches only level-4 headings, not `#####` or `######`.
    """
def _extract_section_map_from_body(block: str) -> dict:
    """Extract logical sections from a person-profile or roster record body."""
def _extract_named_sections(content: str, levels: tuple = (4,)) -> dict:
    """Collect heading blocks into a canonical section map."""
def infer_document_kind(filename: str, content: str) -> str:
    """Infer document type using both filename and content structure."""
def _extract_canonical_faculty(markdown_dir: str, dept_code: str = "ee") -> set:
    """Extract canonical faculty names from faculty-list file."""
```


### [graphrag/llm.py](file:///home/c3i/chatbot/graphrag/llm.py)
> LLM integration for GraphRAG with provider-selectable response generation.

#### `class OllamaLLM:`
  ```python
      def __init__(self, api_url: str = None, model: str = None, api_key: str = None):
      def generate(self, prompt: str, system_prompt: str = None,
                   temperature: float = 0.3, max_tokens: int = 1024) -> str:
      def __call__(self, prompt: str) -> str:
  ```

#### `class GeminiLLM:`
  ```python
      def __init__(
          self,
          api_key: str = None,
          model: str = None,
          api_base: str = None,
      ):
      def generate(
          self,
          prompt: str,
          system_prompt: str = None,
          temperature: float = 0.3,
          max_tokens: int = 1024,
      ) -> str:
      def __call__(self, prompt: str) -> str:
  ```

#### `class BedrockLLM:`
  ```python
      def __init__(
          self,
          api_key: str = None,
          model: str = None,
          region: str = None,
      ):
      def _set_model(self, model: str) -> None:
      def _candidate_inference_profile_models(self):
      def _should_retry_with_inference_profile(self, response) -> bool:
      def _retry_with_inference_profile(self, payload, headers):
      def _should_retry_with_fallback_model(self, response) -> bool:
      def _retry_with_fallback_model(self, payload, headers):
      def generate(
          self,
          prompt: str,
          system_prompt: str = None,
          temperature: float = 0.3,
          max_tokens: int = 1024,
      ) -> str:
      def __call__(self, prompt: str) -> str:
  ```

#### Top-level Functions
```python
def _build_bedrock_api_url(region: str, model: str) -> str:
def _derive_bedrock_inference_prefix(region: str) -> str:
def _summarize_http_error(response) -> str:
def get_llm_provider() -> str:
def get_system_prompt(dept_code: str = "ee") -> str:
def _clean_response_url(url: str) -> str:
    """Remove leaked HTML attributes from a URL-like response fragment."""
def sanitize_response(text: str) -> str:
    """Strip HTML noise from LLM responses while preserving useful content."""
def create_llm_from_env(provider: str = None, model: str = None):
    """Instantiate the configured LLM provider."""
def get_unified_system_prompt() -> str:
    """Get system prompt for the unified IIT Jammu chatbot (cross-department / broadcast mode)."""
def build_chat_prompt(query: str, context: str, dept_code: str = "ee") -> str:
def build_multi_dept_chat_prompt(query: str, dept_contexts: dict) -> str:
    """
    Build a chat prompt for cross-department queries.
    
    Args:
        query: The user's question.
        dept_contexts: Dict of dept_code → {"name": str, "context": str}.
    """
```


### [graphrag/multi_retriever.py](file:///home/c3i/chatbot/graphrag/multi_retriever.py)
> Cross-Department Retrieval Orchestrator for GraphRAG.
> 
> Coordinates retrieval across multiple department-scoped HybridRetriever
> instances. Supports:
>   - Single department delegation
>   - Multi-department merging with department headers
>   - Broadcast search across all departments with relevance ranking

#### `class MultiDepartmentRetriever:`
  > Orchestrates retrieval across 1+ department-scoped HybridRetrievers and SectionRetrievers.
  ```python
      def __init__(self, retrievers: Dict, section_retrievers: Dict = None):
      """
      Args:
          retrievers: Dict mapping dept_code → HybridRetriever instance.
          section_retrievers: Dict mapping sec_code → SectionRetriever instance.
      """
      def retrieve_single(self, query: str, dept_code: str) -> Dict[str, Any]:
      """Retrieve from a single department. Delegates directly."""
      def retrieve_multi(self, query: str, dept_codes: List[str]) -> Dict[str, Any]:
      """Retrieve from multiple departments/sections and merge contexts with headers."""
      def _is_topic_query(query: str) -> bool:
      """
      Detect broad subject/topic queries that need full cross-department coverage.
      
      Examples that should return True:
          - "Who teaches deep learning?"
          - "Who works on machine learning?"
          - "Faculty working on VLSI"
          - "Which professors do research in robotics?"
          - "Computer vision researchers at IIT Jammu"
          - "Experts in deep learning"
      """
      def _is_person_or_comparison_query(self, query: str) -> bool:
      """
      Detect queries about specific people or comparisons between people.
      
      Used to prevent admin-only direct answers from short-circuiting
      broadcast when richer faculty data is available from departments.
      """
      def retrieve_broadcast(self, query: str, top_n: int = 3) -> Dict[str, Any]:
      """Search ALL loaded departments and sections and return the top-N by relevance."""
      def _normalize_token(self, token: str) -> str:
      """Apply lightweight normalization for keyword coverage checks."""
      def _extract_query_focus_terms(self, query: str) -> List[str]:
      """
      Extract important focus keywords from query, ignoring all stopwords/generic terms.
      
      Handles compound research phrases (e.g. "computer vision", "deep learning")
      by scanning for known multi-word topics BEFORE splitting into individual
      words. This prevents stop-word filtering on individual component words
      (e.g. "computer" in "computer vision") from erasing the research topic.
      """
      def _is_bundle_relevant(self, query: str, bundle: Dict[str, Any], is_topic: bool) -> bool:
      """Determine if a department bundle has genuinely relevant information."""
  ```


### [graphrag/person_index.py](file:///home/c3i/chatbot/graphrag/person_index.py)
> Global Person Index for IIT Jammu Unified Chatbot.
> Cross-references and resolves individuals across all loaded department and section graphs
> to ensure factual consistency and complete role mapping.

#### `class GlobalPersonIndex:`
  ```python
      def __init__(self):
      def _is_role_placeholder(cls, name: str) -> bool:
      """Return True if *name* looks like a committee-role slot, not a person."""
      def index_graph(self, graph, source_name: str, is_section: bool = False):
      """Extract person entities from a graph and index them."""
      def lookup(self, name_query: str) -> dict:
      """Look up all roles for a given name query, using entity resolver logic."""
  ```


### [graphrag/retriever.py](file:///home/c3i/chatbot/graphrag/retriever.py)
> Hybrid Retrieval Engine for GraphRAG.
> Combines entity search, vector search, and community summaries
> into a clean, structured context for the LLM.

#### `class HybridRetriever:`
  ```python
      def __init__(self, graph: nx.DiGraph, embedding_engine, community_reports: List[Dict], dept_code: str = "ee"):
      def _build_indexes(self):
      """Pre-build lookup indexes for fast entity search."""
      def _normalize_token(self, token: str) -> str:
      """Apply lightweight normalization for keyword coverage checks."""
      def _query_tokens(self, query: str) -> List[str]:
      """Tokenize a query into normalized keywords."""
      def _is_department_contact_query(self, query: str) -> bool:
      """Detect generic department contact / point-of-contact questions."""
      def _get_hod_member(self) -> Optional[Dict[str, str]]:
      """Return the department HoD if present in the faculty roster."""
      def _build_department_contact_answer(self) -> Optional[str]:
      """Build a deterministic answer for generic department contact requests."""
      def _is_lab_query(self, query: str) -> bool:
      """Detect queries about department labs/laboratories/facilities."""
      def _build_labs_answer(self, query: Optional[str] = None) -> Optional[str]:
      """Build a deterministic answer listing all labs in the department."""
      def _is_address_query(self, query: str) -> bool:
      """Detect queries about department address, location, or contact info."""
      def _build_address_answer(self) -> Optional[str]:
      """Build a deterministic answer with department address/contact info."""
      def _is_graduated_phd_query(self, query: str) -> bool:
      """Detect queries about graduated/passed-out PhD students or PhD alumni."""
      def _build_graduated_phd_answer(self) -> Optional[str]:
      """Build a deterministic answer listing graduated PhD students."""
      def _is_alumni_query(self, query: str) -> bool:
      """Detect queries about department alumni."""
      def _build_alumni_answer(self) -> Optional[str]:
      """Build a deterministic answer listing alumni information."""
      def _build_administration_answer(self, query: str) -> Optional[str]:
      def _is_broad_reasoning_query(self, query: str) -> bool:
      """Relax strict evidence gating for synthesis-heavy prompts."""
      def _infer_query_concepts(self, query: str) -> List[str]:
      """Identify concrete concepts that should be explicitly supported by evidence."""
      def _concept_supported_by_graph(self, concept: str) -> bool:
      """Check whether the department graph contains the concept structurally."""
      def _build_unavailable_response(self, query: str, reason: Optional[str] = None) -> str:
      """Return a safe, department-scoped unavailable-information response."""
      def _build_provenance(
          self,
          direct: bool,
          local_results: List[Dict[str, Any]],
          vector_results: List[Dict[str, Any]],
          global_results: List[Dict[str, Any]],
          section_word_counts: Dict[str, int],
      ) -> Dict[str, Any]:
      """Summarize which retrieval channels contributed evidence."""
      def _assess_answerability(
          self,
          query: str,
          local_results: List[Dict[str, Any]],
          vector_results: List[Dict[str, Any]],
          global_results: List[Dict[str, Any]],
          context: str,
      ) -> Dict[str, Any]:
      """Decide whether the retrieved evidence can safely support an answer."""
      def _is_faculty_roster_query(self, query: str) -> bool:
      """Detect department-level faculty count/list requests."""
      def _get_faculty_structured_fields(self) -> List[str]:
      """Return non-empty structured fields available on faculty nodes."""
      def _extract_requested_faculty_attribute(self, query: str) -> Optional[str]:
      """Map analytic faculty questions to a structured attribute."""
      def _is_faculty_analytics_query(self, query: str) -> bool:
      """Detect faculty analytics/breakdown queries that should be answered structurally."""
      def _normalize_designation(self, designation: str) -> str:
      """Collapse free-form designations into stable analytic buckets."""
      def _compute_faculty_breakdown(self, attribute: str) -> Dict[str, int]:
      """Compute deterministic faculty breakdowns for supported structured attributes."""
      def _faculty_listing_url(self) -> str:
      """Return the canonical faculty listing page for the department."""
      def _build_faculty_analytics_answer(self, query: str) -> Optional[str]:
      """Answer faculty analytics queries without falling through to fuzzy retrieval/LLM counting."""
      def _query_has_count_intent(self, query: str) -> bool:
      def _query_has_list_intent(self, query: str) -> bool:
      def _is_exact_count_query(self, query: str) -> bool:
      """Detect queries where approximate community summaries should not drive the answer."""
      def _is_phd_roster_query(self, query: str) -> bool:
      """Detect department-level PhD scholar/student count/list requests."""
      def _is_mtech_roster_query(self, query: str) -> bool:
      """Detect department-level M.Tech student count/list requests."""
      def _extract_supervisor_query_name(self, query: str) -> Optional[str]:
      """Extract the student name from supervisor/advisor style questions."""
      def _extract_research_area_query_name(self, query: str) -> Optional[str]:
      """Extract the student name from research area style questions."""
      def _extract_email_query_name(self, query: str) -> Optional[str]:
      """Extract the entity name from email/contact style questions."""
      def _extract_supervisor_from_students_query(self, query: str) -> Optional[str]:
      """Extract the supervisor's name from query seeking their students."""
      def _find_entity_by_name(self, raw_name: str, allowed_labels: Optional[Tuple[str, ...]] = None) -> Optional[str]:
      """Resolve an entity name to a graph node id using exact then fuzzy matching."""
      def get_faculty_roster(self) -> List[Dict]:
      """Return the authoritative department faculty roster from graph nodes."""
      def get_phd_roster(self) -> List[Dict]:
      """Return the authoritative PhD scholar roster from graph nodes and supervision edges."""
      def _faculty_roster_context(self) -> str:
      """Build a complete roster context block for the LLM."""
      def _phd_roster_context(self) -> str:
      """Build a complete PhD roster context block for deterministic answering."""
      def get_mtech_roster(self) -> List[Dict]:
      """Return the authoritative M.Tech student roster from graph nodes and supervision edges."""
      def _mtech_roster_context(self) -> str:
      """Build a complete M.Tech roster context block for deterministic answering."""
      def get_direct_answer(self, query: str, suppress_topic_match: bool = False) -> Optional[str]:
      """
      Return deterministic answers for questions that should not rely on LLM inference.
      
      Args:
          query: The user's question.
          suppress_topic_match: If True, skip topic/expert contact matching.
              Used during broadcast to prevent a single department from
              short-circuiting cross-department topic queries.
      """
      def _get_node_display(self, node_id: str) -> str:
      """Get a clean display string for a node."""
      def _get_relationships_display(self, node_id: str) -> str:
      """Get formatted relationships for a node."""
      def _name_match(self, query: str) -> List[str]:
      """Find entities whose names appear in the query using advanced token-based fuzzy matching."""
      def _is_placement_query(self, query: str) -> bool:
      """Detect placement/salary/higher studies queries."""
      def _placement_context(self) -> str:
      """Build structured placement context from graph nodes."""
      def _find_supervisors_by_research_area(self, query: str) -> List[Dict]:
      """
      Find supervisors of PhD students whose research area matches query keywords.
      
      Uses word-boundary-aware matching via _topic_matches_text() to prevent
      false positives from short keywords appearing as substrings in unrelated
      words (e.g. 'ai' matching 'uncertainty').
      """
      def _local_search(self, query: str, top_k: int = 6) -> List[Dict]:
      """Entity search: name matching first, then embedding fallback."""
      def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
      """Semantic chunk search with minimum score threshold."""
      def _global_search(self, query: str, top_k: int = 3) -> List[Dict]:
      """Community summary search."""
      def retrieve(self, query: str, local_top_k: int = 6, vector_top_k: int = 4,
                   global_top_k: int = 2, max_context_words: int = 3000) -> str:
      """Run full hybrid retrieval and return clean formatted context."""
      def retrieve_bundle(
          self,
          query: str,
          local_top_k: int = 6,
          vector_top_k: int = 4,
          global_top_k: int = 2,
          max_context_words: int = 3000,
      ) -> Dict[str, Any]:
      """Run full hybrid retrieval and return context plus provenance metadata."""
  ```

#### Top-level Functions
```python
def _topic_matches_text(topic: str, text: str) -> bool:
    """
    Check if a research topic matches against text using word-boundary-aware matching.
    
    For short topics (<=4 chars like 'ai', 'nlp', 'iot'), uses word-boundary regex
    to prevent substring collisions (e.g. 'ai' matching 'uncertainty').
    For longer topics (>=5 chars like 'machine learning'), uses substring match.
    Also expands known abbreviations (e.g. 'ai' also searches 'artificial intelligence').
    """
def load_retriever(dept_code: str = "ee", data_dir: str = None) -> HybridRetriever:
    """Load all components and create a HybridRetriever."""
```


### [graphrag/rules_db.py](file:///home/c3i/chatbot/graphrag/rules_db.py)
#### `class RulesDB:`
  ```python
      def __init__(self, db_path: str = DB_PATH):
      def get_connection(self):
      def init_db(self):
      """Initialise database schema and tables."""
      def insert_section(self, section_id: str, section_number: str, title: str, full_text: str,
                         parent_id: Optional[str], program: str, source_file: str,
                         last_amended: Optional[str] = None, amendment_note: Optional[str] = None):
      """Insert or replace a rule section."""
      def insert_fact(self, fact_type: str, fact_key: str, fact_value: str, operator: Optional[str],
                      condition_text: Optional[str], section_id: str, program: str):
      """Insert a rule fact."""
      def insert_grade(self, grade: str, grade_point: int, description: Optional[str]):
      """Insert or replace a grade scale item."""
      def insert_credit_requirement(self, program: str, category: str, category_full: Optional[str],
                                    min_credits: float, percentage: Optional[float], notes: Optional[str]):
      """Insert a credit requirement."""
      def insert_program_milestone(self, program: str, milestone: str, deadline: Optional[str], details: Optional[str], section_id: Optional[str]):
      """Insert a program milestone."""
      def lookup_fact(self, fact_key: str, program: Optional[str] = None) -> List[Dict]:
      """Lookup facts by key and program."""
      def get_section(self, section_id: str) -> Optional[Dict]:
      """Retrieve a specific section by its identifier."""
      def get_section_with_children(self, section_id: str) -> List[Dict]:
      """Retrieve a section and all its children/subsections."""
      def search_sections(self, query: str, program: Optional[str] = None, limit: int = 5) -> List[Dict]:
      """Perform full-text search over rule sections."""
      def get_all_sections(self, program: Optional[str] = None) -> List[Dict]:
      """Retrieve all rule sections, optionally scoped to one program."""
      def get_grade_scale(self) -> List[Dict]:
      """Retrieve the grade scale mapping."""
      def get_credit_requirements(self, program: str) -> List[Dict]:
      """Retrieve credit requirements for a given program."""
      def get_program_milestones(self, program: str) -> List[Dict]:
      """Retrieve program milestones for a given program."""
      def clear_all(self):
      """Clear all tables in the database."""
  ```


### [graphrag/rules_parser.py](file:///home/c3i/chatbot/graphrag/rules_parser.py)
#### `class RulesParser:`
  ```python
      def __init__(self):
      def parse_sec_num(self, s: str) -> Optional[Tuple[int, ...]]:
      """Convert section number string to tuple of ints for hierarchy comparisons."""
      def is_toc_line(self, line: str) -> bool:
      """Check if a line looks like it belongs to Table of Contents."""
      def clean_text(self, text: str) -> str:
      """Clean running headers and footer text."""
      def _recover_missing_top_level_sections(
          self,
          content_lines: List[str],
          sections: List[Dict[str, Any]],
          program: str,
          source_file: str,
      ) -> List[Dict[str, Any]]:
      """Recover mixed-case top-level sections skipped after PDF page headers."""
      def _recover_mtech_appendix_sections(
          self,
          content_lines: List[str],
          sections: List[Dict[str, Any]],
          program: str,
          source_file: str,
      ) -> List[Dict[str, Any]]:
      """Recover M.Tech appendix sections that OCR/parser rules swallow into section 13."""
      def parse_file(self, filepath: str, program: str) -> List[Dict[str, Any]]:
      """Parse a rules markdown file into hierarchical sections."""
  ```


### [graphrag/rules_retriever.py](file:///home/c3i/chatbot/graphrag/rules_retriever.py)
#### `class RulesRetriever:`
  ```python
      def __init__(self, db: Optional[RulesDB] = None):
      def _normalize_token(self, token: str) -> str:
      def _canonical_query(self, query: str) -> str:
      def _compact_code_text(self, text: str) -> str:
      def _course_code_needles(self, query: str) -> List[str]:
      """
      Return compact course-code-like strings from user text.
      
      Academic PDFs contain OCR variants such as "M AL055P4I" while users may
      type "M AL055P 4I"; compact comparison lets both resolve to the same
      example without making generic token overlap dominate retrieval.
      """
      def _expanded_terms_and_phrases(self, query: str) -> Tuple[List[str], List[str]]:
      def classify_intent(self, query: str) -> Dict[str, Any]:
      """Classify target program and intent based on query keywords."""
      def _safe_fts_queries(self, query: str) -> List[str]:
      def _section_tokens(self, text: str) -> set:
      def _rank_sections(self, query: str, program: Optional[str], limit: int) -> List[Dict[str, Any]]:
      def retrieve(self, query: str, limit: int = 4) -> Dict[str, Any]:
      """Perform hybrid retrieval: structured lookup + full-text search."""
      def generate_context(self, retrieval_results: Dict[str, Any]) -> str:
      """Format the retrieved results into a readable context block for the LLM."""
  ```


### [graphrag/section_kg_builder.py](file:///home/c3i/chatbot/graphrag/section_kg_builder.py)
> Section Knowledge Graph Builder for IIT Jammu Sections.
> Parses section markdown files and constructs a NetworkX DiGraph with entities,
> relationships, and clean text chunks.

#### `class SectionKGBuilder:`
  ```python
      def __init__(self, section_code: str, markdown_dir: str = None):
      def _add_node(self, node_id: str, label: str, **properties):
      def _add_edge(self, source: str, target: str, rel_type: str, **properties):
      def _create_document_node(self, filename: str, content: str) -> str:
      def _parse_people_list(self, filename: str, content: str, doc_id: str):
      """
      Parse the academics_people-list.md and accounts/e2 people lists.
      Format:
      #### Ajay Singh
      ##### **Designation**
      ###### Associate Dean: Curriculum (PG)
      ##### adpg.acad@iitjammu.ac.in
      """
      def _parse_counselling_team(self, filename: str, content: str, doc_id: str):
      """Parse counselling team tables from counselling_team.md."""
      def _parse_counselor_profiles(self, filename: str, content: str, doc_id: str):
      """Parse bio information from counselling_know-your-counselors.md."""
      def _parse_di_team(self, filename: str, content: str, doc_id: str):
      """Parse team hierarchy from di_team.html.md."""
      def _parse_section_contact(self, filename: str, content: str, doc_id: str):
      """Parse contact details from contact markdown files."""
      def _parse_hod_message(self, filename: str, content: str, doc_id: str):
      """Parse HOD/Dean message file to extract head info."""
      def _parse_curriculum_document(self, filename: str, content: str, doc_id: str):
      """Parse academic curriculum and specialization documents."""
      def build(self) -> nx.DiGraph:
      def _parse_alumni_medalists(self, filename: str, content: str, doc_id: str):
      def _parse_alumni_awards(self, filename: str, content: str, doc_id: str):
      def _parse_alumni_contacts(self, filename: str, content: str, doc_id: str):
      def _parse_past_recruiters(self, filename: str, content: str, doc_id: str):
      def _parse_placement_policy(self, filename: str, content: str, doc_id: str):
      def _parse_aipc_guidelines(self, filename: str, content: str, doc_id: str):
      def _parse_rise_up_details(self, filename: str, content: str, doc_id: str):
      def _parse_cds_contact(self, filename: str, content: str, doc_id: str):
      def _parse_placement_stats(self, filename: str, content: str, doc_id: str):
      def _parse_ir_team(self, filename: str, content: str, doc_id: str):
      def _parse_ir_contact(self, filename: str, content: str, doc_id: str):
      def _parse_mous(self, filename: str, content: str, doc_id: str):
      def _parse_clubs(self, filename: str, content: str, doc_id: str):
      def _parse_sports(self, filename: str, content: str, doc_id: str):
      def _parse_hostels(self, filename: str, content: str, doc_id: str):
      def _parse_fests(self, filename: str, content: str, doc_id: str):
      def _parse_medical_about(self, filename: str, content: str, doc_id: str):
      def _parse_medical_doctors(self, filename: str, content: str, doc_id: str):
      def _parse_medical_collaborations(self, filename: str, content: str, doc_id: str):
      def _parse_medical_contact(self, filename: str, content: str, doc_id: str):
      def _parse_medical_services(self, filename: str, content: str, doc_id: str):
      def _parse_osd_team(self, filename: str, content: str, doc_id: str):
      def _parse_osd_uba(self, filename: str, content: str, doc_id: str):
      def _parse_osd_ces(self, filename: str, content: str, doc_id: str):
      def _parse_osd_events(self, filename: str, content: str, doc_id: str):
      def _parse_osd_contact(self, filename: str, content: str, doc_id: str):
      def save(self, output_dir: str = None):
      def load(data_dir: str):
  ```

#### Top-level Functions
```python
def create_section_entity_descriptions(graph, section_code: str) -> list:
```


### [graphrag/section_retriever.py](file:///home/c3i/chatbot/graphrag/section_retriever.py)
> Section Retriever Engine for GraphRAG.
> Handles Academics, Accounts, Counselling, DI, and E2 sections with direct deterministic answers
> and fallback semantic chunk search.

#### `class SectionRetriever:`
  ```python
      def __init__(self, section_code: str, graph: nx.DiGraph, chunks: List[Dict], embedding_engine=None):
      def _build_provenance(
          self,
          direct: bool,
          local_results: List[Dict[str, Any]],
          vector_results: List[Dict[str, Any]],
          section_word_counts: Dict[str, int],
      ) -> Dict[str, Any]:
      """Summarize which retrieval channels contributed evidence."""
      def get_direct_answer(self, query: str, global_person_index=None) -> Optional[str]:
      """Check for direct, deterministic answers based on section-specific entities."""
      def retrieve_bundle(
          self,
          query: str,
          local_top_k: int = 5,
          vector_top_k: int = 5,
          global_top_k: int = 3,
          max_context_words: int = 4500,
      ) -> Dict[str, Any]:
      """Retrieve relevant context for a section query. Falls back to BM25/vector search over chunks."""
  ```


### [graphrag/verifier.py](file:///home/c3i/chatbot/graphrag/verifier.py)
> Post-Generation Faithfulness Verifier for GraphRAG.
> 
> Layer 4 of the hallucination defense. After the LLM generates a response,
> this module verifies that factual claims in the response are grounded in
> the retrieved context. Unsupported claims are stripped or the response is
> replaced with an "I don't know" fallback.
> 
> Uses the same configured LLM provider unless verification is disabled.

#### `class ClaimVerification:`
  > A single factual claim and its verification status.

#### `class VerificationResult:`
  > Result of faithfulness verification.

#### `class ResponseVerifier:`
  > Post-generation hallucination detector.
  > 
  > Uses the LLM itself to check if its response is grounded
  > in the retrieved context. This catches:
  > - Fabricated names/emails/designations
  > - Invented counts or statistics
  > - Facts merged from different entities
  ```python
      def __init__(self, llm):
      """
      Args:
          llm: The active LLM instance for verification prompts.
      """
      def verify(self, query: str, context: str, response: str) -> VerificationResult:
      """
      Verify that the LLM response is grounded in the retrieved context.
      
      Only verifies factoid queries (who/what/how many/list/name).
      Broad reasoning queries skip verification.
      
      Args:
          query: The original user query.
          context: The retrieved context that was fed to the LLM.
          response: The LLM-generated response.
      
      Returns:
          VerificationResult with faithful status and cleaned response.
      """
      def _is_factoid_query(self, query: str) -> bool:
      """Determine if a query is factoid (requires verification) vs. broad reasoning."""
      def _run_verification(self, query: str, context: str, response: str) -> VerificationResult:
      """Run the actual LLM-based claim verification."""
      def _parse_verification_response(self, raw: str) -> tuple:
      """Parse the JSON output from the verification LLM call."""
  ```

#### Top-level Functions
```python
def is_verification_enabled() -> bool:
```


### [scratch/generate_repomap.py](file:///home/c3i/chatbot/scratch/generate_repomap.py)
#### Top-level Functions
```python
def clean_signature(sig_text):
    """Strip trailing comment lines, empty lines, or docstring starts from a signature block."""
def extract_signature(lines, node):
    """Extract signature lines of a class or function from source lines."""
def parse_python_file(filepath, rel_path):
    """Parse a python file and return structured class/function/docstring info."""
def generate_directory_tree(startpath, exclude_dirs):
    """Generate a clean text-based directory tree."""
def main():
```


### [scratch/test_broadcast.py](file:///home/c3i/chatbot/scratch/test_broadcast.py)
#### Top-level Functions
```python
def main():
def re_findall_words(text):
```


### [scratch/test_retrieval_queries.py](file:///home/c3i/chatbot/scratch/test_retrieval_queries.py)
#### Top-level Functions
```python
def test_query(retriever, query):
def main():
```


### [scratch/test_verify.py](file:///home/c3i/chatbot/scratch/test_verify.py)
#### Top-level Functions
```python
def test_queries():
```


### [scratch/verify_academics_kg.py](file:///home/c3i/chatbot/scratch/verify_academics_kg.py)
#### Top-level Functions
```python
def main():
```


### [scripts/convert_pdfs_to_md.py](file:///home/c3i/chatbot/scripts/convert_pdfs_to_md.py)
> PDF & Excel → Markdown Converter for IIT Jammu Chatbot
> =======================================================
> Handles:
>   - Digitally-typed PDFs  → pdfplumber text + table extraction
>   - Mobile-scanned PDFs   → OCR via pytesseract (pdf2image + tesseract)
>   - Excel (.xlsx) files   → openpyxl → markdown tables
>   
> Architecture:
>   1. Classify each file (digital vs scanned vs excel)
>   2. Extract text using the best strategy
>   3. Detect and render tables as markdown
>   4. Post-process: clean up, add metadata headers
>   5. Write structured .md files preserving directory hierarchy

#### Top-level Functions
```python
def classify_pdf(path: str) -> str:
    """Return 'digital', 'scanned', or 'mixed'."""
def extract_table_as_md(table: list) -> str:
    """Convert a pdfplumber table (list of lists) to markdown table."""
def extract_digital_pdf(path: str) -> str:
    """Extract text + tables from a digitally-typed PDF."""
def preprocess_image(img: Image.Image) -> Image.Image:
    """Enhance scanned image for better OCR accuracy."""
def ocr_page_image(img: Image.Image, page_num: int) -> str:
    """Run OCR on a single page image."""
def extract_scanned_pdf(path: str) -> str:
    """Extract text from scanned PDF via OCR."""
def extract_mixed_pdf(path: str) -> str:
    """Handle PDFs with both digital and scanned pages."""
def extract_excel(path: str) -> str:
    """Convert Excel file to markdown tables."""
def clean_text(text: str) -> str:
    """Clean extracted text for chatbot consumption."""
def generate_metadata_header(
    source_path: str,
    doc_type: str,
    category: str,
    subcategory: str,
) -> str:
    """Generate YAML front-matter for the markdown file."""
def derive_title(filename: str) -> str:
    """Derive a clean title from the filename."""
def determine_category(rel_path: str) -> Tuple[str, str]:
    """Determine category and subcategory from relative path."""
def make_output_path(input_path: str, base_input: str, output_dir: str) -> str:
    """Create output .md path preserving directory structure."""
def process_file(
    input_path: str,
    base_input: str,
    output_dir: str,
    force: bool = False,
) -> Optional[Dict]:
    """Process a single file and return a report dict."""
def run_pipeline(input_dir: str, output_dir: str, force: bool = False):
    """Run the full conversion pipeline."""
```


### [scripts/download_academics_pdfs.py](file:///home/c3i/chatbot/scripts/download_academics_pdfs.py)
> Download and parse all Google Drive PDFs/documents from the IIT Jammu
> Academics section.
> 
> Categories handled:
>   1. Rules & Regulations  (PG / M.Tech / Ph.D / UG)
>   2. General Downloads
>   3. Specialisation & Courses  (PG / UG)
>   4. Academic Notifications
> 
> Outputs beautifully formatted Markdown files organised into subfolders
> under  scraped_data/sections/academics/parsed_documents/

#### Top-level Functions
```python
def extract_file_id(url: str) -> str:
    """Extract Google Drive file id from various URL formats."""
def extract_spreadsheet_id(url: str) -> str:
    """Extract Google Spreadsheet id."""
def extract_folder_id(url: str) -> str:
    """Extract Google Drive folder id."""
def sanitise_filename(name: str) -> str:
    """Create a filesystem-safe filename from a document title."""
def download_gdrive_file(file_id: str, dest: Path, quiet: bool = True) -> bool:
    """Download a file from Google Drive using gdown. Returns True on success."""
def download_spreadsheet_as_csv(sheet_id: str, dest: Path) -> bool:
    """Download a Google Spreadsheet as CSV."""
def list_folder_files(folder_id: str) -> list:
    """
    Use gdown to list files in a public Google Drive folder.
    Returns list of dicts with 'id' and 'name'.
    """
def detect_file_type(path: Path) -> str:
    """Detect file type from magic bytes."""
def pdf_to_markdown(pdf_path: Path, title: str, source_url: str) -> str:
    """Parse PDF using pdfplumber and produce beautifully formatted markdown."""
def format_table(table: list, idx: int) -> str:
    """Format an extracted table as a Markdown table."""
def clean_text(text: str) -> str:
    """Clean extracted text: normalise whitespace, preserve paragraph breaks."""
def csv_to_markdown(csv_path: Path, title: str, source_url: str) -> str:
    """Convert downloaded CSV into markdown table."""
def text_to_markdown(text_path: Path, title: str, source_url: str) -> str:
    """Wrap a plain text file in markdown."""
def extract_links_from_md(md_path: Path) -> list:
    """
    Parse a scraped markdown file and extract all Google Drive links
    with their display names and hierarchy context.
    Returns list of dicts: {title, url, type, category_path}
    """
def process_link(link: dict, output_dir: Path, stats: dict) -> str:
    """Download and parse a single link. Returns output md path or None."""
def main():
def write_index(manifest: dict):
    """Write a combined index markdown file for all parsed documents."""
```


### [scripts/download_gdrive_pass2.py](file:///home/c3i/chatbot/scripts/download_gdrive_pass2.py)
> Second-pass Google Drive downloader using multiple strategies:
> 1. Direct download with confirm=t parameter
> 2. Wget with --no-check-certificate 
> 3. curl with cookie handling
> 
> Then re-parse all successfully downloaded PDFs.

#### Top-level Functions
```python
def detect_file_type(path):
def sanitise_filename(name):
def clean_text(text):
def format_table(table, idx):
def pdf_to_markdown(pdf_path, title, source_url):
def download_with_curl(file_id, dest_path):
    """Use curl with proper cookie handling to download from Google Drive."""
def main():
def reparse_documents():
    """Re-parse all documents, replacing stubs with real content."""
def generate_final_stats():
    """Print comprehensive statistics."""
```


### [scripts/download_gdrive_playwright.py](file:///home/c3i/chatbot/scripts/download_gdrive_playwright.py)
> Download Google Drive files using Playwright to handle the confirmation
> pages. Parses the downloaded PDFs into structured Markdown.
> 
> This is a second-pass script that processes files which the initial
> gdown/requests pass couldn't download (saved as HTML interstitials).

#### Top-level Functions
```python
def detect_file_type(path):
    """Detect file type from magic bytes."""
def sanitise_filename(name):
    """Create a filesystem-safe filename from a document title."""
def clean_text(text):
    """Clean extracted text."""
def format_table(table, idx):
    """Format an extracted table as Markdown."""
def pdf_to_markdown(pdf_path, title, source_url):
    """Parse PDF into beautiful Markdown."""
def get_html_file_ids():
    """Find all cached files that are HTML (need re-downloading)."""
def download_batch_with_playwright(file_ids, batch_size=5):
    """
    Use Playwright to download a batch of Google Drive files.
    Opens browser, navigates to the file view page, and clicks download.
    """
def reparse_all_documents():
    """
    After re-downloading, re-parse all documents and update the
    markdown files.
    """
def main():
```


### [scripts/flatten_curriculum_docs.py](file:///home/c3i/chatbot/scripts/flatten_curriculum_docs.py)
#### Top-level Functions
```python
def main():
```


### [scripts/ingest_rules.py](file:///home/c3i/chatbot/scripts/ingest_rules.py)
#### Top-level Functions
```python
def main():
```


### [tests/conftest.py](file:///home/c3i/chatbot/tests/conftest.py)
> Shared pytest fixtures for the IIT Jammu EE GraphRAG chatbot tests.

#### Top-level Functions
```python
def canonical_faculty():
    """The 24 canonical faculty names."""
def built_graph():
    """Build a fresh knowledge graph for testing."""
def graph(built_graph):
def chunks(built_graph):
def resolver(built_graph):
```


### [tests/test_chunk_quality.py](file:///home/c3i/chatbot/tests/test_chunk_quality.py)
> Tests for chunk data quality — ensures chunks are clean and free of noise.

#### `class TestChunkCleanliness:`
  > Verify chunks don't contain boilerplate or HTML noise.
  ```python
      def test_no_source_url_boilerplate(self, chunks):
      """No chunk should contain '# Source URL:' header."""
      def test_no_image_references(self, chunks):
      """No chunk should contain markdown image references."""
      def test_no_html_noise(self, chunks):
      """No chunk should contain raw HTML tags or attributes."""
      def test_minimum_chunk_length(self, chunks):
      """All chunks should have meaningful content (>30 chars)."""
      def test_no_navigation_breadcrumbs(self, chunks):
      """No chunk should contain navigation breadcrumb patterns."""
      def test_chunks_have_metadata(self, chunks):
      """All chunks should have proper metadata."""
      def test_no_markdown_links_in_chunks(self, chunks):
      """Chunks should not contain raw markdown link syntax (should be stripped)."""
  ```

#### `class TestStructuredRosterChunking:`
  ```python
      def test_repeated_roster_entries_are_chunked_on_record_boundaries(self):
      """Roster pages should be chunked by `####` records, not raw word windows."""
      def test_repeated_roster_entries_work_for_other_heading_levels(self):
      """Generic roster detection should also work when records use `###` headings."""
      def test_table_blocks_are_preserved_when_chunking(self):
      """Adjacent markdown table rows should remain together in structural chunking."""
  ```

#### `class TestGenericExtractionHelpers:`
  ```python
      def test_obfuscated_email_helper_normalizes_common_patterns(self):
      def test_block_section_extractor_handles_inline_labels(self):
      def test_document_kind_inference_uses_content_and_filename(self):
  ```


### [tests/test_crawler_architecture.py](file:///home/c3i/chatbot/tests/test_crawler_architecture.py)
> Unit tests for crawler URL normalization and department resolution.

#### Top-level Functions
```python
def test_resolve_department_alias():
def test_markdown_dir_prefers_scraped_data_when_present(tmp_path, monkeypatch):
def test_canonicalize_root_relative_department_href_from_nested_page():
def test_canonicalize_tilde_profile_href_from_listing_page():
def test_canonicalize_strips_nonbreaking_space_and_index_html():
def test_classify_department_page_and_binary():
def test_generic_page_detection():
def test_section_url_matching():
```


### [tests/test_cse_faculty_profile_parsing.py](file:///home/c3i/chatbot/tests/test_cse_faculty_profile_parsing.py)
> Regression tests for CSE faculty profile ingestion.

#### `class TestCSEFacultyProfiles:`
  ```python
      def test_obfuscated_emails_are_normalized(self, cse_graph):
      def test_link_wrapped_sections_are_extracted(self, cse_graph):
      def test_department_records_available_faculty_schema(self, cse_graph):
  ```

#### Top-level Functions
```python
def cse_graph():
```


### [tests/test_cse_phd_email_retrieval.py](file:///home/c3i/chatbot/tests/test_cse_phd_email_retrieval.py)
> Regression tests for CSE PhD scholar email ingestion and retrieval.

#### `class TestCSEPhDEmails:`
  ```python
      def test_phd_emails_are_extracted_into_graph(self, cse_graph):
      def test_named_email_query_is_answered_directly(self, cse_graph):
      def test_phd_roster_context_can_include_email(self, cse_graph):
  ```

#### Top-level Functions
```python
def cse_graph():
```


### [tests/test_data_integrity.py](file:///home/c3i/chatbot/tests/test_data_integrity.py)
> Tests for data integrity — ensures the knowledge graph accurately represents the source data.

#### `class TestFacultyCount:`
  > Verify the exact faculty count matches the authoritative source.
  ```python
      def test_exactly_24_faculty_nodes(self, graph, canonical_faculty):
      """The graph must have exactly 24 Faculty nodes (matching ee_faculty-list.html.md)."""
      def test_all_canonical_names_present(self, graph, canonical_faculty):
      """Every one of the 24 known faculty names must exist as a graph node."""
      def test_no_external_persons_labeled_as_faculty(self, graph, canonical_faculty):
      """No ExternalPerson should be labeled as Faculty."""
  ```

#### `class TestFacultyAttributes:`
  > Verify each faculty node has essential attributes.
  ```python
      def test_all_faculty_have_email(self, graph, canonical_faculty):
      """Every canonical faculty should have an email address."""
      def test_all_faculty_have_designation(self, graph, canonical_faculty):
      """Every canonical faculty should have a designation."""
      def test_faculty_connected_to_department(self, graph, canonical_faculty):
      """Every faculty should have a MEMBER_OF edge to the department."""
  ```

#### `class TestDepartmentNode:`
  > Verify the department node has correct metadata.
  ```python
      def test_department_exists(self, graph):
      def test_department_faculty_count(self, graph):
      """Department node should store the correct faculty count."""
  ```

#### `class TestPhDStudents:`
  > Verify PhD students are correctly parsed.
  ```python
      def test_phd_students_exist(self, graph):
      def test_exactly_66_phd_students(self, graph):
      """The graph should reflect the full current PhD roster page."""
      def test_phd_students_have_supervisors(self, graph):
      """At least some PhD students should have SUPERVISED_BY edges."""
      def test_department_phd_count(self, graph):
      """Department node should store the exact PhD scholar count."""
  ```

#### `class TestExternalPersons:`
  > Verify external collaborators are correctly separated from faculty.
  ```python
      def test_external_persons_exist(self, graph):
      """There should be ExternalPerson nodes for non-IIT-Jammu supervisors."""
      def test_external_persons_not_in_department(self, graph):
      """ExternalPerson nodes should NOT have MEMBER_OF edges to the department."""
  ```


### [tests/test_department_eval_pipeline.py](file:///home/c3i/chatbot/tests/test_department_eval_pipeline.py)
#### Top-level Functions
```python
def test_allocate_category_counts_covers_all_categories():
def test_extract_json_payload_handles_markdown_fences():
def test_summarize_results_uses_half_credit_for_partial():
```


### [tests/test_entity_resolution.py](file:///home/c3i/chatbot/tests/test_entity_resolution.py)
> Tests for entity resolution — ensures name variants map to canonical names.

#### `class TestNormalizeName:`
  ```python
      def test_strips_dr_prefix(self):
      def test_strips_prof_prefix(self):
      def test_strips_assistant_professor(self):
      def test_capitalizes_words(self):
      def test_strips_extra_whitespace(self):
  ```

#### `class TestInitialsMatch:`
  ```python
      def test_b_n_subudhi_matches_badri(self):
      def test_a_bansal_matches_ankur(self):
      def test_non_matching_initials(self):
      def test_different_last_name(self):
      def test_short_name_no_crash(self):
  ```

#### `class TestEntityResolver:`
  ```python
      def test_resolves_exact_canonical(self):
      def test_resolves_abbreviated_name(self):
      def test_resolves_partial_name(self):
      def test_resolves_optional_middle_name(self):
      def test_is_canonical_faculty_true(self):
      def test_is_canonical_faculty_false_for_external(self):
      def test_is_canonical_via_initials(self):
  ```

#### `class TestCanonicalFacultyExtraction:`
  ```python
      def test_extracts_24_faculty(self):
      def test_extracts_cse_faculty_from_plain_headings(self):
  ```

#### `class TestCSEGraphParsing:`
  ```python
      def test_vinit_is_faculty_not_student(self, cse_graph):
      def test_vinit_supervises_expected_students(self, cse_graph):
      def test_direct_answer_returns_students_under_vinit(self, cse_graph):
  ```

#### Top-level Functions
```python
def cse_graph():
```


### [tests/test_response_quality.py](file:///home/c3i/chatbot/tests/test_response_quality.py)
> Tests for response sanitization — ensures no HTML noise in chatbot responses.

#### `class TestSanitizeResponse:`
  ```python
      def test_strips_target_blank(self):
      def test_strips_rel_noopener(self):
      def test_strips_combined_attributes(self):
      def test_converts_html_anchor_to_markdown(self):
      def test_strips_div_span_tags(self):
      def test_preserves_markdown_links(self):
      def test_cleans_attributes_inside_markdown_link_url(self):
      def test_cleans_malformed_anchor_fragment(self):
      def test_preserves_plain_text(self):
      def test_preserves_email_addresses(self):
      def test_strips_standalone_target_blank(self):
  ```

#### `class TestOllamaLLM:`
  ```python
      def test_successful_response(self, monkeypatch):
      def test_timeout_fallback_message(self, monkeypatch):
      def test_general_exception_fallback_message(self, monkeypatch):
  ```

#### `class TestBedrockLLM:`
  ```python
      def test_successful_response(self, monkeypatch):
      def test_retries_with_inference_profile_when_base_model_is_rejected(self, monkeypatch):
      def test_missing_api_key_fallback_message(self, monkeypatch):
      def test_falls_back_to_configured_backup_model_when_account_cannot_use_primary(self, monkeypatch):
  ```

#### `class TestLLMFactory:`
  ```python
      def test_create_llm_from_env_uses_ollama(self, monkeypatch):
      def test_create_llm_from_env_uses_gemini(self, monkeypatch):
      def test_create_llm_from_env_uses_bedrock(self, monkeypatch):
  ```

#### `class TestSystemPrompt:`
  ```python
      def test_cse_prompt_does_not_invent_hod_alias_email(self):
  ```


### [tests/test_retrieval.py](file:///home/c3i/chatbot/tests/test_retrieval.py)
> Tests for retrieval accuracy — ensures the retriever returns correct results.

#### `class TestHodRetrieval:`
  ```python
      def test_hod_query_returns_ravikant(self, retriever):
      """Asking about HoD should return Ravikant Saini."""
  ```

#### `class TestFacultyCountRetrieval:`
  ```python
      def test_faculty_count_query_returns_complete_roster(self, retriever, canonical_faculty):
      """Faculty count/list questions should include the full authoritative roster."""
      def test_direct_faculty_answer_is_complete(self, retriever, canonical_faculty):
      """The direct graph answer should not rely on LLM counting."""
  ```

#### `class TestPhDRosterRetrieval:`
  ```python
      def test_phd_count_query_returns_authoritative_roster_context(self, retriever):
      """Department-level PhD count questions should bypass fuzzy community summaries."""
      def test_direct_phd_answer_is_exact(self, retriever):
      """The direct graph answer should return the exact PhD scholar count."""
  ```

#### `class TestFacultyProfileRetrieval:`
  ```python
      def test_specific_faculty_query(self, retriever):
      """Querying a specific faculty name should return their info."""
  ```

#### `class TestDirectSupervisorAnswers:`
  ```python
      def test_supervisor_query_is_answered_directly(self, retriever):
      """Supervisor questions should bypass the LLM and use graph edges directly."""
  ```

#### `class TestCSEFacultyAnalytics:`
  ```python
      def test_cse_faculty_roster_count_is_authoritative(self, cse_retriever):
      """CSE faculty count/list should come from the graph roster, not chunk counting."""
      def test_gender_ratio_query_is_rejected_when_attribute_is_missing(self, cse_retriever):
      """Gender analytics must not be guessed from names or partial retrieval."""
      def test_main_point_of_contact_is_answered_directly(self, cse_retriever):
      """Generic contact questions should resolve to the HoD instead of drifting into unrelated facts."""
      def test_missing_startup_data_returns_unavailable_fallback(self, cse_retriever):
      """If the department graph has no startup evidence, the retriever should not hallucinate one."""
      def test_provenance_reports_combined_graph_and_vector_usage(self, cse_retriever):
      """Hybrid retrieval should report whether graph, vector, or both contributed."""
  ```

#### `class TestLaboratoryRetrieval:`
  ```python
      def test_ee_labs_retrieval(self, retriever):
      """Ask about EE labs and verify the correct list is returned."""
      def test_cse_labs_retrieval_negative(self, cse_retriever):
      """Ask about CSE labs specifically and verify deterministic empty message."""
      def test_cse_labs_retrieval_broadcast_ignored(self, cse_retriever):
      """Ask a general lab query to CSE retriever and verify it returns None to not pollute broadcast."""
  ```

#### `class TestFacultyDomainRetrieval:`
  ```python
      def test_rf_domain_retrieval(self, retriever):
      """Verify that searching for 'RF' returns Archana Rajput and Alok Kumar Saxena."""
      def test_microwave_domain_retrieval(self, retriever):
      """Verify that searching for 'Microwave' returns Archana Rajput and Alok Kumar Saxena."""
      def test_antenna_domain_retrieval(self, retriever):
      """Verify that searching for 'Antenna Design' returns Archana Rajput and Alok Kumar Saxena."""
  ```

#### Top-level Functions
```python
def retriever():
    """Load the full retriever (requires data/ to be populated)."""
def cse_retriever():
    """Load the CSE retriever from the checked-in department data."""
```


### [tests/test_rules_chatbot.py](file:///home/c3i/chatbot/tests/test_rules_chatbot.py)
#### Top-level Functions
```python
def test_rules_db_querying():
def test_rules_retriever_intent_classification():
def test_rules_retriever_context_generation():
def test_section_retriever_routing_academics():
def test_academic_policy_queries_route_to_academics(query):
def test_rules_retriever_finds_policy_evidence(query, expected):
def test_parser_keeps_change_of_department_section():
def test_rules_db_contains_amendment_and_department_change():
def test_academic_rules_do_not_mix_notice_list_chunks():
def test_parser_recovers_mtech_appendix_sections():
def test_user_reported_academic_queries_route_to_rules_db(query, expected_anchor):
```


### [tests/test_section_chatbot.py](file:///home/c3i/chatbot/tests/test_section_chatbot.py)
> Tests for section routing, retrieval, and entity resolution in the chatbot.

#### Top-level Functions
```python
def test_section_routing():
def test_global_person_index_role_prioritization():
def test_section_retriever_direct_answers():
def test_new_sections_retriever_direct_answers():
```

