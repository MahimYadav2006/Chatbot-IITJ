# IIT Jammu Chatbot — Scraping Architecture (Part 1: Core Engine)

## Navigation
- **Part 1 (this file)**: Core Engine, URL Pipeline, HTML→Markdown Compiler
- [Part 2](scraping_architecture_part2.md): Domain Crawlers & Document Pipelines
- [Part 3](scraping_architecture_part3.md): Registry, Data Taxonomy & Execution

---

## 1. Why Does This Scraping System Exist?

The IIT Jammu chatbot needs to answer questions about **everything** on the university's web presence — faculty profiles, academic regulations, committee compositions, lab equipment, admission procedures, and more. But this data lives across:

- **13 academic department websites** (EE, CSE, ME, etc.)
- **20 institutional section sites** (library, medical centre, accounts, etc.)
- **11 student-facing pages** (admissions, FAQs, schedules)
- **7 media pages** (events, news, donations)
- **12 quick-link pages** (staff directories, RTI, anti-ragging)
- **3 central facility sites** (CIF, I3C, Central Workshop)
- **100+ Google Drive PDFs** (academic rules, syllabi, notifications)

The scraping system converts ALL of this into clean **Markdown files** that feed into the chatbot's GraphRAG knowledge base. Think of it as: `Raw Web → Structured Markdown → Knowledge Graph → Chatbot Answers`.

---

## 2. The Dual-Architecture Problem

IIT Jammu's web presence is built on **two completely different technologies**, and the scraper must handle both:

### Problem A: The Angular SPA (Main Portal)

The main website (`iitjammu.ac.in/director`, `/faculty`, `/events`, etc.) is an **Angular Single Page Application**. When you make a standard HTTP request to these URLs, you get back:

```html
<html>
  <body>
    <app-root></app-root>  <!-- Empty! -->
    <script src="main.js"></script>
  </body>
</html>
```

There's **no content** in the HTML — JavaScript loads everything dynamically from backend APIs. A simple `requests.get()` would capture nothing useful. This is why the scraper uses **Playwright** (a real headless browser) that executes JavaScript and waits for content to render.

### Problem B: Static Department Microsites

Department sites (`/ee/*`, `/computer_science_engineering/*`, `/chemistry/*`) are traditional **server-rendered HTML** — tables, lists, faculty cards are all in the HTML source. These are simpler to scrape but come with their own challenges: broken relative links, duplicate path segments, and inconsistent HTML structures.

---

## 3. The Core Crawler Engine (`crawler.py`)

This 1,211-line file is the heart of everything. Every other scraping script imports and reuses its functions.

### 3.1 The Two-Phase Pipeline

Every single crawl operation follows the same pattern:

```
 PHASE 1: DISCOVERY                        PHASE 2: CONVERSION
 ════════════════════                      ════════════════════

 Start with seed URLs                      For each discovered URL:
       │                                         │
       ▼                                         ▼
 ┌─────────────┐                           ┌──────────────────┐
 │ Add to BFS  │                           │ Is it a binary   │──Yes──→ Download file
 │ Queue       │                           │ file? (.pdf etc) │         Parse with pdfplumber/
 └──────┬──────┘                           └────────┬─────────┘         openpyxl/tesseract
        │                                          No│
        ▼                                           ▼
 ┌──────────────┐                          ┌────────────────────┐
 │ Pop next URL │                          │ Use cached snapshot│
 │ from queue   │                          │ from Phase 1       │
 └──────┬───────┘                          └────────┬───────────┘
        │                                           │
        ▼                                           ▼
 ┌──────────────────┐                      ┌────────────────────┐
 │ Load in Playwright│                     │ Strip noise tags   │
 │ (headless Chrome) │                     │ (header, footer,   │
 └──────┬────────────┘                     │  nav, scripts)     │
        │                                  └────────┬───────────┘
        ▼                                           │
 ┌──────────────────┐                               ▼
 │ Is this a real   │                      ┌────────────────────┐
 │ content page?    │                      │ Find best content  │
 │ (not generic     │                      │ root node (scoring)│
 │  SPA shell?)     │                      └────────┬───────────┘
 └──┬──────────┬────┘                               │
   Yes         No                                   ▼
    │      (skip it)                       ┌────────────────────┐
    ▼                                      │ Recursive DOM →    │
 ┌──────────────────┐                      │ Markdown compile   │
 │ Extract ALL links│                      └────────┬───────────┘
 │ from page HTML   │                               │
 └──────┬───────────┘                               ▼
        │                                  ┌────────────────────┐
        ▼                                  │ Clean markdown,    │
 ┌──────────────────┐                      │ add quality flags, │
 │ Classify each:   │                      │ save to .md file   │
 │ page? binary?    │                      └────────────────────┘
 │ reject?          │
 └──────┬───────────┘
        │
        ▼
 Queue valid "page" links
 Collect "binary" URLs
 Log every decision
```

**Why two phases?** Phase 1 renders every page via Playwright (expensive — ~2-5 seconds per page). By caching the HTML snapshots, Phase 2 can convert them to Markdown without re-rendering. This also means Phase 2 can be re-run without re-crawling.

### 3.2 The Decision Tracking System

The crawler doesn't just silently accept or reject pages. It creates a **complete audit trail** using two dataclass types:

```python
@dataclass
class LinkDecision:
    source_url: str      # "Where did I find this link?"
    target_url: str      # "What URL was it pointing to?"
    kind: str            # "page" | "binary" | "reject"
    reason: str          # "department-page" | "binary-file" | "external-domain" | etc.
```

**Example:** If page `/ee/faculty` contains a link to `/ee/labs`, a link to `google.com`, and a link to a PDF:
```json
[
  {"source_url": "/ee/faculty", "target_url": "/ee/labs", "kind": "page", "reason": "department-page"},
  {"source_url": "/ee/faculty", "target_url": "google.com", "kind": "reject", "reason": "external-domain"},
  {"source_url": "/ee/faculty", "target_url": "/ee/syllabus.pdf", "kind": "binary", "reason": "binary-file"}
]
```

```python
@dataclass
class PageDecision:
    url: str             # "What URL did I try to load?"
    final_url: str       # "Where did I end up after redirects?"
    title: str           # "What was the <title> tag?"
    accepted: bool       # "Did I accept this page's content?"
    reason: str          # "Why did I accept/reject it?"
    text_length: int     # "How many characters of visible text?"
```

**Example:** A page that redirects and turns out to be generic:
```json
{
  "url": "https://iitjammu.ac.in/ee/nonexistent",
  "final_url": "https://iitjammu.ac.in/ee/",
  "title": "Indian Institute of Technology Jammu | Leading Engineering...",
  "accepted": false,
  "reason": "generic-fallback-page",
  "text_length": 287
}
```

All decisions are saved to `crawl_manifest.json` — you can inspect exactly why any URL was included or excluded.

### 3.3 BFS Discovery In Detail (`discover_site()`)

**Why BFS (Breadth-First Search)?**

Imagine a department site with this structure:
```
/ee/                    (depth 0 — index page)
├── /ee/faculty         (depth 1 — lists all faculty)
│   ├── /ee/faculty/dr-sharma  (depth 2)
│   └── /ee/faculty/dr-gupta   (depth 2)
├── /ee/labs            (depth 1 — lists all labs)
│   └── /ee/labs/vlsi-lab      (depth 2)
└── /ee/programs        (depth 1 — lists programs)
```

BFS processes **all depth-1 pages before any depth-2 pages**. This is crucial because:
1. Index pages (`/ee/faculty`) contain the most information (full faculty list)
2. Individual pages (`/ee/faculty/dr-sharma`) are often sparse
3. If there's a `--max-pages` limit, BFS captures the most valuable pages first

**The queue uses Python's `deque` (double-ended queue) for O(1) FIFO operations:**

```python
queue: deque[Tuple[str, Optional[str]]] = deque()
# Each item is (url_to_visit, source_url_that_linked_to_it)

# Seed the queue with starting URLs
for seed_path in ("", "/", "/index.html"):
    seed_url = canonicalize_url(base_url, base_url, base_url + seed_path)
    queue.append((seed_url, None))

# BFS loop
while queue:
    current_url, _source_url = queue.popleft()  # FIFO: take from FRONT
    # ... process page ...
    for candidate_url in extracted_links:
        queue.append((candidate_url, current_url))  # Add to BACK
```

**Three levels of deduplication prevent wasted work:**

| Check | Data Structure | What It Prevents |
|-------|---------------|-----------------|
| `enqueued` set | `Set[str]` | Same URL being added to queue twice |
| `visited` set | `Set[str]` | Same URL being processed twice |
| `accepted_final_urls` set | `Set[str]` | Two different URLs that redirect to the same final URL |

### 3.4 Playwright Navigation — The 4-Tier Fallback

Loading IIT Jammu pages is unreliable. Analytics trackers, slow font CDNs, and Google Tag Manager scripts can prevent the page from ever reaching `networkidle` state. The crawler tries progressively weaker wait conditions:

```
┌─ Attempt 1: networkidle (60s timeout) ──────────────────────┐
│  "Wait until no network requests for 500ms"                  │
│  Best quality but hangs on analytics/tracking pixels         │
└──────────────────────────────────────────────┬───────────────┘
                                               │ FAILS
┌─ Attempt 2: load (45s timeout) ──────────────▼───────────────┐
│  "Wait until the 'load' event fires"                          │
│  Doesn't wait for async XHR but gets most content            │
└──────────────────────────────────────────────┬───────────────┘
                                               │ FAILS
┌─ Attempt 3: domcontentloaded (45s timeout) ──▼───────────────┐
│  "Wait until HTML is parsed (no images/styles needed)"        │
│  Fast but may miss dynamically loaded content                │
└──────────────────────────────────────────────┬───────────────┘
                                               │ FAILS
┌─ Attempt 4: commit + 10s fixed wait (30s) ───▼───────────────┐
│  "Just start loading and wait 10 seconds"                     │
│  Last resort — captures whatever loaded in 10 seconds        │
└──────────────────────────────────────────────────────────────┘
```

After any successful navigation, there's an additional `PLAYWRIGHT_WAIT_MS` (5000ms) pause to let dynamic content hydrate.

**The browser context** spoofs Chrome 120 on Windows to avoid bot detection:
```python
context = browser.new_context(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
```

---

## 4. URL Management (`utils.py`)

### 4.1 Why URL Canonicalization Is So Critical

IIT Jammu pages emit **broken, inconsistent, and duplicate URLs** constantly. Without normalization, the crawler would:
- Visit the same page multiple times under different URL forms
- Crash on malformed relative paths
- Follow infinite loops of nested path segments

Here's what `canonicalize_url()` handles, with real examples:

**Problem 1: Non-Breaking Spaces**
```
Input:  "https://iitjammu.ac.in/ee/faculty\xa0"  (invisible character!)
Output: "https://iitjammu.ac.in/ee/faculty"
```

**Problem 2: Duplicate Department Prefixes**
A page at `/ee/programs` might contain a relative link `ee/about-us`. Naively resolving this gives `/ee/ee/about-us`. The function detects when the first segment of the href matches the department slug and resolves against the site root instead:
```
Base:    https://iitjammu.ac.in/ee
Current: https://iitjammu.ac.in/ee/programs
Href:    "ee/about-us"
Output:  "https://iitjammu.ac.in/ee/about-us"  (NOT /ee/ee/about-us)
```

**Problem 3: Circular Nested Paths**
```
Input:  "/chemistry/labs/chemistry/labs/chemistry/labs"
Output: "/chemistry/labs"  (consecutive duplicates collapsed)
```

**Problem 4: `index.html` Suffixes**
```
Input:  "/ee/index.html"
Output: "/ee/"
```

**Problem 5: Certificate Program Special Cases**
Certificate program pages live at the site root (`/ai-ml-nanodegree-certification`) but are linked from `/certificate-programs/ai-ml-nanodegree-certification`. The function detects these by keyword matching and strips the `/certificate-programs/` prefix.

### 4.2 URL Classification Pipeline

Every extracted URL goes through this decision tree:

```
URL discovered on a page
         │
         ▼
    Is scheme http or https?
    ├── NO → REJECT (unsupported-scheme)     [catches javascript:, mailto:, tel:]
    │
    ▼ YES
    Does domain match allowed_domain? (ignoring www.)
    ├── NO → REJECT (external-domain)        [catches google.com, youtube.com, etc.]
    │
    ▼ YES
    Is it a static asset? (.css, .js, .svg, .mp4, .woff, .zip, etc.)
    ├── YES → REJECT (static-asset)          [14 extensions blocked]
    │
    ▼ NO
    Is it a binary document? (.pdf, .xlsx, .docx, .csv, .jpg, .png, .gif)
    ├── YES → BINARY (binary-file)           [queued for download + parse]
    │
    ▼ NO
    Does URL path start with base_url path?
    ├── YES → PAGE (department-page)         [queued for BFS crawling]
    │
    ▼ NO
    REJECT (outside-department-scope)        [link to another department/section]
```

### 4.3 Generic Page Detection — Catching the SPA Shell

When you request an invalid URL on IIT Jammu's Angular site, it doesn't return a 404. Instead it returns a **200 OK with a generic landing page**. The crawler must detect and reject these.

Detection rules:
1. Title must exactly equal: `"Indian Institute of Technology Jammu | Leading Engineering Institute for Future Innovators"`
2. If title matches AND total text ≤ 350 chars → **generic** (it's just the header/footer)
3. If title matches AND has copyright footer AND text ≤ 500 chars → **generic**
4. If title doesn't match → **not generic** (real page with unique title)

---

## 5. The DOM-to-Markdown Compiler

This is where raw HTML becomes clean, structured Markdown for the knowledge graph.

### 5.1 Step 1: Strip Noise

Before any conversion, these elements are **completely removed** from the DOM:

| Selector | Why It's Noise |
|----------|----------------|
| `script`, `style`, `noscript` | Code, not content |
| `header`, `footer`, `nav` | Navigation chrome, not page content |
| `form` | Interactive elements, not information |
| `iframe[src*='youtube.com']` | Embedded videos (can't extract text) |
| `.slick-cloned`, `.owl-item.cloned` | Carousel duplicate slides |
| `.sr-only` | Screen-reader-only labels |
| `[hidden]`, `[aria-hidden='true']` | Intentionally hidden content |

Also checks for IIT Jammu-specific CSS classes: `rs-header`, `rs-footer`, `full-width-header`.

### 5.2 Step 2: Find the Best Content Root

Not all content on a page is equally valuable. The sidebar might have 50 navigation links, while the main content area has the actual information. The crawler **scores** candidate container nodes:

```
Score = len(visible_text) − (0.35 × len(anchor_text)) − (25 × image_count)
```

**Why this formula works:**
- `len(visible_text)`: More text = more likely to be the main content
- `−0.35 × len(anchor_text)`: Penalizes navigation menus (mostly links, little text)
- `−25 × image_count`: Penalizes image galleries and slideshows

The function tests these selectors in order and picks the highest-scoring match:
`main` → `[role='main']` → `article` → `.page-content` → `.content` → `.entry-content` → `.rs-services-details` → `.rs-inner-blog` → `.department-content` → `.container`

If nothing scores well, falls back to `<body>` or the entire `<soup>`.

### 5.3 Step 3: Recursive Tag-by-Tag Conversion

The `html_to_markdown()` function walks the DOM tree **recursively** — each element is converted based on its tag name, and its children are converted first (depth-first):

```python
def html_to_markdown(element, base_url):
    # Base cases
    if isinstance(element, Comment): return ""      # Skip HTML comments
    if isinstance(element, str): return str(element) # Raw text nodes
    if element.name in ["script", "style", ...]: return ""  # Noise

    # Recursive case: convert based on tag
    if element.name == "a":
        inner = "".join(html_to_markdown(child, base_url) for child in element.children)
        href = canonicalize_url(base_url, base_url, element.get("href", ""))
        return f"[{inner}]({href})"

    if element.name == "table":
        # ... build pipe-delimited table ...

    # Default: just concatenate children
    return "".join(html_to_markdown(child, base_url) for child in element.children)
```

**Table conversion is particularly sophisticated:**
1. Finds all `<tr>` rows
2. Within each row, finds all `<th>` and `<td>` cells
3. **Recursively converts** cell contents (so links/bold inside cells are preserved)
4. Normalizes all rows to the same column count (padding with empty strings)
5. First row becomes the header, second row is the `|---|---|` separator

### 5.4 Step 4: Post-Processing & Quality Flags

After conversion, `clean_markdown()` normalizes the output:
- Collapses 3+ consecutive blank lines to 2
- Removes duplicate adjacent lines (catches repeated carousel content)
- Strips trailing whitespace

Then quality flags are assigned:

| Flag | Meaning | When Triggered |
|------|---------|---------------|
| `empty-markdown` | No content at all | Markdown string is empty |
| `short-markdown` | Very little content | Less than 200 characters |
| `generic-markdown` | Boilerplate only | Contains the generic IIT Jammu header text |
| `generic-page-shell` | SPA shell page | Title/text matches the Angular fallback page |

If primary conversion produces empty markdown, a **fallback** kicks in that extracts the raw page title and all visible text as a last resort.

---

## 6. Binary File Processing

When the crawler encounters a `.pdf`, `.xlsx`, `.docx`, `.csv`, or image URL, it downloads the file and converts it to text:

### 6.1 Download & Detect

```python
# MIME type → file extension mapping
ext_map = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/csv": ".csv",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}
```

Files are downloaded to a **secure temporary path** within the output directory and deleted after parsing.

### 6.2 Parsing by Format

| Format | Primary Tool | Fallback | Output Quality |
|--------|-------------|----------|---------------|
| **PDF** | `pdfplumber` (coordinate-based extraction) | `pdftotext` CLI command | High — preserves table structure |
| **XLSX** | `openpyxl` (sheet-by-sheet) | None | High — pipe-delimited tables per sheet |
| **DOCX** | `python-docx` (paragraph extraction) | None | Medium — paragraph text, no complex formatting |
| **CSV** | Python `csv` module | None | High — direct pipe table conversion |
| **Images** | `pytesseract` + `Pillow` (OCR) | None | Low-Medium — depends on image quality |

**All libraries are optional** — imported with `try/except ImportError`. If `pdfplumber` isn't installed, the crawler falls back to the `pdftotext` command-line tool. If that's not available either, binary files are skipped gracefully.

---

## 7. Output Format

### 7.1 Individual Markdown Files

Each crawled page becomes a file named `{dept_code}_{url_path}.md`:

```
ee_ee.md                          ← /ee/ (index page)
ee_ee_faculty.md                  ← /ee/faculty
ee_ee_labs_vlsi-lab.md            ← /ee/labs/vlsi-lab
ee_ee_syllabus.pdf.md             ← /ee/syllabus.pdf (parsed)
```

File header format:
```markdown
# Source URL: https://iitjammu.ac.in/ee/faculty
# Page Title: Faculty Members | Department of Electrical Engineering
# Content Flags: none

## Faculty Members
| Name | Designation | Research Area |
|------|------------|---------------|
| Dr. Sharma | Associate Professor | VLSI Design |
...
```

### 7.2 Combined File

`00_combined_{code}_site.md` concatenates all pages with `---` separators. Useful for quick manual review of everything crawled.

### 7.3 Crawl Manifest

`crawl_manifest.json` is the complete audit trail — every URL attempted, every link discovered, and every accept/reject decision with reasons.

---

*Continued in [Part 2](scraping_architecture_part2.md): Domain Crawlers & Document Pipelines*

# IIT Jammu Chatbot — Scraping Architecture (Part 2: Domain Crawlers & Document Pipeline)

*Continued from [Part 1](scraping_architecture_part1.md)*

---

## 8. Why Domain-Specific Crawlers Exist

The core `crawler.py` handles academic departments and institutional sections. But some content doesn't fit that model:

- **Student pages** live at the site root (`/ugadm`, `/phd`), not under a department prefix
- **Media pages** include external sites (`sangam-2-0-psi.vercel.app`)
- **Quick-link pages** are standalone pages with committee rosters and directories
- **Central facility sites** span multiple sub-domains and have calendar trap issues

Each domain crawler lives in `scripts/` and reuses the core engine's functions via imports:
```python
from crawler import discover_site, download_and_convert, _page_urls_to_export, _clear_output_dir
```

They all follow the **same pattern**: define a URL map → launch Playwright → iterate sites → discover + convert → save combined file + manifest.

---

## 9. Student Sites Crawler (`scripts/crawl_students.py`)

### 9.1 What It Crawls

11 student-facing pages that prospective and current students need:

| Slug | URL | What You'll Find |
|------|-----|------------------|
| `pmrf` | `/pmrf` | Prime Minister's Research Fellowship details, eligibility |
| `online-education` | `/online-education/` | Online/distance learning programs |
| `academics-general-downloads` | `/academics/academics-general-downloads` | Downloadable forms (leave applications, grade cards) |
| `calendar-schedule-time-table` | `/calendar-schedule-time-table` | Semester dates, exam schedules, holiday calendars |
| `ugadm` | `/ugadm` | UG admission process, JEE cutoffs, seat matrix |
| `faq-main-website` | `/faq-main-website` | Frequently asked questions about everything |
| `why-iitjammu` | `/why-iitjammu` | NIRF rankings, campus facilities, placement stats |
| `pg-admissions` | `/pg-admissions` | M.Tech/M.Sc admission procedures, GATE requirements |
| `phd` | `/phd` | PhD program structure, fellowship details |
| `visvesvaraya-phd` | `/visvesvaraya-phd/` | Visvesvaraya PhD scheme for Electronics & IT |
| `certificate-programs` | `/certificate-programs` | Certificate courses and nanodegree programs |

### 9.2 How It Works (Step by Step)

```
1. Create output root: scraped_data/students/
2. Launch Playwright (headless Chrome)
3. For each slug in STUDENT_URLS:
   a. Create subdirectory: scraped_data/students/{slug}/
   b. If --clean flag: delete all existing .md files
   c. Call discover_site(base_url, context, max_pages)
      → This runs the full BFS from Part 1
      → Returns: accepted pages, binary URLs, snapshots, decisions
   d. For each accepted page:
      → Call download_and_convert() with cached snapshot
      → Saves individual .md file
   e. For each binary URL:
      → Call download_and_convert() (downloads + parses)
   f. Save 00_combined_{slug}_site.md (all content in one file)
   g. Save crawl_manifest.json (full audit trail)
4. Close browser
```

### 9.3 CLI Usage
```bash
python scripts/crawl_students.py                         # All 11 sites
python scripts/crawl_students.py --slug pmrf             # Just PMRF
python scripts/crawl_students.py --clean                 # Wipe existing data first
python scripts/crawl_students.py --max-pages 3           # Only 3 pages per site (debugging)
python scripts/crawl_students.py --clean --slug ugadm    # Clean + crawl one site
```

---

## 10. Media Sites Crawler (`scripts/crawl_media.py`)

### 10.1 What It Crawls

7 media and communication pages:

| Slug | URL | What You'll Find |
|------|-----|------------------|
| `donations` | `/donations` | How to donate to IIT Jammu |
| `prism` | `/prism` | PRISM research magazine articles |
| `newsdigest` | `/newsdigest` | Institutional news summaries |
| `events` | `/events` | Upcoming and past events calendar |
| `holidays-list-2026` | `/holidays-list-2026` | Official holiday list |
| `mou` | `/mou` | MoU partnerships with other institutions |
| `sangam-2-0` | `sangam-2-0-psi.vercel.app/` | Cultural fest website (EXTERNAL) |

**Note:** The Sangam fest site is hosted on Vercel (external domain), not on `iitjammu.ac.in`. The core engine's domain check (`classify_discovered_url`) will naturally restrict BFS to only that domain's pages.

### 10.2 Output Structure
```
scraped_data/Media/
├── donations/
│   ├── donations_donations.md
│   ├── 00_combined_donations_site.md
│   └── crawl_manifest.json
├── events/
│   ├── events_events.md
│   ├── events_events_some-event.md
│   ├── 00_combined_events_site.md
│   └── crawl_manifest.json
├── prism/
└── ... (7 subdirectories total)
```

---

## 11. Quick Links Crawler (`scripts/crawl_quick.py`)

### 11.1 What It Crawls

12 governance and compliance pages:

| Slug | Content |
|------|---------|
| `institute-honorary-chair-professor` | Distinguished visiting professors |
| `welcome-contacts` | Welcome desk phone numbers, emails |
| `suo-moto-disclosure` | RTI proactive disclosures |
| `st-sc-cell` | SC/ST Cell members, policies |
| `internal-complaint-committee` | ICC for workplace harassment |
| `adjunct-faculty` | Adjunct faculty from industry |
| `anti-ragging` | Anti-ragging committee, rules, helpline |
| `staff-page` | Complete staff directory by department |
| `institute-ethics-committee` | Research ethics committee |
| `equal-opportunity-cell` | Equal opportunity compliance |
| `voip-directory` | VoIP telephone directory for campus |
| `rti` | Right to Information officer details |

### 11.2 PDF Supplements

Some Quick sections have **manually placed PDF files** in `scraped_data/Quick/Pdf-data/` that were converted separately:
- `Some Points to Remember About Ragging.md`
- `Prevention of Caste-Based Discrimination.md`
- `Student Grievance Redressal Committee.md`

These are linked to sections via `QUICK_MULTI_SOURCE` in `departments.py` (covered in Part 3).

---

## 12. Central Instruments Crawler (`scripts/crawl_ci.py`)

### 12.1 What It Crawls

3 sub-sites for research infrastructure:

| Slug | URL | Domain | Content |
|------|-----|--------|---------|
| `cif` | `/cif` | `iitjammu.ac.in` | Central Instruments Facility — equipment booking, charges |
| `i3c` | `i3c-iitjammu.in` | **External domain** | Innovation & Incubation Center |
| `central-workshop` | `/central-workshop` | `iitjammu.ac.in` | Machining, fabrication services |

### 12.2 Unique Design: The Calendar Trap Prevention

The I3C website uses WordPress with the Tribe Events plugin, which creates **infinite pagination loops**:
```
/events/                    → links to →
/events/?tribe-bar-date=2024-01  → links to →
/events/?tribe-bar-date=2024-02  → links to →
/events/?tribe-bar-date=2024-03  → ... forever
```

The crawler would never stop discovering "new" URLs. To prevent this, `crawl_ci.py` **monkeypatches** the URL classifier:

```python
# Save original function
original_classify = utils.classify_discovered_url

# Replace with custom version that blocks calendar URLs
def custom_classify_discovered_url(url, base_url, allowed_domain):
    if any(p in url.lower() for p in (
        "/events/", "tribe-bar-date", "ical=", "outlook-ical", "tribe-venue"
    )):
        return "reject", "calendar-trap"
    return original_classify(url, base_url, allowed_domain)

# Monkeypatch it onto the crawler module
crawler.classify_discovered_url = custom_classify_discovered_url
```

This is the **only** domain crawler that modifies core engine behavior at runtime.

### 12.3 Unique Design: Shared Output Directory

Unlike other crawlers that create per-slug subdirectories, the CI crawler dumps ALL sub-site content into a single `scraped_data/sections/ci/` directory. This is because the chatbot treats CIF + I3C + Central Workshop as one unified "Central Instruments & Innovation" knowledge domain.

---

## 13. The Google Drive PDF Pipeline

Academic regulations, syllabi, and course structures are hosted as PDFs on Google Drive — not on any IIT Jammu webpage. These require a completely different approach.

### 13.1 The Three-Pass Architecture

Google Drive actively resists automated downloads. The pipeline uses **three progressively more aggressive strategies**:

```
Pass 1: download_academics_pdfs.py
  │  Uses gdown library + requests with cookies
  │  Success rate: ~70-80%
  │  Failed files saved as HTML stubs
  │
  ▼
Pass 2a: download_gdrive_playwright.py
  │  Uses real browser (Playwright) to click download buttons
  │  Handles virus scan confirmation pages
  │  Success rate: picks up ~50% of remaining
  │
  ▼
Pass 2b: download_gdrive_pass2.py
     Uses curl and wget with cookie manipulation
     Three strategies: direct confirm=t, two-step cookies, wget fallback
     Picks up remaining accessible files
```

### 13.2 Pass 1 In Detail (`download_academics_pdfs.py`)

**Step 1: Find All Google Drive Links**

The script reads 4 previously-crawled markdown files from the academics section and extracts every Google Drive URL:

```python
SOURCE_FILES = [
    {"md_file": "academics_academics-rules-and-regulations.md",     "output_folder": "rules_and_regulations"},
    {"md_file": "academics_academics-general-downloads.md",          "output_folder": "general_downloads"},
    {"md_file": "academics_academics-specialisation-and-courses.md", "output_folder": "specialisation_and_courses"},
    {"md_file": "academics_academic-notifications.md",               "output_folder": "academic_notifications"},
]
```

The link extractor (`extract_links_from_md()`) is context-aware — it tracks the heading hierarchy so links are categorized properly:

```markdown
## Rules and Regulations           ← context_stack = [(2, "Rules and Regulations")]
### PG                             ← context_stack = [(2, "..."), (3, "PG")]
1. MTech                           ← context_stack = [(2, "..."), (3, "PG"), (99, "MTech")]
   - [M.Tech Rules](https://drive.google.com/file/d/ABC123/view)
                                   ↑ This link gets category_path = ["Rules and Regulations", "PG", "MTech"]
```

Links are classified into 3 types:
- `file` — Google Drive file (`/file/d/{ID}/`)
- `spreadsheet` — Google Sheets (`/spreadsheets/d/{ID}/`)
- `folder` — Google Drive folder (`/folders/{ID}`)

**Step 2: Download Each File**

For `file` type links:
1. Extract the file ID from the URL using regex
2. Check if already cached in `pdf_cache/` directory
3. Try `gdown.download()` first (Python library for Google Drive)
4. If gdown fails → try `requests` with cookie-based virus scan bypass:
   ```python
   # Google shows a "virus scan" interstitial for large files
   # The confirm token is in a cookie
   for key, value in resp.cookies.items():
       if key.startswith('download_warning'):
           url = f"...&confirm={value}"  # Re-request with confirmation
   ```
5. Rate limit: 0.5 second delay between downloads

For `spreadsheet` type: Export as CSV via `https://docs.google.com/spreadsheets/d/{ID}/export?format=csv`

For `folder` type: Parse the folder HTML page to find file IDs, then download each file individually.

**Step 3: Parse Downloaded Files**

File type is detected from **magic bytes** (first 8 bytes), not file extension:
```python
if header[:5] == b'%PDF-':     return 'pdf'
if header[:4] == b'PK\x03\x04': return 'xlsx_or_docx'  # ZIP-based Office formats
if b'<html' in header.lower():  return 'html'           # Google served an interstitial!
```

PDFs are parsed with `pdfplumber`, which provides:
- Page-by-page text extraction
- **Table detection and extraction** — tables are rendered as markdown pipe tables
- **Table deduplication** — after extracting tables, their text is removed from the page text to prevent duplicate content

**Step 4: Generate Outputs**

Each parsed document gets a markdown file with metadata:
```markdown
# UG Rules & Curriculum 2022

> **Source**: [https://drive.google.com/file/d/ABC123/view](...)
> **Pages**: 47

---

## Page 1

### Table 1
| Grade | Points | Description |
|-------|--------|-------------|
| AA    | 10     | Outstanding |
...

[remaining page text]
```

Failed downloads get stub files:
```markdown
> ⚠ **Download failed** — this document could not be retrieved from Google Drive.
> It may require special access permissions.
```

### 13.3 Pass 2a: Playwright Download (`download_gdrive_playwright.py`)

Scans `pdf_cache/` for files that are still HTML (download failed in Pass 1). For each one:

1. Opens a real Chromium browser with `accept_downloads=True`
2. Navigates to the Google Drive export URL
3. Tries **4 strategies** to trigger the download:
   - Click `a[href*="confirm="]` (confirmation link)
   - Submit `form[id="download-form"]` (form-based download)
   - Click `#uc-download-link` (direct download link)
   - Open file viewer page → click `[aria-label="Download"]` button
4. Saves the downloaded file back to `pdf_cache/`
5. Re-parses all stub markdown files with the newly downloaded PDFs

### 13.4 Pass 2b: curl/wget Download (`download_gdrive_pass2.py`)

Alternative to Playwright — uses command-line tools. Three strategies per file:

```bash
# Strategy 1: Direct with confirm=t
curl -L -o file.tmp "https://drive.google.com/uc?export=download&confirm=t&id=FILE_ID"

# Strategy 2: Two-step with cookies
curl -c cookies.txt "https://drive.google.com/uc?export=download&id=FILE_ID"
# Extract confirm token from cookies
curl -L -b cookies.txt -o file.tmp "...&confirm=TOKEN&id=FILE_ID"

# Strategy 3: wget fallback
wget --no-check-certificate -O file.tmp "...&confirm=t&id=FILE_ID"
```

After downloading, validates that the result is actually a PDF (not another HTML interstitial) using magic bytes.

---

## 14. The PDF/Excel → Markdown Converter (`scripts/convert_pdfs_to_md.py`)

This script handles **locally stored** PDFs and Excel files (not from Google Drive). It's especially important for scanned documents that need OCR.

### 14.1 The Classification System

Each PDF is classified before processing by sampling the first 5 pages:

```python
SCANNED_THRESHOLD = 50  # characters per page

def classify_pdf(path):
    pdf = pdfplumber.open(path)
    digital_pages = 0
    sample = min(len(pdf.pages), 5)

    for page in pdf.pages[:sample]:
        text = (page.extract_text() or "").strip()
        if len(text) > SCANNED_THRESHOLD:  # More than 50 chars? Has real text
            digital_pages += 1

    ratio = digital_pages / sample
    if ratio > 0.8:  return "digital"   # 80%+ pages have text → use pdfplumber
    if ratio < 0.2:  return "scanned"   # 80%+ pages are images → use OCR
    return "mixed"                       # Some of each → hybrid approach
```

### 14.2 OCR Pipeline for Scanned Documents

When a PDF is a scan (photo of a printed document), text extraction requires OCR:

```
PDF file
    │
    ▼ pdf2image (rasterize at 300 DPI)
Page images (one per page)
    │
    ▼ For each image:
    │
    ├─ Convert to grayscale (L mode)
    │
    ├─ Increase contrast by 1.5× (makes text darker)
    │
    ├─ Apply sharpen filter (crisper edges)
    │
    ├─ Binarize: pixel < 140 → black, else → white
    │    (removes gray gradients, makes text crisp)
    │
    └─ pytesseract.image_to_string(
         config="--oem 3 --psm 6"
       )
       │  oem 3 = LSTM neural net engine (best accuracy)
       │  psm 6 = "assume uniform block of text" (good for document pages)
       │
       ▼
    Extracted text string
```

### 14.3 Mixed PDF Handling

For PDFs that have both digital and scanned pages (e.g., a typed document with a scanned appendix):

```python
for i, page in enumerate(pdf.pages):
    text = page.extract_text()
    if len(text) > 50:
        # This page has real text → use pdfplumber (fast, accurate)
        use_pdfplumber(page)
    else:
        # This page is scanned → use OCR (slow, but necessary)
        use_tesseract(images[i])
```

### 14.4 Output with YAML Front-Matter

Each converted file gets machine-readable metadata:
```yaml
---
source_file: "UG_Rules_2022.pdf"
source_path: "Rules_and_Regulations/UG/UG_Rules_2022.pdf"
document_type: "pdf"
category: "rules_and_regulations"
subcategory: "undergraduate"
converted_at: "2026-06-29T12:00:00"
institution: "Indian Institute of Technology Jammu"
---
```

Category is auto-detected from the file path — "Rules and Regulations/UG" → category=`rules_and_regulations`, subcategory=`undergraduate`.

---

## 15. Post-Processing Scripts

### 15.1 Curriculum Flattener (`flatten_curriculum_docs.py`)

**Problem:** The Google Drive PDF pipeline creates deeply nested directories:
```
parsed_documents/academics-specialisation-and-courses/PG/MTech/CSE/program.md
```

The ingestion pipeline expects flat files in `scraped_data/sections/academics/`.

**Solution:** This script walks the nested tree and copies files with flattened names:
```
Input:  PG/MTech/CSE/program.md
Output: academics_curriculum_pg_mtech_cse_program.md
```

### 15.2 Rules Ingester (`ingest_rules.py`)

**Purpose:** Converts the free-text academic regulation PDFs into a **structured SQLite database** (`rules.db`) that the chatbot can query precisely.

**What it ingests:**

1. **Section tree** — Parses 4 regulation documents into hierarchical sections:
   - `9.5-IIT_Jammu_Rules___Curriculumn.md` → UG rules (old format)
   - `UG_Curriculum_2022_Scheme_IIT_Jammu.md` → UG rules (2022 scheme)
   - `M.Tech_RRs___Curric..md` → M.Tech regulations
   - `PhD_RRs.md` → PhD regulations

2. **Amendment overlays** — Standalone amendment documents that override specific rules:
   - `Amenedment_in_Rule_2.3.2.2.md` → Open elective credit cap changes
   - `Amedment_UG_Rules.md` → General UG rule amendments

3. **Grade scale** — All 16 grade codes:
   ```
   AA=10, AB=9, BB=8, BC=7, CC=6, CD=5, DD=4, FF=0
   SA=0 (Satisfactory), UA=0 (Unsatisfactory), II=0 (Incomplete), etc.
   ```

4. **Credit requirements** — 11 UG categories:
   ```
   Institute Core: 18.5 credits
   Department Core: 51 credits (range 51-57)
   Open Elective: 18 credits (min 6 from HSS)
   BTech Project: 9 credits (3+3+3 over 3 semesters)
   Total: 132-138 credits depending on branch
   ```

5. **Rule facts** — 22 precise numerical constraints the chatbot needs to answer questions correctly:
   ```
   min_cgpa_minor ≥ 7.0
   max_semester_drop ≤ 1 (≤ 2 for medical)
   max_open_elective_single_department ≤ 9 credits
   department_change_policy == "not_offered"
   ```

6. **Program milestones** — PhD progression timeline:
   ```
   Course Work → Comprehensive Exam (12-24 months) → Research Plan → Candidacy
   ```

---

*Continued in [Part 3](scraping_architecture_part3.md): Registry, Data Taxonomy & Execution*
# IIT Jammu Chatbot — Scraping Architecture (Part 3: Registry, Data Taxonomy & Execution)

*Continued from [Part 2](scraping_architecture_part2.md)*

---

## 16. The Central Registry (`departments.py`)

This 357-line file is the **single source of truth** for every piece of content the scraper knows about. Every crawler, every ingestion script, and the chatbot itself references this registry.

### 16.1 Department Registry (`DEPARTMENTS`)

13 academic departments, each defined as a dictionary:

```python
DEPARTMENTS = {
    "ee": {
        "name": "Electrical Engineering",                    # Short display name
        "full_name": "Department of Electrical Engineering",  # Full formal name
        "base_url": "https://iitjammu.ac.in/ee",            # Crawl starting point
        "template": "A",                                      # Site template variant
        "official_contact_email": "hod.ee@iitjammu.ac.in",   # Optional contact
    },
    "computer_science_engineering": {
        "name": "Computer Science & Engineering",
        "full_name": "Department of Computer Science and Engineering",
        "base_url": "https://iitjammu.ac.in/computer_science_engineering",
        "template": "B",
        "aliases": ["cse"],  # Can be referenced as "cse" in CLI commands
    },
    # ... 11 more departments
}
```

**Complete department list:**

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

**Template A vs B:** Different CSS/HTML layouts used by departments. Template A sites have one DOM structure, Template B sites have another. The crawler's content root scoring handles both automatically.

**Aliases:** Some departments have shorthand codes. `"cse"` resolves to `"computer_science_engineering"`, `"chemical"` resolves to `"chemical-engineering"`. The `resolve_department_code()` function handles this:

```python
def resolve_department_code(code):
    normalized = code.strip().lower()
    if normalized in DEPARTMENTS:      return normalized       # Direct match
    if normalized in DEPARTMENT_ALIASES: return DEPARTMENT_ALIASES[normalized]  # Alias
    raise KeyError(f"Department code '{code}' not found")
```

### 16.2 Lab Registry (`CORRECT_LABS`)

Verified lab names for 8 departments, used for accurate entity resolution in the knowledge graph:

```python
CORRECT_LABS = {
    "ee": [
        "Low Voltage Lab 2",
        "Prototype Design and Development Lab",
        "Underwater Research Lab",
        "High Voltage Lab",
        "IC Reliabality, Security & Quality Laboratory",
        "AADHRIT Lab",
        # ...
    ],
    "mechanical_engineering": [
        "Fluid Mechanics Lab",
        "Control Engineering Lab",
        "Heat & Mass Transfer Lab",
        # ...
    ],
    # ... 6 more departments
}
```

### 16.3 Section Registry (`SECTIONS`)

37 institutional sections spanning 5 categories. Each section has a name and base_url:

**Institutional Sections (20):**
```
academics, alumni-affairs, cds, counselling, di, e2, saral, accounts,
hindicell, ir, library, medical-centre, osd, sp, rc, sw, security, tlu,
tinkerers-lab, ci
```

**Student Sections (11):**
```
students-faq, students-schedule, students-phd-admissions,
students-pg-admissions, students-ug-admissions,
students-certificate-programs, students-online-education,
students-pmrf, students-visvesvaraya, students-why-iitjammu,
students-academic-downloads
```

**Media Section (1):** `media`

**Quick-Link Sections (6):**
```
quick-adjunct-faculty, quick-anti-ragging, quick-committees,
quick-staff, quick-contacts, quick-rti
```

### 16.4 The Folder Mapping System

**The problem:** Not all sections store their data under `scraped_data/sections/`. Student data is under `scraped_data/students/`, media under `scraped_data/Media/`, and quick links under `scraped_data/Quick/`. The chatbot needs to know where to find each section's files.

**The solution:** Three mapping dictionaries translate section codes to filesystem paths:

```python
# Student sections → scraped_data/students/{slug}
STUDENT_SECTION_FOLDER_MAP = {
    "students-faq": "students/faq-main-website",
    "students-ug-admissions": "students/ugadm",
    "students-phd-admissions": "students/phd",
    # ... 11 mappings
}

# Media section → scraped_data/Media
MEDIA_SECTION_FOLDER_MAP = {
    "media": "Media",
}

# Quick sections → scraped_data/Quick/{slug}
QUICK_SECTION_FOLDER_MAP = {
    "quick-adjunct-faculty": "Quick/adjunct-faculty",
    "quick-staff": "Quick/staff-page",
    "quick-committees": "Quick",        # Multi-source!
    "quick-contacts": "Quick",          # Multi-source!
    # ... 6 mappings
}
```

The `get_section_markdown_dir()` function resolves the correct path:
```python
def get_section_markdown_dir(code):
    if code in STUDENT_SECTION_FOLDER_MAP:
        return os.path.join(SCRAPED_DATA_ROOT, STUDENT_SECTION_FOLDER_MAP[code])
    if code in MEDIA_SECTION_FOLDER_MAP:
        return os.path.join(SCRAPED_DATA_ROOT, MEDIA_SECTION_FOLDER_MAP[code])
    if code in QUICK_SECTION_FOLDER_MAP:
        return os.path.join(SCRAPED_DATA_ROOT, QUICK_SECTION_FOLDER_MAP[code])
    return os.path.join(SCRAPED_DATA_ROOT, "sections", code)  # Default
```

### 16.5 Multi-Source Sections

Some sections pull content from **multiple subdirectories**. For example, `quick-committees` needs data from 5 different sources:

```python
QUICK_MULTI_SOURCE = {
    "quick-committees": [
        "Quick/equal-opportunity-cell",        # Equal Opportunity Cell
        "Quick/institute-ethics-committee",     # Ethics Committee
        "Quick/internal-complaint-committee",   # ICC
        "Quick/st-sc-cell",                    # SC/ST Cell
        "Quick/Pdf-data",                      # Manually placed PDFs
    ],
    "quick-contacts": [
        "Quick/voip-directory",                # Phone numbers
        "Quick/welcome-contacts",              # Welcome desk contacts
    ],
    "quick-anti-ragging": [
        "Quick/anti-ragging",                  # Committee page
        "Quick/Pdf-data",                      # Anti-ragging regulation PDFs
    ],
    "quick-rti": [
        "Quick/rti",                           # RTI main page
        "Quick/suo-moto-disclosure",           # Proactive disclosures
    ],
}
```

The ingestion pipeline iterates over all source directories for these sections, merging content into a unified knowledge domain.

### 16.6 Skip Patterns

Certain files should be excluded during ingestion (boilerplate, duplicates, or misplaced content):

```python
STUDENT_SECTION_SKIP_PATTERNS = [
    "_nirf_", "_rti_", "_SGRC_", "_index.md",
    "nirf_2026", "CERT-IN_Certificate",
    "Student_Grievance_Redressal_Committee",
]

MEDIA_SECTION_SKIP_PATTERNS = [
    "_index.md", "crawl_manifest.json",
    "CERT-IN", "SGRC", "nirf_2026",
]

QUICK_SECTION_SKIP_PATTERNS = [
    "_index.md", "crawl_manifest.json",
    "00_combined_",  # Skip combined files (ingested individually)
]
```

---

## 17. Complete Output Data Taxonomy

Here's the full filesystem tree of everything the scraping system produces:

```
scraped_data/
│
├── administration/                    ← crawl_administration() in crawler.py
│   ├── administration_board-of-governors.md
│   ├── administration_director.md
│   ├── administration_deans-and-associate-deans.md
│   ├── administration_registrar.md
│   ├── administration_finance-committee.md
│   ├── 00_combined_administration_site.md
│   └── crawl_manifest.json
│
├── ee/                                ← crawl_department("ee")
├── computer_science_engineering/      ← crawl_department("cse")
├── mechanical_engineering/            ← crawl_department("mechanical_engineering")
├── civil_engineering/
├── chemical-engineering/
├── bsbe/
├── chemistry/
├── hss/
├── idp/
├── materials-engineering/
├── mathematics/
├── physics/
│   (Each contains: individual .md files + 00_combined + manifest)
│
├── sections/                          ← crawl_section() in crawler.py
│   ├── academics/                     ← The richest section
│   │   ├── academics_academics.md     ← Main page
│   │   ├── academics_academics-rules-and-regulations.md
│   │   ├── academics_academic-notifications.md
│   │   ├── academics_curriculum_*.md  ← Flattened by flatten_curriculum_docs.py
│   │   ├── parsed_documents/          ← Google Drive PDFs → markdown
│   │   │   ├── rules_and_regulations/
│   │   │   │   ├── UG/
│   │   │   │   │   ├── 9.5-IIT_Jammu_Rules___Curriculumn.md
│   │   │   │   │   ├── UG_Curriculum_2022_Scheme_IIT_Jammu.md
│   │   │   │   │   └── Amenedment_in_Rule_2.3.2.2.md
│   │   │   │   └── PG/
│   │   │   │       ├── MTech/M.Tech_RRs___Curric..md
│   │   │   │       └── PhD/PhD_RRs.md
│   │   │   ├── general_downloads/
│   │   │   ├── specialisation_and_courses/
│   │   │   │   ├── UG/Specialization/
│   │   │   │   └── PG/MTech/
│   │   │   ├── academic_notifications/
│   │   │   ├── download_manifest.json
│   │   │   └── 00_index.md
│   │   └── pdf_cache/                 ← Raw downloaded binaries (cached)
│   │
│   ├── ci/                            ← crawl_ci.py (CIF + I3C + Workshop)
│   ├── accounts/
│   ├── alumni-affairs/
│   ├── cds/
│   ├── counselling/
│   ├── di/
│   ├── e2/
│   ├── hindicell/
│   ├── ir/
│   ├── library/
│   ├── medical-centre/
│   ├── osd/
│   ├── rc/
│   ├── saral/
│   ├── security/
│   ├── sp/
│   ├── sw/
│   ├── tinkerers-lab/
│   └── tlu/
│
├── students/                          ← crawl_students.py
│   ├── faq-main-website/
│   ├── ugadm/
│   ├── pg-admissions/
│   ├── phd/
│   ├── pmrf/
│   ├── certificate-programs/
│   ├── online-education/
│   ├── visvesvaraya-phd/
│   ├── why-iitjammu/
│   ├── calendar-schedule-time-table/
│   └── academics-general-downloads/
│
├── Media/                             ← crawl_media.py
│   ├── donations/
│   ├── events/
│   ├── holidays-list-2026/
│   ├── mou/
│   ├── newsdigest/
│   ├── prism/
│   └── sangam-2-0/
│
└── Quick/                             ← crawl_quick.py
    ├── Pdf-data/                      ← Manually placed PDF conversions
    │   ├── Some Points to Remember About Ragging.md
    │   ├── Prevention of Caste-Based Discrimination.md
    │   └── Student Grievance Redressal Committee.md
    ├── adjunct-faculty/
    ├── anti-ragging/
    ├── equal-opportunity-cell/
    ├── institute-ethics-committee/
    ├── institute-honorary-chair-professor/
    ├── internal-complaint-committee/
    ├── rti/
    ├── st-sc-cell/
    ├── staff-page/
    ├── suo-moto-disclosure/
    ├── voip-directory/
    └── welcome-contacts/
```

---

## 18. Complete Execution Guide

### 18.1 Full Crawl (From Scratch)

Run these in order. Each step depends on the previous ones:

```bash
# ─── STEP 1: Academic Departments (13 departments) ───
# Time: ~30-60 min depending on site speed
python crawler.py --all --clean-output

# ─── STEP 2: Administration Pages ───
# Time: ~2-3 min (only 9 pages, no BFS)
python crawler.py --dept administration --clean-output

# ─── STEP 3: Institutional Sections (20 sections) ───
# Time: ~60-90 min
python crawler.py --all-sections --clean-output

# ─── STEP 4: Student Pages (11 sites) ───
# Time: ~15-20 min
python scripts/crawl_students.py --clean

# ─── STEP 5: Media Pages (7 sites) ───
# Time: ~10-15 min
python scripts/crawl_media.py --clean

# ─── STEP 6: Quick Link Pages (12 sites) ───
# Time: ~10-15 min
python scripts/crawl_quick.py --clean

# ─── STEP 7: Central Instruments (3 sub-sites) ───
# Time: ~10-15 min
python scripts/crawl_ci.py --clean

# ─── STEP 8: Download Academic PDFs from Google Drive ───
# Time: ~20-40 min (network dependent)
python scripts/download_academics_pdfs.py

# ─── STEP 9: Re-download Failed PDFs (Playwright) ───
# Time: ~10-20 min
python scripts/download_gdrive_playwright.py

# ─── STEP 10: Re-download Failed PDFs (curl/wget) ───
# Time: ~5-10 min
python scripts/download_gdrive_pass2.py

# ─── STEP 11: Convert Local PDFs/Excel to Markdown ───
# Time: ~5-15 min (longer if OCR needed)
python scripts/convert_pdfs_to_md.py

# ─── STEP 12: Flatten Curriculum Documents ───
# Time: < 1 min
python scripts/flatten_curriculum_docs.py

# ─── STEP 13: Ingest Rules into Structured DB ───
# Time: < 1 min
python scripts/ingest_rules.py
```

### 18.2 Incremental Updates

To update just one department or section without re-crawling everything:

```bash
# Re-crawl just EE department
python crawler.py --dept ee --clean-output

# Re-crawl just the library section
python crawler.py --section library --clean-output

# Re-crawl just the UG admissions student page
python scripts/crawl_students.py --slug ugadm --clean

# Re-crawl just the CIF sub-site of CI
python scripts/crawl_ci.py --slug cif --clean
```

### 18.3 Debugging a Crawl

```bash
# Crawl only 3 pages (fast test)
python crawler.py --dept ee --max-pages 3

# Check what the crawler decided about each URL
cat scraped_data/ee/crawl_manifest.json | python -m json.tool | head -100
```

---

## 19. Key Design Decisions Summary

| Decision | Why |
|----------|-----|
| **Playwright over requests/Scrapy** | IIT Jammu uses Angular SPAs — static HTTP gets empty pages |
| **BFS over DFS** | High-value index pages are processed first; `--max-pages` captures the best content |
| **Full audit trail** | Every URL decision is logged in `crawl_manifest.json` for debugging |
| **4-tier navigation fallback** | IIT Jammu pages often hang on analytics scripts |
| **Content root scoring** | Navigation sidebars have many links but little text; scoring formula penalizes this |
| **Recursive DOM compiler** | Naive `.get_text()` destroys structure; recursion preserves tables, lists, links |
| **3-pass PDF download** | Google Drive actively blocks automated downloads |
| **OCR with image preprocessing** | Scanned PDFs need contrast enhancement and binarization for readable OCR |
| **Monkeypatch for calendar traps** | WordPress Tribe Events creates infinite URL pagination |
| **Multi-source sections** | Some chatbot knowledge domains span multiple scraped directories |
| **Optional library imports** | `try/except ImportError` ensures graceful degradation on minimal installs |
| **Structured rules DB** | Free-text regulations are ambiguous; SQLite with grounded facts enables precise answers |

---

## 20. Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| `playwright` | Headless browser for JavaScript rendering | **Yes** |
| `beautifulsoup4` | HTML parsing and DOM traversal | **Yes** |
| `requests` | HTTP downloads for binary files | **Yes** |
| `pdfplumber` | PDF text and table extraction | Recommended |
| `openpyxl` | Excel file parsing | Optional |
| `python-docx` | Word document parsing | Optional |
| `pytesseract` | OCR for scanned PDFs/images | Optional |
| `Pillow` | Image processing for OCR | Optional |
| `pdf2image` | PDF → image conversion for OCR | Optional |
| `gdown` | Google Drive file downloads | For PDF pipeline |
| `pdfminer` | Fallback PDF text extraction | Optional fallback |

Install Playwright browsers after pip install:
```bash
playwright install chromium
```
