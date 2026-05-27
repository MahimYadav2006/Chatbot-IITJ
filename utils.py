import posixpath
import re
import urllib.parse
from typing import Optional, Tuple

# Constants used by the crawler for content validation and timing
PLAYWRIGHT_WAIT_MS = 5000  # milliseconds to wait for dynamic content to load
MIN_CONTENT_LEN = 200      # minimum character length to consider a page non-empty
GENERIC_TITLE = "Indian Institute of Technology Jammu | Leading Engineering Institute for Future Innovators"
GENERIC_HEADER = GENERIC_TITLE

BINARY_EXTENSIONS = {
    ".pdf",
    ".xlsx",
    ".docx",
    ".csv",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
}

STATIC_ASSET_EXTENSIONS = {
    ".css",
    ".js",
    ".json",
    ".xml",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp4",
    ".webm",
    ".mp3",
    ".zip",
}

SKIP_SCHEMES = ("javascript:", "mailto:", "tel:", "data:")


def _normalize_href(href: str) -> str:
    """Strip invisible whitespace and normalize HTML oddities from hrefs."""
    href = urllib.parse.unquote(str(href or ""))
    href = href.replace("\xa0", " ").strip()
    href = re.sub(r"\s+", " ", href)
    return href


def _collapse_duplicate_segments(path: str) -> str:
    """Collapse duplicate consecutive path segments and repeated department prefixes."""
    parts = [part for part in path.split("/") if part and part != "."]
    cleaned_parts = []

    for part in parts:
        if part == "..":
            if cleaned_parts:
                cleaned_parts.pop()
            continue
        if cleaned_parts and cleaned_parts[-1] == part:
            continue
        cleaned_parts.append(part)

    return "/" + "/".join(cleaned_parts)


def canonicalize_url(base_url: str, current_url: str, href: str) -> Optional[str]:
    """Resolve an href into a canonical absolute URL within the department site.

    IIT Jammu pages often emit broken relative links such as:
    - ``computer_science_engineering/about-us`` from nested pages
    - trailing non-breaking spaces in URLs
    - duplicated department prefixes

    This helper normalizes those cases so we do not invent URLs like
    ``.../program-list/computer_science_engineering/program-list/...``.
    """
    href = _normalize_href(href)
    if not href or href == "#":
        return None

    lowered = href.lower()
    if lowered.startswith(SKIP_SCHEMES):
        return None

    base_parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    current_parsed = urllib.parse.urlparse(current_url)
    base_segments = [segment for segment in base_parsed.path.split("/") if segment]
    dept_slug = base_segments[-1] if base_segments else ""
    site_root = f"{base_parsed.scheme}://{base_parsed.netloc}/"

    if href.startswith(("http://", "https://")):
        absolute = href
    elif href.startswith("/"):
        absolute = urllib.parse.urljoin(site_root, href.lstrip("/"))
    else:
        href_first_segment = href.split("/", 1)[0]
        if href_first_segment == dept_slug:
            absolute = urllib.parse.urljoin(site_root, href)
        elif href.startswith(("./", "../")):
            absolute = urllib.parse.urljoin(current_url.rstrip("/") + "/", href)
        elif href.startswith("~"):
            current_dir = current_parsed.path.rstrip("/") + "/"
            absolute = urllib.parse.urljoin(
                f"{current_parsed.scheme}://{current_parsed.netloc}{current_dir}",
                href,
            )
        else:
            absolute = urllib.parse.urljoin(base_url.rstrip("/") + "/", href)

    parsed = urllib.parse.urlparse(absolute)
    normalized_path = posixpath.normpath(parsed.path or "/")
    if parsed.path.endswith("/") and not normalized_path.endswith("/"):
        normalized_path += "/"
    if normalized_path.endswith("/index.html"):
        normalized_path = normalized_path[:-11] or "/"
    normalized_path = _collapse_duplicate_segments(normalized_path)
    if normalized_path == "/.":
        normalized_path = "/"

    normalized_query = urllib.parse.quote_plus(
        urllib.parse.unquote_plus(parsed.query),
        safe="=&~_-.",
    )

    return parsed._replace(
        path=normalized_path,
        query=normalized_query,
        fragment="",
    ).geturl()


def is_binary_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in BINARY_EXTENSIONS)


def is_static_asset_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in STATIC_ASSET_EXTENSIONS)


def is_same_department_url(url: str, base_url: str) -> bool:
    base = urllib.parse.urlparse(base_url.rstrip("/"))
    candidate = urllib.parse.urlparse(url)
    base_prefix = base.path.rstrip("/")
    candidate_path = candidate.path.rstrip("/")
    return (
        candidate.scheme in {"http", "https"}
        and candidate.netloc == base.netloc
        and (
            candidate_path == base_prefix
            or candidate_path.startswith(base_prefix + "/")
        )
    )


def classify_discovered_url(url: str, base_url: str, allowed_domain: str) -> Tuple[str, str]:
    """Classify a normalized URL for crawling."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "reject", "unsupported-scheme"
    if parsed.netloc != allowed_domain:
        return "reject", "external-domain"
    if is_static_asset_url(url):
        return "reject", "static-asset"
    if is_binary_url(url):
        return "binary", "binary-file"
    if is_same_department_url(url, base_url):
        return "page", "department-page"
    return "reject", "outside-department-scope"


def is_generic_page(title: str, text: str) -> bool:
    compact_text = re.sub(r"\s+", " ", text or "").strip()
    has_generic_title = (title or "").strip() == GENERIC_TITLE
    has_generic_header = GENERIC_HEADER in compact_text
    has_footer_only_signature = "Copyright © 2020 IIT Jammu, all rights reserved" in compact_text

    if not has_generic_title and not has_generic_header:
        return False
    if len(compact_text) <= 350:
        return True
    if has_footer_only_signature and len(compact_text) <= 500:
        return True
    return False


def is_generic_content(markdown: str) -> bool:
    """Return ``True`` if the extracted markdown looks like boilerplate only."""
    return GENERIC_HEADER in (markdown or "")
