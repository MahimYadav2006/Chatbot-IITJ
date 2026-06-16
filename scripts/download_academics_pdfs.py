#!/usr/bin/env python3
"""
Download and parse all Google Drive PDFs/documents from the IIT Jammu
Academics section.

Categories handled:
  1. Rules & Regulations  (PG / M.Tech / Ph.D / UG)
  2. General Downloads
  3. Specialisation & Courses  (PG / UG)
  4. Academic Notifications

Outputs beautifully formatted Markdown files organised into subfolders
under  scraped_data/sections/academics/parsed_documents/
"""

import hashlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import gdown
import pdfplumber
import requests

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ACADEMICS = BASE_DIR / "scraped_data" / "sections" / "academics"
OUT_ROOT  = ACADEMICS / "parsed_documents"
DL_CACHE  = ACADEMICS / "pdf_cache"          # raw downloads go here

# ── helpers ──────────────────────────────────────────────────────────────

def extract_file_id(url: str) -> str:
    """Extract Google Drive file id from various URL formats."""
    # drive.google.com/file/d/FILE_ID/...
    m = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    # drive.google.com/open?id=FILE_ID
    m = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    return None


def extract_spreadsheet_id(url: str) -> str:
    """Extract Google Spreadsheet id."""
    m = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    return None


def extract_folder_id(url: str) -> str:
    """Extract Google Drive folder id."""
    m = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    return None


def sanitise_filename(name: str) -> str:
    """Create a filesystem-safe filename from a document title."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_. ')
    if len(name) > 180:
        name = name[:180]
    return name


def download_gdrive_file(file_id: str, dest: Path, quiet: bool = True) -> bool:
    """Download a file from Google Drive using gdown. Returns True on success."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        result = gdown.download(url, str(dest), quiet=quiet, fuzzy=True)
        if result and dest.exists() and dest.stat().st_size > 0:
            return True
    except Exception as e:
        print(f"    ⚠  gdown failed for {file_id}: {e}")

    # fallback: direct requests download
    try:
        export_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        sess = requests.Session()
        resp = sess.get(export_url, stream=True, timeout=60)
        # handle Google's virus scan confirmation
        for key, value in resp.cookies.items():
            if key.startswith('download_warning'):
                export_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                resp = sess.get(export_url, stream=True, timeout=60)
                break
        if resp.status_code == 200:
            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(32768):
                    f.write(chunk)
            if dest.stat().st_size > 0:
                return True
    except Exception as e2:
        print(f"    ⚠  requests fallback failed for {file_id}: {e2}")
    return False


def download_spreadsheet_as_csv(sheet_id: str, dest: Path) -> bool:
    """Download a Google Spreadsheet as CSV."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 10:
            dest.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"    ⚠  Spreadsheet download failed for {sheet_id}: {e}")
    return False


def list_folder_files(folder_id: str) -> list:
    """
    Use gdown to list files in a public Google Drive folder.
    Returns list of dicts with 'id' and 'name'.
    """
    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        files = gdown.download_folder(url, quiet=True, skip_download=True,
                                       output=str(DL_CACHE / "folder_scan"))
        # gdown.download_folder with skip_download returns list of paths – 
        # but we need the IDs. Let's parse the folder page instead.
    except Exception:
        pass

    # Parse the folder HTML directly to find file IDs
    file_list = []
    try:
        resp = requests.get(
            f"https://drive.google.com/drive/folders/{folder_id}",
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if resp.status_code == 200:
            # Find all file IDs in the page source
            file_ids = re.findall(r'/file/d/([a-zA-Z0-9_-]{20,})', resp.text)
            # Find file names from the page
            names = re.findall(r'data-tooltip="([^"]+?\.(?:pdf|docx?|xlsx?|pptx?|csv))"', resp.text, re.I)
            seen = set()
            for fid in file_ids:
                if fid not in seen:
                    seen.add(fid)
                    file_list.append({"id": fid, "name": None})
    except Exception as e:
        print(f"    ⚠  Folder listing failed for {folder_id}: {e}")

    return file_list


# ── PDF → Markdown ──────────────────────────────────────────────────────

def detect_file_type(path: Path) -> str:
    """Detect file type from magic bytes."""
    with open(path, 'rb') as f:
        header = f.read(8)
    if header[:5] == b'%PDF-':
        return 'pdf'
    if header[:4] == b'PK\x03\x04':
        return 'xlsx_or_docx'
    if header[:2] in (b'\xff\xfe', b'\xfe\xff') or header[:3] == b'\xef\xbb\xbf':
        return 'text'
    if b'<html' in header.lower() or b'<!doc' in header.lower():
        return 'html'
    # Try to read as text
    try:
        path.read_text(encoding='utf-8', errors='strict')[:200]
        return 'text'
    except Exception:
        pass
    return 'unknown'


def pdf_to_markdown(pdf_path: Path, title: str, source_url: str) -> str:
    """Parse PDF using pdfplumber and produce beautifully formatted markdown."""
    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"> **Source**: [{source_url}]({source_url})\n")
    lines.append("---\n")

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            lines.append(f"> **Pages**: {total_pages}\n")

            for page_num, page in enumerate(pdf.pages, 1):
                # ── tables ──
                tables = page.extract_tables()
                # ── text ──
                text = page.extract_text() or ""

                if total_pages > 1:
                    lines.append(f"\n## Page {page_num}\n")

                if tables:
                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue
                        lines.append(format_table(table, table_idx + 1))
                        # remove table text from the page text to avoid duplication
                        for row in table:
                            for cell in (row or []):
                                if cell:
                                    text = text.replace(str(cell), "", 1)

                # clean up remaining text
                cleaned = clean_text(text)
                if cleaned.strip():
                    lines.append(cleaned + "\n")

    except Exception as e:
        lines.append(f"\n> ⚠ **PDF parsing error**: {e}\n")
        # Fallback: try pdfminer
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(pdf_path))
            if text and text.strip():
                lines.append("\n### Content (fallback extraction)\n")
                lines.append(clean_text(text) + "\n")
        except Exception:
            lines.append("> Could not extract text from this PDF.\n")

    return "\n".join(lines)


def format_table(table: list, idx: int) -> str:
    """Format an extracted table as a Markdown table."""
    if not table or not table[0]:
        return ""
    parts = []
    parts.append(f"\n### Table {idx}\n")

    # Normalise rows to same width
    max_cols = max(len(row or []) for row in table)
    normalised = []
    for row in table:
        if not row:
            continue
        cells = [(str(c).replace('\n', ' ').strip() if c else "") for c in row]
        while len(cells) < max_cols:
            cells.append("")
        normalised.append(cells)

    if not normalised:
        return ""

    # Header row
    header = normalised[0]
    parts.append("| " + " | ".join(header) + " |")
    parts.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in normalised[1:]:
        parts.append("| " + " | ".join(row) + " |")
    parts.append("")
    return "\n".join(parts)


def clean_text(text: str) -> str:
    """Clean extracted text: normalise whitespace, preserve paragraph breaks."""
    if not text:
        return ""
    # Collapse runs of 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove runs of spaces but keep single newlines
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Remove leading/trailing whitespace per line
    lines = [l.strip() for l in text.split('\n')]
    return '\n'.join(lines)


def csv_to_markdown(csv_path: Path, title: str, source_url: str) -> str:
    """Convert downloaded CSV into markdown table."""
    import csv
    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"> **Source**: [{source_url}]({source_url})\n")
    lines.append("---\n")
    try:
        content = csv_path.read_text(encoding='utf-8', errors='replace')
        reader = csv.reader(content.splitlines())
        rows = list(reader)
        if not rows:
            lines.append("> Empty spreadsheet.\n")
            return "\n".join(lines)
        max_cols = max(len(r) for r in rows)
        # Header
        header = rows[0]
        while len(header) < max_cols:
            header.append("")
        lines.append("| " + " | ".join(h.replace('\n', ' ').strip() for h in header) + " |")
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in rows[1:]:
            while len(row) < max_cols:
                row.append("")
            lines.append("| " + " | ".join(c.replace('\n', ' ').strip() for c in row) + " |")
        lines.append("")
    except Exception as e:
        lines.append(f"> ⚠ CSV parsing error: {e}\n")
    return "\n".join(lines)


def text_to_markdown(text_path: Path, title: str, source_url: str) -> str:
    """Wrap a plain text file in markdown."""
    content = text_path.read_text(encoding='utf-8', errors='replace')
    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"> **Source**: [{source_url}]({source_url})\n")
    lines.append("---\n")
    lines.append(clean_text(content))
    return "\n".join(lines)


# ── Link extraction from source MD files ────────────────────────────────

def extract_links_from_md(md_path: Path) -> list:
    """
    Parse a scraped markdown file and extract all Google Drive links
    with their display names and hierarchy context.
    Returns list of dicts: {title, url, type, category_path}
    """
    content = md_path.read_text(encoding='utf-8')
    links = []
    context_stack = []

    for line in content.split('\n'):
        # Track heading context
        hm = re.match(r'^(#+)\s+(.*)', line)
        if hm:
            level = len(hm.group(1))
            heading = hm.group(2).strip()
            # Keep only higher-level headings
            context_stack = [c for c in context_stack if c[0] < level]
            context_stack.append((level, heading))
            continue

        # Track numbered list context (like "1. PG", "2. Ph.D", etc.)
        nm = re.match(r'^\s*\d+\.\s+(?!\[)(.+)', line)
        if nm:
            ctx = nm.group(1).strip()
            if not re.search(r'https?://', ctx):
                context_stack = [c for c in context_stack if c[0] < 99]
                context_stack.append((99, ctx))

        # Find all markdown links
        for match in re.finditer(r'\[([^\]]+)\]\((https?://[^)]+)\)', line):
            title = match.group(1).strip()
            url = match.group(2).strip()

            link_type = 'unknown'
            if 'drive.google.com/file/d/' in url:
                link_type = 'file'
            elif 'docs.google.com/spreadsheets' in url:
                link_type = 'spreadsheet'
            elif 'drive.google.com/drive/folders/' in url:
                link_type = 'folder'
            else:
                continue  # skip non-google-drive links

            cat_parts = [c[1] for c in context_stack if c[1]]
            links.append({
                'title': title,
                'url': url,
                'type': link_type,
                'category_path': cat_parts,
            })

    return links


# ── Source files config ──────────────────────────────────────────────────

SOURCE_FILES = [
    {
        "md_file": "academics_academics-rules-and-regulations.md",
        "output_folder": "rules_and_regulations",
        "label": "Rules & Regulations",
    },
    {
        "md_file": "academics_academics-general-downloads.md",
        "output_folder": "general_downloads",
        "label": "General Downloads",
    },
    {
        "md_file": "academics_academics-specialisation-and-courses.md",
        "output_folder": "specialisation_and_courses",
        "label": "Specialisation & Courses",
    },
    {
        "md_file": "academics_academic-notifications.md",
        "output_folder": "academic_notifications",
        "label": "Academic Notifications",
    },
]


# ── Main pipeline ────────────────────────────────────────────────────────

def process_link(link: dict, output_dir: Path, stats: dict) -> str:
    """Download and parse a single link. Returns output md path or None."""
    title = link['title']
    url = link['url']
    link_type = link['type']

    # Build subfolder from category path
    cat_parts = link.get('category_path', [])
    sub_dir = output_dir
    for part in cat_parts:
        safe_part = sanitise_filename(part)
        if safe_part:
            sub_dir = sub_dir / safe_part
    sub_dir.mkdir(parents=True, exist_ok=True)

    safe_title = sanitise_filename(title)
    if not safe_title:
        safe_title = "document"
    md_out = sub_dir / f"{safe_title}.md"

    # Skip if already exists and non-empty
    if md_out.exists() and md_out.stat().st_size > 200:
        print(f"  ✓ Already parsed: {safe_title}")
        stats['skipped'] += 1
        return str(md_out)

    if link_type == 'file':
        file_id = extract_file_id(url)
        if not file_id:
            print(f"  ✗ Cannot extract file ID from {url}")
            stats['failed'] += 1
            return None

        # download
        dl_dest = DL_CACHE / f"{file_id}"
        if not dl_dest.exists() or dl_dest.stat().st_size == 0:
            print(f"  ↓ Downloading: {safe_title[:60]}...")
            ok = download_gdrive_file(file_id, dl_dest)
            if not ok:
                stats['failed'] += 1
                # Write a stub so we know it failed
                md_out.write_text(
                    f"# {title}\n\n> **Source**: [{url}]({url})\n\n"
                    f"> ⚠ **Download failed** — this document could not be "
                    f"retrieved from Google Drive. It may require special "
                    f"access permissions.\n",
                    encoding='utf-8',
                )
                return str(md_out)
            time.sleep(0.5)  # rate limit kindness

        # detect and parse
        ftype = detect_file_type(dl_dest)
        if ftype == 'pdf':
            md_content = pdf_to_markdown(dl_dest, title, url)
        elif ftype == 'text':
            md_content = text_to_markdown(dl_dest, title, url)
        elif ftype == 'html':
            # Google might serve an HTML interstitial
            md_content = (
                f"# {title}\n\n> **Source**: [{url}]({url})\n\n"
                f"> This file requires direct Google Drive access to view.\n"
            )
        else:
            md_content = (
                f"# {title}\n\n> **Source**: [{url}]({url})\n\n"
                f"> File type: {ftype}. Binary content not rendered.\n"
            )

        md_out.write_text(md_content, encoding='utf-8')
        stats['success'] += 1
        print(f"  ✓ Parsed: {safe_title[:60]}")
        return str(md_out)

    elif link_type == 'spreadsheet':
        sheet_id = extract_spreadsheet_id(url)
        if not sheet_id:
            print(f"  ✗ Cannot extract spreadsheet ID from {url}")
            stats['failed'] += 1
            return None

        csv_dest = DL_CACHE / f"{sheet_id}.csv"
        if not csv_dest.exists() or csv_dest.stat().st_size == 0:
            print(f"  ↓ Downloading spreadsheet: {safe_title[:60]}...")
            ok = download_spreadsheet_as_csv(sheet_id, csv_dest)
            if not ok:
                stats['failed'] += 1
                md_out.write_text(
                    f"# {title}\n\n> **Source**: [{url}]({url})\n\n"
                    f"> ⚠ **Download failed** — spreadsheet could not be exported.\n",
                    encoding='utf-8',
                )
                return str(md_out)
            time.sleep(0.5)

        md_content = csv_to_markdown(csv_dest, title, url)
        md_out.write_text(md_content, encoding='utf-8')
        stats['success'] += 1
        print(f"  ✓ Parsed spreadsheet: {safe_title[:60]}")
        return str(md_out)

    elif link_type == 'folder':
        folder_id = extract_folder_id(url)
        if not folder_id:
            stats['failed'] += 1
            return None

        print(f"  📁 Processing folder: {safe_title[:60]}...")
        folder_dir = sub_dir / safe_title
        folder_dir.mkdir(parents=True, exist_ok=True)

        file_list = list_folder_files(folder_id)
        if not file_list:
            # Write a placeholder
            readme = folder_dir / "_README.md"
            readme.write_text(
                f"# {title}\n\n> **Source**: [{url}]({url})\n\n"
                f"> This Google Drive folder could not be enumerated. "
                f"Visit the link above to access its contents.\n",
                encoding='utf-8',
            )
            stats['folder_stubs'] += 1
            return str(readme)

        folder_outputs = []
        for finfo in file_list:
            fid = finfo['id']
            fname = finfo.get('name') or fid
            dl_dest = DL_CACHE / fid
            if not dl_dest.exists() or dl_dest.stat().st_size == 0:
                ok = download_gdrive_file(fid, dl_dest)
                if not ok:
                    continue
                time.sleep(0.5)

            ftype = detect_file_type(dl_dest)
            safe_fname = sanitise_filename(fname)
            if not safe_fname:
                safe_fname = fid
            file_md = folder_dir / f"{safe_fname}.md"
            if ftype == 'pdf':
                mc = pdf_to_markdown(dl_dest, fname, f"https://drive.google.com/file/d/{fid}/view")
            elif ftype == 'text':
                mc = text_to_markdown(dl_dest, fname, f"https://drive.google.com/file/d/{fid}/view")
            else:
                mc = f"# {fname}\n\n> File type: {ftype}\n"
            file_md.write_text(mc, encoding='utf-8')
            folder_outputs.append(str(file_md))
            stats['success'] += 1

        return str(folder_dir)

    return None


def main():
    print("=" * 70)
    print("  IIT Jammu Academics PDF Downloader & Parser")
    print("=" * 70)

    DL_CACHE.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    global_stats = {
        'total_links': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'folder_stubs': 0,
    }
    manifest = {}  # category → list of output files

    for src in SOURCE_FILES:
        md_path = ACADEMICS / src['md_file']
        if not md_path.exists():
            print(f"\n⚠  Source file not found: {md_path}")
            continue

        label = src['label']
        out_dir = OUT_ROOT / src['output_folder']
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─' * 70}")
        print(f"  📂 {label}")
        print(f"     Source: {src['md_file']}")
        print(f"     Output: {out_dir.relative_to(BASE_DIR)}")
        print(f"{'─' * 70}")

        links = extract_links_from_md(md_path)
        print(f"  Found {len(links)} links\n")
        global_stats['total_links'] += len(links)

        cat_outputs = []
        for i, link in enumerate(links, 1):
            print(f"  [{i}/{len(links)}] {link['title'][:70]}")
            result = process_link(link, out_dir, global_stats)
            if result:
                cat_outputs.append({
                    'title': link['title'],
                    'url': link['url'],
                    'output': result,
                    'category': ' > '.join(link.get('category_path', [])),
                })

        manifest[label] = cat_outputs

    # ── write manifest ──
    manifest_path = OUT_ROOT / "download_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── write combined index ──
    write_index(manifest)

    # ── summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total links found : {global_stats['total_links']}")
    print(f"  Successfully parsed: {global_stats['success']}")
    print(f"  Skipped (cached)  : {global_stats['skipped']}")
    print(f"  Failed downloads  : {global_stats['failed']}")
    print(f"  Folder stubs      : {global_stats['folder_stubs']}")
    print(f"  Manifest          : {manifest_path}")
    print(f"{'=' * 70}")


def write_index(manifest: dict):
    """Write a combined index markdown file for all parsed documents."""
    idx = OUT_ROOT / "00_index.md"
    lines = [
        "# IIT Jammu Academics — Parsed Documents Index\n",
        "> Auto-generated index of all downloaded and parsed academic documents.\n",
        "---\n",
    ]
    for category, entries in manifest.items():
        lines.append(f"\n## {category}\n")
        if not entries:
            lines.append("_No documents parsed._\n")
            continue
        # Group by subcategory
        by_cat: dict = {}
        for e in entries:
            cat = e.get('category', '') or 'General'
            by_cat.setdefault(cat, []).append(e)
        for subcat, items in by_cat.items():
            if subcat and subcat != 'General':
                lines.append(f"\n### {subcat}\n")
            for item in items:
                rel = os.path.relpath(item['output'], OUT_ROOT)
                lines.append(f"- [{item['title']}]({rel})")
        lines.append("")

    idx.write_text("\n".join(lines), encoding='utf-8')
    print(f"\n  📋 Index written: {idx}")


if __name__ == "__main__":
    main()
