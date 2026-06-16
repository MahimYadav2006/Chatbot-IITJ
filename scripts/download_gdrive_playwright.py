#!/usr/bin/env python3
"""
Download Google Drive files using Playwright to handle the confirmation
pages. Parses the downloaded PDFs into structured Markdown.

This is a second-pass script that processes files which the initial
gdown/requests pass couldn't download (saved as HTML interstitials).
"""

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pdfplumber

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ACADEMICS = BASE_DIR / "scraped_data" / "sections" / "academics"
DL_CACHE  = ACADEMICS / "pdf_cache"
OUT_ROOT  = ACADEMICS / "parsed_documents"


def detect_file_type(path):
    """Detect file type from magic bytes."""
    with open(path, 'rb') as f:
        header = f.read(8)
    if header[:5] == b'%PDF-':
        return 'pdf'
    if header[:4] == b'PK\x03\x04':
        return 'xlsx_or_docx'
    if b'<html' in header.lower() or b'<!doc' in header.lower() or b'<HTML' in header:
        return 'html'
    try:
        path.read_text(encoding='utf-8', errors='strict')[:200]
        return 'text'
    except Exception:
        pass
    return 'unknown'


def sanitise_filename(name):
    """Create a filesystem-safe filename from a document title."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_. ')
    if len(name) > 180:
        name = name[:180]
    return name


def clean_text(text):
    """Clean extracted text."""
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    lines = [l.strip() for l in text.split('\n')]
    return '\n'.join(lines)


def format_table(table, idx):
    """Format an extracted table as Markdown."""
    if not table or not table[0]:
        return ""
    parts = []
    parts.append("\n### Table {}\n".format(idx))

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

    header = normalised[0]
    parts.append("| " + " | ".join(header) + " |")
    parts.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in normalised[1:]:
        parts.append("| " + " | ".join(row) + " |")
    parts.append("")
    return "\n".join(parts)


def pdf_to_markdown(pdf_path, title, source_url):
    """Parse PDF into beautiful Markdown."""
    lines = []
    lines.append("# {}\n".format(title))
    lines.append("> **Source**: [{}]({})\n".format(source_url, source_url))
    lines.append("---\n")

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            lines.append("> **Pages**: {}\n".format(total_pages))

            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                text = page.extract_text() or ""

                if total_pages > 1:
                    lines.append("\n## Page {}\n".format(page_num))

                if tables:
                    for table_idx, table in enumerate(tables):
                        if not table:
                            continue
                        lines.append(format_table(table, table_idx + 1))
                        for row in table:
                            for cell in (row or []):
                                if cell:
                                    text = text.replace(str(cell), "", 1)

                cleaned = clean_text(text)
                if cleaned.strip():
                    lines.append(cleaned + "\n")

    except Exception as e:
        lines.append("\n> PDF parsing error: {}\n".format(e))
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(str(pdf_path))
            if text and text.strip():
                lines.append("\n### Content (fallback extraction)\n")
                lines.append(clean_text(text) + "\n")
        except Exception:
            lines.append("> Could not extract text from this PDF.\n")

    return "\n".join(lines)


def get_html_file_ids():
    """Find all cached files that are HTML (need re-downloading)."""
    html_ids = []
    for fname in sorted(os.listdir(str(DL_CACHE))):
        fpath = DL_CACHE / fname
        if not fpath.is_file() or fname.startswith('.') or fname.endswith('.csv'):
            continue
        ftype = detect_file_type(fpath)
        if ftype == 'html':
            html_ids.append(fname)
    return html_ids


def download_batch_with_playwright(file_ids, batch_size=5):
    """
    Use Playwright to download a batch of Google Drive files.
    Opens browser, navigates to the file view page, and clicks download.
    """
    from playwright.sync_api import sync_playwright

    success = 0
    failed = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for i, file_id in enumerate(file_ids):
            dest_path = DL_CACHE / file_id
            url = "https://drive.google.com/uc?export=download&id={}".format(file_id)

            print("  [{}/{}] Downloading {}...".format(i + 1, len(file_ids), file_id[:30]))
            context = None
            try:
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()

                # Navigate to the direct download URL
                page.goto(url, wait_until='domcontentloaded', timeout=30000)
                time.sleep(1)

                # Check if there's a confirmation page (virus scan warning)
                download = None

                # Try to find and click download button/link
                # Google shows different pages:
                # 1. Direct download (small files) - file is downloaded immediately
                # 2. Virus scan warning page - needs clicking "Download anyway"
                # 3. Sign-in required page

                # Method 1: Try the direct export URL with confirmation
                confirm_link = page.query_selector('a[href*="confirm="]')
                if confirm_link:
                    with page.expect_download(timeout=60000) as dl_info:
                        confirm_link.click()
                    download = dl_info.value
                else:
                    # Method 2: Try form submit button
                    form_btn = page.query_selector('form[id="download-form"] input[type="submit"]')
                    if not form_btn:
                        form_btn = page.query_selector('form input[type="submit"]')
                    if not form_btn:
                        form_btn = page.query_selector('#uc-download-link')

                    if form_btn:
                        with page.expect_download(timeout=60000) as dl_info:
                            form_btn.click()
                        download = dl_info.value
                    else:
                        # Method 3: Check if file already downloaded (redirect)
                        current_url = page.url
                        if 'drive.usercontent.google.com' in current_url:
                            # We're on the download page; the file should auto-download
                            # or we need to submit the form
                            any_btn = page.query_selector('button, input[type="submit"], a[href*="download"]')
                            if any_btn:
                                with page.expect_download(timeout=60000) as dl_info:
                                    any_btn.click()
                                download = dl_info.value

                if download:
                    download.save_as(str(dest_path))
                    if dest_path.stat().st_size > 100:
                        ftype = detect_file_type(dest_path)
                        if ftype == 'pdf':
                            success += 1
                            print("    ✓ Downloaded PDF ({} bytes)".format(dest_path.stat().st_size))
                        else:
                            print("    ~ Downloaded but type is: {}".format(ftype))
                            success += 1
                    else:
                        print("    ✗ Downloaded file too small")
                        failed += 1
                else:
                    # Try alternative: use the full viewer URL and extract
                    print("    ~ No download button found, trying viewer...")
                    page2 = context.new_page()
                    view_url = "https://drive.google.com/file/d/{}/view".format(file_id)
                    page2.goto(view_url, wait_until='domcontentloaded', timeout=30000)
                    time.sleep(2)

                    # Look for download menu item
                    # Try the kebab menu or download icon
                    dl_btn = page2.query_selector('[aria-label="Download"], [data-tooltip="Download"]')
                    if dl_btn:
                        with page2.expect_download(timeout=60000) as dl_info:
                            dl_btn.click()
                        download = dl_info.value
                        download.save_as(str(dest_path))
                        success += 1
                        print("    ✓ Downloaded via viewer ({} bytes)".format(dest_path.stat().st_size))
                    else:
                        failed += 1
                        print("    ✗ Could not find download mechanism")
                    page2.close()

            except Exception as e:
                print("    ✗ Error: {}".format(str(e)[:100]))
                failed += 1
            finally:
                if context:
                    try:
                        context.close()
                    except Exception:
                        pass

            # Rate limit
            time.sleep(1)

        browser.close()

    return success, failed


def reparse_all_documents():
    """
    After re-downloading, re-parse all documents and update the
    markdown files.
    """
    # Load the manifest to know which file IDs map to which output paths
    manifest_path = OUT_ROOT / "download_manifest.json"
    if not manifest_path.exists():
        print("No manifest found!")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    updated = 0
    still_html = 0

    for category, entries in manifest.items():
        for entry in entries:
            url = entry.get('url', '')
            output_path = entry.get('output', '')
            title = entry.get('title', '')

            if not url or not output_path:
                continue

            # Extract file ID from URL
            m = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
            if not m:
                continue
            file_id = m.group(1)

            cached_file = DL_CACHE / file_id
            if not cached_file.exists():
                continue

            ftype = detect_file_type(cached_file)
            if ftype != 'pdf':
                still_html += 1
                continue

            # Parse the PDF and update the markdown
            output = Path(output_path)
            if not output.exists():
                continue

            # Check if current content is a stub (from HTML file)
            current = output.read_text(encoding='utf-8', errors='replace')
            if 'direct Google Drive access' in current or len(current) < 300:
                # This was a stub, update it
                md_content = pdf_to_markdown(cached_file, title, url)
                output.write_text(md_content, encoding='utf-8')
                updated += 1
                print("  ✓ Updated: {}".format(title[:60]))

    print("\nRe-parse summary: {} updated, {} still HTML".format(updated, still_html))
    return updated


def main():
    print("=" * 70)
    print("  Google Drive Playwright Downloader (Pass 2)")
    print("=" * 70)

    html_ids = get_html_file_ids()
    print("  Found {} HTML files that need re-downloading\n".format(len(html_ids)))

    if not html_ids:
        print("  No HTML files to process!")
        return

    # Download in batches
    total_success = 0
    total_failed = 0

    success, failed = download_batch_with_playwright(html_ids)
    total_success += success
    total_failed += failed

    print("\n" + "=" * 70)
    print("  Download Results: {} success, {} failed".format(total_success, total_failed))
    print("=" * 70)

    # Re-parse all documents
    print("\n  Re-parsing documents...\n")
    reparse_all_documents()


if __name__ == "__main__":
    main()
