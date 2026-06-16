#!/usr/bin/env python3
"""
Second-pass Google Drive downloader using multiple strategies:
1. Direct download with confirm=t parameter
2. Wget with --no-check-certificate 
3. curl with cookie handling

Then re-parse all successfully downloaded PDFs.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pdfplumber

BASE_DIR = Path(__file__).resolve().parent.parent
ACADEMICS = BASE_DIR / "scraped_data" / "sections" / "academics"
DL_CACHE  = ACADEMICS / "pdf_cache"
OUT_ROOT  = ACADEMICS / "parsed_documents"


def detect_file_type(path):
    with open(path, 'rb') as f:
        header = f.read(8)
    if header[:5] == b'%PDF-':
        return 'pdf'
    if header[:4] == b'PK\x03\x04':
        return 'xlsx_or_docx'
    if b'<html' in header.lower() or b'<!doc' in header.lower() or b'<HTML' in header:
        return 'html'
    return 'unknown'


def sanitise_filename(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_. ')
    if len(name) > 180:
        name = name[:180]
    return name


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    lines = [l.strip() for l in text.split('\n')]
    return '\n'.join(lines)


def format_table(table, idx):
    if not table or not table[0]:
        return ""
    parts = ["\n### Table {}\n".format(idx)]
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
    parts.append("| " + " | ".join(normalised[0]) + " |")
    parts.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in normalised[1:]:
        parts.append("| " + " | ".join(row) + " |")
    parts.append("")
    return "\n".join(parts)


def pdf_to_markdown(pdf_path, title, source_url):
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


def download_with_curl(file_id, dest_path):
    """Use curl with proper cookie handling to download from Google Drive."""
    tmp = str(dest_path) + ".tmp"
    
    # Strategy 1: Direct download with confirm=t
    cmd = [
        'curl', '-L', '-o', tmp,
        '-s', '--max-time', '120',
        'https://drive.google.com/uc?export=download&confirm=t&id={}'.format(file_id)
    ]
    try:
        subprocess.run(cmd, timeout=130, capture_output=True)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            ftype = detect_file_type(Path(tmp))
            if ftype == 'pdf':
                os.rename(tmp, str(dest_path))
                return True
    except Exception:
        pass

    # Strategy 2: Two-step curl with cookies
    cookie_file = str(dest_path) + ".cookies"
    try:
        # First request to get cookies
        cmd1 = [
            'curl', '-s', '-c', cookie_file, '-L',
            'https://drive.google.com/uc?export=download&id={}'.format(file_id),
            '-o', '/dev/null'
        ]
        subprocess.run(cmd1, timeout=30, capture_output=True)

        # Extract confirm token from cookie file
        confirm = 't'
        if os.path.exists(cookie_file):
            with open(cookie_file) as f:
                for line in f:
                    if 'download_warning' in line:
                        parts = line.strip().split('\t')
                        if parts:
                            confirm = parts[-1]

        # Second request with cookies and confirm
        cmd2 = [
            'curl', '-L', '-o', tmp,
            '-s', '--max-time', '120',
            '-b', cookie_file,
            'https://drive.google.com/uc?export=download&confirm={}&id={}'.format(confirm, file_id)
        ]
        subprocess.run(cmd2, timeout=130, capture_output=True)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            ftype = detect_file_type(Path(tmp))
            if ftype == 'pdf':
                os.rename(tmp, str(dest_path))
                return True
    except Exception:
        pass
    finally:
        for f in [cookie_file, tmp]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    # Strategy 3: Try wget
    try:
        cmd = [
            'wget', '--no-check-certificate', '-q',
            '-O', tmp,
            'https://drive.google.com/uc?export=download&confirm=t&id={}'.format(file_id)
        ]
        subprocess.run(cmd, timeout=130, capture_output=True)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            ftype = detect_file_type(Path(tmp))
            if ftype == 'pdf':
                os.rename(tmp, str(dest_path))
                return True
    except Exception:
        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

    return False


def main():
    print("=" * 70)
    print("  Google Drive Downloader (Pass 2 - curl/wget)")
    print("=" * 70)

    # Find all HTML files in cache that need re-downloading
    html_ids = []
    for fname in sorted(os.listdir(str(DL_CACHE))):
        fpath = DL_CACHE / fname
        if not fpath.is_file() or fname.startswith('.') or fname.endswith('.csv'):
            continue
        ftype = detect_file_type(fpath)
        if ftype == 'html':
            html_ids.append(fname)

    print("  Found {} HTML files that need re-downloading\n".format(len(html_ids)))

    success = 0
    failed = 0
    
    for i, file_id in enumerate(html_ids):
        dest = DL_CACHE / file_id
        print("  [{}/{}] {}".format(i + 1, len(html_ids), file_id[:40]), end='')

        ok = download_with_curl(file_id, dest)
        if ok:
            success += 1
            print("  ✓ PDF")
        else:
            failed += 1
            print("  ✗")

        # Rate limit
        if (i + 1) % 10 == 0:
            time.sleep(2)
        else:
            time.sleep(0.5)

    print("\n  Download: {} success, {} failed".format(success, failed))

    # Now re-parse all downloaded PDFs
    print("\n  Re-parsing documents...")
    reparse_documents()

    # Generate final statistics
    generate_final_stats()


def reparse_documents():
    """Re-parse all documents, replacing stubs with real content."""
    manifest_path = OUT_ROOT / "download_manifest.json"
    if not manifest_path.exists():
        print("  No manifest found!")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    updated = 0
    for category, entries in manifest.items():
        for entry in entries:
            url = entry.get('url', '')
            output_path = entry.get('output', '')
            title = entry.get('title', '')
            if not url or not output_path:
                continue
            m = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
            if not m:
                continue
            file_id = m.group(1)
            cached = DL_CACHE / file_id
            if not cached.exists():
                continue
            ftype = detect_file_type(cached)
            if ftype != 'pdf':
                continue
            out = Path(output_path)
            if not out.exists():
                continue
            current = out.read_text(encoding='utf-8', errors='replace')
            if 'direct Google Drive access' in current or len(current) < 500:
                md = pdf_to_markdown(cached, title, url)
                out.write_text(md, encoding='utf-8')
                updated += 1
                if updated <= 10:
                    print("    ✓ Updated: {}".format(title[:60]))
    
    print("  Total updated: {}".format(updated))


def generate_final_stats():
    """Print comprehensive statistics."""
    pdf_count = 0
    html_count = 0
    other_count = 0
    total = 0
    
    for fname in os.listdir(str(DL_CACHE)):
        fpath = DL_CACHE / fname
        if not fpath.is_file() or fname.startswith('.') or fname.endswith('.csv'):
            continue
        total += 1
        ftype = detect_file_type(fpath)
        if ftype == 'pdf':
            pdf_count += 1
        elif ftype == 'html':
            html_count += 1
        else:
            other_count += 1

    md_count = 0
    stub_count = 0
    good_count = 0
    for root, dirs, files in os.walk(str(OUT_ROOT)):
        for f in files:
            if f.endswith('.md') and f != '00_index.md':
                md_count += 1
                fpath = Path(root) / f
                content = fpath.read_text(encoding='utf-8', errors='replace')
                if 'direct Google Drive access' in content or 'Download failed' in content:
                    stub_count += 1
                elif len(content) > 500:
                    good_count += 1

    print("\n" + "=" * 70)
    print("  FINAL STATISTICS")
    print("=" * 70)
    print("  Cache files     : {} total ({} PDFs, {} HTML, {} other)".format(
        total, pdf_count, html_count, other_count))
    print("  Markdown files  : {} total".format(md_count))
    print("  Good (parsed)   : {}".format(good_count))
    print("  Stubs (no data) : {}".format(stub_count))
    print("=" * 70)


if __name__ == "__main__":
    main()
