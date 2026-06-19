#!/usr/bin/env python3
"""
Crawl Student-related websites at IIT Jammu.
Uses the modular scraping and conversion logic from crawler.py.
"""

import json
import os
import sys
import urllib.parse
from typing import Dict, List, Optional, Set
from playwright.sync_api import sync_playwright

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from crawler import (
    discover_site,
    download_and_convert,
    _page_urls_to_export,
    _clear_output_dir
)
from departments import get_scraped_data_root

# Map of slug to base URL
STUDENT_URLS = {
    "pmrf": "https://iitjammu.ac.in/pmrf",
    "online-education": "https://iitjammu.ac.in/online-education/",
    "academics-general-downloads": "https://www.iitjammu.ac.in/academics/academics-general-downloads",
    "calendar-schedule-time-table": "https://iitjammu.ac.in/calendar-schedule-time-table",
    "ugadm": "https://iitjammu.ac.in/ugadm",
    "faq-main-website": "https://iitjammu.ac.in/faq-main-website",
    "why-iitjammu": "https://www.iitjammu.ac.in/why-iitjammu",
    "pg-admissions": "https://iitjammu.ac.in/pg-admissions",
    "phd": "https://iitjammu.ac.in/phd",
    "visvesvaraya-phd": "https://iitjammu.ac.in/visvesvaraya-phd/",
    "certificate-programs": "https://iitjammu.ac.in/certificate-programs"
}

def crawl_student_sites(clean: bool = False, max_pages: Optional[int] = None, slug_to_crawl: Optional[str] = None):
    students_root = os.path.join(get_scraped_data_root(), "students")
    os.makedirs(students_root, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )

        for slug, base_url in STUDENT_URLS.items():
            if slug_to_crawl and slug != slug_to_crawl:
                continue
            output_dir = os.path.join(students_root, slug)
            os.makedirs(output_dir, exist_ok=True)

            if clean:
                print(f"Cleaning previous output in: {output_dir}")
                _clear_output_dir(output_dir)

            print("\n==================================================")
            print(f"Crawling student site: {slug}")
            print(f"Base URL: {base_url}")
            print(f"Output Directory: {output_dir}")
            print("==================================================")

            try:
                pages, binaries, page_snapshots, page_decisions, link_decisions = discover_site(
                    base_url,
                    context,
                    max_pages=max_pages,
                )
                page_urls_to_export = _page_urls_to_export(page_decisions, page_snapshots)
                print(
                    f"Validated {len(pages)} HTML pages, rendered {len(page_urls_to_export)} HTML pages, "
                    f"and discovered {len(binaries)} binary files.\n"
                )

                print("Phase 2: Converting discovered content to Markdown...")
                combined_sections = [
                    f"# IIT Jammu Student Site: {slug} – Crawled Content",
                    "",
                    f"Base URL: {base_url}",
                ]

                # Convert pages to markdown
                for url in page_urls_to_export:
                    result = download_and_convert(
                        url,
                        output_dir,
                        context,
                        slug,
                        base_url,
                        cached_snapshot=page_snapshots.get(url),
                    )
                    if result:
                        markdown, content_flags = result
                        combined_sections.extend([
                            "",
                            "---",
                            "",
                            f"## {urllib.parse.urlparse(url).path or '/'}",
                            "",
                            f"Source URL: {url}",
                            "",
                            f"Content Flags: {', '.join(content_flags) if content_flags else 'none'}",
                            "",
                            markdown,
                        ])

                # Convert binaries to markdown
                for url in binaries:
                    print(f"Downloading binary file: {url}")
                    result = download_and_convert(
                        url,
                        output_dir,
                        context,
                        slug,
                        base_url,
                    )
                    if result:
                        markdown, content_flags = result
                        combined_sections.extend([
                            "",
                            "---",
                            "",
                            f"## {urllib.parse.urlparse(url).path or '/'}",
                            "",
                            f"Source URL: {url}",
                            "",
                            f"Content Flags: {', '.join(content_flags) if content_flags else 'none'}",
                            "",
                            markdown,
                        ])

                # Save combined file
                combined_path = os.path.join(output_dir, f"00_combined_{slug}_site.md")
                with open(combined_path, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(combined_sections).strip() + "\n")

                # Save manifest
                manifest = {
                    "slug": slug,
                    "base_url": base_url,
                    "accepted_pages": pages,
                    "rendered_pages": page_urls_to_export,
                    "accepted_binaries": binaries,
                    "page_decisions": [asdict_to_dict(item) for item in page_decisions],
                    "link_decisions": [asdict_to_dict(item) for item in link_decisions],
                }
                manifest_path = os.path.join(output_dir, "crawl_manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2)

                print(f"✓ Crawl completed for {slug}.")
            except Exception as e:
                print(f"✗ Failed crawling student site {slug}: {e}")

        browser.close()

def asdict_to_dict(obj) -> dict:
    from dataclasses import asdict
    return asdict(obj)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IIT Jammu Student Sites Crawler")
    parser.add_argument("--clean", action="store_true", help="Clean output directories before crawling")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to crawl per site (for debugging)")
    parser.add_argument("--slug", type=str, default=None, help="Specific slug to crawl (e.g. certificate-programs)")
    args = parser.parse_args()

    crawl_student_sites(clean=args.clean, max_pages=args.max_pages, slug_to_crawl=args.slug)
