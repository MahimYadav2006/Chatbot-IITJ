#!/usr/bin/env python3
"""
Crawl Central Instruments Facility (CIF), Central Workshop, and I3C websites at IIT Jammu.
Uses the modular scraping and conversion logic from crawler.py and saves all results under
the 'ci' section.
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

# Monkeypatch classify_discovered_url to avoid events calendar loops
import crawler
import utils
original_classify = utils.classify_discovered_url

def custom_classify_discovered_url(url, base_url, allowed_domain):
    # Reject WordPress/Tribe calendar pages & feeds
    if any(p in url.lower() for p in ("/events/", "tribe-bar-date", "ical=", "outlook-ical", "tribe-venue")):
        print(f"Skipping calendar-trap link: {url}")
        return "reject", "calendar-trap"
    return original_classify(url, base_url, allowed_domain)

crawler.classify_discovered_url = custom_classify_discovered_url

# Map of slug to base URL for the central instrumentation & facilities / innovation section
CI_URLS = {
    "cif": "https://iitjammu.ac.in/cif",
    "i3c": "https://i3c-iitjammu.in",
    "central-workshop": "https://iitjammu.ac.in/central-workshop"
}

def crawl_ci_sites(clean: bool = False, max_pages: Optional[int] = None, slug_to_crawl: Optional[str] = None):
    ci_root = os.path.join(get_scraped_data_root(), "sections", "ci")
    os.makedirs(ci_root, exist_ok=True)

    if clean and not slug_to_crawl:
        print(f"Cleaning previous output in: {ci_root}")
        _clear_output_dir(ci_root)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )

        all_pages: List[str] = []
        all_binaries: List[str] = []
        all_page_decisions = []
        all_link_decisions = []
        combined_sections = [
            "# IIT Jammu Central Facilities & Innovation (CI) – Crawled Content",
            "",
        ]

        for slug, base_url in CI_URLS.items():
            if slug_to_crawl and slug != slug_to_crawl:
                continue

            print("\n==================================================")
            print(f"Crawling CI sub-site: {slug}")
            print(f"Base URL: {base_url}")
            print(f"Output Directory: {ci_root}")
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
                    f"and discovered {len(binaries)} binary files for {slug}.\n"
                )

                all_pages.extend(pages)
                all_binaries.extend(binaries)
                all_page_decisions.extend([asdict_to_dict(item) for item in page_decisions])
                all_link_decisions.extend([asdict_to_dict(item) for item in link_decisions])

                print(f"Phase 2: Converting discovered content for {slug} to Markdown...")
                combined_sections.append(f"## Sub-site: {slug.upper()} ({base_url})")
                combined_sections.append("")

                # Convert pages to markdown
                for url in page_urls_to_export:
                    result = download_and_convert(
                        url,
                        ci_root,
                        context,
                        "ci",
                        base_url,
                        cached_snapshot=page_snapshots.get(url),
                    )
                    if result:
                        markdown, content_flags = result
                        combined_sections.extend([
                            "",
                            "---",
                            "",
                            f"### {urllib.parse.urlparse(url).path or '/'}",
                            "",
                            f"Source URL: {url}",
                            "",
                            f"Content Flags: {', '.join(content_flags) if content_flags else 'none'}",
                            "",
                            markdown,
                            "",
                        ])

                # Convert binaries to markdown
                for url in binaries:
                    print(f"Downloading binary file: {url}")
                    result = download_and_convert(
                        url,
                        ci_root,
                        context,
                        "ci",
                        base_url,
                    )
                    if result:
                        markdown, content_flags = result
                        combined_sections.extend([
                            "",
                            "---",
                            "",
                            f"### {urllib.parse.urlparse(url).path or '/'}",
                            "",
                            f"Source URL: {url}",
                            "",
                            f"Content Flags: {', '.join(content_flags) if content_flags else 'none'}",
                            "",
                            markdown,
                            "",
                        ])

                print(f"✓ Crawl completed for {slug}.")
            except Exception as e:
                print(f"✗ Failed crawling CI sub-site {slug}: {e}")

        # Save combined file for the entire ci section
        combined_path = os.path.join(ci_root, "00_combined_ci_site.md")
        with open(combined_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(combined_sections).strip() + "\n")

        # Save unified manifest for the ci section
        manifest = {
            "section_code": "ci",
            "section_name": "Central Instruments & Innovation",
            "accepted_pages": all_pages,
            "accepted_binaries": all_binaries,
            "page_decisions": all_page_decisions,
            "link_decisions": all_link_decisions,
        }
        manifest_path = os.path.join(ci_root, "crawl_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        browser.close()

def asdict_to_dict(obj) -> dict:
    from dataclasses import asdict
    return asdict(obj)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IIT Jammu Central Facilities & Innovation Crawler")
    parser.add_argument("--clean", action="store_true", help="Clean output directories before crawling")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to crawl per site (for debugging)")
    parser.add_argument("--slug", type=str, default=None, help="Specific sub-site to crawl (cif, i3c, central-workshop)")
    args = parser.parse_args()

    crawl_ci_sites(clean=args.clean, max_pages=args.max_pages, slug_to_crawl=args.slug)
