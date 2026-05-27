"""Tests for chunk data quality — ensures chunks are clean and free of noise."""

import re

from graphrag.kg_builder import (
    _extract_first_email,
    _extract_section_map_from_body,
    chunk_text,
    infer_document_kind,
    smart_chunk_text,
)


class TestChunkCleanliness:
    """Verify chunks don't contain boilerplate or HTML noise."""

    def test_no_source_url_boilerplate(self, chunks):
        """No chunk should contain '# Source URL:' header."""
        for chunk_id, text, meta in chunks:
            assert '# Source URL:' not in text, f"Chunk {chunk_id} has Source URL boilerplate"

    def test_no_image_references(self, chunks):
        """No chunk should contain markdown image references."""
        for chunk_id, text, meta in chunks:
            if re.search(r'!\[.*?\]\(https?://', text):
                assert False, f"Chunk {chunk_id} has image reference: {text[:100]}"

    def test_no_html_noise(self, chunks):
        """No chunk should contain raw HTML tags or attributes."""
        html_patterns = ['noopener', 'target=', '_blank', '<a ', '<div', '<span']
        for chunk_id, text, meta in chunks:
            for pat in html_patterns:
                assert pat not in text.lower(), f"Chunk {chunk_id} contains HTML noise: '{pat}'"

    def test_minimum_chunk_length(self, chunks):
        """All chunks should have meaningful content (>30 chars)."""
        for chunk_id, text, meta in chunks:
            assert len(text.strip()) >= 30, f"Chunk {chunk_id} is too short: '{text}'"

    def test_no_navigation_breadcrumbs(self, chunks):
        """No chunk should contain navigation breadcrumb patterns."""
        nav_patterns = ['- [Home]', '- [Brief Info]', '- [Profile]', '- [Current Openings]']
        for chunk_id, text, meta in chunks:
            for pat in nav_patterns:
                assert pat not in text, f"Chunk {chunk_id} has nav breadcrumb: '{pat}'"

    def test_chunks_have_metadata(self, chunks):
        """All chunks should have proper metadata."""
        for chunk_id, text, meta in chunks:
            assert 'doc' in meta, f"Chunk {chunk_id} missing 'doc' metadata"
            assert 'url' in meta, f"Chunk {chunk_id} missing 'url' metadata"
            assert 'title' in meta, f"Chunk {chunk_id} missing 'title' metadata"

    def test_no_markdown_links_in_chunks(self, chunks):
        """Chunks should not contain raw markdown link syntax (should be stripped)."""
        for chunk_id, text, meta in chunks:
            links = re.findall(r'\[[^\]]+\]\(https?://[^)]+\)', text)
            assert not links, f"Chunk {chunk_id} has markdown links: {links[:3]}"


class TestStructuredRosterChunking:
    def test_repeated_roster_entries_are_chunked_on_record_boundaries(self):
        """Roster pages should be chunked by `####` records, not raw word windows."""
        roster = ["# PhD Students"]
        for idx in range(1, 18):
            roster.append(
                f"#### Student {idx}\n\n"
                f"**Supervisor**\nDr. Faculty {idx}\n\n"
                f"Research Area\nTopic {idx}\n"
            )
        chunks = chunk_text("\n\n".join(roster), chunk_size=120, overlap=20)
        assert len(chunks) > 1
        for chunk in chunks[1:]:
            assert chunk.startswith("#### "), f"Chunk does not start at a record boundary: {chunk[:80]}"

    def test_repeated_roster_entries_work_for_other_heading_levels(self):
        """Generic roster detection should also work when records use `###` headings."""
        roster = ["# Faculty Roster"]
        for idx in range(1, 12):
            roster.append(
                f"### Faculty Member {idx}\n\n"
                f"Assistant Professor\n\n"
                f"Research Interest\nTopic {idx}\n"
            )
        chunks = smart_chunk_text("\n\n".join(roster), chunk_size=80, overlap=10)
        assert len(chunks) > 1
        assert all(item["meta"]["strategy"] == "repeated_heading_records" for item in chunks)
        for chunk in chunks[1:]:
            assert chunk["text"].startswith("### "), f"Chunk does not start at detected record boundary: {chunk['text'][:80]}"

    def test_table_blocks_are_preserved_when_chunking(self):
        """Adjacent markdown table rows should remain together in structural chunking."""
        text = """
# Placement Data

## 2024

| Program | Percentage | Mean |
| --- | --- | --- |
| UG | 90 | 20 |
| PG | 80 | 15 |

Additional commentary about placements and outcomes that makes the section long enough
to require structure-aware chunking rather than a single flat word window repeated
across unrelated lines and rows.
"""
        chunks = smart_chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        assert any("| Program | Percentage | Mean |" in item["text"] for item in chunks)


class TestGenericExtractionHelpers:
    def test_obfuscated_email_helper_normalizes_common_patterns(self):
        text = "alex [DOT] joseph [AT] iitjammu [DOT] ac [DOT] in"
        assert _extract_first_email(text) == "alex.joseph@iitjammu.ac.in"

    def test_block_section_extractor_handles_inline_labels(self):
        block = """
##### **Supervisor**
###### Dr. Suman Banerjee
##### **Research Area**
###### Differential Privacy
##### alex.joseph@iitjammu.ac.in
"""
        sections = _extract_section_map_from_body(block)
        assert "Suman Banerjee" in sections["supervisor"]
        assert "Differential Privacy" in sections["research_area"]

    def test_document_kind_inference_uses_content_and_filename(self):
        phd_doc = """
# Source URL: https://iitjammu.ac.in/computer_science_engineering/phd-list

#### Alex Joseph
##### **Supervisor**
###### Dr. Suman Banerjee
##### **Research Area**
###### Differential Privacy
"""
        assert infer_document_kind("computer_science_engineering_phd-list.md", phd_doc) == "phd_roster"
