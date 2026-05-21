"""Tests for chunk data quality — ensures chunks are clean and free of noise."""

import re

from graphrag.kg_builder import chunk_text


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
