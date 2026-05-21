"""Tests for entity resolution — ensures name variants map to canonical names."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.kg_builder import normalize_name, _initials_match, EntityResolver, _extract_canonical_faculty

MARKDOWN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "iitjammu_ee_markdown")


class TestNormalizeName:
    def test_strips_dr_prefix(self):
        assert normalize_name("Dr. Ajay Singh") == "Ajay Singh"

    def test_strips_prof_prefix(self):
        assert normalize_name("Prof. Ravikant Saini") == "Ravikant Saini"

    def test_strips_assistant_professor(self):
        assert normalize_name("Assistant Professor Ankur Bansal") == "Ankur Bansal"

    def test_capitalizes_words(self):
        assert normalize_name("ajay singh") == "Ajay Singh"

    def test_strips_extra_whitespace(self):
        assert normalize_name("  Ajay   Singh  ") == "Ajay Singh"


class TestInitialsMatch:
    def test_b_n_subudhi_matches_badri(self):
        assert _initials_match("B. N Subudhi", "Badri Narayan Subudhi")

    def test_a_bansal_matches_ankur(self):
        assert _initials_match("A. Bansal", "Ankur Bansal")

    def test_non_matching_initials(self):
        assert not _initials_match("X. Y Subudhi", "Badri Narayan Subudhi")

    def test_different_last_name(self):
        assert not _initials_match("B. N Kumar", "Badri Narayan Subudhi")

    def test_short_name_no_crash(self):
        assert not _initials_match("A", "Ajay Singh")


class TestEntityResolver:
    def test_resolves_exact_canonical(self):
        canonical = {"Ajay Singh", "Badri Narayan Subudhi"}
        resolver = EntityResolver(canonical)
        assert resolver.resolve("Ajay Singh") == "Ajay Singh"

    def test_resolves_abbreviated_name(self):
        canonical = {"Badri Narayan Subudhi"}
        resolver = EntityResolver(canonical)
        result = resolver.resolve("B. N Subudhi")
        assert result == "Badri Narayan Subudhi"

    def test_resolves_partial_name(self):
        canonical = {"Alok Kumar Saxena"}
        resolver = EntityResolver(canonical)
        result = resolver.resolve("Alok Saxena")
        assert result == "Alok Kumar Saxena"

    def test_resolves_optional_middle_name(self):
        canonical = {"Anup Shukla"}
        resolver = EntityResolver(canonical)
        result = resolver.resolve("Anup Kumar Shukla")
        assert result == "Anup Shukla"

    def test_is_canonical_faculty_true(self):
        canonical = {"Ajay Singh", "Ankur Bansal"}
        resolver = EntityResolver(canonical)
        assert resolver.is_canonical_faculty("Ajay Singh")
        assert resolver.is_canonical_faculty("Dr. Ajay Singh")

    def test_is_canonical_faculty_false_for_external(self):
        canonical = {"Ajay Singh"}
        resolver = EntityResolver(canonical)
        assert not resolver.is_canonical_faculty("Chinmoy Kundu")

    def test_is_canonical_via_initials(self):
        canonical = {"Ankur Bansal"}
        resolver = EntityResolver(canonical)
        assert resolver.is_canonical_faculty("A. Bansal")


class TestCanonicalFacultyExtraction:
    def test_extracts_24_faculty(self):
        names = _extract_canonical_faculty(MARKDOWN_DIR)
        assert len(names) == 24, f"Expected 24 canonical faculty, got {len(names)}"
