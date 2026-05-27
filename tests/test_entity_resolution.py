"""Tests for entity resolution — ensures name variants map to canonical names."""

import sys
import os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.kg_builder import (
    normalize_name,
    _initials_match,
    EntityResolver,
    KnowledgeGraphBuilder,
    _extract_canonical_faculty,
)
from graphrag.retriever import HybridRetriever

MARKDOWN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "iitjammu_ee_markdown")
CSE_MARKDOWN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "iitjammu_computer_science_engineering_markdown",
)


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

    def test_extracts_cse_faculty_from_plain_headings(self):
        names = _extract_canonical_faculty(CSE_MARKDOWN_DIR, "computer_science_engineering")
        assert "Vinit Jakhetiya" in names
        assert "Research Experience" not in names


@pytest.fixture(scope="module")
def cse_graph():
    builder = KnowledgeGraphBuilder(dept_code="computer_science_engineering")
    return builder.build()


class TestCSEGraphParsing:
    def test_vinit_is_faculty_not_student(self, cse_graph):
        node_id = "computer_science_engineering:Vinit Jakhetiya"
        assert cse_graph.has_node(node_id)
        assert cse_graph.nodes[node_id]["label"] == "Faculty"
        assert not cse_graph.has_node("computer_science_engineering:Dr. Vinit Jakhetiya")

    def test_vinit_supervises_expected_students(self, cse_graph):
        node_id = "computer_science_engineering:Vinit Jakhetiya"
        supervised_students = {
            cse_graph.nodes[source]["name"]
            for source, _, edge_data in cse_graph.in_edges(node_id, data=True)
            if edge_data.get("type") == "SUPERVISED_BY"
        }
        assert {"Ajeet Kumar Verma", "Ambreen Bashir", "Deebha Mumtaz"} <= supervised_students

    def test_direct_answer_returns_students_under_vinit(self, cse_graph):
        retriever = HybridRetriever(cse_graph, embedding_engine=None, community_reports=[],
                                    dept_code="computer_science_engineering")
        answer = retriever.get_direct_answer("PhD students under Dr. Vinit Jakhetiya")
        assert answer is not None
        assert "Ajeet Kumar Verma" in answer
        assert "Ambreen Bashir" in answer
        assert "Deebha Mumtaz" in answer
        assert "is a PhD student" not in answer
