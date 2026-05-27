"""Regression tests for CSE faculty profile ingestion."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def cse_graph():
    from graphrag.kg_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder(dept_code="computer_science_engineering")
    return builder.build()


class TestCSEFacultyProfiles:
    def test_obfuscated_emails_are_normalized(self, cse_graph):
        node = cse_graph.nodes["computer_science_engineering:Harkeerat Kaur"]
        assert node["email"] == "harkeerat.kaur@iitjammu.ac.in"

    def test_link_wrapped_sections_are_extracted(self, cse_graph):
        node = cse_graph.nodes["computer_science_engineering:Aroof Aimen"]
        assert any(term in node["education"] for term in ("IIT Ropar", "Indian Institute of Technology Ropar"))
        assert "University of Wisconsin" in node["research_experience"]

    def test_department_records_available_faculty_schema(self, cse_graph):
        dept = cse_graph.nodes["IIT Jammu COMPUTER_SCIENCE_ENGINEERING Department"]
        assert "email" in dept["faculty_structured_fields"]
        assert "education" in dept["faculty_structured_fields"]
        assert "research_experience" in dept["faculty_structured_fields"]
