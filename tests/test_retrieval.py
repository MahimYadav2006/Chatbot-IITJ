"""Tests for retrieval accuracy — ensures the retriever returns correct results."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture(scope="module")
def retriever():
    """Load the full retriever (requires data/ to be populated)."""
    if not os.path.exists(os.path.join(DATA_DIR, "graph.pkl")):
        pytest.skip("Data directory not populated — run ingest.py first")
    from graphrag.retriever import load_retriever
    return load_retriever(DATA_DIR)


class TestHodRetrieval:
    def test_hod_query_returns_ravikant(self, retriever):
        """Asking about HoD should return Ravikant Saini."""
        context = retriever.retrieve("Who is the Head of Department?")
        assert "Ravikant Saini" in context, f"HoD not found in context: {context[:300]}"


class TestFacultyCountRetrieval:
    def test_faculty_count_query_returns_complete_roster(self, retriever, canonical_faculty):
        """Faculty count/list questions should include the full authoritative roster."""
        context = retriever.retrieve("What is the faculty count and list in the EE department?")
        assert "24 faculty members" in context
        missing = [name for name in canonical_faculty if name not in context]
        assert not missing, f"Faculty roster context is missing: {missing}"

    def test_direct_faculty_answer_is_complete(self, retriever, canonical_faculty):
        """The direct graph answer should not rely on LLM counting."""
        answer = retriever.get_direct_answer("Give me all faculty members count and list")
        assert answer is not None
        assert "**24 faculty members**" in answer
        missing = [name for name in canonical_faculty if name not in answer]
        assert not missing, f"Direct faculty answer is missing: {missing}"


class TestPhDRosterRetrieval:
    def test_phd_count_query_returns_authoritative_roster_context(self, retriever):
        """Department-level PhD count questions should bypass fuzzy community summaries."""
        context = retriever.retrieve("total number of phd scholar in ee at iit jammu")
        assert "66 PhD scholars" in context
        assert "Authoritative PhD Scholar Roster" in context
        assert "Department Overview" not in context

    def test_direct_phd_answer_is_exact(self, retriever):
        """The direct graph answer should return the exact PhD scholar count."""
        answer = retriever.get_direct_answer("What is the total number of PhD scholars in EE at IIT Jammu?")
        assert answer is not None
        assert "**66 PhD scholars**" in answer
        assert "phd-list.html" in answer


class TestFacultyProfileRetrieval:
    def test_specific_faculty_query(self, retriever):
        """Querying a specific faculty name should return their info."""
        context = retriever.retrieve("Tell me about Ankit Dubey")
        assert "Ankit Dubey" in context
        # Should have some profile info
        assert any(kw in context for kw in ["Professor", "email", "research", "IIT"])


class TestDirectSupervisorAnswers:
    def test_supervisor_query_is_answered_directly(self, retriever):
        """Supervisor questions should bypass the LLM and use graph edges directly."""
        answer = retriever.get_direct_answer("Who supervises Ritujoy Biswas?")
        assert answer is not None
        assert "Ritujoy Biswas" in answer
        assert "Karan Nathwani" in answer
        assert "Speech Intelligibility Improvement" in answer
