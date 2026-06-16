"""Tests for retrieval accuracy — ensures the retriever returns correct results."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture(scope="module")
def retriever():
    """Load the full retriever (requires data/ to be populated)."""
    if not os.path.exists(os.path.join(DATA_DIR, "ee", "graph.pkl")):
        pytest.skip("Data directory not populated — run ingest.py first")
    from graphrag.retriever import load_retriever
    return load_retriever("ee")


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
        answer = retriever.get_deterministic_context("Give me all faculty members count and list")
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
        answer = retriever.get_deterministic_context("What is the total number of PhD scholars in EE at IIT Jammu?")
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
        answer = retriever.get_deterministic_context("Who supervises Ritujoy Biswas?")
        assert answer is not None
        assert "Ritujoy Biswas" in answer
        assert "Karan Nathwani" in answer
        assert "Speech Intelligibility Improvement" in answer


@pytest.fixture(scope="module")
def cse_retriever():
    """Load the CSE retriever from the checked-in department data."""
    cse_dir = os.path.join(DATA_DIR, "computer_science_engineering")
    if not os.path.exists(os.path.join(cse_dir, "graph.pkl")):
        pytest.skip("CSE data directory not populated — run ingest.py --dept cse first")
    from graphrag.retriever import load_retriever
    return load_retriever(dept_code="computer_science_engineering")


class TestCSEFacultyAnalytics:
    def test_cse_faculty_roster_count_is_authoritative(self, cse_retriever):
        """CSE faculty count/list should come from the graph roster, not chunk counting."""
        answer = cse_retriever.get_deterministic_context("Give me the faculty count and list in the CSE department")
        assert answer is not None
        assert "**15 faculty members**" in answer
        for name in (
            "Aroof Aimen",
            "Harkeerat Kaur",
            "Sidharth Maheshwari",
            "Vinit Jakhetiya",
            "Yamuna Prasad",
        ):
            assert name in answer

    def test_gender_ratio_query_is_rejected_when_attribute_is_missing(self, cse_retriever):
        """Gender analytics must not be guessed from names or partial retrieval."""
        answer = cse_retriever.get_deterministic_context(
            "What is the ratio of male to female in CSE department faculties? Calculate it."
        )
        assert answer is not None
        assert "can't calculate" in answer.lower()
        assert "gender is not stored" in answer.lower()
        assert "**15 members**" in answer
        assert "To avoid guessing" in answer

    def test_main_point_of_contact_is_answered_directly(self, cse_retriever):
        """Generic contact questions should resolve to the HoD instead of drifting into unrelated facts."""
        answer = cse_retriever.get_deterministic_context("Who is the main point of contact?")
        assert answer is not None
        assert "Yamuna Prasad" in answer
        assert "Head of Department" in answer
        assert "yamuna.prasad@iitjammu.ac.in" in answer

    def test_missing_startup_data_returns_unavailable_fallback(self, cse_retriever):
        """If the department graph has no startup evidence, the retriever should not hallucinate one."""
        bundle = cse_retriever.retrieve_bundle(
            "Startups by faculty of CSE Dept at IIT Jammu",
            local_top_k=5,
            vector_top_k=5,
            global_top_k=3,
            max_context_words=1200,
        )
        assert bundle["answerability"]["answerable"] is False
        assert "startup" in bundle["answerability"]["reason"].lower()
        assert bundle["fallback_response"] is not None
        assert "don't have that specific information" in bundle["fallback_response"].lower()

    def test_provenance_reports_combined_graph_and_vector_usage(self, cse_retriever):
        """Hybrid retrieval should report whether graph, vector, or both contributed."""
        bundle = cse_retriever.retrieve_bundle(
            "Tell me about Samaresh Bera",
            local_top_k=5,
            vector_top_k=5,
            global_top_k=3,
            max_context_words=1200,
        )
        provenance = bundle["provenance"]
        assert provenance["source_mode"] in {"graph", "both"}
        assert provenance["graph"]["items"] >= 1


class TestLaboratoryRetrieval:
    def test_ee_labs_retrieval(self, retriever):
        """Ask about EE labs and verify the correct list is returned."""
        answer = retriever.get_deterministic_context("What labs are there in the EE department?")
        assert answer is not None
        assert "Low Voltage Lab1" in answer
        assert "AADHRIT Lab" in answer
        assert "Fluid Mechanics Lab" not in answer

    def test_cse_labs_retrieval_negative(self, cse_retriever):
        """Ask about CSE labs specifically and verify deterministic empty message."""
        answer = cse_retriever.get_deterministic_context("Are there labs in the CSE department?")
        assert answer is not None
        assert "no laboratories" in answer.lower()
        assert "Computer Science and Engineering" in answer

    def test_cse_labs_retrieval_broadcast_ignored(self, cse_retriever):
        """Ask a general lab query to CSE retriever and verify it returns None to not pollute broadcast."""
        answer = cse_retriever.get_deterministic_context("List all department labs")
        assert answer is None


class TestFacultyDomainRetrieval:
    def test_rf_domain_retrieval(self, retriever):
        """Verify that searching for 'RF' returns Archana Rajput and Alok Kumar Saxena."""
        context = retriever.retrieve("Which faculty members work on RF?")
        assert "Archana Rajput" in context
        assert "Alok Kumar Saxena" in context

    def test_microwave_domain_retrieval(self, retriever):
        """Verify that searching for 'Microwave' returns Archana Rajput and Alok Kumar Saxena."""
        context = retriever.retrieve("Who does research in Microwave?")
        assert "Archana Rajput" in context
        assert "Alok Kumar Saxena" in context

    def test_antenna_domain_retrieval(self, retriever):
        """Verify that searching for 'Antenna Design' returns Archana Rajput and Alok Kumar Saxena."""
        context = retriever.retrieve("Find professors working on Antenna Design")
        assert "Archana Rajput" in context
        assert "Alok Kumar Saxena" in context
