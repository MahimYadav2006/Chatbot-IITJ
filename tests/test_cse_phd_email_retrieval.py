"""Regression tests for CSE PhD scholar email ingestion and retrieval."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def cse_graph():
    from graphrag.kg_builder import KnowledgeGraphBuilder

    builder = KnowledgeGraphBuilder(dept_code="computer_science_engineering")
    return builder.build()


class TestCSEPhDEmails:
    def test_phd_emails_are_extracted_into_graph(self, cse_graph):
        node = cse_graph.nodes["computer_science_engineering:Alex Joseph"]
        assert node["email"] == "alex.joseph@iitjammu.ac.in"

    def test_named_email_query_is_answered_directly(self, cse_graph):
        from graphrag.retriever import HybridRetriever

        class DummyEmbeddings:
            def search(self, *args, **kwargs):
                return []

        retriever = HybridRetriever(cse_graph, DummyEmbeddings(), [], dept_code="computer_science_engineering")
        answer = retriever.get_direct_answer("What is the email of Alex Joseph?")
        assert answer == "Alex Joseph's official email is alex.joseph@iitjammu.ac.in."

    def test_phd_roster_context_can_include_email(self, cse_graph):
        from graphrag.retriever import HybridRetriever

        class DummyEmbeddings:
            def search(self, *args, **kwargs):
                return []

        retriever = HybridRetriever(cse_graph, DummyEmbeddings(), [], dept_code="computer_science_engineering")
        answer = retriever.get_direct_answer("List all PhD scholars in CSE")
        assert answer is not None
        assert "alex.joseph@iitjammu.ac.in" in answer
