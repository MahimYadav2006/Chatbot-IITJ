"""Shared pytest fixtures for the IIT Jammu EE GraphRAG chatbot tests."""

import os
import sys
import pytest

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MARKDOWN_DIR = os.path.join(PROJECT_ROOT, "scraped_data", "ee")

# The 24 canonical faculty names from ee_faculty-list.html.md
CANONICAL_FACULTY = [
    "Ajay Singh",
    "Alok Kumar Saxena",
    "Ambika Prasad Shah",
    "Ankit Dubey",
    "Ankur Bansal",
    "Anup Shukla",
    "Archana Rajput",
    "Arun Kumar Verma",
    "Badri Narayan Subudhi",
    "Chandan Yadav",
    "Ibhanchand Rath",
    "Kankat Ghosh",
    "Karan Nathwani",
    "Kushmanda Saurav",
    "Nalin Kumar Sharma",
    "Padmini Singh",
    "Priyanka Mishra",
    "Priyatosh Jena",
    "Ravikant Saini",
    "Rohit Buddhiram Chaurasiya",
    "Satyadev Ahlawat",
    "Shikha Baghel",
    "Soma S Dhavala",
    "Sudhakar Modem",
]


@pytest.fixture(scope="session")
def canonical_faculty():
    """The 24 canonical faculty names."""
    return CANONICAL_FACULTY


@pytest.fixture(scope="session")
def built_graph():
    """Build a fresh knowledge graph for testing."""
    from graphrag.kg_builder import KnowledgeGraphBuilder
    builder = KnowledgeGraphBuilder(MARKDOWN_DIR)
    graph = builder.build()
    return graph, builder.chunks, builder.resolver


@pytest.fixture(scope="session")
def graph(built_graph):
    return built_graph[0]


@pytest.fixture(scope="session")
def chunks(built_graph):
    return built_graph[1]


@pytest.fixture(scope="session")
def resolver(built_graph):
    return built_graph[2]
