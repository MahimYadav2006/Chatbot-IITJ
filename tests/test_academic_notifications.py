"""Unit tests for Academic Notifications parsing and retrieval."""

import sys
import os
import pytest
from networkx import DiGraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.section_retriever import SectionRetriever

def test_academic_notifications_committees():
    # Setup mock graphs for academics
    academics_graph = DiGraph()
    
    # 1. DPGC member
    academics_graph.add_node(
        "committee_member:dpgc:chemical_engineering:dr_aditya_shankar_sandupatla",
        label="CommitteeMember",
        name="Dr. Aditya Shankar Sandupatla",
        designation="External Member",
        department="Chemical Engineering",
        committee_type="DPGC",
        committee_name="DPGC Chemical Engineering",
        notification_date="22 September 2025",
        source_file="DPGC.md"
    )
    
    # 2. DPGC chairperson
    academics_graph.add_node(
        "committee_member:dpgc:chemical_engineering:dr_shirsha_bose",
        label="CommitteeMember",
        name="Dr. Shirsha Bose",
        designation="Chairperson",
        department="Chemical Engineering",
        committee_type="DPGC",
        committee_name="DPGC Chemical Engineering",
        notification_date="22 September 2025",
        source_file="DPGC.md"
    )

    # 3. Faculty Advisor
    academics_graph.add_node(
        "facultyadvisor:btech_in_civil_engineering:dr_ved_prakash_ranjan",
        label="FacultyAdvisor",
        name="Dr. Ved Prakash Ranjan",
        programme="B.Tech. in Civil Engineering",
        batch_year="2025",
        notification_date="05 May 2025",
        source_file="Faculty_Advisor_-_2025.md"
    )

    # 4. Fee Structure
    academics_graph.add_node(
        "fee_structure:btech:2025:all",
        label="FeeStructure",
        programme="B.Tech",
        entry_year="2025",
        income_category="All",
        fee_gen_obc_ews="₹129,410",
        fee_sc_st_pwd="₹29,410",
        category="B.Tech Programmes",
        notification_date="12 January 2025",
        source_file="Fee_Notification_2025-02_Sem.md"
    )

    retriever = SectionRetriever("academics", academics_graph, [], None)
    
    # Test DPGC members
    ans = retriever.get_deterministic_context("List all members of DPGC Committee of Chemical Engineering")
    assert ans is not None
    assert "Dr. Aditya Shankar Sandupatla" in ans
    assert "Dr. Shirsha Bose" in ans
    
    # Test DPGC chairperson
    ans = retriever.get_deterministic_context("Who is DPGC Chairperson of chemical engineering")
    assert ans is not None
    assert "Dr. Shirsha Bose" in ans
    assert "Dr. Aditya Shankar Sandupatla" not in ans
    
    # Test Faculty Advisor
    ans = retriever.get_deterministic_context("Who is the faculty advisor of B.Tech. in Civil Engineering?")
    assert ans is not None
    assert "Dr. Ved Prakash Ranjan" in ans
    
    # Test Fee structure
    ans = retriever.get_deterministic_context("What is the fee structure of B.Tech for entry year 2025?")
    assert ans is not None
    assert "₹129,410" in ans
    assert "₹29,410" in ans
