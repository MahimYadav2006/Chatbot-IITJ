"""Tests for section routing, retrieval, and entity resolution in the chatbot."""

import sys
import os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dept_router import DepartmentRouter
from graphrag.person_index import GlobalPersonIndex
from graphrag.section_retriever import SectionRetriever
from graphrag.section_kg_builder import SectionKGBuilder
from graphrag.embeddings import EmbeddingEngine
from networkx import DiGraph


def test_section_routing():
    router = DepartmentRouter()
    
    # Test Academics routing
    res = router.route("Who is the head of academics section?")
    assert "academics" in res.sections
    
    # Test Accounts routing
    res = router.route("Where is the accounts department located?")
    assert "accounts" in res.sections
    
    # Test Counselling routing
    res = router.route("How to book a slot with the counsellor?")
    assert "counselling" in res.sections
    
    # Test Digital Infrastructure routing
    res = router.route("Who is incharge of network and security in DI?")
    assert "di" in res.sections
    
    # Test Establishment II routing
    res = router.route("Contact details of Establishment II section")
    assert "e2" in res.sections


def test_global_person_index_role_prioritization():
    # Construct a mock graph with a person holding academic role and administrative role
    academic_graph = DiGraph()
    academic_graph.add_node("ee:Ravikant Saini", label="Faculty", name="Ravikant Saini", designation="Associate Professor", email="saini@iitjammu.ac.in")
    
    section_graph = DiGraph()
    section_graph.add_node("academics:Ravikant Saini", label="SectionHead", name="Ravikant Saini", designation="Associate Dean (Academics)", email="adean.acad@iitjammu.ac.in")
    
    index = GlobalPersonIndex()
    index.index_graph(academic_graph, "ee", is_section=False)
    index.index_graph(section_graph, "academics", is_section=True)
    
    res = index.lookup("Ravikant Saini")
    assert res is not None
    assert len(res["roles"]) == 2
    
    # Check prioritization logic using the helper from app.py
    from app import get_global_person_direct_answer, person_index
    # Temporarily set the global person_index to our test index
    import app
    old_index = app.person_index
    app.person_index = index
    
    try:
        # Standard faculty/academic query: EE / Faculty role should be first
        ans = get_global_person_direct_answer("who is Ravikant Saini?")
        assert ans is not None
        # Since it is a general query, EE Associate Professor role (academic/dept) should appear first in the text
        first_role_idx = ans.find("Associate Professor")
        second_role_idx = ans.find("Associate Dean (Academics)")
        assert first_role_idx < second_role_idx
        
        # Administrative query: Section role should be first
        ans_admin = get_global_person_direct_answer("who is Dean Ravikant Saini?")
        assert ans_admin is not None
        first_role_idx_admin = ans_admin.find("Associate Dean (Academics)")
        second_role_idx_admin = ans_admin.find("Associate Professor")
        assert first_role_idx_admin < second_role_idx_admin
    finally:
        app.person_index = old_index


def test_section_retriever_direct_answers():
    # Setup mock graphs for sections
    e2_graph = DiGraph()
    # AR Puja Rajyaguru
    e2_graph.add_node(
        "e2:Puja Rajyaguru",
        label="SectionHead",
        name="Puja Rajyaguru",
        designation="Assistant Registrar",
        email="puja.rajyaguru@iitjammu.ac.in",
        phone="0191-257-1234"
    )
    
    counselling_graph = DiGraph()
    counselling_graph.add_node(
        "counselling:Counselling Room",
        label="SectionContact",
        name="Counselling Room",
        email="counselling@iitjammu.ac.in",
        phone="0191-257-0002"
    )
    
    academics_graph = DiGraph()
    
    # Mock chunks for academics spec catalog
    acad_chunks = [
        {
            "id": "chunk_1",
            "text": "Computer Science: https://drive.google.com/drive/folders/1abc123\nElectrical Engineering: https://drive.google.com/drive/folders/456",
            "metadata": {"doc": "specialisation-and-courses.md"}
        }
    ]
    
    # Test E2 Section Head answer
    e2_retriever = SectionRetriever("e2", e2_graph, [], None)
    ans = e2_retriever.get_direct_answer("Who is the head of E2 section?")
    assert ans is not None
    assert "Puja Rajyaguru" in ans
    assert "Assistant Registrar" in ans
    
    # Test Counselling contact info answer
    counselling_retriever = SectionRetriever("counselling", counselling_graph, [], None)
    ans = counselling_retriever.get_direct_answer("counselling contact email")
    assert ans is not None
    assert "counselling@iitjammu.ac.in" in ans
    
    # Test Academics doc link redirection answer
    acad_retriever = SectionRetriever("academics", academics_graph, acad_chunks, None)
    ans = acad_retriever.get_direct_answer("computer science specialisation curriculum link")
    assert ans is not None
    assert "https://drive.google.com/drive/folders/1abc123" in ans


def test_new_sections_retriever_direct_answers():
    # Setup mock graphs for new sections
    alumni_graph = DiGraph()
    alumni_graph.add_node("medalist:john_doe:2024", label="AlumniMedalist", name="John Doe", award="President Gold Medal", year=2024, department="CSE")
    
    cds_graph = DiGraph()
    cds_graph.add_node("recruiter:google", label="Recruiter", name="Google")
    cds_graph.add_node("policy:cgpa_cutoff", label="PlacementPolicy", name="CGPA Cutoff", description="Minimum 6 CGPA", category="Eligibility")
    
    ir_graph = DiGraph()
    ir_graph.add_node("club:coding", label="Club", name="Coding Club", category="Technical", description="Coding club")
    ir_graph.add_node("hostel:canary", label="Hostel", name="Canary", gender="Boys", description="Boys Hostel")
    
    medical_graph = DiGraph()
    medical_graph.add_node("doc:dr_smith", label="MedicalDoctor", name="Dr. Smith", designation="Medical Officer", email="smith@iitjammu.ac.in")
    
    osd_graph = DiGraph()
    osd_graph.add_node("uba:main", label="UBAProgram", name="Unnat Bharat Abhiyan", description="UBA", focus_areas="Water", coordinator="Dr. Sameer")
    
    # 1. Alumni Affairs
    retriever = SectionRetriever("alumni-affairs", alumni_graph, [], None)
    ans = retriever.get_direct_answer("Who won the President Gold Medal in 2024?")
    assert ans is not None and "John Doe" in ans
    
    # 2. CDS
    retriever = SectionRetriever("cds", cds_graph, [], None)
    ans = retriever.get_direct_answer("list recruiters")
    assert ans is not None and "Google" in ans
    ans = retriever.get_direct_answer("placement policy CGPA cutoff")
    assert ans is not None and "Minimum 6 CGPA" in ans
    
    # 3. IR
    retriever = SectionRetriever("ir", ir_graph, [], None)
    ans = retriever.get_direct_answer("list student clubs")
    assert ans is not None and "Coding Club" in ans
    ans = retriever.get_direct_answer("hostels canary")
    assert ans is not None and "Canary" in ans
    
    # 4. Medical Centre
    retriever = SectionRetriever("medical-centre", medical_graph, [], None)
    ans = retriever.get_direct_answer("list health centre doctors")
    assert ans is not None and "Dr. Smith" in ans
    
    # 5. OSD
    retriever = SectionRetriever("osd", osd_graph, [], None)
    ans = retriever.get_direct_answer("what is Unnat Bharat Abhiyan?")
    assert ans is not None and "Unnat Bharat Abhiyan" in ans

