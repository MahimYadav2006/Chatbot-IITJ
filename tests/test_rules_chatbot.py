import pytest
from graphrag.rules_db import RulesDB
from graphrag.rules_parser import RulesParser
from graphrag.rules_retriever import RulesRetriever
from graphrag.section_retriever import SectionRetriever
from dept_router import DepartmentRouter

def test_rules_db_querying():
    db = RulesDB()
    # Verify grade scale has values seeded
    grades = db.get_grade_scale()
    assert len(grades) > 0
    assert any(g["grade"] == "AA" and g["grade_point"] == 10 for g in grades)

    # Verify credit requirements can be fetched
    ug_reqs = db.get_credit_requirements("UG")
    assert len(ug_reqs) > 0
    assert any(r["category"] == "Minor" and r["min_credits"] == 12.0 for r in ug_reqs)

    # Verify facts lookup works
    facts = db.lookup_fact("min_cgpa_minor", "UG")
    assert len(facts) > 0
    assert float(facts[0]["fact_value"]) == 7.0

def test_rules_retriever_intent_classification():
    retriever = RulesRetriever()
    
    # Test UG intent
    intent_ug = retriever.classify_intent("what is the minimum CGPA required for minor in B.Tech?")
    assert intent_ug["program"] == "UG"
    assert "min_cgpa_minor" in intent_ug["facts"]
    
    # Test PhD intent
    intent_phd = retriever.classify_intent("what are the milestones for a Ph.D. candidate?")
    assert intent_phd["program"] == "PhD"
    assert intent_phd["milestones"] is True
    
    # Test grading intent
    intent_grades = retriever.classify_intent("show me the grading scale and pointers")
    assert intent_grades["grades"] is True

def test_rules_retriever_context_generation():
    retriever = RulesRetriever()
    
    # Retrieve minor CGPA details
    res = retriever.retrieve("what is the minimum CGPA required for minor in B.Tech?")
    context = retriever.generate_context(res)
    
    assert "7.0" in context
    assert "Minor" in context
    assert "Relevant Rules & Regulations Document Sections" in context

def test_section_retriever_routing_academics():
    # Construct a dummy SectionRetriever for academics to verify the routing intercept
    from unittest.mock import MagicMock
    import networkx as nx
    
    graph = nx.DiGraph()
    chunks = [{"id": "c1", "text": "Dummy text", "metadata": {"doc": "rules-and-regulations"}}]
    
    retriever = SectionRetriever(section_code="academics", graph=graph, chunks=chunks, embedding_engine=None)
    
    # Factual query should bypass direct template answers and return rules database context
    bundle = retriever.retrieve_bundle("what is the minimum CGPA required for minor in B.Tech?")
    assert bundle["provenance"]["route"] in ("rules_db", "rules_db+chunks")
    assert "7.0" in bundle["context"]
    
    # Explicit link query should still return the direct template answer
    bundle_link = retriever.retrieve_bundle("give me the link for the academic rules manual pdf")
    assert bundle_link["provenance"]["route"] in ("direct_graph", "rules_db", "rules_db+chunks")


@pytest.mark.parametrize("query", [
    "Procedure of withdrawal of courses during semester",
    "Are IDP and HSS considered same or different",
    "What are tthe compulsoy HSS Courses for minimum degree requirements",
    "How many maximum credits can I take from a single department as open electives",
    "Procedure of BTP allotment",
    "What is the Provision of semester internship",
    "Is Change of department possible at iit jammu",
    "Is changing branch after admission permitted at iit jammu",
    "Is change of branch possible after admission?",
])
def test_academic_policy_queries_route_to_academics(query):
    route = DepartmentRouter().route(query)
    assert route.departments == []
    assert route.sections == ["academics"]


@pytest.mark.parametrize("query,expected", [
    ("Procedure of withdrawal of courses during semester", "Withdrawal of course"),
    ("Are IDP and HSS considered same or different", "not clubbed"),
    ("What are tthe compulsoy HSS Courses for minimum degree requirements", "Humanities and Social Sciences core"),
    ("How many maximum credits can I take from a single department as open electives", "maximum of 9 credits"),
    ("Procedure of BTP allotment", "BTP allotment"),
    ("What is the Provision of semester internship", "Provision of semester internship"),
    ("Is Change of department possible at iit jammu", "not to offer the change of department"),
    ("Is changing branch after admission permitted at iit jammu", "not to offer the change of department"),
    ("Is change of branch possible after admission?", "not to offer the change of department"),
])
def test_rules_retriever_finds_policy_evidence(query, expected):
    retriever = RulesRetriever()
    result = retriever.retrieve(query, limit=6)
    context = retriever.generate_context(result)
    assert expected.lower() in context.lower()


def test_parser_keeps_change_of_department_section():
    parser = RulesParser()
    sections = parser.parse_file(
        "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/UG/9.5-IIT_Jammu_Rules___Curriculumn.md",
        "undergraduate",
    )
    assert any(
        section["section_number"].strip().rstrip(".") == "9"
        and "change of department" in section["title"].lower()
        for section in sections
    )


def test_rules_db_contains_amendment_and_department_change():
    db = RulesDB()
    amendment_rows = db.search_sections("open elective", program="UG", limit=20)
    assert any("Amenedment_in_Rule_2.3.2.2.md" == row["source_file"] for row in amendment_rows)
    change_rows = db.search_sections("change department", program="UG", limit=20)
    assert any(row["section_number"].strip().rstrip(".") == "9" for row in change_rows)


def test_academic_rules_do_not_mix_notice_list_chunks():
    from graphrag.section_kg_builder import SectionKGBuilder
    from departments import get_section_data_dir

    graph, chunks = SectionKGBuilder.load(get_section_data_dir("academics"))
    retriever = SectionRetriever(section_code="academics", graph=graph, chunks=chunks, embedding_engine=None)
    bundle = retriever.retrieve_bundle("Procedure of BTP allotment")

    assert bundle["provenance"]["route"] in ("rules_db", "rules_db+chunks")
    assert "BTP allotment" in bundle["context"]
    assert "Committee for framing policy of grading for the PG Thesis and BTP" not in bundle["context"]


def test_parser_recovers_mtech_appendix_sections():
    parser = RulesParser()
    sections = parser.parse_file(
        "/home/c3i/chatbot/scraped_data/sections/academics/parsed_documents/Rules_and_Regulations/PG/MTech/M.Tech_RRs___Curric..md",
        "mtech",
    )
    by_number = {section["section_number"].strip().rstrip("."): section for section in sections}

    assert by_number["A.2"]["title"] == "Course structure for students under RA category"
    assert "Table 7" in by_number["A.2"]["full_text"]
    assert by_number["B"]["title"] == "Course code convention"
    assert "AL055P" in by_number["B"]["full_text"]


@pytest.mark.parametrize("query,expected_anchor", [
    (
        "If my CGPA is 9 Will I be considered first division, second divison or third divison",
        "first, second and third divisions",
    ),
    (
        "Describe Evaluation scheme of the theoretical courses",
        "Evaluation modes of theoretical courses",
    ),
    (
        "Different kind of recognition in MTech students",
        "Institute gold medal",
    ),
    (
        "Course structure for students under RA category",
        "Section A.2: Course structure for students under RA category",
    ),
    (
        "Course code convention at iit jammu",
        "Course code convention",
    ),
    (
        "A course if coded as M AL055P 4I will indicate what???",
        "AL055P",
    ),
    (
        "How can I get Silver Medal",
        "Institute silver medal",
    ),
    (
        "How can I get Gold Medal",
        "President of India gold medal",
    ),
])
def test_user_reported_academic_queries_route_to_rules_db(query, expected_anchor):
    from graphrag.section_kg_builder import SectionKGBuilder
    from departments import get_section_data_dir

    route = DepartmentRouter().route(query)
    assert route.departments == []
    assert route.sections == ["academics"]

    graph, chunks = SectionKGBuilder.load(get_section_data_dir("academics"))
    retriever = SectionRetriever(section_code="academics", graph=graph, chunks=chunks, embedding_engine=None)
    bundle = retriever.retrieve_bundle(query)

    assert bundle["provenance"]["route"] in ("rules_db", "rules_db+chunks")
    assert bundle["answerability"]["answerable"] is True
    assert expected_anchor.lower() in bundle["context"].lower()
    assert "Page Title: Indian Institute of Technology Jammu" not in bundle["context"]
    assert "Committee for framing policy of grading for the PG Thesis and BTP" not in bundle["context"]
