import sys
import os

# Set up environment variables
os.environ["DEPT_CONFIG_PATH"] = "config/departments.json"

sys.path.insert(0, os.path.abspath("."))

from graphrag.section_retriever import SectionRetriever
from graphrag.person_index import GlobalPersonIndex
from app import get_global_person_direct_answer, init_app
import app

def test_queries():
    print("Initializing App...")
    # Initialize person index and other items
    init_app()
    
    print("\n--- TEST 1: specific doctor details (who is Dr. Karunika) ---")
    ans1 = get_global_person_direct_answer("who is Dr. Karunika")
    print(ans1)
    assert ans1 is not None
    assert "MBBS, MD" in ans1
    assert "Over 10 years of experience" in ans1
    
    print("\n--- TEST 2: multi-role query (who is Dr. Sanat Kumar Tiwari) ---")
    ans2 = get_global_person_direct_answer("who is Dr. Sanat Kumar Tiwari")
    print(ans2)
    assert ans2 is not None
    assert "Faculty" in ans2 or "Physics" in ans2
    assert "Counselling" in ans2
    assert "Chairperson - Medical Unit" in ans2 or "Medical Unit" in ans2
    
    print("\n--- TEST 3: list of doctors (should exclude Sanat Kumar Tiwari) ---")
    retriever_medical = app.section_retrievers["medical-centre"]
    ans3 = retriever_medical.get_direct_answer("list of doctors", global_person_index=app.person_index)
    print(ans3)
    assert ans3 is not None
    assert "Dr. Karunika Sharma" in ans3
    assert "Dr. Pawanjeet Kour" in ans3
    assert "Sanat Kumar Tiwari" not in ans3
    
    print("\n--- TEST 4: list all clubs ---")
    retriever_ir = app.section_retrievers["ir"]
    ans4 = retriever_ir.get_direct_answer("list all clubs")
    # count lines starting with "- **" to verify number of clubs
    club_lines = [l for l in ans4.splitlines() if l.startswith("- **")]
    print(f"Total clubs returned: {len(club_lines)}")
    assert len(club_lines) == 17
    
    print("\n--- TEST 5: specific club details ---")
    ans5 = retriever_ir.get_direct_answer("tell me about coding club")
    print(ans5)
    assert "Coding Club" in ans5
    assert "Fintech Club" not in ans5
    
    print("\n--- TEST 6: specific hostel details ---")
    ans6 = retriever_ir.get_direct_answer("tell me about canary hostel")
    print(ans6)
    assert "Canary Hostel" in ans6
    assert "Fulgar Hostel" not in ans6

    print("\n--- TEST 7: chairperson of medical unit ---")
    ans7 = retriever_medical.get_direct_answer("who is the chairperson of the medical unit")
    print(ans7)
    assert "Sanat Kumar Tiwari" in ans7

    print("\n--- TEST 8: check generic query does not trigger global person index ---")
    ans8 = get_global_person_direct_answer("Clubs at IIT Jammu")
    print(f"Clubs at IIT Jammu response: {ans8}")
    assert ans8 is None

    print("\n--- TEST 9: check generic query does not trigger global person index ---")
    ans9 = get_global_person_direct_answer("Hostels at iit jammu")
    print(f"Hostels at iit jammu response: {ans9}")
    assert ans9 is None
    
    print("\nAll verification checks passed successfully!")

if __name__ == "__main__":
    test_queries()
