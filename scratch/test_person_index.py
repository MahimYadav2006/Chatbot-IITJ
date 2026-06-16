import sys
sys.path.append("/home/c3i/chatbot")
from app import get_global_person_direct_answer, init_app

init_app()
queries = [
    "What about minnor in computer science engineering HOw can I actually do it considering that I am currently an electrical engineering student",
    "What about minors in computer science engineering",
    "I have Design and Analysis of Algorithms, Computer Vision and Operating Systems Am I eligible for minors in CSE",
    "How many credits do a UG EE student needs to graduate??",
    "Semester wise credit distribution for MTech in Communication and Signal Processing",
    "Courses and their course code for minors in Quantum Technology"
]

for q in queries:
    ans = get_global_person_direct_answer(q)
    print(f"Q: {q}")
    if ans:
        print(f"  Ans: {ans[:100]}...")
    else:
        print("  Ans: None")
