import sys
import os
sys.path.append("/home/c3i/chatbot")
from dept_router import DepartmentRouter

router = DepartmentRouter()
queries = [
    "What about minnor in computer science engineering HOw can I actually do it considering that I am currently an electrical engineering student",
    "What about minors in computer science engineering",
    "I have Design and Analysis of Algorithms, Computer Vision and Operating Systems Am I eligible for minors in CSE",
    "How many credits do a UG EE student needs to graduate??",
    "Semester wise credit distribution for MTech in Communication and Signal Processing",
    "Courses and their course code for minors in Quantum Technology"
]

for q in queries:
    res = router.route(q)
    print(f"Q: {q}")
    print(f"  Depts: {res.departments}")
    print(f"  Secs: {res.sections}")
    print(f"  Confidence: {res.confidence}")
    print(f"  Reason: {res.reason}\n")
