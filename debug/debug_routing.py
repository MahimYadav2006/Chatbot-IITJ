import sys, os
os.environ["DEPT_CONFIG_PATH"] = "config/departments.json"
from dept_router import DepartmentRouter
import app

app.init_app()
router = DepartmentRouter()

queries = [
    "What about minnor in computer science engineering HOw can I actually do it considering that I am currently an electrical engineering student",
    "I have Design and Analysis of Algorithms, Computer Vision and Operating Systems Am I eligible for minors in CSE",
    "How many credits do a UG EE student needs to graduate??",
    "Semester wise credit distribution for MTech in Communication and Signal Processing",
    "Courses and their course code for minors in Quantum Technology"
]

for q in queries:
    print(f"\nQuery: {q}")
    route = router.route(q)
    print(f"Confidence: {route.confidence}")
    print(f"Departments: {route.departments}")
    print(f"Sections: {route.sections}")
    print(f"Reason: {route.reason}")
