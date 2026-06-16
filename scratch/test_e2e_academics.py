import sys
sys.path.append("/home/c3i/chatbot")
from app import init_app, app

init_app()

with app.test_client() as client:
    q1 = "What are the eligibility requirements for a minor in computer science engineering?"
    res1 = client.post("/api/chat", json={"message": q1})
    print(f"--- Q1: {q1} ---")
    print(res1.get_json()["response"][:500])
