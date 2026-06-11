import os
import sys

sys.path.append("/home/c3i/chatbot")
os.environ["GEMINI_API_KEY"] = "mock_key" # mock key if needed, or if it reads from environment

from app import app, init_app

init_app()

client = app.test_client()

queries = [
    "List of Institute medalists",
    "List of doctors at IIT Jammu",
    "List of all clubs at iit jammu",
    "List of hostels at iit jammu",
    "What are the fests organized at iit jammu",
    "timings of dental services",
    "Who is the coordinator of counselling services?",
]

for q in queries:
    print(f"\n==========================================")
    print(f"QUERY: {q}")
    print(f"==========================================")
    res = client.post("/api/chat", json={"message": q})
    print(res.get_json())
