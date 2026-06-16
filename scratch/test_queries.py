import requests
import json

queries = [
    "Semester wise credit distribution for MTech in Communication and Signal Processing",
    "How many credits do a UG EE student needs to graduate??",
    "Doctors at iit jammu"
]

for q in queries:
    print(f"\n--- Query: {q}")
    res = requests.post("http://127.0.0.1:5050/api/chat", json={"message": q})
    print(res.json().get('response'))
