import sys
import json
sys.path.append("/home/c3i/chatbot")
from graphrag.rules_retriever import RulesRetriever
import pprint

rr = RulesRetriever()

queries = [
    "What about minnor in computer science engineering",
    "Semester wise credit distribution for MTech in Communication and Signal Processing",
    "Courses and their course code for minors in Quantum Technology"
]

for q in queries:
    res = rr.retrieve(q)
    print(f"--- QUERY: {q} ---")
    print(f"Terms & Phrases: {rr._expanded_terms_and_phrases(q)}")
    print("FTS Results:")
    for f in res['fts_results'][:3]:
        print(f"  [{f['_retrieval_score']}] {f['title']} (from {f['source_file']})")
    print("\n")
