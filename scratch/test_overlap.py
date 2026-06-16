import sys
import re
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "What about minnor in computer science engineering HOw can I actually do it considering that I am currently an electrical engineering student"

print("\n--- Local results (word overlap) ---")
q_words = set(re.findall(r"\w+", q.lower()))
local_results = []
for chunk in retriever.chunks:
    chunk_text = chunk["text"]
    chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
    overlap = len(q_words.intersection(chunk_words))
    if overlap > 0:
        local_results.append({
            "doc": chunk.get("metadata", {}).get("doc", "Unknown"),
            "text": chunk_text[:150].replace("\n", " "),
            "score": overlap / len(q_words)
        })
local_results.sort(key=lambda x: x["score"], reverse=True)
for i, lr in enumerate(local_results[:20]):
    print(f"[{i+1}] [{lr['score']:.4f}] {lr['doc']}: {lr['text']}...")
