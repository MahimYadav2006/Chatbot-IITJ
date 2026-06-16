import sys
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "What about minnor in computer science engineering HOw can I actually do it considering that I am currently an electrical engineering student"

print("--- Vector search for Q ---")
search_res = retriever.embeddings.search(q, top_k=5, department_filter="academics")
for item, score in search_res:
    print(f"[{score:.4f}] {item['metadata'].get('name', 'Unknown')}: {item['text'][:150]}...")

print("\n--- Local results (word overlap) ---")
q_words = set(retriever._canonical_query(q).split())
local_results = []
for chunk in retriever.chunks:
    chunk_text = chunk["text"]
    chunk_words = set(retriever._canonical_query(chunk_text).split())
    overlap = len(q_words.intersection(chunk_words))
    if overlap > 0:
        local_results.append({
            "name": chunk.get("metadata", {}).get("name", "Unknown"),
            "text": chunk_text[:150],
            "score": overlap / len(q_words)
        })
local_results.sort(key=lambda x: x["score"], reverse=True)
for lr in local_results[:5]:
    print(f"[{lr['score']:.4f}] {lr['name']}: {lr['text']}...")
