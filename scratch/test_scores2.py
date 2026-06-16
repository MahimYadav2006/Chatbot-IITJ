import sys
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "What about minor in computer science engineering How can I actually do it considering that I am currently an electrical engineering student"

print("--- Vector search top 20 ---")
search_res = retriever.embeddings.search(q, top_k=20, department_filter="academics")
for i, (item, score) in enumerate(search_res):
    doc = item['metadata'].get('doc', item.get('id', 'N/A'))
    print(f"[{i+1}] [{score:.4f}] {doc}: {item['text'][:100]}...")
