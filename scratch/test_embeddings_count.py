import sys
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")
print("FAISS Index vectors:", retriever.embeddings.index.ntotal)
print("Chunks list size:", len(retriever.chunks))
print("First 5 items in embeddings:")
for i in range(min(5, len(retriever.embeddings.items))):
    item = retriever.embeddings.items[i]
    print(item['metadata'].get('name', 'Unknown'))
