import sys
import re
import math
from collections import Counter
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "What about minnor in computer science engineering HOw can I actually do it considering that I am currently an electrical engineering student"

# Normalize typos
q_clean = re.sub(r'\bminnors?\b', 'minor', q.lower())

# Extract words
all_words = re.findall(r"\w+", q_clean)

# Stop words
STOP_WORDS = {
    "what", "about", "in", "how", "can", "i", "actually", "do", "it", 
    "considering", "that", "am", "currently", "an", "student", "the", 
    "a", "of", "and", "to", "for", "on", "with", "is", "are", "was", 
    "were", "be", "been", "have", "has", "had", "you", "your", "we", "our",
    "they", "their", "he", "she", "it", "its", "me", "my", "myself"
}

q_words = {w for w in all_words if w not in STOP_WORDS}
print("Filtered query words:", q_words)

# Precompute document frequencies
doc_frequencies = Counter()
for chunk in retriever.chunks:
    words = set(re.findall(r"\w+", chunk["text"].lower()))
    doc_frequencies.update(words)
num_docs = len(retriever.chunks)

# Calculate IDF
def get_idf(word):
    df = doc_frequencies.get(word, 0)
    if df == 0:
        return 0.0
    return math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

# Print IDFs of query words
for w in q_words:
    print(f"IDF({w}) = {get_idf(w):.4f} (DF={doc_frequencies.get(w, 0)})")

local_results = []
for chunk in retriever.chunks:
    chunk_text = chunk["text"]
    chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
    matching_words = q_words.intersection(chunk_words)
    if matching_words:
        score = sum(get_idf(w) for w in matching_words)
        local_results.append({
            "doc": chunk.get("metadata", {}).get("doc", "Unknown"),
            "text": chunk_text[:150].replace("\n", " "),
            "score": score
        })

local_results.sort(key=lambda x: x["score"], reverse=True)
print("\n--- IDF-weighted Local results top 15 ---")
for i, lr in enumerate(local_results[:15]):
    print(f"[{i+1}] [{lr['score']:.4f}] {lr['doc']}: {lr['text']}...")
