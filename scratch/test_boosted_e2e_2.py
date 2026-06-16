import sys
import re
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "I have Design and Analysis of Algorithms, Computer Vision and Operating Systems Am I eligible for minors in CSE"

# Typo correction in query
q_clean = re.sub(r'\bminnors?\b', 'minor', q.lower())

boosted_chunks = []
if "minor" in q_clean:
    for chunk in retriever.chunks:
        doc_name = chunk.get("metadata", {}).get("doc", "").lower()
        if "minor" in doc_name:
            topic_matches = []
            if "computer" in q_clean or "cse" in q_clean:
                topic_matches = ["computer", "cse"]
            elif "quantum" in q_clean:
                topic_matches = ["quantum"]
            elif "economics" in q_clean:
                topic_matches = ["economics"]
            elif "mathematics" in q_clean or "math" in q_clean:
                topic_matches = ["mathematics", "math"]
            elif "physics" in q_clean:
                topic_matches = ["physics"]
            elif "nuclear" in q_clean:
                topic_matches = ["nuclear"]
            elif "chemistry" in q_clean:
                topic_matches = ["chemistry"]
            elif "bioengineering" in q_clean or "biology" in q_clean:
                topic_matches = ["bioengineering", "biology", "bio"]
            
            if topic_matches and any(tm in doc_name for tm in topic_matches):
                boosted_chunks.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "score": 2.0,
                    "label": "TextChunk"
                })

print("Number of boosted chunks:", len(boosted_chunks))

# We simulate the retrieve_bundle where we prepend these boosted chunks
bundle = retriever.retrieve_bundle(q, vector_top_k=5, local_top_k=5)

# Prepend boosted chunks to context if they are not already there
context_blocks = [bc["text"] for bc in boosted_chunks]
for block in bundle["context"].split("\n\n"):
    if block not in context_blocks and block.strip():
        context_blocks.append(block)

combined_context = "\n\n".join(context_blocks)

# Generate response
from graphrag.llm import create_llm_from_env, build_chat_prompt, get_system_prompt
llm = create_llm_from_env()
system_instruction = get_system_prompt("academics")
prompt = build_chat_prompt(q, combined_context)

response = llm.generate(prompt, system_prompt=system_instruction)
print("\n--- Response ---")
print(response)
