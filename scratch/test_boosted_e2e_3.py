import sys
import re
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "Semester wise credit distribution for MTech in Communication and Signal Processing"

# Typo correction in query
q_clean = re.sub(r'\bminnors?\b', 'minor', q.lower())

boosted_chunks = []
if "mtech" in q_clean or "m.tech" in q_clean or "master of technology" in q_clean:
    for chunk in retriever.chunks:
        doc_name = chunk.get("metadata", {}).get("doc", "").lower()
        if "mtech" in doc_name or "m.tech" in doc_name or "master_of_technology" in doc_name:
            topic_matches = []
            if "communication" in q_clean or "signal" in q_clean or "csp" in q_clean:
                topic_matches = ["csp", "communication"]
            elif "vlsi" in q_clean:
                topic_matches = ["vlsi"]
            elif "mechanical" in q_clean or "msd" in q_clean:
                topic_matches = ["mechanical", "msd"]
            elif "chemical" in q_clean:
                topic_matches = ["chemical"]
            elif "tunnel" in q_clean:
                topic_matches = ["tunnel"]
            elif "geotechnical" in q_clean:
                topic_matches = ["geotechnical"]
            elif "structural" in q_clean:
                topic_matches = ["structural"]
            elif "computer science" in q_clean or "cse" in q_clean:
                topic_matches = ["computer_science", "cse"]
            
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
