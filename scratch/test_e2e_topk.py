import sys
sys.path.append("/home/c3i/chatbot")
from app import init_app, app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "What about minnor in computer science engineering HOw can I actually do it considering that I am currently an electrical engineering student"
bundle = retriever.retrieve_bundle(q, vector_top_k=10, local_top_k=10)

from graphrag.llm import create_llm_from_env, build_chat_prompt, get_system_prompt
llm = create_llm_from_env()

system_instruction = get_system_prompt("academics")
prompt = build_chat_prompt(q, bundle["context"])

print("--- System prompt length:", len(system_instruction))
print("--- User prompt length:", len(prompt))

response = llm.generate(prompt, system_prompt=system_instruction)
print("--- Response ---")
print(response)
