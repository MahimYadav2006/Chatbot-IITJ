import sys
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever
from graphrag.llm import build_chat_prompt, get_system_prompt
import json

init_app()
retriever = get_section_retriever("academics")

q = "What about minnor in computer science engineering HOw can I actually do it considering that I am currently an electrical engineering student"
bundle = retriever.retrieve_bundle(q)
print("Context:")
print(bundle["context"])
