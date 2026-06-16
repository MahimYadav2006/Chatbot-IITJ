import sys
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "How many credits do a Civil Engineering student require to graduate"
bundle = retriever.retrieve_bundle(q)

with open("/home/c3i/chatbot/scratch/credits_context.txt", "w") as f:
    f.write(bundle['context'])

print("Saved context to scratch/credits_context.txt. Length:", len(bundle['context']))
