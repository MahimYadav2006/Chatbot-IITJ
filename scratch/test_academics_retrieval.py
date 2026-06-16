import sys
sys.path.append("/home/c3i/chatbot")
from app import init_app, get_section_retriever

init_app()
retriever = get_section_retriever("academics")

q = "Semester wise credit distribution for MTech in Communication and Signal Processing"
bundle = retriever.retrieve_bundle(q)

ctx = bundle['context']
print("Does it contain 'Semester-wise Credit Distribution'?")
print("Semester-wise Credit Distribution" in ctx)
print("Number of words:", len(ctx.split()))
