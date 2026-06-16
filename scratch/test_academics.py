import sys, os
os.environ["DEPT_CONFIG_PATH"] = "config/departments.json"
from app import init_app
import app

init_app()
sec = app.section_retrievers["academics"]
bundle = sec.retrieve_bundle("Semester wise credit distribution for MTech in Communication and Signal Processing")
print("Context length:", len(bundle['context']))
print("Answerable:", bundle['answerability']['answerable'])
if bundle['context']:
    print("Context snippet:", bundle['context'][:200])

