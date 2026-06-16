import sys, os
os.environ["DEPT_CONFIG_PATH"] = "config/departments.json"
from app import retrievers, section_retrievers, multi_retriever, init_app

init_app()

query = "Doctors at iit jammu"
try:
    print(multi_retriever.retrieve_broadcast(query, top_n=20))
except Exception as e:
    import traceback
    traceback.print_exc()

