import pickle

path = "/home/c3i/chatbot/data/sections/medical-centre/graph.pkl"
with open(path, "rb") as f:
    g = pickle.load(f)

for node, data in g.nodes(data=True):
    if data.get("label") == "MedicalDoctor":
        print(f"Doctor Node: {node} -> Name: {data.get('name')}, Designation: {data.get('designation')}, Email: {data.get('email')}")
