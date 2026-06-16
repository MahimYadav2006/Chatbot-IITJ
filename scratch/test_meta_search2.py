import json

with open("/home/c3i/chatbot/data/sections/academics/embeddings_meta.json", "r") as f:
    meta = json.load(f)

found_cse = 0
for i, m in enumerate(meta):
    doc = m.get("metadata", {}).get("doc", "")
    if "computer" in doc.lower() and "minor" in doc.lower():
        found_cse += 1
        print(f"[{i}] {m.get('type')}: doc={doc} | {m['text'][:150]}...")
print("Total matches:", found_cse)
