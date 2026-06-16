import json

with open("/home/c3i/chatbot/data/sections/academics/chunks.json", "r") as f:
    chunks = json.load(f)

print("Total chunks in chunks.json:", len(chunks))

found_cse = 0
for c in chunks:
    doc = c.get("metadata", {}).get("doc", "")
    if "computer" in doc.lower() and "minor" in doc.lower():
        found_cse += 1
        if found_cse <= 3:
            print(f"Doc: {doc} | {c['text'][:150]}...")
print("Total cse minor chunks in chunks.json:", found_cse)

found_quantum = 0
for c in chunks:
    doc = c.get("metadata", {}).get("doc", "")
    if "quantum" in doc.lower():
        found_quantum += 1
        if found_quantum <= 3:
            print(f"Doc: {doc} | {c['text'][:150]}...")
print("Total quantum chunks in chunks.json:", found_quantum)
