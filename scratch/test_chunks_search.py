import json

with open("/home/c3i/chatbot/data/sections/academics/chunks.json", "r") as f:
    chunks = json.load(f)

print("Total chunks in chunks.json:", len(chunks))

found_quantum = 0
for c in chunks:
    src = c.get("metadata", {}).get("source_file", "")
    if "quantum" in src.lower() or "quantum" in c["text"].lower():
        found_quantum += 1
        if found_quantum <= 3:
            print(f"Source: {src} | {c['text'][:150]}...")
print("Total quantum chunks in chunks.json:", found_quantum)

found_cse = 0
for c in chunks:
    src = c.get("metadata", {}).get("source_file", "")
    if "computer" in src.lower() and "minor" in src.lower():
        found_cse += 1
        if found_cse <= 3:
            print(f"Source: {src} | {c['text'][:150]}...")
print("Total cse minor chunks in chunks.json:", found_cse)
