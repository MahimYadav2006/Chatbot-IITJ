import json

with open("/home/c3i/chatbot/data/sections/academics/embeddings_meta.json", "r") as f:
    meta = json.load(f)

print("Total metadata items:", len(meta))
types = {}
for m in meta:
    t = m.get("type")
    types[t] = types.get(t, 0) + 1
print("Types breakdown:", types)

print("\nSearch for 'quantum':")
found_quantum = 0
for i, m in enumerate(meta):
    if "quantum" in m["text"].lower():
        found_quantum += 1
        if found_quantum <= 5:
            print(f"[{i}] {m.get('type')}: {m.get('metadata', {}).get('name', 'N/A')} | {m['text'][:150]}...")
print("Total quantum matching:", found_quantum)

print("\nSearch for 'computer science and engineering' or 'minor in computer':")
found_cse = 0
for i, m in enumerate(meta):
    if "computer science" in m["text"].lower() and "minor" in m["text"].lower():
        found_cse += 1
        if found_cse <= 5:
            print(f"[{i}] {m.get('type')}: {m.get('metadata', {}).get('name', 'N/A')} | {m['text'][:150]}...")
print("Total cse minor matching:", found_cse)
