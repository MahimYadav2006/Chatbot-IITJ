import sys
import sqlite3
import json
sys.path.append("/home/c3i/chatbot")
from graphrag.section_kg_builder import SectionKGBuilder
from departments import get_section_data_dir

# 1. Search in chunks.json
graph, chunks = SectionKGBuilder.load(get_section_data_dir("academics"))
print("Search in chunks:")
found_in_chunks = False
for chunk in chunks:
    if "Committee for framing policy of grading for the PG Thesis and BTP" in chunk["text"]:
        print(f"  Found in chunk ID: {chunk['id']}, source: {chunk.get('metadata', {}).get('source_file')}")
        found_in_chunks = True
if not found_in_chunks:
    print("  Not found in chunks.")

# 2. Search in rules.db
print("\nSearch in rules.db:")
conn = sqlite3.connect("/home/c3i/chatbot/data/sections/academics/rules.db")
cursor = conn.cursor()
cursor.execute("SELECT source_file, section_number, title, full_text FROM rule_sections")
found_in_db = False
for source_file, sec_num, title, full_text in cursor.fetchall():
    if "Committee for framing policy of grading for the PG Thesis and BTP" in full_text:
        print(f"  Found in DB: file={source_file}, section={sec_num}, title={title}")
        found_in_db = True
if not found_in_db:
    print("  Not found in DB.")
conn.close()
