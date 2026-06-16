from graphrag.rules_db import RulesDB

db = RulesDB()

# Search sections
print("Search: 'comprehensive'")
res = db.search_sections("comprehensive", program="PhD", limit=3)
for r in res:
    print(f"  ID: {r['id']}, Title: {r['title']}, File: {r['source_file']}")
    print(f"    Text: {r['full_text'][:150]}...")

# Lookup fact
print("\nLookup fact: 'min_cgpa_minor'")
facts = db.lookup_fact("min_cgpa_minor")
for f in facts:
    print(f"  Key: {f['fact_key']}, Value: {f['fact_value']}, Operator: {f['operator']}, Source Section: {f['section_title']}")

# Get grades
print("\nGrade scale:")
grades = db.get_grade_scale()
for g in grades[:5]:
    print(f"  Grade: {g['grade']}, Point: {g['grade_point']}, Desc: {g['description']}")

# Get milestones
print("\nPhD Milestones:")
milestones = db.get_program_milestones("PhD")
for m in milestones:
    print(f"  Milestone: {m['milestone']}, Deadline: {m['deadline']}, Details: {m['details']}")
