from graphrag.rules_db import RulesDB

db = RulesDB()
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT id, section_number, title FROM rule_sections WHERE source_file='UG_Curriculum_2022_Scheme_IIT_Jammu.md' AND (section_number LIKE '2.%' OR title LIKE '%Minor%' OR title LIKE '%Specialization%')")
for row in cursor.fetchall():
    print(dict(row))
conn.close()
