from graphrag.rules_db import RulesDB

db = RulesDB()
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT id, section_number, title FROM rule_sections WHERE program='UG' AND (title LIKE '%Minor%' OR title LIKE '%Specialization%')")
for row in cursor.fetchall():
    print(dict(row))
conn.close()
