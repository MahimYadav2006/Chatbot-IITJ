from graphrag.rules_db import RulesDB

db = RulesDB()
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT id, section_number, title, parent_id, length(full_text) as len FROM rule_sections WHERE program = 'PhD'")
for row in cursor.fetchall():
    print(dict(row))
conn.close()
