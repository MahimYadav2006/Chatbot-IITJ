from graphrag.rules_db import RulesDB

db = RulesDB()
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT id, section_number, title FROM rule_sections WHERE id LIKE 'undergraduate_2022_%' LIMIT 15")
for row in cursor.fetchall():
    print(dict(row))
conn.close()
