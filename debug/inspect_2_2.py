from graphrag.rules_db import RulesDB

db = RulesDB()
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT id, section_number, title FROM rule_sections WHERE section_number LIKE '2.2.%'")
for row in cursor.fetchall():
    print(dict(row))
conn.close()
