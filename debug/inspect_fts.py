from graphrag.rules_db import RulesDB

db = RulesDB()
conn = db.get_connection()
cursor = conn.cursor()
cursor.execute("SELECT count(*) FROM rule_sections_fts")
print("FTS count:", cursor.fetchone()[0])

cursor.execute("SELECT rowid, * FROM rule_sections_fts LIMIT 3")
for row in cursor.fetchall():
    print(dict(row))
conn.close()
