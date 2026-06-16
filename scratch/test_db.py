import sys
import sqlite3
sys.path.append("/home/c3i/chatbot")

conn = sqlite3.connect("/home/c3i/chatbot/graphrag/rules.db")
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", c.fetchall())

c.execute("PRAGMA table_info(rules);")
print("Rules schema:", c.fetchall())

c.execute("PRAGMA table_info(rule_sections);")
print("Rule Sections schema:", c.fetchall())
