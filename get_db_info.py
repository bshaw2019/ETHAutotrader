import sqlite3

conn = sqlite3.connect('historical.db')

cursor = conn.cursor()

cursor.execute("SELECT * FROM historical")

out = cursor.fetchall()

for entry in out:
    print(entry)

conn.close()
