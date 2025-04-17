import sqlite3

db_path = "./retrival_database.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS CAR")
conn.commit()
conn.close()