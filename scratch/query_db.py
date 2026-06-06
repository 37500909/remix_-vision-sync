import sqlite3
import os

DB_FILE = r"c:\Users\andres.salgado\Downloads\remix_-vision-sync\backend\biometrics.db"
if os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, first_name, last_name, dni, image_path FROM users")
        rows = cursor.fetchall()
        print("Registered Users:")
        for r in rows:
            print(f"ID: {r[0]} | Name: {r[1]} {r[2]} | DNI: {r[3]} | Path: {r[4]}")
    except Exception as e:
        print("Database query failed:", e)
    conn.close()
else:
    print("No DB file found at:", DB_FILE)
