import sqlite3
import os

DB_PATH = "world.db"
SCHEMA_PATH = "schema.sql"

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, 'r') as f:
        schema = f.read()
    
    conn.executescript(schema)
    print("Schema initialized.")
    return conn

def seed_data(conn):
    cursor = conn.cursor()
    
    # Create Location
    cursor.execute("INSERT INTO Locations (name, connected_to_ids) VALUES (?, ?)", ("The Rusted Tavern", ""))
    tavern_id = cursor.lastrowid
    print(f"Created Location: The Rusted Tavern (ID: {tavern_id})")
    
    # Create Entity
    cursor.execute("INSERT INTO Entities (name, type, hp, max_hp, location_id) VALUES (?, ?, ?, ?, ?)", 
                   ("Orc Brawler", "NPC", 50, 50, tavern_id))
    print(f"Created Entity: Orc Brawler (ID: {cursor.lastrowid}) in Tavern")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    conn = init_db()
    seed_data(conn)
