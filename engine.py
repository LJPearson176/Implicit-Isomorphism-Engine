import sqlite3
from typing import Dict, Any, Optional

DB_PATH = "world.db"

def connect_db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def apply_damage(target_id: int, amount: int) -> Optional[Dict[str, Any]]:
    """
    Applies damage to an entity with clamping logic.
    Query -> Clamp HP (min 0) -> Update DB (HP, Status) -> Return Outcome.
    """
    conn = connect_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        conn.execute("BEGIN")

        # 1. Query Current HP
        cursor.execute("SELECT id, name, hp, max_hp, status FROM Entities WHERE id = ?", (target_id,))
        row = cursor.fetchone()
        
        if not row:
            raise ValueError(f"Entity with ID {target_id} not found.")

        current_hp = row['hp']
        name = row['name']
        
        # 2. Clamping Logic
        new_hp = max(0, current_hp - amount)
        
        # 3. Determine Outcome and Status
        outcome = "wound"
        status = "Alive"
        
        if new_hp == 0:
            outcome = "kill"
            status = "Dead"
        
        # 4. Update DB
        cursor.execute("UPDATE Entities SET hp = ?, status = ? WHERE id = ?", (new_hp, status, target_id))
        conn.commit()

        return {
            "outcome": outcome,
            "current_hp": new_hp,
            "target": name,
            "status": status
        }

    except Exception as e:
        conn.rollback()
        print(f"Transaction Failed: {e}")
        return None
    finally:
        conn.close()

if __name__ == "__main__":
    # Simple test if run directly
    pass
