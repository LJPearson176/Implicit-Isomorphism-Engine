from engine import apply_damage
import sqlite3

def run_test():
    print("Testing Apply Damage (Clamped)...")
    
    # Test 1: Wound
    print("\n--- Test 1: Apply 10 Damage (Wound) ---")
    res = apply_damage(1, 10)
    print(f"Result: {res}")
    assert res['outcome'] == 'wound'
    assert res['current_hp'] == 40
    assert res['status'] == 'Alive'
    
    # Test 2: Fatal Damage (Clamped)
    print("\n--- Test 2: Apply 100 Damage (Kill) ---")
    res_kill = apply_damage(1, 100)
    print(f"Result: {res_kill}")
    assert res_kill['outcome'] == 'kill'
    assert res_kill['current_hp'] == 0
    assert res_kill['status'] == 'Dead'
    
    # Verify DB state
    conn = sqlite3.connect("world.db")
    cur = conn.cursor()
    cur.execute("SELECT hp, status FROM Entities WHERE id=1")
    row = cur.fetchone()
    print(f"DB State: HP={row[0]}, Status={row[1]}")
    assert row[0] == 0
    assert row[1] == 'Dead'

if __name__ == "__main__":
    run_test()
