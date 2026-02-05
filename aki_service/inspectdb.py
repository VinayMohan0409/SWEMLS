import sqlite3
import os
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "aki_service.db"

def clear_database(conn):
    """Wipes all data from the tables but keeps the schema."""
    tables = ['patients', 'labs', 'alerts']
    for table in tables:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()
    print("--- Database Cleared: All tables are now empty ---")

def inspect(reset=False):
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    if reset:
        clear_database(conn)
    
    tables = ['patients', 'labs', 'alerts']
    for table in tables:
        print(f"\n--- Table: {table} ---")
        # Get total row count
        count = conn.execute(f"SELECT COUNT(*) as total FROM {table}").fetchone()
        print(f"Total Rows: {count['total']}")
        
        # Get sample data
        rows = conn.execute(f"SELECT * FROM {table} LIMIT 5").fetchall()
        for row in rows:
            print(dict(row))
    conn.close()

if __name__ == "__main__":
    # Change this to True if you want to wipe the data
    import sys
    should_reset = "--reset" in sys.argv
    inspect(reset=should_reset)