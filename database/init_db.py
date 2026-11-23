# ==============================================================================
# DATABASE INITIALIZATION SCRIPT: database/init_db.py
# Creates database tables and default admin user safely without dropping data.
# Optional --reset flag explicitly drops and recreates tables.
# ==============================================================================

import os
import sys
import sqlite3
from werkzeug.security import generate_password_hash

# Add parent directory to python path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DB_DIR, DB_PATH, SCHEMA_PATH


def init_db(reset=False):
    """
    Initializes the SQLite database tables and default admin user.
    
    Parameters:
        reset (bool): If True, drops existing tables before recreating them.
                      If False (default), safely creates missing tables without destroying existing data.
    """
    # Fetch current DB_PATH (respecting DROPOUT_DB_PATH environment variable if set)
    current_db_path = os.environ.get("DROPOUT_DB_PATH", DB_PATH)
    current_db_dir = os.path.dirname(current_db_path)
    if current_db_dir:
        os.makedirs(current_db_dir, exist_ok=True)
    
    conn = sqlite3.connect(current_db_path)
    cursor = conn.cursor()
    
    if reset:
        print(f"WARNING: --reset specified! Dropping all existing tables in '{current_db_path}'...")
        cursor.execute("DROP TABLE IF EXISTS students;")
        cursor.execute("DROP TABLE IF EXISTS users;")
        conn.commit()
    
    # Read and execute schema.sql
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_script = f.read()
    cursor.executescript(schema_script)
    
    # Check if default admin user exists
    default_username = "admin"
    default_password = "admin123"
    
    cursor.execute("SELECT id FROM users WHERE username = ?", (default_username,))
    admin_user = cursor.fetchone()
    
    if not admin_user:
        hashed_password = generate_password_hash(default_password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (default_username, hashed_password)
        )
        print(f"Default admin user created: Username='{default_username}', Password='{default_password}'")
    
    conn.commit()
    conn.close()
    print(f"Database initialization complete for '{current_db_path}'.")


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    init_db(reset=reset_flag)
