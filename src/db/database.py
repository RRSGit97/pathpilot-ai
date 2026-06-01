import sqlite3
from src.config.settings import settings

def get_db_connection():
    # Placeholder database connection logic
    # In SQLite, URL might need parsing or we just use path
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Placeholder for database initialisation
    pass
