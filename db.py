"""
Database manager for storing places and embeddings

IMPROVEMENTS:
- Bug Fix: `insert_place` now correctly handles database connections.
- Schema Init: `user_searches` table creation moved to `init_database`.
- New Method: Added `get_user_search_history` to retrieve recent user searches.
- Robustness: Added try-except blocks for database operations.
"""

import sqlite3
import json
from typing import List, Dict, Optional
import logging
from datetime import datetime
from config import DATABASE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize database and create tables if they don't exist"""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create all database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Places table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS places (
                id INTEGER PRIMARY KEY AUTOINCREMENT, place_id TEXT UNIQUE,
                name TEXT NOT NULL, description TEXT, address TEXT,
                latitude REAL, longitude REAL, rating REAL,
                user_ratings_total INTEGER, price_level INTEGER,
                types TEXT, tags TEXT, vibe_tags TEXT, difficulty_tags TEXT,
                best_time_tags TEXT, accessibility_tags TEXT, cost_tags TEXT,
                duration_tags TEXT, phone TEXT, website TEXT, opening_hours TEXT,
                photos TEXT, reviews TEXT, source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                attributes TEXT
            )
        ''')
        
        # User interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, place_id TEXT,
                interaction_type TEXT, rating INTEGER, feedback TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (place_id) REFERENCES places(place_id)
            )
        ''')

        # User searches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, query TEXT,
                location TEXT, timestamp TEXT, detected_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def insert_place(self, place_data: Dict):
        """Insert a place with duplicate prevention (by name)"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM places WHERE name = ?", (place_data['name'],))
            if cursor.fetchone():
                return

            cursor.execute("""
                INSERT INTO places (place_id, name, description, address, rating, attributes, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                place_data.get('place_id', f"db_{hash(place_data['name'])}"),
                place_data['name'],
                place_data.get('description'),
                place_data.get('address'),
                place_data.get('rating'),
                json.dumps(place_data.get('attributes')),
                place_data.get('source')
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"DB Error on insert: {e}")
        finally:
            if conn:
                conn.close()

    def log_user_search(self, search_data: Dict):
        """Log a user's search query to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_searches (user_id, query, location, timestamp, detected_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    search_data['user_id'], search_data['query'],
                    search_data['location'], search_data['timestamp'],
                    search_data['detected_type']
                ))
        except Exception as e:
            logger.error(f"Error logging search: {e}")

    def get_user_search_history(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Retrieve the most recent search history for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT query, location, timestamp FROM user_searches WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                    (user_id, limit)
                )
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError: # Table might not exist yet
            return []
        except Exception as e:
            logger.error(f"Error getting search history: {e}")
            return []

    def clear_user_search_history(self, user_id: str):
        """Clear all search history for a specific user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM user_searches WHERE user_id = ?", (user_id,))
        except Exception as e:
            logger.error(f"Error clearing history: {e}")

    def get_all_places(self, limit: int = 1000) -> List[Dict]:
        """Get all places"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM places LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
