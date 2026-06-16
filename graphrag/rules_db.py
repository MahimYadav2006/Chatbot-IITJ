import sqlite3
import os
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sections",
    "academics",
    "rules.db"
)

class RulesDB:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialise database schema and tables."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rule_sections (
            id TEXT PRIMARY KEY,
            section_number TEXT NOT NULL,
            title TEXT NOT NULL,
            full_text TEXT NOT NULL,
            parent_id TEXT,
            program TEXT NOT NULL,         -- 'UG', 'MTech', 'PhD'
            source_file TEXT NOT NULL,
            last_amended TEXT,             -- ISO format date
            amendment_note TEXT,
            FOREIGN KEY (parent_id) REFERENCES rule_sections(id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rule_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_type TEXT NOT NULL,        -- 'threshold', 'credit', 'grade_map', 'eligibility'
            fact_key TEXT NOT NULL,         -- unique identifier
            fact_value TEXT NOT NULL,       -- the value
            operator TEXT,                  -- '>=', '<=', '==', etc.
            condition_text TEXT,            -- context/conditions
            section_id TEXT NOT NULL,
            program TEXT NOT NULL,
            FOREIGN KEY (section_id) REFERENCES rule_sections(id)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS grade_scale (
            grade TEXT PRIMARY KEY,
            grade_point INTEGER NOT NULL,
            description TEXT
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS credit_requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            program TEXT NOT NULL,          -- 'UG', 'MTech', 'PhD'
            category TEXT NOT NULL,         -- 'IC-EF', etc.
            category_full TEXT,
            min_credits REAL NOT NULL,
            percentage REAL,
            notes TEXT
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS program_milestones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            program TEXT NOT NULL,
            milestone TEXT NOT NULL,
            deadline TEXT,
            details TEXT,
            section_id TEXT,
            FOREIGN KEY (section_id) REFERENCES rule_sections(id)
        );
        """)

        # Check if virtual table for FTS5 already exists, if not, create it
        try:
            cursor.execute("SELECT 1 FROM rule_sections_fts LIMIT 1;")
        except sqlite3.OperationalError:
            cursor.execute("""
            CREATE VIRTUAL TABLE rule_sections_fts USING fts5(
                section_number, title, full_text, program,
                content='rule_sections',
                content_rowid='rowid'
            );
            """)
            # Create triggers to keep FTS table in sync
            cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS t_rule_sections_ai AFTER INSERT ON rule_sections BEGIN
                INSERT INTO rule_sections_fts(rowid, section_number, title, full_text, program)
                VALUES (new.rowid, new.section_number, new.title, new.full_text, new.program);
            END;
            """)
            cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS t_rule_sections_ad AFTER DELETE ON rule_sections BEGIN
                INSERT INTO rule_sections_fts(rule_sections_fts, rowid, section_number, title, full_text, program)
                VALUES ('delete', old.rowid, old.section_number, old.title, old.full_text, old.program);
            END;
            """)
            cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS t_rule_sections_au AFTER UPDATE ON rule_sections BEGIN
                INSERT INTO rule_sections_fts(rule_sections_fts, rowid, section_number, title, full_text, program)
                VALUES ('delete', old.rowid, old.section_number, old.title, old.full_text, old.program);
                INSERT INTO rule_sections_fts(rowid, section_number, title, full_text, program)
                VALUES (new.rowid, new.section_number, new.title, new.full_text, new.program);
            END;
            """)

        conn.commit()
        conn.close()
        logger.info("Rules SQLite Database initialized successfully.")

    def insert_section(self, section_id: str, section_number: str, title: str, full_text: str,
                       parent_id: Optional[str], program: str, source_file: str,
                       last_amended: Optional[str] = None, amendment_note: Optional[str] = None):
        """Insert or replace a rule section."""
        conn = self.get_connection()
        try:
            with conn:
                conn.execute("""
                INSERT OR REPLACE INTO rule_sections (id, section_number, title, full_text, parent_id, program, source_file, last_amended, amendment_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (section_id, section_number, title, full_text, parent_id, program, source_file, last_amended, amendment_note))
        finally:
            conn.close()

    def insert_fact(self, fact_type: str, fact_key: str, fact_value: str, operator: Optional[str],
                    condition_text: Optional[str], section_id: str, program: str):
        """Insert a rule fact."""
        conn = self.get_connection()
        try:
            with conn:
                conn.execute("""
                INSERT OR REPLACE INTO rule_facts (fact_type, fact_key, fact_value, operator, condition_text, section_id, program)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (fact_type, fact_key, fact_value, operator, condition_text, section_id, program))
        finally:
            conn.close()

    def insert_grade(self, grade: str, grade_point: int, description: Optional[str]):
        """Insert or replace a grade scale item."""
        conn = self.get_connection()
        try:
            with conn:
                conn.execute("""
                INSERT OR REPLACE INTO grade_scale (grade, grade_point, description)
                VALUES (?, ?, ?)
                """, (grade, grade_point, description))
        finally:
            conn.close()

    def insert_credit_requirement(self, program: str, category: str, category_full: Optional[str],
                                  min_credits: float, percentage: Optional[float], notes: Optional[str]):
        """Insert a credit requirement."""
        conn = self.get_connection()
        try:
            with conn:
                conn.execute("""
                INSERT INTO credit_requirements (program, category, category_full, min_credits, percentage, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (program, category, category_full, min_credits, percentage, notes))
        finally:
            conn.close()

    def insert_program_milestone(self, program: str, milestone: str, deadline: Optional[str], details: Optional[str], section_id: Optional[str]):
        """Insert a program milestone."""
        conn = self.get_connection()
        try:
            with conn:
                conn.execute("""
                INSERT INTO program_milestones (program, milestone, deadline, details, section_id)
                VALUES (?, ?, ?, ?, ?)
                """, (program, milestone, deadline, details, section_id))
        finally:
            conn.close()

    def lookup_fact(self, fact_key: str, program: Optional[str] = None) -> List[Dict]:
        """Lookup facts by key and program."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if program:
                cursor.execute("""
                SELECT f.*, s.title as section_title, s.source_file, s.section_number
                FROM rule_facts f
                JOIN rule_sections s ON f.section_id = s.id
                WHERE f.fact_key = ? AND f.program = ?
                """, (fact_key, program))
            else:
                cursor.execute("""
                SELECT f.*, s.title as section_title, s.source_file, s.section_number
                FROM rule_facts f
                JOIN rule_sections s ON f.section_id = s.id
                WHERE f.fact_key = ?
                """, (fact_key,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_section(self, section_id: str) -> Optional[Dict]:
        """Retrieve a specific section by its identifier."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM rule_sections WHERE id = ?", (section_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_section_with_children(self, section_id: str) -> List[Dict]:
        """Retrieve a section and all its children/subsections."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            SELECT * FROM rule_sections 
            WHERE id = ? OR parent_id = ?
            ORDER BY id ASC
            """, (section_id, section_id))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def search_sections(self, query: str, program: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Perform full-text search over rule sections."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if program:
                cursor.execute("""
                SELECT s.*, fts.rank
                FROM rule_sections s
                JOIN rule_sections_fts fts ON s.rowid = fts.rowid
                WHERE rule_sections_fts MATCH ? AND s.program = ?
                ORDER BY rank
                LIMIT ?
                """, (query, program, limit))
            else:
                cursor.execute("""
                SELECT s.*, fts.rank
                FROM rule_sections s
                JOIN rule_sections_fts fts ON s.rowid = fts.rowid
                WHERE rule_sections_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """, (query, limit))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_all_sections(self, program: Optional[str] = None) -> List[Dict]:
        """Retrieve all rule sections, optionally scoped to one program."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if program:
                cursor.execute("""
                SELECT * FROM rule_sections
                WHERE program = ?
                ORDER BY source_file, section_number
                """, (program,))
            else:
                cursor.execute("""
                SELECT * FROM rule_sections
                ORDER BY program, source_file, section_number
                """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_grade_scale(self) -> List[Dict]:
        """Retrieve the grade scale mapping."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM grade_scale ORDER BY grade_point DESC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_credit_requirements(self, program: str) -> List[Dict]:
        """Retrieve credit requirements for a given program."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM credit_requirements WHERE program = ?", (program,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_program_milestones(self, program: str) -> List[Dict]:
        """Retrieve program milestones for a given program."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM program_milestones WHERE program = ?", (program,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def clear_all(self):
        """Clear all tables in the database."""
        conn = self.get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM rule_facts")
                conn.execute("DELETE FROM rule_sections")
                conn.execute("DELETE FROM grade_scale")
                conn.execute("DELETE FROM credit_requirements")
                conn.execute("DELETE FROM program_milestones")
                try:
                    conn.execute("DELETE FROM rule_sections_fts")
                except sqlite3.OperationalError:
                    pass
        finally:
            conn.close()
