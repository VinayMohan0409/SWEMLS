import sqlite3
import logging
from typing import List, Tuple, Optional

from aki_service.pager import PagerClient

log = logging.getLogger("aki_service")

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None  # Single connection, reused
        self._init_db()

    def _get_conn(self):
        """
        Returns a SINGLE reusable connection.
        
        Why this matters:
        - Each sqlite3.connect() creates a new "view" of the database
        - With WAL mode, a new connection might not see recent writes immediately
        - By reusing ONE connection, all reads see all previous writes
        
        This prevents non-deterministic behavior where:
        - insert_lab() writes with Connection A
        - get_history() reads with Connection B (might miss the write!)
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path, 
                timeout=10,
                check_same_thread=False  # Safe since we're single-threaded message processing
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")  # Good balance of safety/speed
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def close(self):
        """Explicitly close the connection when done."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _init_db(self):
        with self._get_conn() as conn:
            # Patients: tracks admission status and demographics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    mrn TEXT PRIMARY KEY,
                    is_admitted BOOLEAN DEFAULT 1,
                    last_admit_time TEXT,
                    last_discharge_time TEXT,
                    dob TEXT,
                    sex TEXT
                )
            """)
            # Labs: stores unpivoted creatinine history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS labs (
                    mrn TEXT,
                    test_time TEXT,
                    value REAL,
                    PRIMARY KEY (mrn, test_time)
                )
            """)
            # Alerts: critical for idempotency (one page per test)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    mrn TEXT, 
                    test_time TEXT,
                    status TEXT, 
                    attempt_count INTEGER DEFAULT 0,
                    last_attempt_time TEXT,
                    PRIMARY KEY (mrn, test_time)
                )
            """)
            conn.commit()

    def has_alert_for_current_admission(self, mrn: str) -> bool:
        """Checks if a 'sent' alert exists since the patient's last admission."""
        conn = self._get_conn()
        # Compare alert test_time with patient's last_admit_time
        # Both are in HL7 format (YYYYMMDDHHMMSS), so string comparison works
        row = conn.execute("""
            SELECT 1 FROM alerts a
            JOIN patients p ON a.mrn = p.mrn
            WHERE a.mrn = ? 
              AND a.status = 'sent'
              AND a.test_time >= p.last_admit_time
            LIMIT 1
        """, (mrn,)).fetchone()
        return row is not None

    def insert_labs_bulk(self, rows: List[Tuple[str, str, float]]) -> int:
        """Bulk insert labs efficiently. Duplicates ignored."""
        if not rows:
            return 0
        conn = self._get_conn()
        conn.executemany(
            "INSERT OR IGNORE INTO labs (mrn, test_time, value) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        return len(rows)



    def insert_lab(self, mrn: str, test_time: str, value: float) -> bool:
        """Returns True if new data was saved; False if duplicate."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO labs (mrn, test_time, value) VALUES (?, ?, ?)",
                    (mrn, test_time, value)
                )
                return True
        except sqlite3.IntegrityError:
            return False

    def get_history(self, mrn: str) -> List[Tuple[str, float]]:
        """Queries all historical results for a specific patient."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT test_time, value FROM labs WHERE mrn = ? ORDER BY test_time ASC",
                (mrn,)
            ).fetchall()
            return [(row["test_time"], row["value"]) for row in rows]

    def update_alert(self, mrn: str, test_time: str, status: str):
        """
        Durable alert tracking. 
        When a NEW alert is created, attempt_count is initialized to 0.
        """
        with self._get_conn() as conn:
            # We insert with attempt_count=0. 
            # ON CONFLICT only updates the status, leaving attempt_count alone.
            conn.execute("""
                INSERT INTO alerts (mrn, test_time, status, attempt_count) 
                VALUES (?, ?, ?, 0)
                ON CONFLICT(mrn, test_time) DO UPDATE SET status=excluded.status
            """, (mrn, test_time, status))
    
    def update_patient(self, mrn: str, is_admitted: bool, dob: str = None, sex: str = None, 
                       admit_time: str = None, discharge_time: str = None):
        """Updates or inserts patient admission, demographics, and visit timestamps."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO patients (mrn, is_admitted, dob, sex, last_admit_time, last_discharge_time) 
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(mrn) DO UPDATE SET 
                    is_admitted=excluded.is_admitted,
                    dob=COALESCE(excluded.dob, dob),
                    sex=COALESCE(excluded.sex, sex),
                    last_admit_time=COALESCE(excluded.last_admit_time, last_admit_time),
                    last_discharge_time=COALESCE(excluded.last_discharge_time, last_discharge_time)
            """, (mrn, int(is_admitted), dob, sex, admit_time, discharge_time))

    def get_patient_state(self, mrn: str) -> Optional[dict]:
        """Retrieves patient demographics and admission status."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM patients WHERE mrn = ?", (mrn,)).fetchone()
            return dict(row) if row else None

    def retry_failed_alerts(self, pager: PagerClient) -> int:
        count = 0
        with self._get_conn() as conn:
            # Only retry alerts that haven't failed too many times (e.g., max 3 attempts)
            failed = conn.execute("""
                SELECT mrn, test_time, attempt_count 
                FROM alerts 
                WHERE status = 'failed' AND attempt_count < 3
            """).fetchall()
            
            for row in failed:
                import time
                current_ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                
                ok, _ = pager.send_page(row["mrn"], row["test_time"])
                
                new_status = "sent" if ok else "failed"
                new_count = row["attempt_count"] + 1
                
                conn.execute("""
                    UPDATE alerts 
                    SET status = ?, attempt_count = ?, last_attempt_time = ?
                    WHERE mrn = ? AND test_time = ?
                """, (new_status, new_count, current_ts, row["mrn"], row["test_time"]))
                
                if ok:
                    count += 1
            conn.commit()
        return count