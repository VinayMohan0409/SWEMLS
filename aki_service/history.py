from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

from .model_compat import date_to_ordinal_from_any


@dataclass
class PatientHistory:
    # list of (date_ordinal, value)
    creatinine: List[Tuple[int, float]]


class HistoryStore:
    """In-memory history store loaded from history.csv (wide format).

    This is intentionally simple so you can later replace it with SQLite
    without changing parsing and inference layers.
    """

    def __init__(self, db) -> None:
        # db connection
        self.db = db 
    
    def load_history_csv(self, path: str) -> int:
            p = Path(path)
            if not p.exists():
                return 0
                
            # Updated check: Just try to get any history from a known MRN or check labs count
            # For simplicity, we can let the SQL 'INSERT OR IGNORE' handles duplicates, 
            # but to follow your 'Optimization' logic:
            # Use a simple query to see if the table is already populated.
            
            n = 0
            with p.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mrn = row["mrn"].strip()
                    # Unpivot the wide format 
                    for k in range(26):
                        date_key = f"creatinine_date_{k}"
                        res_key = f"creatinine_result_{k}"
                        
                        if date_key in row and row[date_key] and row[res_key]:
                            # insert_lab already handles duplicates via IntegrityError
                            if self.db.insert_lab(mrn, row[date_key], float(row[res_key])):
                                n += 1
            return n
    
    def get_history_from_db(self, mrn: str) -> List[Tuple[int, float]]:
        """Fetches labs from DB and converts dates to ordinals for the model."""
        raw_history = self.db.get_history(mrn)
        # Convert the ISO/HL7 string from DB to the ordinal integer the model expects
        return [(date_to_ordinal_from_any(ts), val) for ts, val in raw_history]
