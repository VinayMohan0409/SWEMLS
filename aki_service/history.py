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
            
            n = 0
            with p.open("r", newline="") as f:
                reader = csv.DictReader(f)
                
                # Get all column names once
                fieldnames = reader.fieldnames or []
                
                # Dynamically find all creatinine column pairs
                # Look for columns matching pattern: creatinine_date_N, creatinine_result_N
                date_columns = sorted([
                    col for col in fieldnames 
                    if col.startswith("creatinine_date_")
                ])
                
                for row in reader:
                    mrn = row["mrn"].strip()
                    
                    # Process each date column dynamically
                    for date_col in date_columns:
                        # Extract the index (e.g., "creatinine_date_5" -> "5")
                        idx = date_col.replace("creatinine_date_", "")
                        result_col = f"creatinine_result_{idx}"
                        
                        if result_col in row and row[date_col] and row[result_col]:
                            try:
                                value = float(row[result_col])
                                if self.db.insert_lab(mrn, row[date_col], value):
                                    n += 1
                            except (ValueError, TypeError):
                                # Skip rows with invalid numeric values
                                pass
            return n
    
    def get_history_from_db(self, mrn: str) -> List[Tuple[int, float]]:
        """Fetches labs from DB and converts dates to ordinals for the model."""
        raw_history = self.db.get_history(mrn)
        # Convert the ISO/HL7 string from DB to the ordinal integer the model expects
        return [(date_to_ordinal_from_any(ts), val) for ts, val in raw_history]
