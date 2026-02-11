from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

from .model_compat import date_to_ordinal_from_any


@dataclass
class PatientHistory:
    creatinine: List[Tuple[int, float]]


class HistoryStore:
    """In-memory history store loaded from history.csv (wide format).

    This is intentionally simple so you can later replace it with SQLite
    without changing parsing and inference layers.
    """

    def __init__(self, db) -> None:
        self.db = db 
    
    def load_history_csv(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            return 0

        n = 0
        batch: List[Tuple[str, str, float]] = []
        batch_size = 5000

        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            date_columns = sorted([c for c in fieldnames if c.startswith("creatinine_date_")])

            for row in reader:
                mrn = (row.get("mrn") or "").strip()
                if not mrn:
                    continue

                for date_col in date_columns:
                    idx = date_col.replace("creatinine_date_", "")
                    result_col = f"creatinine_result_{idx}"
                    d = row.get(date_col)
                    v = row.get(result_col)
                    if not d or not v:
                        continue
                    try:
                        value = float(v)
                    except (ValueError, TypeError):
                        continue

                    batch.append((mrn, d, value))
                    if len(batch) >= batch_size:
                        n += self.db.insert_labs_bulk(batch)
                        batch.clear()

        if batch:
            n += self.db.insert_labs_bulk(batch)

        return n

    
    def get_history_from_db(self, mrn: str) -> List[Tuple[int, float]]:
        """Fetches labs from DB and converts dates to ordinals for the model."""
        raw_history = self.db.get_history(mrn)
        # Convert the ISO/HL7 string from DB to the ordinal integer the model expects
        return [(date_to_ordinal_from_any(ts), val) for ts, val in raw_history]
