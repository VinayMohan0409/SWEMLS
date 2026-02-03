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

    def __init__(self) -> None:
        self._h: Dict[str, PatientHistory] = {}

    def get(self, mrn: str) -> PatientHistory:
        return self._h.setdefault(mrn, PatientHistory(creatinine=[]))

    def add_creatinine(self, mrn: str, date_any: str, value: float) -> None:
        d = date_to_ordinal_from_any(date_any)
        if d is None:
            return
        ph = self.get(mrn)
        ph.creatinine.append((d, float(value)))

    def load_history_csv(self, path: str) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        n = 0
        with p.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                mrn = (row.get("mrn") or "").strip()
                if not mrn:
                    continue
                # collect all creatinine_date_k/result_k pairs
                for k in range(0, 10_000):
                    dk = f"creatinine_date_{k}"
                    vk = f"creatinine_result_{k}"
                    if dk not in row and vk not in row:
                        # assume contiguous indices; break early
                        if k > 0:
                            break
                    d = (row.get(dk) or "").strip()
                    v = (row.get(vk) or "").strip()
                    if not d or not v:
                        continue
                    try:
                        vf = float(v)
                    except ValueError:
                        continue
                    self.add_creatinine(mrn, d, vf)
                n += 1
        return n
