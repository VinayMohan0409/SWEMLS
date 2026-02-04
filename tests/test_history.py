import csv
from pathlib import Path
import pytest

from aki_service.history import HistoryStore


class FakeDB:
    """fake DB for HistoryStore unit tests"""
    def __init__(self):
        self.inserted = []  # (mrn, date_str, value)
        self.history_by_mrn = {}  # mrn -> [(timestamp_str, value)]
        self.duplicate_keys = set()  # keys that should return False on insert

    def insert_lab(self, mrn: str, date_str: str, value: float) -> bool:
        key = (mrn, date_str, value)
        if key in self.duplicate_keys:
            return False
        self.inserted.append((mrn, date_str, value))
        return True

    def get_history(self, mrn: str):
        return self.history_by_mrn.get(mrn, [])


def write_history_csv(tmp_path: Path, rows):
    p = tmp_path / "history.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def test_load_history_csv_missing_file_returns_zero(tmp_path):
    # tests that missing history file returns 0 inserts
    db = FakeDB()
    hs = HistoryStore(db)
    n = hs.load_history_csv(str(tmp_path / "does_not_exist.csv"))
    assert n == 0
    assert db.inserted == []


def test_load_history_csv_inserts_only_nonempty_date_and_value(tmp_path):
    # tests that only (date,value) pairs present in the wide row are inserted
    db = FakeDB()
    hs = HistoryStore(db)

    # build row with two valid pairs and one incomplete pair
    row = {"mrn": " 123 "}
    for k in range(26):
        row[f"creatinine_date_{k}"] = ""
        row[f"creatinine_result_{k}"] = ""

    row["creatinine_date_0"] = "20240101"
    row["creatinine_result_0"] = "100.0"
    row["creatinine_date_1"] = "20240102"
    row["creatinine_result_1"] = "110.0"
    row["creatinine_date_2"] = "20240103"
    row["creatinine_result_2"] = ""  # incomplete -> should not insert

    path = write_history_csv(tmp_path, [row])
    n = hs.load_history_csv(path)

    assert n == 2
    assert db.inserted == [
        ("123", "20240101", 100.0),
        ("123", "20240102", 110.0),
    ]


def test_load_history_csv_counts_only_new_inserts(tmp_path):
    # tests that HistoryStore counts only DB inserts that return True (skips duplicates)
    db = FakeDB()
    hs = HistoryStore(db)

    row = {"mrn": "1"}
    for k in range(26):
        row[f"creatinine_date_{k}"] = ""
        row[f"creatinine_result_{k}"] = ""

    row["creatinine_date_0"] = "20240101"
    row["creatinine_result_0"] = "100.0"

    # mark that exact lab insertion as duplicate in fake DB
    db.duplicate_keys.add(("1", "20240101", 100.0))

    path = write_history_csv(tmp_path, [row])
    n = hs.load_history_csv(path)

    assert n == 0
    assert db.inserted == []


def test_get_history_from_db_converts_to_ordinals(monkeypatch):
    # tests that get_history_from_db converts DB timestamps to date ordinals
    db = FakeDB()
    hs = HistoryStore(db)
    db.history_by_mrn["1"] = [("20240101000000", 100.0), ("20240102000000", 110.0)]

    # patch the imported conversion function inside aki_service.history module
    import aki_service.history as history_mod
    monkeypatch.setattr(history_mod, "date_to_ordinal_from_any", lambda ts: 123 if ts.endswith("100000") else 124)

    out = hs.get_history_from_db("1")
    assert out == [(123, 100.0), (124, 110.0)]
