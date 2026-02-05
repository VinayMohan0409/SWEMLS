# tests/unit/test_history.py

import csv

import pytest

import aki_service.history as h


class FakeDB:
    def __init__(self):
        self.bulk_calls = []
        self.history_map = {}

        self.bulk_return = 0

    def insert_labs_bulk(self, batch):
        # Record bulk inserts
        self.bulk_calls.append(list(batch))
        return self.bulk_return if self.bulk_return else len(batch)

    def get_history(self, mrn):
        # Return raw history rows
        return list(self.history_map.get(mrn, []))


def write_csv(tmp_path, name, fieldnames, rows):
    p = tmp_path / name
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def test_load_history_csv_missing_file_returns_0(tmp_path):
    # Missing file returns 0
    db = FakeDB()
    store = h.HistoryStore(db=db)

    n = store.load_history_csv(str(tmp_path / "nope.csv"))
    assert n == 0
    assert db.bulk_calls == []


def test_load_history_csv_skips_blank_mrn(tmp_path):
    # Blank MRN rows ignored
    db = FakeDB()
    store = h.HistoryStore(db=db)

    path = write_csv(
        tmp_path,
        "history.csv",
        ["mrn", "creatinine_date_1", "creatinine_result_1"],
        [
            {"mrn": " ", "creatinine_date_1": "2025-01-01", "creatinine_result_1": "1.0"},
            {"mrn": "", "creatinine_date_1": "2025-01-01", "creatinine_result_1": "1.0"},
        ],
    )

    n = store.load_history_csv(path)
    assert n == 0
    assert db.bulk_calls == []


def test_load_history_csv_inserts_valid_pairs(tmp_path):
    # Valid date and value inserted
    db = FakeDB()
    store = h.HistoryStore(db=db)

    path = write_csv(
        tmp_path,
        "history.csv",
        [
            "mrn",
            "creatinine_date_1",
            "creatinine_result_1",
            "creatinine_date_2",
            "creatinine_result_2",
        ],
        [
            {
                "mrn": "123",
                "creatinine_date_1": "2025-01-01",
                "creatinine_result_1": "1.2",
                "creatinine_date_2": "2025-01-03",
                "creatinine_result_2": "2.4",
            }
        ],
    )

    n = store.load_history_csv(path)
    assert n == 2
    assert len(db.bulk_calls) == 1
    assert db.bulk_calls[0] == [("123", "2025-01-01", 1.2), ("123", "2025-01-03", 2.4)]


def test_load_history_csv_skips_bad_values_and_missing_pairs(tmp_path):
    # Bad floats and missing pairs ignored
    db = FakeDB()
    store = h.HistoryStore(db=db)

    path = write_csv(
        tmp_path,
        "history.csv",
        [
            "mrn",
            "creatinine_date_1",
            "creatinine_result_1",
            "creatinine_date_2",
            "creatinine_result_2",
            "creatinine_date_3",
            "creatinine_result_3",
        ],
        [
            {
                "mrn": "123",
                "creatinine_date_1": "2025-01-01",
                "creatinine_result_1": "bad",
                "creatinine_date_2": "2025-01-02",
                "creatinine_result_2": "",
                "creatinine_date_3": "",
                "creatinine_result_3": "1.0",
            }
        ],
    )

    n = store.load_history_csv(path)
    assert n == 0
    assert db.bulk_calls == []


def test_load_history_csv_batches_at_5000(tmp_path):
    # Batches split at 5000
    db = FakeDB()
    store = h.HistoryStore(db=db)

    rows = []
    for i in range(5001):
        rows.append(
            {
                "mrn": "123",
                "creatinine_date_1": f"2025-01-{(i % 28) + 1:02d}",
                "creatinine_result_1": "1.0",
            }
        )

    path = write_csv(
        tmp_path,
        "history.csv",
        ["mrn", "creatinine_date_1", "creatinine_result_1"],
        rows,
    )

    n = store.load_history_csv(path)
    assert n == 5001
    assert len(db.bulk_calls) == 2
    assert len(db.bulk_calls[0]) == 5000
    assert len(db.bulk_calls[1]) == 1


def test_load_history_csv_date_columns_sorted(tmp_path):
    # Date columns sorted by name
    db = FakeDB()
    store = h.HistoryStore(db=db)

    path = write_csv(
        tmp_path,
        "history.csv",
        [
            "mrn",
            "creatinine_date_2",
            "creatinine_result_2",
            "creatinine_date_1",
            "creatinine_result_1",
        ],
        [
            {
                "mrn": "123",
                "creatinine_date_2": "2025-01-02",
                "creatinine_result_2": "2.0",
                "creatinine_date_1": "2025-01-01",
                "creatinine_result_1": "1.0",
            }
        ],
    )

    store.load_history_csv(path)
    assert db.bulk_calls[0] == [("123", "2025-01-01", 1.0), ("123", "2025-01-02", 2.0)]


def test_get_history_from_db_converts_to_ordinals(monkeypatch):
    # Converts timestamps to ordinals
    db = FakeDB()
    db.history_map["123"] = [("2025-01-01 00:00:00", 1.0), ("20250102000000", 2.0)]
    store = h.HistoryStore(db=db)

    seen = []

    def fake_date_to_ordinal(x):
        seen.append(x)
        return 111 if "2025-01-01" in x else 222

    monkeypatch.setattr(h, "date_to_ordinal_from_any", fake_date_to_ordinal, raising=True)

    out = store.get_history_from_db("123")
    assert out == [(111, 1.0), (222, 2.0)]
    assert seen == ["2025-01-01 00:00:00", "20250102000000"]
