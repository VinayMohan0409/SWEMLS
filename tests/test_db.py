# tests/unit/test_db.py

import sqlite3
import time

import pytest

import aki_service.db as db_mod


def make_db(tmp_path):
    # Creates real sqlite file db
    p = tmp_path / "t.sqlite"
    return db_mod.Database(str(p))


def test_get_conn_reused_and_row_factory(tmp_path):
    # Same connection reused
    db = make_db(tmp_path)
    c1 = db._get_conn()
    c2 = db._get_conn()
    assert c1 is c2
    assert c1.row_factory is sqlite3.Row


def test_close_resets_connection(tmp_path):
    # Close drops connection handle
    db = make_db(tmp_path)
    c1 = db._get_conn()
    db.close()
    c2 = db._get_conn()
    assert c1 is not c2


def test_init_db_creates_tables(tmp_path):
    # Tables exist after init
    db = make_db(tmp_path)
    conn = db._get_conn()
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "patients" in names
    assert "labs" in names
    assert "alerts" in names


def test_insert_lab_and_get_history_sorted(tmp_path):
    # Inserts and returns ordered history
    db = make_db(tmp_path)

    assert db.insert_lab("1", "20250102000000", 2.0) is True
    assert db.insert_lab("1", "20250101000000", 1.0) is True

    hist = db.get_history("1")
    assert hist == [("20250101000000", 1.0), ("20250102000000", 2.0)]


def test_insert_lab_duplicate_returns_false(tmp_path):
    # Duplicate blocked by PK
    db = make_db(tmp_path)

    assert db.insert_lab("1", "20250101000000", 1.0) is True
    assert db.insert_lab("1", "20250101000000", 1.0) is False


def test_insert_labs_bulk_empty_returns_0(tmp_path):
    # Empty bulk no op
    db = make_db(tmp_path)
    assert db.insert_labs_bulk([]) == 0


def test_insert_labs_bulk_inserts_and_ignores_dupes(tmp_path):
    # Bulk insert ignores duplicates
    db = make_db(tmp_path)

    n = db.insert_labs_bulk(
        [
            ("1", "20250101000000", 1.0),
            ("1", "20250101000000", 1.0),
            ("1", "20250102000000", 2.0),
        ]
    )
    assert n == 3

    hist = db.get_history("1")
    assert hist == [("20250101000000", 1.0), ("20250102000000", 2.0)]


def test_update_patient_insert_and_update_rules(tmp_path):
    # Upsert keeps existing dob and sex when None
    db = make_db(tmp_path)

    db.update_patient("1", is_admitted=True, dob="20000101", sex="M", admit_time="20250101120000")
    st = db.get_patient_state("1")
    assert st["mrn"] == "1"
    assert st["is_admitted"] == 1
    assert st["dob"] == "20000101"
    assert st["sex"] == "M"
    assert st["last_admit_time"] == "20250101120000"
    assert st["last_discharge_time"] is None

    # Update admission status and discharge time without overwriting demo
    db.update_patient("1", is_admitted=False, discharge_time="20250102120000")
    st2 = db.get_patient_state("1")
    assert st2["is_admitted"] == 0
    assert st2["dob"] == "20000101"
    assert st2["sex"] == "M"
    assert st2["last_discharge_time"] == "20250102120000"


def test_get_patient_state_missing_returns_none(tmp_path):
    # Missing patient yields None
    db = make_db(tmp_path)
    assert db.get_patient_state("nope") is None


def test_update_alert_insert_then_update_keeps_attempt_count(tmp_path):
    # Update alert keeps attempt_count when updating
    db = make_db(tmp_path)
    conn = db._get_conn()

    db.update_alert("1", "20250101000000", "pending")
    row = conn.execute(
        "SELECT status, attempt_count FROM alerts WHERE mrn=? AND test_time=?",
        ("1", "20250101000000"),
    ).fetchone()
    assert row["status"] == "pending"
    assert row["attempt_count"] == 0

    # Bump attempt_count manually then update status
    conn.execute(
        "UPDATE alerts SET attempt_count=2 WHERE mrn=? AND test_time=?",
        ("1", "20250101000000"),
    )
    conn.commit()

    db.update_alert("1", "20250101000000", "sent")
    row2 = conn.execute(
        "SELECT status, attempt_count FROM alerts WHERE mrn=? AND test_time=?",
        ("1", "20250101000000"),
    ).fetchone()
    assert row2["status"] == "sent"
    assert row2["attempt_count"] == 2


def test_has_alert_for_current_admission_true_only_after_admit_time(tmp_path):
    # Checks test_time against last admit time
    db = make_db(tmp_path)
    conn = db._get_conn()

    db.update_patient("1", is_admitted=True, admit_time="20250101120000")

    conn.execute(
        "INSERT INTO alerts (mrn, test_time, status, attempt_count) VALUES (?, ?, 'sent', 1)",
        ("1", "20250101110000"),
    )
    conn.execute(
        "INSERT INTO alerts (mrn, test_time, status, attempt_count) VALUES (?, ?, 'sent', 1)",
        ("1", "20250101130000"),
    )
    conn.commit()

    assert db.has_alert_for_current_admission("1") is True

    # If admit time later than all sent alerts then False
    db.update_patient("1", is_admitted=True, admit_time="20250101140000")
    assert db.has_alert_for_current_admission("1") is False


def test_retry_failed_alerts_updates_status_and_attempts(monkeypatch, tmp_path):
    # Retries only failed with attempt_count below 3
    db = make_db(tmp_path)
    conn = db._get_conn()

    conn.execute(
        "INSERT INTO alerts (mrn, test_time, status, attempt_count) VALUES (?, ?, 'failed', 0)",
        ("1", "20250101000000"),
    )
    conn.execute(
        "INSERT INTO alerts (mrn, test_time, status, attempt_count) VALUES (?, ?, 'failed', 2)",
        ("2", "20250101000000"),
    )
    conn.execute(
        "INSERT INTO alerts (mrn, test_time, status, attempt_count) VALUES (?, ?, 'failed', 3)",
        ("3", "20250101000000"),
    )
    conn.commit()

    monkeypatch.setattr(time, "strftime", lambda *_a, **_k: "20000101000000", raising=True)

    class FakePager:
        def __init__(self):
            self.calls = []

        def send_page(self, mrn, test_time):
            self.calls.append((mrn, test_time))
            if mrn == "1":
                return True, "ok"
            return False, "no"

    pager = FakePager()
    sent = db.retry_failed_alerts(pager)

    assert sent == 1
    assert pager.calls == [("1", "20250101000000"), ("2", "20250101000000")]

    r1 = conn.execute(
        "SELECT status, attempt_count, last_attempt_time FROM alerts WHERE mrn=?",
        ("1",),
    ).fetchone()
    assert r1["status"] == "sent"
    assert r1["attempt_count"] == 1
    assert r1["last_attempt_time"] == "20000101000000"

    r2 = conn.execute(
        "SELECT status, attempt_count, last_attempt_time FROM alerts WHERE mrn=?",
        ("2",),
    ).fetchone()
    assert r2["status"] == "failed"
    assert r2["attempt_count"] == 3
    assert r2["last_attempt_time"] == "20000101000000"

    r3 = conn.execute(
        "SELECT status, attempt_count FROM alerts WHERE mrn=?",
        ("3",),
    ).fetchone()
    assert r3["status"] == "failed"
    assert r3["attempt_count"] == 3
