# tests/unit/test_router.py

import sqlite3
from dataclasses import dataclass

import pytest

import aki_service.router as router_mod


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE alerts (
            mrn TEXT NOT NULL,
            test_time TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL,
            last_attempt_time TEXT,
            UNIQUE(mrn, test_time)
        )
        """
    )
    conn.commit()
    return conn


class FakeDB:
    def __init__(self, conn):
        self._conn = conn
        self.updated = []
        self.labs = []
        self.patient_state = {}

        self.insert_lab_return = True

    def _get_conn(self):
        return self._conn

    def update_patient(self, mrn, **kwargs):
        self.updated.append((mrn, kwargs))

    def insert_lab(self, mrn, test_time, value):
        self.labs.append((mrn, test_time, value))
        return self.insert_lab_return

    def get_patient_state(self, mrn):
        return self.patient_state.get(mrn)


class FakeHistory:
    def __init__(self, history=None):
        self.history = history if history is not None else []
        self.calls = []

    def get_history_from_db(self, mrn):
        self.calls.append(mrn)
        return list(self.history)


class FakeInference:
    def __init__(self, ready=True, should_page=False, crash=False):
        self._ready = ready
        self._should_page = should_page
        self._crash = crash
        self.calls = []

    def is_ready(self):
        return self._ready

    def predict_aki(self, *, mrn, test_time_hl7, history_ord_vals, demographics):
        self.calls.append(
            dict(
                mrn=mrn,
                test_time_hl7=test_time_hl7,
                history_ord_vals=history_ord_vals,
                demographics=demographics,
            )
        )
        if self._crash:
            raise RuntimeError("boom")
        return self._should_page


class FakePager:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def send_page(self, mrn, test_time):
        self.calls.append((mrn, test_time))
        return (self.ok, "info")


def patch_hl7(monkeypatch):
    class HL7ParseError(Exception):
        pass

    @dataclass
    class UnknownEvent:
        msg_type: str = "ZZZ"

    @dataclass
    class AdmitEvent:
        mrn: str
        dob: str
        sex: str
        msg_time: str

    @dataclass
    class DischargeEvent:
        mrn: str
        msg_time: str

    @dataclass
    class CreatinineEvent:
        mrn: str
        test_time: str
        value: float

    monkeypatch.setattr(router_mod.hl7, "HL7ParseError", HL7ParseError, raising=True)
    monkeypatch.setattr(router_mod.hl7, "UnknownEvent", UnknownEvent, raising=True)
    monkeypatch.setattr(router_mod.hl7, "AdmitEvent", AdmitEvent, raising=True)
    monkeypatch.setattr(router_mod.hl7, "DischargeEvent", DischargeEvent, raising=True)
    monkeypatch.setattr(router_mod.hl7, "CreatinineEvent", CreatinineEvent, raising=True)

    return dict(
        HL7ParseError=HL7ParseError,
        UnknownEvent=UnknownEvent,
        AdmitEvent=AdmitEvent,
        DischargeEvent=DischargeEvent,
        CreatinineEvent=CreatinineEvent,
    )


def patch_date_to_ordinal(monkeypatch):
    monkeypatch.setattr(router_mod, "date_to_ordinal_from_any", lambda _: 10, raising=True)


def test_handle_message_parse_error_returns_ae(monkeypatch):
    types = patch_hl7(monkeypatch)

    def parse_event(_):
        raise types["HL7ParseError"]("bad")

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    r = router_mod.Router(db=FakeDB(conn), history=FakeHistory(), inference=FakeInference())

    assert r.handle_message(b"x") == "AE"


def test_handle_message_unknown_event_returns_aa(monkeypatch):
    types = patch_hl7(monkeypatch)

    def parse_event(_):
        return types["UnknownEvent"]("ABC")

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    db = FakeDB(conn)
    r = router_mod.Router(db=db, history=FakeHistory(), inference=FakeInference())

    assert r.handle_message(b"x") == "AA"
    assert db.updated == []


def test_handle_message_admit_updates_patient(monkeypatch):
    types = patch_hl7(monkeypatch)

    def parse_event(_):
        return types["AdmitEvent"](mrn="1", dob="20000101", sex="M", msg_time="20250101120000")

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    db = FakeDB(conn)
    r = router_mod.Router(db=db, history=FakeHistory(), inference=FakeInference())

    assert r.handle_message(b"x") == "AA"
    assert db.updated == [
        (
            "1",
            dict(is_admitted=True, dob="20000101", sex="M", admit_time="20250101120000"),
        )
    ]


def test_handle_message_discharge_updates_patient(monkeypatch):
    types = patch_hl7(monkeypatch)

    def parse_event(_):
        return types["DischargeEvent"](mrn="1", msg_time="20250102120000")

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    db = FakeDB(conn)
    r = router_mod.Router(db=db, history=FakeHistory(), inference=FakeInference())

    assert r.handle_message(b"x") == "AA"
    assert db.updated == [("1", dict(is_admitted=False, discharge_time="20250102120000"))]


def test_creatinine_duplicate_lab_skips_inference(monkeypatch):
    types = patch_hl7(monkeypatch)
    patch_date_to_ordinal(monkeypatch)

    def parse_event(_):
        return types["CreatinineEvent"](mrn="1", test_time="20250103120000", value=1.2)

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    db = FakeDB(conn)
    db.insert_lab_return = False
    inf = FakeInference(ready=True, should_page=True)
    r = router_mod.Router(db=db, history=FakeHistory(), inference=inf)

    assert r.handle_message(b"x") == "AA"
    assert inf.calls == []
    assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0


def test_creatinine_model_not_ready_skips_prediction(monkeypatch):
    types = patch_hl7(monkeypatch)
    patch_date_to_ordinal(monkeypatch)

    def parse_event(_):
        return types["CreatinineEvent"](mrn="1", test_time="20250103120000", value=1.2)

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    db = FakeDB(conn)
    db.patient_state["1"] = dict(is_admitted=1, dob="20000101", sex="M")
    inf = FakeInference(ready=False, should_page=True)
    r = router_mod.Router(db=db, history=FakeHistory(history=[(9, 1.0), (10, 1.2)]), inference=inf)

    assert r.handle_message(b"x") == "AA"
    assert inf.calls == []
    assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0


def test_creatinine_inference_crash_returns_ae(monkeypatch):
    types = patch_hl7(monkeypatch)
    patch_date_to_ordinal(monkeypatch)

    def parse_event(_):
        return types["CreatinineEvent"](mrn="1", test_time="20250103120000", value=1.2)

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    db = FakeDB(conn)
    db.patient_state["1"] = dict(is_admitted=1, dob="20000101", sex="M")
    inf = FakeInference(ready=True, crash=True)
    r = router_mod.Router(db=db, history=FakeHistory(history=[(9, 1.0)]), inference=inf)

    assert r.handle_message(b"x") == "AE"
    assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0


def test_creatinine_suppressed_when_discharged(monkeypatch):
    types = patch_hl7(monkeypatch)
    patch_date_to_ordinal(monkeypatch)

    def parse_event(_):
        return types["CreatinineEvent"](mrn="1", test_time="20250103120000", value=1.2)

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    db = FakeDB(conn)
    db.patient_state["1"] = dict(is_admitted=0, dob="20000101", sex="M")
    inf = FakeInference(ready=True, should_page=True)
    r = router_mod.Router(db=db, history=FakeHistory(history=[(9, 1.0)]), inference=inf)

    assert r.handle_message(b"x") == "AA"
    assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0


def test_creatinine_dry_run_pages_sets_sent(monkeypatch):
    types = patch_hl7(monkeypatch)
    patch_date_to_ordinal(monkeypatch)

    def parse_event(_):
        return types["CreatinineEvent"](mrn="1", test_time="20250103120000", value=1.2)

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    db = FakeDB(conn)
    db.patient_state["1"] = dict(is_admitted=1, dob="20000101", sex="M")
    inf = FakeInference(ready=True, should_page=True)
    r = router_mod.Router(db=db, history=FakeHistory(history=[(9, 1.0)]), inference=inf, pager=None)

    assert r.handle_message(b"x") == "AA"
    row = conn.execute(
        "SELECT status, attempt_count FROM alerts WHERE mrn = ? AND test_time = ?",
        ("1", "20250103120000"),
    ).fetchone()
    assert row == ("sent", 0)


def test_creatinine_already_paged_once_per_mrn_suppresses_new_event(monkeypatch):
    types = patch_hl7(monkeypatch)
    patch_date_to_ordinal(monkeypatch)

    def parse_event(_):
        return types["CreatinineEvent"](mrn="1", test_time="20250104120000", value=1.3)

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    conn.execute(
        "INSERT INTO alerts (mrn, test_time, status, attempt_count) VALUES (?, ?, 'sent', 1)",
        ("1", "20250103120000"),
    )
    conn.commit()

    db = FakeDB(conn)
    db.patient_state["1"] = dict(is_admitted=1, dob="20000101", sex="M")
    inf = FakeInference(ready=True, should_page=True)
    r = router_mod.Router(db=db, history=FakeHistory(history=[(9, 1.0)]), inference=inf)

    assert r.handle_message(b"x") == "AA"
    cnt = conn.execute("SELECT COUNT(*) FROM alerts WHERE mrn = ?", ("1",)).fetchone()[0]
    assert cnt == 1


def test_creatinine_claim_race_integrity_error_skips_send(monkeypatch):
    types = patch_hl7(monkeypatch)
    patch_date_to_ordinal(monkeypatch)

    def parse_event(_):
        return types["CreatinineEvent"](mrn="1", test_time="20250103120000", value=1.2)

    monkeypatch.setattr(router_mod.hl7, "parse_event", parse_event, raising=True)

    conn = make_conn()
    conn.execute(
        "INSERT INTO alerts (mrn, test_time, status, attempt_count) VALUES (?, ?, 'pending', 0)",
        ("1", "20250103120000"),
    )
    conn.commit()

    db = FakeDB(conn)
    db.patient_state["1"] = dict(is_admitted=1, dob="20000101", sex="M")
    inf = FakeInference(ready=True, should_page=True)
    pager = FakePager(ok=True)
    r = router_mod.Router(db=db, history=FakeHistory(history=[(9, 1.0)]), inference=inf, pager=pager)

    assert r.handle_message(b"x") == "AA"
    assert pager.calls == []


def test_parse_hl7_timestamp(monkeypatch):
    assert router_mod.Router.parse_hl7_timestamp("20250101").strftime("%Y-%m-%d") == "2025-01-01"
    assert router_mod.Router.parse_hl7_timestamp("202501011230").strftime("%Y-%m-%d %H:%M") == "2025-01-01 12:30"
    assert (
        router_mod.Router.parse_hl7_timestamp("20250101123059").strftime("%Y-%m-%d %H:%M:%S")
        == "2025-01-01 12:30:59"
    )

    with pytest.raises(ValueError):
        router_mod.Router.parse_hl7_timestamp("2025-01-01")

    with pytest.raises(ValueError):
        router_mod.Router.parse_hl7_timestamp("2025010112")


def test_normalize_to_iso(monkeypatch):
    assert router_mod.Router.normalize_to_iso("2025-01-01 12:30") == "2025-01-01 12:30:00"
    assert router_mod.Router.normalize_to_iso("2025-01-01 12:30:59") == "2025-01-01 12:30:59"
    assert router_mod.Router.normalize_to_iso("2025-01-01T12:30:59Z") == "2025-01-01 12:30:59"
    assert router_mod.Router.normalize_to_iso("20250101123059") == "2025-01-01 12:30:59"
