import pytest

from aki_service.router import Router
from aki_service.inference import Demographics


# event types (match the isinstance checks in Router.handle_message)
class FakeUnknownEvent:
    def __init__(self, msg_type="ZZZ^ZZZ"):
        self.msg_type = msg_type


class FakeAdmitEvent:
    def __init__(self, mrn="1", dob="20000101", sex="M", msg_time="20240101000000"):
        self.mrn = mrn
        self.dob = dob
        self.sex = sex
        self.msg_time = msg_time


class FakeDischargeEvent:
    def __init__(self, mrn="1", msg_time="20240102000000"):
        self.mrn = mrn
        self.msg_time = msg_time


class FakeCreatinineEvent:
    def __init__(self, mrn="1", test_time="20240103000000", value=150.0):
        self.mrn = mrn
        self.test_time = test_time
        self.value = value


class FakeHL7ParseError(Exception):
    pass


class FakeHistory:
    def __init__(self, history_map=None):
        self.history_map = history_map or {}

    def get_history_from_db(self, mrn: str):
        return self.history_map.get(mrn, [])


class FakeInference:
    def __init__(self, will_page: bool):
        self.will_page = will_page
        self.calls = []

    def predict_aki(self, *, mrn, test_time_hl7, history_ord_vals, demographics):
        self.calls.append((mrn, test_time_hl7, history_ord_vals, demographics))
        return self.will_page


class FakePager:
    def __init__(self, ok=True, info="OK"):
        self.ok = ok
        self.info = info
        self.calls = []

    def send_page(self, mrn, test_time):
        self.calls.append((mrn, test_time))
        return self.ok, self.info


class FakeConnCtx:
    """context manager returned by db._get_conn() used in Router"""
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.db.executed.append((sql.strip(), params))

    def commit(self):
        self.db.commits += 1


class FakeDB:
    def __init__(self):
        self.updated_patients = []
        self.insert_labs = []   # (mrn, test_time, value) calls
        self.history = {}
        self.patient_state = {} # mrn -> dict
        self.alert_updates = [] # (mrn, test_time, status)
        self.executed = []      # SQL executed via _get_conn()
        self.commits = 0

        # Control flags
        self.insert_lab_returns = True      # can flip to simulate duplicates

    def update_patient(self, mrn, **kwargs):
        self.updated_patients.append((mrn, kwargs))

    def insert_lab(self, mrn, test_time, value):
        self.insert_labs.append((mrn, test_time, value))
        return self.insert_lab_returns

    def get_patient_state(self, mrn):
        return self.patient_state.get(mrn)

    def update_alert(self, mrn, test_time, status):
        self.alert_updates.append((mrn, test_time, status))

    def _get_conn(self):
        return FakeConnCtx(self)



@pytest.fixture
def make_router(monkeypatch):
    def _make(*, inference_will_page=False, dry_run_pager=False, pager=None, patient_state=None, history_map=None):
        db = FakeDB()
        if patient_state is not None:
            db.patient_state.update(patient_state)

        history = FakeHistory(history_map=history_map)
        inference = FakeInference(will_page=inference_will_page)

        r = Router(history=history, inference=inference, pager=pager, dry_run_pager=dry_run_pager, db=db)
        return r, db, inference
    return _make


def patch_hl7(monkeypatch, event_obj):
    # Patch Router's hl7 module so isinstance checks work against fake classes
    import aki_service.router as router_mod

    class FakeHL7Module:
        HL7ParseError = FakeHL7ParseError
        UnknownEvent = FakeUnknownEvent
        AdmitEvent = FakeAdmitEvent
        DischargeEvent = FakeDischargeEvent
        CreatinineEvent = FakeCreatinineEvent

        @staticmethod
        def parse_event(_bytes):
            return event_obj

    monkeypatch.setattr(router_mod, "hl7", FakeHL7Module)


def test_handle_message_parse_error_returns_AE(monkeypatch, make_router):
    # Tests that HL7 parse errors return AE
    r, db, inf = make_router()

    import aki_service.router as router_mod

    class FakeHL7Module:
        HL7ParseError = FakeHL7ParseError

        @staticmethod
        def parse_event(_bytes):
            raise FakeHL7ParseError("bad")

        UnknownEvent = FakeUnknownEvent
        AdmitEvent = FakeAdmitEvent
        DischargeEvent = FakeDischargeEvent
        CreatinineEvent = FakeCreatinineEvent

    monkeypatch.setattr(router_mod, "hl7", FakeHL7Module)

    assert r.handle_message(b"...") == "AE"


def test_handle_message_unknown_event_returns_AA(monkeypatch, make_router):
    # Tests that unknown message types are accepted (AA) and ignored
    r, db, inf = make_router()
    patch_hl7(monkeypatch, FakeUnknownEvent("FOO^BAR"))

    assert r.handle_message(b"...") == "AA"
    assert db.updated_patients == []
    assert db.insert_labs == []


def test_handle_message_admit_updates_patient(monkeypatch, make_router):
    # Tests that AdmitEvent updates DB patient admitted=True and returns AA
    r, db, inf = make_router()
    ev = FakeAdmitEvent(mrn="7", dob="19900101", sex="F", msg_time="20240101010101")
    patch_hl7(monkeypatch, ev)

    assert r.handle_message(b"...") == "AA"
    assert db.updated_patients == [("7", {"is_admitted": True, "dob": "19900101", "sex": "F", "admit_time": "20240101010101"})]


def test_handle_message_discharge_updates_patient(monkeypatch, make_router):
    # Tests that DischargeEvent updates DB patient admitted=False and returns AA
    r, db, inf = make_router()
    ev = FakeDischargeEvent(mrn="7", msg_time="20240102020202")
    patch_hl7(monkeypatch, ev)

    assert r.handle_message(b"...") == "AA"
    assert db.updated_patients == [("7", {"is_admitted": False, "discharge_time": "20240102020202"})]


def test_creatinine_duplicate_lab_skips_inference(monkeypatch, make_router):
    # Tests that duplicate labs (insert_lab False) skip inference and still return AA
    r, db, inf = make_router()
    db.insert_lab_returns = False

    patch_hl7(monkeypatch, FakeCreatinineEvent(mrn="1", test_time="20240103000000", value=123.0))
    assert r.handle_message(b"...") == "AA"

    assert len(db.insert_labs) == 1
    assert inf.calls == []  # inference not called


def test_creatinine_pages_in_dry_run(monkeypatch, make_router):
    # Tests that when inference says page and dry_run_pager=True, alert is marked sent without pager call
    patient_state = {"1": {"dob": "20000101", "sex": "M", "is_admitted": 1}}
    history_map = {"1": [(100, 1.0), (101, 2.0)]}

    r, db, inf = make_router(
        inference_will_page=True,
        dry_run_pager=True,
        pager=None,
        patient_state=patient_state,
        history_map=history_map,
    )
    patch_hl7(monkeypatch, FakeCreatinineEvent(mrn="1", test_time="20240103000000", value=150.0))

    assert r.handle_message(b"...") == "AA"

    # pending then sent
    assert ("1", "20240103000000", "pending") in db.alert_updates
    assert ("1", "20240103000000", "sent") in db.alert_updates

    # inference called once with Demographics
    assert len(inf.calls) == 1
    mrn, t, hist, demo = inf.calls[0]
    assert mrn == "1"
    assert t == "20240103000000"
    assert hist == history_map["1"]
    assert isinstance(demo, Demographics)


def test_creatinine_suppressed_if_discharged(monkeypatch, make_router):
    # Tests that discharged patients suppress paging even if inference says page
    patient_state = {"1": {"dob": "20000101", "sex": "M", "is_admitted": 0}}
    r, db, inf = make_router(inference_will_page=True, dry_run_pager=True, patient_state=patient_state)
    patch_hl7(monkeypatch, FakeCreatinineEvent(mrn="1", test_time="20240103000000", value=150.0))

    assert r.handle_message(b"...") == "AA"
    # No alerts created because suppressed
    assert db.alert_updates == []


def test_creatinine_pages_via_pager_updates_status(monkeypatch, make_router):
    # Tests that pager send updates alerts status to sent/failed and increments attempt count via SQL
    patient_state = {"1": {"dob": "20000101", "sex": "M", "is_admitted": 1}}
    pager = FakePager(ok=True, info="OK")
    r, db, inf = make_router(inference_will_page=True, dry_run_pager=False, pager=pager, patient_state=patient_state)
    patch_hl7(monkeypatch, FakeCreatinineEvent(mrn="1", test_time="20240103000000", value=150.0))

    assert r.handle_message(b"...") == "AA"

    # pending alert before send
    assert ("1", "20240103000000", "pending") in db.alert_updates
    # pager called
    assert pager.calls == [("1", "20240103000000")]
    # SQL update executed once and commit called once
    assert db.commits == 1
    assert len(db.executed) == 1
    sql, params = db.executed[0]
    assert "UPDATE alerts" in sql
    assert params[0] == "sent"  # status
    assert params[2:] == ("1", "20240103000000")
