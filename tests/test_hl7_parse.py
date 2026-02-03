import pytest
from aki_service.hl7 import parse_event, AdmitEvent, DischargeEvent, CreatinineEvent, HL7ParseError


def test_parse_admit():
    hl7 = (
        b"MSH|^~\\&|SIM|HOSP|||20240120163000||ADT^A01|||2.5\r"
        b"PID|1||478237423||ELIZABETH HOLMES||19840203|F\r"
    )
    ev = parse_event(hl7)
    assert isinstance(ev, AdmitEvent)
    assert ev.mrn == "478237423"
    assert ev.dob == "19840203"
    assert ev.sex == "F"


def test_parse_discharge():
    hl7 = (
        b"MSH|^~\\&|SIM|HOSP|||20240120163000||ADT^A03|||2.5\r"
        b"PID|1||478237423\r"
    )
    ev = parse_event(hl7)
    assert isinstance(ev, DischargeEvent)
    assert ev.mrn == "478237423"


def test_parse_creatinine():
    hl7 = (
        b"MSH|^~\\&|SIM|HOSP|||20240120163000||ORU^R01|||2.5\r"
        b"PID|1||478237423\r"
        b"OBR|1||||||20240120224300\r"
        b"OBX|1|SN|CREATININE||103.4\r"
    )
    ev = parse_event(hl7)
    assert isinstance(ev, CreatinineEvent)
    assert ev.mrn == "478237423"
    assert ev.test_time == "20240120224300"
    assert abs(ev.value - 103.4) < 1e-6


def test_parse_rejects_missing_msh():
    with pytest.raises(HL7ParseError):
        parse_event(b"PID|1||123\r")
