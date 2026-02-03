from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

class HL7ParseError(Exception):
    pass


def _split_segments(hl7: bytes) -> List[bytes]:
    # HL7 requires CR separators. Some sources might include trailing CR.
    segs = [s for s in hl7.split(b"\r") if s]
    if not segs:
        raise HL7ParseError("empty message")
    return segs


def _fields(seg: bytes) -> List[bytes]:
    return seg.split(b"|")


def _msh_get(msh_fields: List[bytes], field_num: int) -> Optional[str]:
    # MSH.1 is field separator (implicit). MSH.2 is encoding chars => msh_fields[1].
    # So for field_num >= 2, index = field_num - 1.
    if field_num == 1:
        return "|"
    idx = field_num - 1
    if idx < 0 or idx >= len(msh_fields):
        return None
    v = msh_fields[idx].decode("ascii", errors="ignore").strip()
    return v or None


def _get(seg_fields: List[bytes], field_num: int) -> Optional[str]:
    # For non-MSH segments, field 1 is at index 1.
    idx = field_num
    if idx < 0 or idx >= len(seg_fields):
        return None
    v = seg_fields[idx].decode("ascii", errors="ignore").strip()
    return v or None


@dataclass(frozen=True)
class AdmitEvent:
    mrn: str
    msg_time: str  # MSH.7
    dob: str       # PID.7
    sex: str       # PID.8


@dataclass(frozen=True)
class DischargeEvent:
    mrn: str
    msg_time: str  # MSH.7


@dataclass(frozen=True)
class CreatinineEvent:
    mrn: str
    msg_time: str     # MSH.7
    test_time: str    # OBR.7
    value: float      # OBX.5


@dataclass(frozen=True)
class UnknownEvent:
    msg_type: str
    msg_time: Optional[str]


def parse_message_type(hl7: bytes) -> Tuple[str, str]:
    segs = _split_segments(hl7)
    msh = _fields(segs[0])
    if not msh or msh[0] != b"MSH":
        raise HL7ParseError("missing MSH segment")
    msg_time = _msh_get(msh, 7)
    msg_type = _msh_get(msh, 9)
    if not msg_type:
        raise HL7ParseError("missing MSH.9 message type")
    if not msg_time:
        # simulator guarantees, but be defensive
        msg_time = ""
    return msg_type, msg_time


def parse_event(hl7: bytes):
    msg_type, msg_time = parse_message_type(hl7)
    segs = _split_segments(hl7)
    by_type: Dict[bytes, List[List[bytes]]] = {}
    for seg in segs:
        f = _fields(seg)
        by_type.setdefault(f[0], []).append(f)

    def require_one(seg_name: bytes) -> List[bytes]:
        arr = by_type.get(seg_name)
        if not arr or len(arr) < 1:
            raise HL7ParseError(f"missing {seg_name.decode()} segment")
        return arr[0]

    if msg_type == "ADT^A01":
        pid = require_one(b"PID")
        mrn = _get(pid, 3)
        dob = _get(pid, 7)
        sex = _get(pid, 8)
        if not mrn or not dob or not sex:
            raise HL7ParseError("ADT^A01 missing required PID fields")
        return AdmitEvent(mrn=mrn, msg_time=msg_time, dob=dob, sex=sex)

    if msg_type == "ADT^A03":
        pid = require_one(b"PID")
        mrn = _get(pid, 3)
        if not mrn:
            raise HL7ParseError("ADT^A03 missing PID.3 MRN")
        return DischargeEvent(mrn=mrn, msg_time=msg_time)

    if msg_type == "ORU^R01":
        pid = require_one(b"PID")
        mrn = _get(pid, 3)
        obr = require_one(b"OBR")
        test_time = _get(obr, 7)
        if not mrn or not test_time:
            raise HL7ParseError("ORU^R01 missing PID.3 or OBR.7")

        # Find creatinine OBX (OBX.3 == CREATININE)
        obxs = by_type.get(b"OBX", [])
        for obx in obxs:
            test_type = _get(obx, 3)
            if test_type != "CREATININE":
                continue
            val_s = _get(obx, 5)
            if not val_s:
                raise HL7ParseError("CREATININE OBX missing OBX.5")
            try:
                value = float(val_s)
            except ValueError as e:
                raise HL7ParseError(f"OBX.5 not a float: {val_s}") from e
            return CreatinineEvent(mrn=mrn, msg_time=msg_time, test_time=test_time, value=value)

        # No creatinine in this ORU: treat as unknown-but-ackable.
        return UnknownEvent(msg_type=msg_type, msg_time=msg_time)

    return UnknownEvent(msg_type=msg_type, msg_time=msg_time)
