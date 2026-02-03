from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

from . import hl7
from .history import HistoryStore
from .inference import Demographics, InferenceService
from .pager import PagerClient
from .model_compat import date_to_ordinal_from_any


log = logging.getLogger("aki_service")


@dataclass
class PatientState:
    admitted: Optional[bool] = None
    demographics: Optional[Demographics] = None


class Router:
    """Layer B: parse HL7 messages and route to handlers.

    For now, this includes an in-memory state + inference + (optional) pager calls,
    so you can test end-to-end. You can later replace the state/pager parts with SQLite.
    """

    def __init__(
        self,
        *,
        history: HistoryStore,
        inference: InferenceService,
        pager: Optional[PagerClient] = None,
        dry_run_pager: bool = False,
    ) -> None:
        self.history = history
        self.inference = inference
        self.pager = pager
        self.dry_run_pager = dry_run_pager

        self.patients: Dict[str, PatientState] = {}
        self.paged: Set[Tuple[str, str]] = set()  # (mrn, test_time_hl7)

    def _get_patient(self, mrn: str) -> PatientState:
        return self.patients.setdefault(mrn, PatientState())

    def handle_message(self, hl7_bytes: bytes) -> str:
        """Return ACK code: AA (accept) or AE (error)."""
        try:
            ev = hl7.parse_event(hl7_bytes)
        except hl7.HL7ParseError as e:
            log.warning("parse_error: %s", e)
            return "AE"

        # Unknown message type: accept (avoid resend loops), ignore content.
        if isinstance(ev, hl7.UnknownEvent):
            log.info("unknown_msg: %s", ev.msg_type)
            return "AA"

        if isinstance(ev, hl7.AdmitEvent):
            ps = self._get_patient(ev.mrn)
            # DOB in HL7 admits is PID.7 format YYYYMMDD
            ps.demographics = Demographics(dob_yyyymmdd=ev.dob, sex=ev.sex)
            ps.admitted = True
            return "AA"

        if isinstance(ev, hl7.DischargeEvent):
            ps = self._get_patient(ev.mrn)
            ps.admitted = False
            return "AA"

        if isinstance(ev, hl7.CreatinineEvent):
            # Update history
            self.history.add_creatinine(ev.mrn, ev.test_time, ev.value)
            ph = self.history.get(ev.mrn)
            ps = self._get_patient(ev.mrn)

            # Build history list INCLUDING this test (already appended)
            history_ord_vals = ph.creatinine

            should_page = self.inference.predict_aki(
                mrn=ev.mrn,
                test_time_hl7=ev.test_time,
                history_ord_vals=history_ord_vals,
                demographics=ps.demographics,
            )

            # suppress if definitely discharged
            if ps.admitted is False:
                should_page = False

            if should_page:
                key = (ev.mrn, ev.test_time)
                if key not in self.paged:
                    self.paged.add(key)
                    if self.pager and not self.dry_run_pager:
                        ok, info = self.pager.send_page(ev.mrn, ev.test_time)
                        if ok:
                            log.info("paged %s,%s", ev.mrn, ev.test_time)
                        else:
                            log.warning("page_failed %s,%s: %s", ev.mrn, ev.test_time, info)
                    else:
                        log.info("dry_paged %s,%s", ev.mrn, ev.test_time)
            return "AA"

        # default accept
        return "AA"
