from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from . import hl7
from .history import HistoryStore
from .inference import Demographics, InferenceService
from .pager import PagerClient
from .model_compat import date_to_ordinal_from_any
from datetime import datetime
import time

log = logging.getLogger("aki_service")


@dataclass
class PatientState:
    admitted: Optional[bool] = None
    demographics: Optional[Demographics] = None


class Router:
    """Layer B: parse HL7 messages and route to handlers."""

    def __init__(
        self,
        *,
        db,
        history: HistoryStore,
        inference: InferenceService,
        pager: Optional[PagerClient] = None,
        dry_run_pager: bool = False,
    ) -> None:
        self.history = history
        self.inference = inference
        self.pager = pager
        self.dry_run_pager = dry_run_pager
        self.db = db

        self.patients: Dict[str, PatientState] = {}

    def _get_patient(self, mrn: str) -> PatientState:
        return self.patients.setdefault(mrn, PatientState())

    def handle_message(self, hl7_bytes: bytes) -> str:
        """Return ACK code: AA (accept) or AE (error)."""

        start_time = time.perf_counter()
        result = "AA"

        try:
            ev = hl7.parse_event(hl7_bytes)
        except hl7.HL7ParseError as e:
            log.warning("parse_error: %s", e)
            result = "AE"
            ev = None 

        if ev:
            if isinstance(ev, hl7.UnknownEvent):
                log.info("unknown_msg: %s", ev.msg_type)
                result = "AA"

            elif isinstance(ev, hl7.AdmitEvent):
                self.db.update_patient(
                    ev.mrn, is_admitted=True, dob=ev.dob, sex=ev.sex, admit_time=ev.msg_time
                )
                result = "AA"

            elif isinstance(ev, hl7.DischargeEvent):
                self.db.update_patient(
                    ev.mrn, is_admitted=False, discharge_time=ev.msg_time
                )
                result = "AA"

            elif isinstance(ev, hl7.CreatinineEvent):
                is_new = self.db.insert_lab(ev.mrn, ev.test_time, ev.value)
                if not is_new:
                    log.debug("Duplicate lab for MRN %s; skipping.", ev.mrn)
                    result = "AA"
                else:
                    event_ord = date_to_ordinal_from_any(ev.test_time)
                    full_history = self.history.get_history_from_db(ev.mrn)
                    history_ord_vals = [(d, v) for (d, v) in full_history if d is not None and d <= event_ord]
                    ps_dict = self.db.get_patient_state(ev.mrn)
                    
                    demo = None
                    if ps_dict and ps_dict['dob']:
                        demo = Demographics(dob_yyyymmdd=ps_dict['dob'], sex=ps_dict['sex'])

                    if not self.inference.is_ready():
                        result = "AA"
                    else:
                        try:
                            should_page = self.inference.predict_aki(
                                mrn=ev.mrn, test_time_hl7=ev.test_time,
                                history_ord_vals=history_ord_vals, demographics=demo
                            )
                        except Exception:
                            log.exception("Inference crashed for MRN %s", ev.mrn)
                            should_page = False
                            return "AE"

                        if ps_dict and ps_dict['is_admitted'] == 0:
                            should_page = False

                        if should_page:
                            # Check if already paged
                            conn = self.db._get_conn()
                            already = conn.execute("SELECT 1 FROM alerts WHERE mrn=? AND status='sent' LIMIT 1", (ev.mrn,)).fetchone()
                            if already:
                                should_page = False

                        if should_page:
                            # Paging execution
                            current_ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                            try:
                                conn.execute("INSERT INTO alerts (mrn, test_time, status, attempt_count) VALUES (?, ?, 'pending', 0)", (ev.mrn, ev.test_time))
                                conn.commit()
                                
                                if self.pager and not self.dry_run_pager:
                                    ok, info = self.pager.send_page(ev.mrn, ev.test_time)
                                    status = "sent" if ok else "failed"
                                    conn.execute("UPDATE alerts SET status=?, attempt_count=1, last_attempt_time=? WHERE mrn=? AND test_time=?", (status, current_ts, ev.mrn, ev.test_time))
                                    conn.commit()
                                else:
                                    conn.execute("UPDATE alerts SET status='sent' WHERE mrn=? AND test_time=?", (ev.mrn, ev.test_time))
                                    conn.commit()
                            except sqlite3.IntegrityError:
                                pass
                        result = "AA"

        # 3. Calculate latency and log before the single return
        latency_ms = (time.perf_counter() - start_time) * 1000
        log.info("Message processed in %.2f ms", latency_ms)
        
        return result
    
    @staticmethod
    def parse_hl7_timestamp(ts: str) -> datetime:
        s = str(ts).strip()
        if not s.isdigit():
            raise ValueError(f"Not an HL7 timestamp: {ts}")

        if len(s) == 8:
            return datetime.strptime(s, "%Y%m%d")
        if len(s) == 12:
            return datetime.strptime(s, "%Y%m%d%H%M")
        if len(s) == 14:
            return datetime.strptime(s, "%Y%m%d%H%M%S")
        raise ValueError(f"Unsupported HL7 timestamp length: {ts}")

    @staticmethod
    def normalize_to_iso(ts: str) -> str:
        s = str(ts).strip()

        # ISO with space, seconds optional
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        # ISO with T / timezone
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        # HL7 fallback
        dt = Router.parse_hl7_timestamp(s)
        return dt.strftime("%Y-%m-%d %H:%M:%S")