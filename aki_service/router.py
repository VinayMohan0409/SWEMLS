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
        db
    ) -> None:
        self.history = history
        self.inference = inference
        self.pager = pager
        self.dry_run_pager = dry_run_pager
        #db addition
        self.db = db

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
            self.db.update_patient(
                ev.mrn, 
                is_admitted=True, 
                dob=ev.dob, 
                sex=ev.sex, 
                admit_time=ev.msg_time
            )
            return "AA"

        if isinstance(ev, hl7.DischargeEvent):
            self.db.update_patient(
                ev.mrn, 
                is_admitted=False, 
                discharge_time=ev.msg_time
            )
            return "AA"

        if isinstance(ev, hl7.CreatinineEvent):

            is_new = self.db.insert_lab(ev.mrn, ev.test_time, ev.value)
            if not is_new:
                log.info("Duplicate lab received for MRN %s at %s; skipping inference.", ev.mrn, ev.test_time)
                return "AA"
            
            # 2. Fetch history and state from DB
            history_ord_vals = self.history.get_history_from_db(ev.mrn)
            ps_dict = self.db.get_patient_state(ev.mrn)
            
            # Map DB dict to Demographics object for the model
            demo = None
            if ps_dict and ps_dict['dob']:
                demo = Demographics(dob_yyyymmdd=ps_dict['dob'], sex=ps_dict['sex'])

            # 3. Predict
            should_page = self.inference.predict_aki(
                mrn=ev.mrn,
                test_time_hl7=ev.test_time,
                history_ord_vals=history_ord_vals,
                demographics=demo,
            )

            # 4. Suppress if discharged
            if ps_dict and ps_dict['is_admitted'] == 0:
                log.info("Suppressed alert for MRN %s (Discharged)", ev.mrn)
                should_page = False

            if should_page:
                import time
                current_ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                
                # Initialize alert with status 'pending' before trying to send
                self.db.update_alert(ev.mrn, ev.test_time, "pending")
                
                if self.pager and not self.dry_run_pager:
                    ok, info = self.pager.send_page(ev.mrn, ev.test_time)
                    status = "sent" if ok else "failed"
                    
                    # Log the attempt and update the status in SQL
                    with self.db._get_conn() as conn:
                        conn.execute("""
                            UPDATE alerts 
                            SET status = ?, attempt_count = attempt_count + 1, last_attempt_time = ?
                            WHERE mrn = ? AND test_time = ?
                        """, (status, current_ts, ev.mrn, ev.test_time))
                        conn.commit()
                    
                    log.info("PAGED MRN %s: %s (Attempt 1)", ev.mrn, info)
                else:
                    # In dry-run mode, we mark as sent but don't increment attempt counts the same way
                    self.db.update_alert(ev.mrn, ev.test_time, "sent")
                    log.info("DRY-RUN PAGE for MRN %s (Test Time: %s)", ev.mrn, ev.test_time)
            
            return "AA"
        return "AA"
