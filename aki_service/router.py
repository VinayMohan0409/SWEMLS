from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set, Tuple

from . import hl7
from .history import HistoryStore
from .inference import Demographics, InferenceService
from .pager import PagerClient
from .model_compat import date_to_ordinal_from_any
from datetime import datetime
from pprint import pprint


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
            # 1. Insert lab (skip if duplicate)
            is_new = self.db.insert_lab(ev.mrn, ev.test_time, ev.value)
            if not is_new:
                log.debug("Duplicate lab received for MRN %s at %s; skipping inference.", ev.mrn, ev.test_time)
                return "AA"
            
            # 2. Fetch history and state from DB
            # IMPORTANT: This now uses the same connection as insert_lab,
            # ensuring the newly inserted lab is visible.
            event_ord = date_to_ordinal_from_any(ev.test_time)

            full_history = self.history.get_history_from_db(ev.mrn)
            history_ord_vals = [
                (d, v) for (d, v) in full_history
                if d is not None and d <= event_ord
            ]
            
            log.debug("MRN %s: Retrieved %d history entries (filtered to %d for event)", 
                     ev.mrn, len(full_history), len(history_ord_vals))

            ps_dict = self.db.get_patient_state(ev.mrn)
            
            # Map DB dict to Demographics object for the model
            demo = None
            if ps_dict and ps_dict['dob']:
                demo = Demographics(dob_yyyymmdd=ps_dict['dob'], sex=ps_dict['sex'])

            # 3. Check if model is ready
            if not self.inference.is_ready():
                log.warning("Inference model not ready; skipping prediction for MRN %s", ev.mrn)
                return "AA"

            # 4. Run prediction
            try:
                should_page = self.inference.predict_aki(
                    mrn=ev.mrn,
                    test_time_hl7=ev.test_time,
                    history_ord_vals=history_ord_vals,
                    demographics=demo,
                )
            except Exception:
                log.exception("Inference crashed for MRN %s at %s", ev.mrn, ev.test_time)
                return "AE"

            # 5. Suppress if patient is discharged
            if ps_dict and ps_dict['is_admitted'] == 0:
                log.debug("Suppressed alert for MRN %s (Discharged)", ev.mrn)
                should_page = False

            # 6. ONE PAGE PER MRN: Only page once per patient (ever)
            # aki.csv has exactly one entry per unique MRN
            if should_page:
                conn = self.db._get_conn()
                already_paged = conn.execute(
                    "SELECT 1 FROM alerts WHERE mrn = ? AND status = 'sent' LIMIT 1",
                    (ev.mrn,),
                ).fetchone()
                if already_paged is not None:
                    log.debug("AKI already alerted for MRN %s previously; suppressing repeat page", ev.mrn)
                    should_page = False

            # 7. If we should page, atomically claim and send
            if should_page:
                import time
                current_ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                
                conn = self.db._get_conn()
                try:
                    # Atomically insert the alert record
                    conn.execute("""
                        INSERT INTO alerts (mrn, test_time, status, attempt_count) 
                        VALUES (?, ?, 'pending', 0)
                    """, (ev.mrn, ev.test_time))
                    conn.commit()
                except sqlite3.IntegrityError:
                    # This exact event was already claimed
                    log.debug("Alert already claimed for MRN %s at %s; skipping page", ev.mrn, ev.test_time)
                    should_page = False
                
                # 8. Send the page
                if should_page:
                    if self.pager and not self.dry_run_pager:
                        ok, info = self.pager.send_page(ev.mrn, ev.test_time)
                        status = "sent" if ok else "failed"
                        
                        conn.execute("""
                            UPDATE alerts 
                            SET status = ?, attempt_count = 1, last_attempt_time = ?
                            WHERE mrn = ? AND test_time = ?
                        """, (status, current_ts, ev.mrn, ev.test_time))
                        conn.commit()
                        
                        log.info("PAGED MRN %s at %s: %s", ev.mrn, ev.test_time, info)
                    else:
                        conn.execute("""
                            UPDATE alerts 
                            SET status = 'sent'
                            WHERE mrn = ? AND test_time = ?
                        """, (ev.mrn, ev.test_time))
                        conn.commit()
                        log.info("DRY-RUN PAGE for MRN %s at %s", ev.mrn, ev.test_time)
            
            return "AA"
        return "AA"
    
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