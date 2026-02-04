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
        print("ROUTER CODE VERSION: TRY/EXCEPT ACTIVE", flush=True)
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
                log.info("Duplicate lab received for MRN %s at %s; skipping inference.", ev.mrn, ev.test_time)
                return "AA"
            
            # 2. Fetch history and state from DB
            event_ord = date_to_ordinal_from_any(ev.test_time)

            full_history = self.history.get_history_from_db(ev.mrn)
            history_ord_vals = [
                (d, v) for (d, v) in full_history
                if d is not None and d <= event_ord
            ]

            ps_dict = self.db.get_patient_state(ev.mrn)
            
            # Map DB dict to Demographics object for the model
            demo = None
            if ps_dict and ps_dict['dob']:
                demo = Demographics(dob_yyyymmdd=ps_dict['dob'], sex=ps_dict['sex'])

            # 3. Check if model is ready
            if not self.inference.is_ready():
                log.warning("Inference model not ready; skipping prediction for MRN %s", ev.mrn)
                return "AA"

            # 4. Predict
            # 4. Predict
            log.debug("Calling predict_aki for MRN %s at %s", ev.mrn, ev.test_time)
            try:
                should_page = self.inference.predict_aki(
                    mrn=ev.mrn,
                    test_time_hl7=ev.test_time,
                    history_ord_vals=history_ord_vals,
                    demographics=demo,
                )
            except Exception:
                log.exception(
                    "Inference crashed for MRN %s at %s", ev.mrn, ev.test_time
                )
                return "AE"


            # 5. Suppress if not admitted or discharged
            if ps_dict and ps_dict['is_admitted'] == 0:
                log.info("Suppressed alert for MRN %s (Discharged)", ev.mrn)
                should_page = False

            # Remove this entire section:
            # NEW: suppress repeat AKI alerts for same MRN
            if should_page:
                with self.db._get_conn() as conn:
                    already_paged = conn.execute(
                        "SELECT 1 FROM alerts WHERE mrn = ? LIMIT 1",
                        (ev.mrn,),
                    ).fetchone()
                if already_paged is not None:
                    log.info(
                        "AKI already alerted for MRN %s previously; suppressing repeat page",
                        ev.mrn,
                    )
                    should_page = False


            # 6. CRITICAL FIX: Check for existing alert for THIS SPECIFIC EVENT (MRN + test_time)
            # This allows multiple AKI events for the same patient at different times
            if should_page:
                with self.db._get_conn() as conn:
                    existing_alert = conn.execute(
                        "SELECT 1 FROM alerts WHERE mrn = ? AND test_time = ? LIMIT 1",
                        (ev.mrn, ev.test_time),
                    ).fetchone()
                    
                    if existing_alert is not None:
                        log.info("Alert already exists for MRN %s at %s; skipping page", ev.mrn, ev.test_time)
                        should_page = False

            # 7. Try to atomically claim this alert to prevent race conditions
            if should_page:
                import time
                current_ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                
                # Try to insert the alert record atomically
                # The PRIMARY KEY (mrn, test_time) ensures only one alert per event
                with self.db._get_conn() as conn:
                    try:
                        conn.execute("""
                            INSERT INTO alerts (mrn, test_time, status, attempt_count) 
                            VALUES (?, ?, 'pending', 0)
                        """, (ev.mrn, ev.test_time))
                        conn.commit()
                    except sqlite3.IntegrityError:
                        # This exact event was already claimed (defensive check)
                        log.info("Alert already claimed for MRN %s at %s; skipping page", ev.mrn, ev.test_time)
                        should_page = False
                
                # 8. Now actually send the page
                if should_page:
                    if self.pager and not self.dry_run_pager:
                        ok, info = self.pager.send_page(ev.mrn, ev.test_time)
                        status = "sent" if ok else "failed"
                        
                        # Update the status in SQL
                        with self.db._get_conn() as conn:
                            conn.execute("""
                                UPDATE alerts 
                                SET status = ?, attempt_count = 1, last_attempt_time = ?
                                WHERE mrn = ? AND test_time = ?
                            """, (status, current_ts, ev.mrn, ev.test_time))
                            conn.commit()
                        
                        log.info("PAGED MRN %s at %s: %s", ev.mrn, ev.test_time, info)
                    else:
                        # In dry-run mode, mark as sent
                        with self.db._get_conn() as conn:
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
