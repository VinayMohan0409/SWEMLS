from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from typing import Tuple

from .history import HistoryStore
from .inference import InferenceService
from .mllp import MLLPClient
from .pager import PagerClient
from .router import Router
from .db import Database


def parse_hostport(s: str) -> Tuple[str, int]:
    if ":" not in s:
        raise ValueError(f"expected host:port, got {s!r}")
    host, port_s = s.rsplit(":", 1)
    return host, int(port_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="SWEMLS Task 3 AKI inference service (initial commit).")
    ap.add_argument("--mllp", default=os.environ.get("MLLP_ADDRESS", "localhost:8440"), help="MLLP server host:port")
    ap.add_argument("--pager", default=os.environ.get("PAGER_ADDRESS", "localhost:8441"), help="Pager server host:port")
    ap.add_argument("--history", default="/data/history.csv", help="Path to history.csv (wide format)")
    ap.add_argument("--model-bundle", default=os.environ.get("AKI_MODEL_BUNDLE", "/app/model/aki_model.pt"), help="Path to exported model bundle (.pt)")
    ap.add_argument("--device", default=os.environ.get("AKI_DEVICE", "cpu"), help="torch device (cpu)")
    ap.add_argument("--dry-run-pager", action="store_true", help="Do not send POSTs to pager (log only)")
    ap.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    ap.add_argument("--db", default=os.environ.get("DB_PATH", "aki_service.db"), help="Path to SQLite database")
    args = ap.parse_args()

    logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO), 
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    log = logging.getLogger("aki_service")

    # 1. Initialize the Database
    db = Database(args.db)

    stop_event = threading.Event()

    def _request_shutdown(signum: int, _frame) -> None:
        # Note: keep this handler minimal and signal-safe.
        log.warning("received signal %s; shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    # 2. Setup Pager Client early for recovery use
    pager = None
    if args.pager:
        host, port = parse_hostport(args.pager)
        pager = PagerClient(host=host, port=port)

    # 3. Handle History (Imports CSV to SQL if necessary) 
    history = HistoryStore(db)
    n = history.load_history_csv(args.history)
    log.info("history loaded into database: rows processed=%d", n)

    # 4. Startup Recovery: Resend alerts marked as 'failed' 
    if pager and not args.dry_run_pager:
        recovered = db.retry_failed_alerts(pager)
        if recovered > 0:
            log.info("recovered %d failed alerts on startup", recovered)

    inf = InferenceService(bundle_path=args.model_bundle, device=args.device)
    if inf.is_ready():
        log.info("model bundle loaded: %s", args.model_bundle)
    else:
        log.warning("model bundle not loaded (missing or invalid): %s. Running with no-op predictions.", args.model_bundle)

    # 6. Setup Router with DB for persistent state
    router = Router(
        db=db, 
        history=history, 
        inference=inf, 
        pager=pager, 
        dry_run_pager=args.dry_run_pager
    )

    host, port = parse_hostport(args.mllp)
    client = MLLPClient((host, port))
    log.info("connecting to MLLP at %s:%d", host, port)
    client.run(router.handle_message, stop_event=stop_event)

    log.info("shutdown complete")


if __name__ == "__main__":
    main()
