#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Tuple


# Simulator prints pager receipts like:
#   pager: paging for MRN 185983563 at 2024-12-21 09:44:00
PAGER_RE = re.compile(
    r"pager:\s+paging for MRN\s+(?P<mrn>\d+)\s+at\s+(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
)


def normalize_timestamp(ts: str) -> str:
    """
    Normalize timestamps to YYYY-MM-DD HH:MM:SS format.
    Handles both ISO (2025-03-30 22:58:00) and HL7 (20250330225800) formats.
    """
    s = str(ts).strip()
    
    # Already in ISO format with seconds
    if len(s) == 19 and s[4] == '-' and s[7] == '-':
        return s
    
    # ISO format without seconds  
    if len(s) == 16 and s[4] == '-' and s[7] == '-':
        return s + ":00"
    
    # HL7 format (YYYYMMDDHHMMSS, YYYYMMDDHHMM, or YYYYMMDD)
    if s.isdigit():
        if len(s) == 8:  # YYYYMMDD
            dt = datetime.strptime(s, "%Y%m%d")
        elif len(s) == 12:  # YYYYMMDDHHMM
            dt = datetime.strptime(s, "%Y%m%d%H%M")
        elif len(s) == 14:  # YYYYMMDDHHMMSS
            dt = datetime.strptime(s, "%Y%m%d%H%M%S")
        else:
            raise ValueError(f"Unsupported HL7 timestamp length: {s}")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Try generic ISO parsing
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        raise ValueError(f"Cannot parse timestamp: {ts}")


def repo_root_from_here() -> Path:
    """
    This file is located at: <repo_root>/aki_service/integration_test.py
    So repo_root is one parent up from this file's directory.
    """
    return Path(__file__).resolve().parents[1]


def load_expected_pages(expected_csv: Path) -> Set[Tuple[str, str]]:
    """
    expected aki.csv format:
      mrn,date
      123765409,2024-03-31 22:09:00
      
    Returns: Set of (mrn_str, normalized_timestamp) tuples
    """
    out: Set[Tuple[str, str]] = set()
    with expected_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mrn_s = (row.get("mrn") or "").strip()
            date_s = (row.get("date") or "").strip()
            if not mrn_s or not date_s:
                continue
            
            # Normalize the timestamp from aki.csv
            normalized_date = normalize_timestamp(date_s)
            out.add((mrn_s, normalized_date))
    
    return out


def fbeta(tp: int, fp: int, fn: int, beta: float) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


@dataclass
class ProcCapture:
    proc: subprocess.Popen
    lines: list[str]
    lock: threading.Lock
    stop: threading.Event


def start_process(cmd: list[str], *, cwd: Path, env: dict) -> ProcCapture:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    cap = ProcCapture(proc=proc, lines=[], lock=threading.Lock(), stop=threading.Event())

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cap.stop.is_set():
                break
            with cap.lock:
                cap.lines.append(line.rstrip("\n"))

    threading.Thread(target=reader, daemon=True).start()
    return cap


def tail_lines(cap: ProcCapture, n: int) -> list[str]:
    with cap.lock:
        return cap.lines[-n:]


def http_post(url: str, body: bytes = b"") -> None:
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=2) as resp:
        resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end integration test: simulator -> service -> pager compare to aki.csv")
    ap.add_argument("--mllp-port", type=int, default=8440)
    ap.add_argument("--pager-port", type=int, default=8441)
    ap.add_argument("--timeout-s", type=int, default=1200)

    # Optional overrides (defaults are repo-root paths)
    ap.add_argument("--simulator", default=None)
    ap.add_argument("--messages", default=None)
    ap.add_argument("--expected", default=None)
    ap.add_argument("--history", default=None)
    ap.add_argument("--model-bundle", default=None)
    ap.add_argument("--service-module", default="aki_service")

    args = ap.parse_args()

    repo = repo_root_from_here()

    simulator = Path(args.simulator) if args.simulator else repo / "simulator.py"
    messages = Path(args.messages) if args.messages else repo / "messages.mllp"
    expected_csv = Path(args.expected) if args.expected else repo / "aki.csv"
    history_csv = Path(args.history) if args.history else repo / "history.csv"
    model_bundle = Path(args.model_bundle) if args.model_bundle else repo / "model" / "model.pt"

    # Basic existence checks
    for p, name in [
        (simulator, "simulator.py"),
        (messages, "messages.mllp"),
        (expected_csv, "aki.csv"),
        (history_csv, "history.csv"),
        (model_bundle, "model bundle (.pt)"),
    ]:
        if not p.exists():
            print(f"[FAIL] missing {name}: {p}", file=sys.stderr)
            return 2

    expected_pages = load_expected_pages(expected_csv)
    print(f"[info] Loaded {len(expected_pages)} expected events from {expected_csv.name}", flush=True)

    # reset SQLite DB so runs are repeatable
    inspect_db = repo / "aki_service" / "inspectdb.py"
    if inspect_db.exists():
        print(f"[debug] resetting DB via: {inspect_db} --reset", flush=True)
        subprocess.run([sys.executable, str(inspect_db), "--reset"], cwd=str(repo), check=True)
    else:
        print(f"[warn] inspect_db.py not found at {inspect_db}; DB not reset", flush=True)

    HOST = "127.0.0.1"
    # Environment: ensure repo is importable
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    env["MLLP_ADDRESS"] = f"{HOST}:{args.mllp_port}"
    env["PAGER_ADDRESS"] = f"{HOST}:{args.pager_port}"
    env["AKI_MODEL_BUNDLE"] = str(model_bundle)
    env["LOG_LEVEL"] = env.get("LOG_LEVEL", "INFO")

    # Start simulator
    sim_cmd = [
        sys.executable,
        "-u",
        str(simulator),
        "--messages",
        str(messages),
        "--mllp",
        str(args.mllp_port),
        "--pager",
        str(args.pager_port),
    ]

    print(f"[info] starting simulator: {' '.join(sim_cmd)}", flush=True)
    sim = start_process(sim_cmd, cwd=repo, env=env)
    print(f"[debug] simulator pid={sim.proc.pid}", flush=True)
    time.sleep(1.0)


    # Give simulator a moment to bind ports
    time.sleep(0.4)

    # Start service (REAL service entrypoint)
    svc_cmd = [
        sys.executable,
        "-u",
        "-m",
        args.service_module,
        "--mllp",
        f"{HOST}:{args.mllp_port}",
        "--pager",
        f"{HOST}:{args.pager_port}",
        "--history",
        str(history_csv),
        "--model-bundle",
        str(model_bundle),
    ]

    print(f"[info] starting service: {' '.join(svc_cmd)}", flush=True)
    svc = start_process(svc_cmd, cwd=repo, env=env)

    # Track (MRN, normalized_timestamp) tuples
    received_pages: Set[Tuple[str, str]] = set()
    mllp_ack_errors: list[str] = []
    last_sim_idx = 0


    deadline = time.time() + args.timeout_s
    sim_done = False
    sim_done_at: Optional[float] = None


    while time.time() < deadline:
        sim_rc = sim.proc.poll()
        svc_rc = svc.proc.poll()
        if sim_rc is not None or svc_rc is not None:
            print(f"[debug] sim exited rc={sim_rc}, svc exited rc={svc_rc}", flush=True)
            break

        # Parse simulator output incrementally (no tail rescans)
        with sim.lock:
            new_lines = sim.lines[last_sim_idx:]
            last_sim_idx = len(sim.lines)

        # Parse pager output for (MRN, timestamp) pairs
        for ln in new_lines:
            if "not acknowledged" in ln.lower():
                mllp_ack_errors.append(ln)

            m = PAGER_RE.search(ln)
            if m:
                mrn = m.group("mrn")  # Keep as string
                dt = m.group("dt")    # Already in YYYY-MM-DD HH:MM:SS format
                received_pages.add((mrn, dt))


        # Stop when simulator finishes replay
        if (not sim_done) and any("closing connection: end of messages" in ln for ln in new_lines):
            sim_done = True
            sim_done_at = time.time()
            print("[debug] simulator finished sending messages; waiting for service to page...", flush=True)

        # after simulator is done, keep waiting a bit for the service to emit pages
        if sim_done and sim_done_at is not None:
            # wait up to 10 seconds for late pages
            if time.time() - sim_done_at > 10.0:
                print("[debug] grace period over; stopping.", flush=True)
                break


        time.sleep(0.2)

    # Try to shut down simulator (best effort)
    try:
        http_post(f"http://{HOST}:{args.pager_port}/shutdown")
    except Exception:
        pass

    # Stop processes
    for cap in (svc, sim):
        cap.stop.set()
        if cap.proc.poll() is None:
            try:
                cap.proc.send_signal(signal.SIGTERM)
            except Exception:
                pass

    for cap in (svc, sim):
        try:
            cap.proc.wait(timeout=5)
        except Exception:
            try:
                cap.proc.kill()
            except Exception:
                pass

    # Compute metrics
    tp_set = received_pages & expected_pages
    fp_set = received_pages - expected_pages
    fn_set = expected_pages - received_pages

    tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = fbeta(tp, fp, fn, beta=1.0)
    f3 = fbeta(tp, fp, fn, beta=3.0)

    print("\n=== Integration Test Results ===")
    print(f"Expected pages: {len(expected_pages)}")
    print(f"Received pages: {len(received_pages)}")
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} F3={f3:.4f}")

    # Additional analysis
    expected_mrns = set(mrn for mrn, _ in expected_pages)
    received_mrns = set(mrn for mrn, _ in received_pages)
    
    print(f"\nMRN-level stats:")
    print(f"  Expected unique MRNs: {len(expected_mrns)}")
    print(f"  Received unique MRNs: {len(received_mrns)}")
    print(f"  Correct MRNs: {len(expected_mrns & received_mrns)}")
    print(f"  Wrong MRNs (paged but shouldn't): {len(received_mrns - expected_mrns)}")
    print(f"  Missed MRNs (should page but didn't): {len(expected_mrns - received_mrns)}")

    ok_mllp = len(mllp_ack_errors) == 0
    ok_pages = received_pages == expected_pages

    if not ok_mllp:
        print("\n[FAIL] MLLP ACK errors detected (simulator said messages were not acknowledged).")
        for ln in mllp_ack_errors[:10]:
            print("  " + ln)

    if not ok_pages:
        print("\n[FAIL] Paging mismatch vs aki.csv")
        
        # Categorize FPs: wrong MRN vs repeat page
        wrong_mrn_fps = [(mrn, dt) for mrn, dt in fp_set if mrn not in expected_mrns]
        repeat_fps = [(mrn, dt) for mrn, dt in fp_set if mrn in expected_mrns]
        
        if wrong_mrn_fps:
            print(f"\n  Wrong MRN False Positives (shouldn't page at all): {len(wrong_mrn_fps)}")
            print(f"    Sample (up to 10): {sorted(wrong_mrn_fps)[:10]}")
        
        if repeat_fps:
            print(f"\n  Repeat/Wrong-Time False Positives (correct MRN, wrong time): {len(repeat_fps)}")
            print(f"    Sample (up to 10): {sorted(repeat_fps)[:10]}")
        
        if fn_set:
            print(f"\n  Missing (FN) events (up to 10): {sorted(list(fn_set))[:10]}")

    if not (ok_mllp and ok_pages):
        print("\n--- simulator stdout (tail) ---")
        for ln in tail_lines(sim, 120):
            print(ln)
        print("\n--- service stdout (tail) ---")
        for ln in tail_lines(svc, 160):
            print(ln)

    return 0 if (ok_mllp and ok_pages) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    raise SystemExit(main())
