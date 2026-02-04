#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
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
from pathlib import Path
from typing import Optional, Set, Tuple


# Simulator prints pager receipts like:
#   pager: paging for MRN 185983563 at 2024-12-21 09:44:00
PAGER_RE = re.compile(
    r"pager:\s+paging for MRN\s+(?P<mrn>\d+)\s+at\s+(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
)


def repo_root_from_here() -> Path:
    """
    This file is located at: <repo_root>/aki_service/integration_test.py
    So repo_root is one parent up from this file's directory.
    """
    return Path(__file__).resolve().parents[1]


def load_expected_pages(expected_csv: Path) -> Set[Tuple[int, str]]:
    """
    expected aki.csv format:
      mrn,date
      123765409,2024-03-31 22:09:00
    """
    out: Set[int] = set() # Changed to set of ints
    with expected_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mrn_s = (row.get("mrn") or "").strip()
            if not mrn_s:
                continue
            out.add(int(mrn_s)) # Only add the MRN
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
    ap.add_argument("--timeout-s", type=int, default=60)

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
    model_bundle = Path(args.model_bundle) if args.model_bundle else repo / "model" / "aki_model.pt"

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

    # reset SQLite DB so runs are repeatable
    inspect_db = repo / "aki_service" / "inspectdb.py"
    if inspect_db.exists():
        print(f"[debug] resetting DB via: {inspect_db} --reset", flush=True)
        subprocess.run([sys.executable, str(inspect_db), "--reset"], cwd=str(repo), check=True)
    else:
        print(f"[warn] inspect_db.py not found at {inspect_db}; DB not reset", flush=True)


    # Environment: ensure repo is importable
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    env["MLLP_ADDRESS"] = f"localhost:{args.mllp_port}"
    env["PAGER_ADDRESS"] = f"localhost:{args.pager_port}"
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
    print("[debug] simulator output sample:", flush=True)
    for ln in tail_lines(sim, 20):
        print("  " + ln, flush=True)


    # Give simulator a moment to bind ports
    time.sleep(0.4)

    # Start service (REAL service entrypoint)
    svc_cmd = [
        sys.executable,
        "-u",
        "-m",
        args.service_module,
        "--mllp",
        f"localhost:{args.mllp_port}",
        "--pager",
        f"localhost:{args.pager_port}",
        "--history",
        str(history_csv),
        "--model-bundle",
        str(model_bundle),
    ]

    print(f"[info] starting service: {' '.join(svc_cmd)}", flush=True)
    svc = start_process(svc_cmd, cwd=repo, env=env)

    # Inside main()
    received_pages: Set[int] = set() # Changed from Set[Tuple[int, str]]
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

            print("\n--- simulator stdout (tail on exit) ---", flush=True)
            for ln in tail_lines(sim, 80):
                print(ln, flush=True)

            print("\n--- service stdout (tail on exit) ---", flush=True)
            for ln in tail_lines(svc, 80):
                print(ln, flush=True)

            break

        # Parse latest simulator output
        # Parse simulator output incrementally (no tail rescans)
        with sim.lock:
            new_lines = sim.lines[last_sim_idx:]
            last_sim_idx = len(sim.lines)

        # Inside the while time.time() < deadline: loop
        for ln in new_lines:
            if "not acknowledged" in ln.lower():
                mllp_ack_errors.append(ln)

            m = PAGER_RE.search(ln)
            if m:
                mrn = int(m.group("mrn"))
                # We ignore m.group("dt") as requested
                received_pages.add(mrn)


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
        http_post(f"http://localhost:{args.pager_port}/shutdown")
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

    ok_mllp = len(mllp_ack_errors) == 0
    ok_pages = received_pages == expected_pages

    if not ok_mllp:
        print("\n[FAIL] MLLP ACK errors detected (simulator said messages were not acknowledged).")
        for ln in mllp_ack_errors[:10]:
            print("  " + ln)

    if not ok_pages:
        print("\n[FAIL] Paging mismatch vs aki.csv")
        if fn_set:
            # sorted(list(fn_set)) will now just be a list of MRNs
            print(f"Missing (FN) MRNs (up to 10): {sorted(list(fn_set))[:10]}")
        if fp_set:
            print(f"Unexpected (FP) MRNs (up to 10): {sorted(list(fp_set))[:10]}")

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
