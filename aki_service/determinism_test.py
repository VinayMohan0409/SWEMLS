#!/usr/bin/env python3
"""
Determinism Test: Verifies that predictions are consistent.

Run this on multiple machines - if the checksum differs, there's a reproducibility issue.
"""

import hashlib
import csv
import sys
from pathlib import Path

def repo_root():
    return Path(__file__).resolve().parents[1]

def main():
    # Add repo to path
    sys.path.insert(0, str(repo_root()))
    
    from aki_service.db import Database
    from aki_service.history import HistoryStore
    from aki_service.inference import InferenceService, Demographics
    from aki_service.model_compat import date_to_ordinal_from_any
    
    # Setup
    model_path = repo_root() / "model" / "model.pt"
    history_path = repo_root() / "history.csv"
    
    # Use in-memory DB for isolation
    db = Database(":memory:")
    history = HistoryStore(db)
    
    print(f"Loading history from {history_path}...")
    n = history.load_history_csv(str(history_path))
    print(f"Loaded {n} lab entries")
    
    print(f"Loading model from {model_path}...")
    inf = InferenceService(bundle_path=str(model_path), device="cpu")
    if not inf.is_ready():
        print("ERROR: Model not loaded!")
        return 1
    
    print(f"Model threshold: {inf.bundle.threshold}")
    
    # Test a fixed set of MRNs with known data
    # We'll compute predictions and hash them
    test_cases = []
    
    # Get all unique MRNs from history
    with db._get_conn() as conn:
        mrns = [row[0] for row in conn.execute("SELECT DISTINCT mrn FROM labs ORDER BY mrn").fetchall()]
    
    print(f"Testing {len(mrns)} unique MRNs...")
    
    predictions = []
    for mrn in mrns[:500]:  # Test first 500 for speed
        history_data = history.get_history_from_db(mrn)
        if not history_data:
            continue
            
        # Use the last entry's date as test time
        last_date_ord, last_val = history_data[-1]
        
        # Filter history up to this point
        history_ord_vals = [(d, v) for d, v in history_data if d <= last_date_ord]
        
        # Make prediction (no demographics for simplicity)
        try:
            from aki_service.model_compat import predict_prob, build_vinay_features_from_history
            
            # Get raw probability
            prob = predict_prob(
                inf.bundle,
                history_ord_vals=history_ord_vals,
                age_years=50.0,  # Fixed for determinism
                sex="M",
            )
            
            # Round like we do in production
            prob_rounded = round(prob, 8)
            threshold = round(float(inf.bundle.threshold), 8)
            decision = prob_rounded >= threshold
            
            predictions.append(f"{mrn}:{prob_rounded:.10f}:{decision}")
            
        except Exception as e:
            predictions.append(f"{mrn}:ERROR:{e}")
    
    # Create deterministic hash
    predictions.sort()  # Ensure order is consistent
    prediction_str = "\n".join(predictions)
    checksum = hashlib.sha256(prediction_str.encode()).hexdigest()[:16]
    
    print(f"\n{'='*60}")
    print(f"DETERMINISM CHECK")
    print(f"{'='*60}")
    print(f"Predictions made: {len(predictions)}")
    print(f"Checksum: {checksum}")
    print(f"{'='*60}")
    print(f"\nIf this checksum differs across machines, predictions are non-deterministic!")
    print(f"\nSample predictions (first 10):")
    for p in predictions[:10]:
        print(f"  {p}")
    
    # Also show predictions near threshold
    print(f"\nPredictions near threshold (|prob - threshold| < 0.01):")
    threshold = float(inf.bundle.threshold)
    near_threshold = []
    for p in predictions:
        parts = p.split(":")
        if len(parts) >= 2 and parts[1] != "ERROR":
            prob = float(parts[1])
            if abs(prob - threshold) < 0.01:
                near_threshold.append(p)
    
    for p in near_threshold[:20]:
        print(f"  {p}")
    
    if near_threshold:
        print(f"\n WARNING  {len(near_threshold)} predictions are near the threshold boundary!")
        print("   These are most likely to flip between machines.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
