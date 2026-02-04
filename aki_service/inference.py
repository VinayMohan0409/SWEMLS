from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

import torch

from .history import HistoryStore
from .model_compat import ModelBundle, build_single_example_tensors, load_bundle, parse_hl7_timestamp


@dataclass
class Demographics:
    dob_yyyymmdd: str
    sex: str  # 'M'/'F'/etc


def compute_age_years(dob_yyyymmdd: str, at_hl7_ts: str) -> Optional[float]:
    try:
        dob = datetime.strptime(dob_yyyymmdd, "%Y%m%d").date()
        at = parse_hl7_timestamp(at_hl7_ts).date()
        return (at - dob).days / 365.25
    except Exception:
        return None


class InferenceService:
    """Loads a Task-1-exported model bundle and runs per-event inference."""

    def __init__(self, *, bundle_path: Optional[str], device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.bundle: Optional[ModelBundle] = None
        self.bundle_path = bundle_path
        if bundle_path:
            try:
                self.bundle = load_bundle(bundle_path, self.device)
            except FileNotFoundError:
                self.bundle = None

    def is_ready(self) -> bool:
        return self.bundle is not None

    def predict_aki(
        self,
        *,
        mrn: str,
        test_time_hl7: str,
        history_ord_vals: Sequence[Tuple[int, float]],
        demographics: Optional[Demographics],
    ) -> bool:
        if self.bundle is None:
            return False

        age_years = 0.0
        sex = "f"
        if demographics is not None:
            a = compute_age_years(demographics.dob_yyyymmdd, test_time_hl7)
            if a is not None:
                age_years = a
            sex = demographics.sex

        vals, times, mask, age, sex_t = build_single_example_tensors(
            history_ord_vals=history_ord_vals,
            age_years=age_years,
            sex=sex,
            device=self.device,
        )
        with torch.no_grad():
            logit = self.bundle.model(vals, times, mask, age, sex_t)
            prob = torch.sigmoid(logit).item()
        
        result = prob >= float(self.bundle.threshold)
        print(f"MRN: {mrn} | Prediction: {result} | Probability: {prob:.4f}")
        return result
