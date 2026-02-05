from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch


# ----------------------------
# HL7 / date helpers
# ----------------------------
def parse_hl7_timestamp(ts: str) -> datetime:
    """Parse HL7 timestamps that may omit seconds.

    Common forms:
      - YYYYMMDD
      - YYYYMMDDHHMM
      - YYYYMMDDHHMMSS
    """
    s = str(ts).strip()
    if len(s) == 8:
        return datetime.strptime(s, "%Y%m%d")
    if len(s) == 12:
        return datetime.strptime(s, "%Y%m%d%H%M")
    if len(s) == 14:
        return datetime.strptime(s, "%Y%m%d%H%M%S")
    try:
        return datetime.strptime(s, "%Y%m%d%H%M%S")
    except ValueError:
        return datetime.strptime(s, "%Y%m%d%H%M")
    
def date_to_ordinal_from_any(value: str) -> Optional[int]:
    """Match Task 1 behaviour: use date only (YYYY-MM-DD...) if present; else HL7 timestamp."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # ISO-ish date (YYYY-MM-DD...) -> take date only
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        s_date = s[:10]
        try:
            return datetime.strptime(s_date, "%Y-%m-%d").date().toordinal()
        except ValueError:
            pass

    # HL7 timestamp
    try:
        return parse_hl7_timestamp(s).date().toordinal()
    except Exception:
        # ISO fallback (handles 'Z')
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date().toordinal()
        except Exception:
            return None



# ----------------------------
# Vinay features (online)
# Mirrors vinay_model.make_features(), but uses history list.
# ----------------------------
MAP_SEX = {"M": 1.0, "F": 0.0}


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float("nan")


def build_vinay_features_from_history(
    *,
    history_ord_vals: Sequence[Tuple[int, float]],
    age_years: float,
    sex: str,
) -> np.ndarray:
    """
    Return shape (12,) features in the same order as vinay_model.make_features():
      age
      sex_binary
      n_creatinine_tests
      cr_baseline
      cr_index
      cr_min
      cr_max
      cr_mean
      cr_std
      cr_range
      change_from_baseline
      rel_change_from_baseline
    """
    vals = np.array([_safe_float(v) for _, v in history_ord_vals], dtype=float)
    valid = np.isfinite(vals)

    n_tests = float(valid.sum())

    baseline = float("nan")
    index = float("nan")
    if valid.any():
        # baseline = first available, index = last available
        first_i = int(np.argmax(valid))
        last_i = int(len(vals) - 1 - np.argmax(valid[::-1]))
        baseline = float(vals[first_i])
        index = float(vals[last_i])

    vm = np.where(valid, vals, np.nan)

    cr_min = float(np.nanmin(vm)) if valid.any() else float("nan")
    cr_max = float(np.nanmax(vm)) if valid.any() else float("nan")
    cr_mean = float(np.nanmean(vm)) if valid.any() else float("nan")
    cr_std = float(np.nanstd(vm)) if valid.any() else float("nan")
    cr_range = (cr_max - cr_min) if np.isfinite(cr_max) and np.isfinite(cr_min) else float("nan")

    change = (index - baseline) if np.isfinite(index) and np.isfinite(baseline) else float("nan")
    rel_change = (
        (change / baseline)
        if np.isfinite(change) and np.isfinite(baseline) and baseline != 0
        else float("nan")
    )

    sex_raw = str(sex).strip().upper()
    sex_bin = MAP_SEX.get(sex_raw, float("nan"))

    x = np.array(
        [
            float(age_years),
            float(sex_bin),
            float(n_tests),
            float(baseline),
            float(index),
            float(cr_min),
            float(cr_max),
            float(cr_mean),
            float(cr_std),
            float(cr_range),
            float(change),
            float(rel_change),
        ],
        dtype=float,
    )
    return x


# ----------------------------
# sklearn pipeline loader (robust)
# ----------------------------
def _load_sklearn_pipeline(model_path: Path) -> Any:
    """
    Load model.pt that was saved as an sklearn Pipeline.
    Supports: joblib, pickle, torch.save.
    """
    # 1) joblib
    try:
        import joblib  # type: ignore

        return joblib.load(str(model_path))
    except Exception:
        pass

    # 2) pickle
    try:
        import pickle

        with model_path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        pass

    # 3) torch.load (sometimes people torch.save(pipeline,...))
    try:
        return torch.load(str(model_path), map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load sklearn pipeline from {model_path}: {e}") from e


def _load_threshold(thr_path: Path) -> float:
    """
    Load threshold.pt robustly:
      - torch.save(float/tensor/dict)
      - pickle / joblib dump
      - plain text float
    """
    if not thr_path.exists():
        raise FileNotFoundError(f"threshold file not found: {thr_path}")

    # 1) torch.load (covers most cases)
    try:
        obj = torch.load(str(thr_path), map_location="cpu")
        if isinstance(obj, dict):
            for k in ("threshold", "thr", "t"):
                if k in obj:
                    return float(obj[k])
            if len(obj) == 1:
                return float(next(iter(obj.values())))
            raise ValueError(f"threshold dict missing key: keys={list(obj.keys())}")
        if hasattr(obj, "item"):
            return float(obj.item())
        return float(obj)
    except Exception:
        pass

    # 2) joblib
    try:
        import joblib  # type: ignore
        obj = joblib.load(str(thr_path))
        if isinstance(obj, dict):
            for k in ("threshold", "thr", "t"):
                if k in obj:
                    return float(obj[k])
            if len(obj) == 1:
                return float(next(iter(obj.values())))
        if hasattr(obj, "item"):
            return float(obj.item())
        return float(obj)
    except Exception:
        pass

    # 3) pickle
    try:
        import pickle
        with thr_path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for k in ("threshold", "thr", "t"):
                if k in obj:
                    return float(obj[k])
            if len(obj) == 1:
                return float(next(iter(obj.values())))
        if hasattr(obj, "item"):
            return float(obj.item())
        return float(obj)
    except Exception:
        pass

    # 4) text only if file looks like text
    try:
        raw = thr_path.read_bytes()
        # if it contains many non-printable bytes, don't treat as text
        if any(b < 9 or (13 < b < 32) for b in raw[:200]):
            raise RuntimeError("threshold.pt appears binary; cannot parse as text")
        return float(raw.decode("utf-8").strip())
    except Exception as e:
        raise RuntimeError(f"Failed to load threshold from {thr_path}: {e}") from e


# ----------------------------
# Public bundle API used by inference.py
# ----------------------------
@dataclass(frozen=True)
class ModelBundle:
    model: Any  # sklearn Pipeline
    threshold: float


def load_bundle(path: str, device: torch.device) -> ModelBundle:
    """
    For your Vinay LR deployment, we treat `path` as:
      - either /.../model.pt
      - or a directory containing model.pt and threshold.pt

    device is unused (kept for compatibility).
    """
    p = Path(path)

    if p.is_dir():
        model_path = p / "model.pt"
        thr_path = p / "threshold.pt"
    else:
        # If user passes model.pt directly, find sibling threshold.pt
        model_path = p
        thr_path = p.with_name("threshold.pt")

    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    pipeline = _load_sklearn_pipeline(model_path)
    threshold = _load_threshold(thr_path)

    return ModelBundle(model=pipeline, threshold=threshold)


def predict_prob(
    bundle: ModelBundle,
    *,
    history_ord_vals: Sequence[Tuple[int, float]],
    age_years: float,
    sex: str,
) -> float:
    """
    Runs sklearn Pipeline predict_proba on Vinay features.
    """
    x = build_vinay_features_from_history(
        history_ord_vals=history_ord_vals,
        age_years=age_years,
        sex=sex,
    )
    # Pipeline expects 2D
    X = x.reshape(1, -1)

    # predict_proba -> [:, 1] positive class
    p = bundle.model.predict_proba(X)[:, 1]
    return float(p[0])
