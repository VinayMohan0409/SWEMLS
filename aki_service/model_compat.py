from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


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
    # fall back: try seconds, then minutes
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
    # Task1 treated ISO-ish strings by taking the first 10 chars (date only).
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
        # ISO fallback
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date().toordinal()
        except Exception:
            return None


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        # mask: [B, L] bool
        logits = self.score(x).squeeze(-1)               # [B, L]
        logits = logits.masked_fill(~mask, -1e9)         # ignore padding
        weights = torch.softmax(logits, dim=-1)          # [B, L]
        pooled = torch.bmm(weights.unsqueeze(1), x)      # [B, 1, D]
        return pooled.squeeze(1)                         # [B, D]


class AKIModel(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.pool = AttentionPool(d_model=hidden * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 2, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        vals: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([vals, times], dim=-1)   # [B, L, 2]
        x = self.in_proj(x)                      # [B, L, H]
        x, _ = self.rnn(x)                       # [B, L, 2H]
        pooled = self.pool(x, mask)              # [B, 2H]
        demo = torch.cat([age, sex], dim=-1)     # [B, 2]
        z = torch.cat([pooled, demo], dim=-1)    # [B, 2H+2]
        return self.head(z).squeeze(-1)          # [B]


@dataclass(frozen=True)
class ModelBundle:
    model: AKIModel
    threshold: float


def load_bundle(path: str, device: torch.device) -> ModelBundle:
    """Load a model bundle saved by Task 1 export."""
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("model bundle must be a dict")
    hidden = int(obj.get("hidden", 96))
    thr = float(obj["threshold"])
    state = obj["state_dict"]
    m = AKIModel(hidden=hidden)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return ModelBundle(model=m, threshold=thr)


def build_single_example_tensors(
    *,
    history_ord_vals: Sequence[Tuple[int, float]],
    age_years: float,
    sex: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build tensors shaped like Task 1 tensorize() for a single patient.

    history_ord_vals: sequence of (date_ordinal, value) including current result.
    """
    if not history_ord_vals:
        # Task1 would error for empty sequences, but online we may see this; build a dummy.
        history_ord_vals = [(datetime.utcnow().date().toordinal(), 0.0)]

    pairs = sorted(history_ord_vals, key=lambda x: x[0])
    first_day = pairs[0][0]
    times = np.array([float(d - first_day) for d, _ in pairs], dtype=np.float32)
    vals = np.array([float(v) for _, v in pairs], dtype=np.float32)

    # scaling (match Task 1)
    vals = np.log1p(np.clip(vals, 0.0, 2000.0)).astype(np.float32)
    times = (np.clip(times, 0.0, 2000.0) / 2000.0).astype(np.float32)
    age = (np.clip(np.array([age_years], dtype=np.float32), 0.0, 120.0) / 120.0).astype(np.float32)

    sex_raw = str(sex).strip().lower()
    sex_v = np.array([1.0 if sex_raw == "m" else 0.0], dtype=np.float32)

    # Pad to L (no padding needed for single example)
    L = vals.shape[0]
    vals_t = torch.from_numpy(vals.reshape(1, L)).to(device)
    times_t = torch.from_numpy(times.reshape(1, L)).to(device)
    mask_t = torch.ones((1, L), dtype=torch.bool, device=device)
    age_t = torch.from_numpy(age.reshape(1, 1)).to(device)
    sex_t = torch.from_numpy(sex_v.reshape(1, 1)).to(device)
    return vals_t, times_t, mask_t, age_t, sex_t
