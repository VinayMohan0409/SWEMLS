#!/usr/bin/env python3


"""


Hello, thank you for reading this! This is Amin's AKI prediction model, hope u like it (:

What my code does:
*  Trains on /data/training.csv (or TRAINING_PATH)
*  Predicts for --input CSV
*  Writes predictions to --output as a single-column CSV with header "aki" and values "y"/"n"

Optional, I did this just for visualization:
* --plot writes a simple figure showing validation probability distributions + chosen threshold.
* Plotting is OFF by default and does not run in evaluation unless you enable it.


"""

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# some utilities


def die(message: str, exit_code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(exit_code)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_date_to_ordinal(value: object) -> Optional[int]:
    """
    supports "YYYY-MM-DD", "YYYY-MM-DD HH:MM:SS", and basic datetime strings.
    returns date ordinal or None.
    """
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    # as per the data: "YYYY-MM-DD HH:MM:SS"
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        s_date = s[:10]
        try:
            return datetime.strptime(s_date, "%Y-%m-%d").date().toordinal()
        except ValueError:
            pass

    # fallback: try ISO format
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date().toordinal()
    except Exception:
        return None



# data parsing


def find_creatinine_indices(columns: List[str]) -> List[int]:
    """
    returns sorted indices k for which both creatinine_date_k and creatinine_result_k exist
    """
    colset = set(columns)
    indices: List[int] = []

    for c in columns:
        if not c.startswith("creatinine_date_"):
            continue
        try:
            k = int(c.split("_")[-1])
        except ValueError:
            continue
        if f"creatinine_result_{k}" in colset:
            indices.append(k)

    indices = sorted(set(indices))
    return indices


def extract_sequence(row: pd.Series, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a patient's sequence as:
      values: creatinine results
      times: days since first test
    """
    pairs: List[Tuple[int, float]] = []

    for k in indices:
        d = parse_date_to_ordinal(row.get(f"creatinine_date_{k}", ""))
        v = row.get(f"creatinine_result_{k}", "")

        if d is None:
            continue

        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if str(v).strip() == "":
            continue

        try:
            vf = float(v)
        except ValueError:
            continue

        pairs.append((d, vf))

    if not pairs:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pairs.sort(key=lambda x: x[0])
    first_day = pairs[0][0]

    times = np.array([d - first_day for (d, _) in pairs], dtype=np.float32)
    values = np.array([val for (_, val) in pairs], dtype=np.float32)
    return values, times


class TensorBundle:
    """
    container for model inputs.
    """
    def __init__(
        self,
        vals: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
        y: Optional[torch.Tensor],
    ):
        self.vals = vals
        self.times = times
        self.mask = mask
        self.age = age
        self.sex = sex
        self.y = y


def tensorize(df: pd.DataFrame, indices: List[int], has_label: bool) -> TensorBundle:
    if "age" not in df.columns or "sex" not in df.columns:
        die("Input is missing required columns: age and sex.")

    # Extract sequences
    seq_vals: List[np.ndarray] = []
    seq_times: List[np.ndarray] = []

    for _, row in df.iterrows():
        v, t = extract_sequence(row, indices)
        seq_vals.append(v)
        seq_times.append(t)

    lengths = np.array([len(v) for v in seq_vals], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) else 0
    if max_len == 0:
        die("No creatinine sequences found (all patients appear to have zero valid tests).")

    n = len(df)
    vals_pad = np.zeros((n, max_len), dtype=np.float32)
    times_pad = np.zeros((n, max_len), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=np.bool_)

    for i in range(n):
        v = seq_vals[i]
        t = seq_times[i]
        L = len(v)

        if L == 0:
            # for robustness: avoid all-false mask
            vals_pad[i, 0] = 0.0
            times_pad[i, 0] = 0.0
            mask[i, 0] = True
            continue

        vals_pad[i, :L] = v
        times_pad[i, :L] = t
        mask[i, :L] = True

    # demographics
    age = df["age"].astype(np.float32).to_numpy().reshape(-1, 1)
    sex_raw = df["sex"].astype(str).str.lower().to_numpy()
    sex = np.where(sex_raw == "m", 1.0, 0.0).astype(np.float32).reshape(-1, 1)

    # scaling
    vals_pad = np.log1p(np.clip(vals_pad, 0.0, 2000.0)).astype(np.float32)
    times_pad = (np.clip(times_pad, 0.0, 2000.0) / 2000.0).astype(np.float32)
    age = (np.clip(age, 0.0, 120.0) / 120.0).astype(np.float32)

    y = None
    if has_label:
        if "aki" not in df.columns:
            die("Expected label column 'aki' but it was not found.")
        y_np = (df["aki"].astype(str).str.lower() == "y").astype(np.float32).to_numpy().reshape(-1, 1)
        y = torch.from_numpy(y_np)

    return TensorBundle(
        vals=torch.from_numpy(vals_pad),
        times=torch.from_numpy(times_pad),
        mask=torch.from_numpy(mask),
        age=torch.from_numpy(age),
        sex=torch.from_numpy(sex),
        y=y,
    )


class AKIDataset(Dataset):
    def __init__(self, t: TensorBundle):
        self.t = t

    def __len__(self) -> int:
        return self.t.vals.shape[0]

    def __getitem__(self, idx: int):
        if self.t.y is None:
            return (
                self.t.vals[idx],
                self.t.times[idx],
                self.t.mask[idx],
                self.t.age[idx],
                self.t.sex[idx],
            )
        return (
            self.t.vals[idx],
            self.t.times[idx],
            self.t.mask[idx],
            self.t.age[idx],
            self.t.sex[idx],
            self.t.y[idx],
        )



# model


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



# Metric + threshold (same behavior)


def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 3.0) -> float:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0

    return (1 + b2) * precision * recall / denom


def best_threshold_for_f3(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_t = 0.5
    best_score = -1.0

    for t in np.linspace(0.05, 0.95, 91):
        pred = (prob >= t).astype(np.int64)
        score = fbeta_score(y_true, pred, beta=3.0)
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t



# Train / predict (same model, same epochs)


def stratified_split_indices(y: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    simple stratified split
    """
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_val = int(round(len(pos_idx) * val_frac))
    n_neg_val = int(round(len(neg_idx) * val_frac))

    val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
    train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])

    rng.shuffle(val_idx)
    rng.shuffle(train_idx)

    return train_idx, val_idx


def train_model(train: TensorBundle, val: TensorBundle, device: torch.device, seed: int) -> Tuple[AKIModel, float, float, np.ndarray, np.ndarray]:
    seed_everything(seed)

    model = AKIModel(hidden=96).to(device)

    y_train = train.y.cpu().numpy().reshape(-1)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())

    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    train_loader = DataLoader(AKIDataset(train), batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(AKIDataset(val), batch_size=256, shuffle=False, num_workers=0)

    best_state = None
    best_f3 = -1.0
    best_thr = 0.5
    best_val_prob = None
    best_val_true = None

    epochs = 8  # same as before

    for epoch in range(1, epochs + 1):
        model.train()
        for vals, times, mask, age, sex, y in train_loader:
            vals = vals.to(device)
            times = times.to(device)
            mask = mask.to(device)
            age = age.to(device)
            sex = sex.to(device)
            y = y.to(device).view(-1)

            opt.zero_grad(set_to_none=True)
            logits = model(vals, times, mask, age, sex)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validate
        model.eval()
        all_true: List[np.ndarray] = []
        all_prob: List[np.ndarray] = []

        with torch.no_grad():
            for vals, times, mask, age, sex, y in val_loader:
                vals = vals.to(device)
                times = times.to(device)
                mask = mask.to(device)
                age = age.to(device)
                sex = sex.to(device)

                logits = model(vals, times, mask, age, sex)
                prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)

                all_true.append(y.numpy().reshape(-1))
                all_prob.append(prob)

        y_true = np.concatenate(all_true, axis=0)
        prob = np.concatenate(all_prob, axis=0)

        thr = best_threshold_for_f3(y_true, prob)
        y_pred = (prob >= thr).astype(np.int64)
        f3 = fbeta_score(y_true, y_pred, beta=3.0)

        if f3 > best_f3:
            best_f3 = f3
            best_thr = thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_prob = prob.copy()
            best_val_true = y_true.copy()

    if best_state is None:
        die("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # best_val_prob/best_val_true are guaranteed set if best_state set
    return model, best_thr, best_f3, best_val_true, best_val_prob


def predict_labels(model: AKIModel, inputs: TensorBundle, thr: float, device: torch.device) -> np.ndarray:
    loader = DataLoader(AKIDataset(inputs), batch_size=256, shuffle=False, num_workers=0)

    probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                vals, times, mask, age, sex, _y = batch
            else:
                vals, times, mask, age, sex = batch

            vals = vals.to(device)
            times = times.to(device)
            mask = mask.to(device)
            age = age.to(device)
            sex = sex.to(device)

            p = torch.sigmoid(model(vals, times, mask, age, sex)).cpu().numpy().reshape(-1)
            probs.append(p)

    prob = np.concatenate(probs, axis=0)
    pred = (prob >= thr).astype(np.int64)
    return pred



# Optional plotting (simple, local only)


def save_validation_plot(y_true: np.ndarray, prob: np.ndarray, thr: float, out_path: str) -> None:
    """
    Simple histogram plot: predicted probabilities for negatives vs positives + threshold.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] Plotting requested but matplotlib unavailable: {e}", file=sys.stderr)
        return

    neg = prob[y_true == 0]
    pos = prob[y_true == 1]

    plt.figure()
    plt.hist(neg, bins=30, alpha=0.6, label="true n")
    plt.hist(pos, bins=30, alpha=0.6, label="true y")
    plt.axvline(thr, linestyle="--", label=f"threshold={thr:.3f}")
    plt.xlabel("Predicted probability of AKI")
    plt.ylabel("Count")
    plt.title("Validation probability distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()



# Main (spec I/O)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    parser.add_argument("--plot", action="store_true", help="Save a simple validation plot (local use).")
    parser.add_argument("--export-model", default=None, help="Export trained model bundle (.pt) for Task 3.")
    parser.add_argument("--no-predict", action="store_true", help="Train/export only; skip writing aki.csv (local use).")
    args = parser.parse_args()

    training_path = os.environ.get("TRAINING_PATH", "/data/training.csv")

    if not os.path.exists(training_path):
        die(f"Training data not found at {training_path}. (Expected /data/training.csv in the container.)")
    if not os.path.exists(args.input):
        die(f"Input data not found: {args.input}")

    # Load data
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(args.input)

    # Use union of indices to support test having more history columns than training.
    # This does not use the 'aki' label as input (no leakage).
    train_idx = find_creatinine_indices(list(train_df.columns))
    test_idx = find_creatinine_indices(list(test_df.columns))
    indices = sorted(set(train_idx) | set(test_idx))

    if not indices:
        die("No creatinine_date_k/creatinine_result_k columns found.")

    # Tensorize
    train_all = tensorize(train_df, indices, has_label=True)
    test_has_label = "aki" in test_df.columns  # for local convenience only
    test_inputs = tensorize(test_df, indices, has_label=test_has_label)

    # Split train/val (stratified)
    y_all = train_all.y.numpy().reshape(-1)
    train_ids, val_ids = stratified_split_indices(y_all, val_frac=0.2, seed=1337)

    def slice_bundle(t: TensorBundle, ids: np.ndarray) -> TensorBundle:
        ids_t = torch.as_tensor(ids, dtype=torch.long)
        y_sliced = None if t.y is None else t.y[ids_t]
        return TensorBundle(
            vals=t.vals[ids_t],
            times=t.times[ids_t],
            mask=t.mask[ids_t],
            age=t.age[ids_t],
            sex=t.sex[ids_t],
            y=y_sliced,
        )

    train_part = slice_bundle(train_all, train_ids)
    val_part = slice_bundle(train_all, val_ids)

    device = torch.device("cpu")  # grading environment likely CPU

    model, thr, val_f3, val_true, val_prob = train_model(train_part, val_part, device=device, seed=1337)

# Optional export for Task 3: save a lightweight model bundle (weights + threshold).
if args.export_model:
    export_path = os.path.abspath(args.export_model)
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    torch.save(
        {
            "version": 1,
            "hidden": 96,
            "state_dict": model.state_dict(),
            "threshold": float(thr),
        },
        export_path,
    )
    print(f"[info] exported model bundle to {export_path}", file=sys.stderr)

if args.no_predict:
    return
    print(f"[info] internal val F3={val_f3:.4f} thr={thr:.3f}", file=sys.stderr)

    # Optional plot (local use)
    if args.plot:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        plot_path = os.path.join(out_dir, "val_prob_hist.png")
        save_validation_plot(val_true, val_prob, thr, plot_path)
        print(f"[info] saved plot to {plot_path}", file=sys.stderr)

    # Predict and write output
    pred = predict_labels(model, test_inputs, thr, device=device)

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("aki",))
        for p in pred:
            w.writerow(("y" if int(p) == 1 else "n",))


if __name__ == "__main__":
    main()
