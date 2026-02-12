# tests/unit/test_model_compat.py

import sys
import types
import pickle
from dataclasses import dataclass

import numpy as np
import pytest
import torch

import aki_service.model_compat as m


def test_parse_hl7_timestamp_basic():
    # Parses YYYYMMDD
    dt = m.parse_hl7_timestamp("20250101")
    assert dt.strftime("%Y-%m-%d") == "2025-01-01"

    # Parses YYYYMMDDHHMM
    dt = m.parse_hl7_timestamp("202501011230")
    assert dt.strftime("%Y-%m-%d %H:%M") == "2025-01-01 12:30"

    # Parses YYYYMMDDHHMMSS
    dt = m.parse_hl7_timestamp("20250101123059")
    assert dt.strftime("%Y-%m-%d %H:%M:%S") == "2025-01-01 12:30:59"


def test_date_to_ordinal_from_any_none_and_empty():
    # None stays None
    assert m.date_to_ordinal_from_any(None) is None

    # Empty stays None
    assert m.date_to_ordinal_from_any("") is None
    assert m.date_to_ordinal_from_any("   ") is None


def test_date_to_ordinal_from_any_iso_date_takes_date_only():
    # ISO date with time uses date only
    ord1 = m.date_to_ordinal_from_any("2025-01-02 12:30:00")
    ord2 = m.date_to_ordinal_from_any("2025-01-02T23:59:59Z")
    assert ord1 == ord2


def test_date_to_ordinal_from_any_hl7_and_iso_fallback():
    # HL7 path
    a = m.date_to_ordinal_from_any("20250103120000")

    # ISO fallback path
    b = m.date_to_ordinal_from_any("2025-01-03T00:00:00Z")

    assert a == b


def test_date_to_ordinal_from_any_bad_returns_none():
    # Unparseable returns None
    assert m.date_to_ordinal_from_any("not a date") is None


def test_safe_float_handles_non_finite_and_bad():
    # Finite passes through
    assert m._safe_float("1.5") == 1.5

    # Inf becomes nan
    assert np.isnan(m._safe_float(float("inf")))

    # Bad becomes nan
    assert np.isnan(m._safe_float("nope"))


def test_build_features_all_valid():
    # Full valid history gives full stats
    hist = [(1, 1.0), (2, 2.0), (3, 3.0)]
    x = m.build_features_from_history(history_ord_vals=hist, age_years=10.0, sex="M")

    assert x.shape == (12,)
    assert x[0] == 10.0
    assert x[1] == 1.0
    assert x[2] == 3.0

    assert x[3] == 1.0  # baseline
    assert x[4] == 3.0  # index
    assert x[5] == 1.0  # min
    assert x[6] == 3.0  # max
    assert x[7] == 2.0  # mean
    assert pytest.approx(x[8], rel=1e-6) == np.std([1.0, 2.0, 3.0])  # std
    assert x[9] == 2.0  # range
    assert x[10] == 2.0  # change
    assert x[11] == 2.0  # rel change


def test_build_features_with_nans_and_unknown_sex():
    # Mixed history keeps baseline first finite and index last finite
    hist = [(1, 1.0), (2, "bad"), (3, 5.0)]
    x = m.build_features_from_history(history_ord_vals=hist, age_years=40.0, sex="X")

    assert x[2] == 2.0  # n tests
    assert x[3] == 1.0  # baseline
    assert x[4] == 5.0  # index
    assert np.isnan(x[1])  # unknown sex


def test_load_threshold_from_torch_dict(monkeypatch, tmp_path):
    # torch dict key threshold
    p = tmp_path / "threshold.pt"
    p.write_bytes(b"placeholder")

    def fake_torch_load(*_a, **_k):
        return {"threshold": 0.7}

    monkeypatch.setattr(torch, "load", fake_torch_load, raising=True)

    assert m._load_threshold(p) == 0.7


def test_load_threshold_from_torch_tensor(monkeypatch, tmp_path):
    # torch tensor item
    p = tmp_path / "threshold.pt"
    p.write_bytes(b"placeholder")

    def fake_torch_load(*_a, **_k):
        return torch.tensor(0.33)

    monkeypatch.setattr(torch, "load", fake_torch_load, raising=True)

    assert pytest.approx(m._load_threshold(p), rel=1e-9) == 0.33


def test_load_threshold_from_joblib(monkeypatch, tmp_path):
    # torch fails then joblib works
    p = tmp_path / "threshold.pt"
    p.write_bytes(b"placeholder")

    def fake_torch_load(*_a, **_k):
        raise RuntimeError("nope")

    monkeypatch.setattr(torch, "load", fake_torch_load, raising=True)

    fake_joblib = types.SimpleNamespace(load=lambda _path: {"thr": 0.25})
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

    assert m._load_threshold(p) == 0.25


def test_load_threshold_from_pickle(monkeypatch, tmp_path):
    # torch and joblib fail then pickle works
    p = tmp_path / "threshold.pt"
    with p.open("wb") as f:
        pickle.dump(0.44, f)

    def fake_torch_load(*_a, **_k):
        raise RuntimeError("nope")

    monkeypatch.setattr(torch, "load", fake_torch_load, raising=True)

    # Make joblib import fail
    monkeypatch.delitem(sys.modules, "joblib", raising=False)

    assert pytest.approx(m._load_threshold(p), rel=1e-9) == 0.44


def test_load_threshold_from_text(monkeypatch, tmp_path):
    # Binary loaders fail then text parse works
    p = tmp_path / "threshold.pt"
    p.write_text("0.42", encoding="utf-8")

    def fake_torch_load(*_a, **_k):
        raise RuntimeError("nope")

    monkeypatch.setattr(torch, "load", fake_torch_load, raising=True)
    monkeypatch.delitem(sys.modules, "joblib", raising=False)

    assert pytest.approx(m._load_threshold(p), rel=1e-9) == 0.42


def test_load_threshold_rejects_binary_as_text(monkeypatch, tmp_path):
    # Binary looking file should not be parsed as text
    p = tmp_path / "threshold.pt"
    p.write_bytes(b"\x00\x01\x02\x03binary")

    def fake_torch_load(*_a, **_k):
        raise RuntimeError("nope")

    monkeypatch.setattr(torch, "load", fake_torch_load, raising=True)
    monkeypatch.delitem(sys.modules, "joblib", raising=False)

    with pytest.raises(RuntimeError):
        m._load_threshold(p)


def test_load_sklearn_pipeline_joblib_first(monkeypatch, tmp_path):
    # joblib wins
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")

    fake_pipeline = object()
    fake_joblib = types.SimpleNamespace(load=lambda _p: fake_pipeline)
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)

    assert m._load_sklearn_pipeline(model_path) is fake_pipeline


def test_load_sklearn_pipeline_pickle_fallback(monkeypatch, tmp_path):
    # joblib fails then pickle works
    model_path = tmp_path / "model.pt"
    with model_path.open("wb") as f:
        pickle.dump({"p": 1}, f)

    class BadJoblib:
        def load(self, _p):
            raise RuntimeError("nope")

    monkeypatch.setitem(sys.modules, "joblib", BadJoblib())

    out = m._load_sklearn_pipeline(model_path)
    assert out == {"p": 1}


def test_load_sklearn_pipeline_torch_fallback(monkeypatch, tmp_path):
    # joblib and pickle fail then torch works
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")

    class BadJoblib:
        def load(self, _p):
            raise RuntimeError("nope")

    monkeypatch.setitem(sys.modules, "joblib", BadJoblib())

    def fake_pickle_load(_f):
        raise RuntimeError("nope")

    monkeypatch.setattr(pickle, "load", fake_pickle_load, raising=True)

    fake_pipeline = object()

    def fake_torch_load(*_a, **_k):
        return fake_pipeline

    monkeypatch.setattr(torch, "load", fake_torch_load, raising=True)

    assert m._load_sklearn_pipeline(model_path) is fake_pipeline


def test_load_bundle_dir_and_file(monkeypatch, tmp_path):
    # load_bundle picks correct paths
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "model.pt").write_bytes(b"x")
    (d / "threshold.pt").write_bytes(b"y")

    got = {}

    def fake_load_pipeline(p):
        got["model_path"] = p
        return "PIPE"

    def fake_load_thr(p):
        got["thr_path"] = p
        return 0.5

    monkeypatch.setattr(m, "_load_sklearn_pipeline", fake_load_pipeline, raising=True)
    monkeypatch.setattr(m, "_load_threshold", fake_load_thr, raising=True)

    b = m.load_bundle(str(d), device=torch.device("cpu"))
    assert b.model == "PIPE"
    assert b.threshold == 0.5
    assert got["model_path"].name == "model.pt"
    assert got["thr_path"].name == "threshold.pt"

    got.clear()
    f = tmp_path / "model.pt"
    f.write_bytes(b"x")
    (tmp_path / "threshold.pt").write_bytes(b"y")

    b2 = m.load_bundle(str(f), device=torch.device("cpu"))
    assert b2.model == "PIPE"
    assert b2.threshold == 0.5
    assert got["model_path"].name == "model.pt"
    assert got["thr_path"].name == "threshold.pt"


def test_load_bundle_missing_model_raises(tmp_path):
    # Missing model should raise
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "threshold.pt").write_bytes(b"y")

    with pytest.raises(FileNotFoundError):
        m.load_bundle(str(d), device=torch.device("cpu"))


def test_predict_prob_calls_model_with_2d_features():
    # predict_prob uses 2D X and returns positive class prob
    class FakeModel:
        def __init__(self):
            self.last_X = None

        def predict_proba(self, X):
            self.last_X = X
            return np.array([[0.1, 0.9]], dtype=float)

    bundle = m.ModelBundle(model=FakeModel(), threshold=0.5)

    p = m.predict_prob(
        bundle,
        history_ord_vals=[(1, 1.0), (2, 2.0)],
        age_years=30.0,
        sex="F",
    )

    assert p == 0.9
    assert bundle.model.last_X.shape == (1, 12)
