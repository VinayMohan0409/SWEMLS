# tests/unit/test_inference.py

from datetime import date

import pytest
import torch

import aki_service.inference as inf_mod


def test_compute_age_years_valid(monkeypatch):
    # Basic age calculation
    monkeypatch.setattr(
        inf_mod,
        "parse_hl7_timestamp",
        lambda _ts: type("X", (), {"date": lambda self: date(2025, 1, 2)})(),
        raising=True,
    )

    age = inf_mod.compute_age_years("20000101", "20250102000000")
    assert age is not None
    assert age > 0


def test_compute_age_years_invalid_returns_none(monkeypatch):
    # Bad dob returns None
    monkeypatch.setattr(
        inf_mod,
        "parse_hl7_timestamp",
        lambda _ts: type("X", (), {"date": lambda self: date(2025, 1, 2)})(),
        raising=True,
    )
    assert inf_mod.compute_age_years("bad", "20250102000000") is None

    # Bad timestamp returns None
    def boom(_ts):
        raise ValueError("nope")

    monkeypatch.setattr(inf_mod, "parse_hl7_timestamp", boom, raising=True)
    assert inf_mod.compute_age_years("20000101", "bad") is None


def test_inference_service_not_ready_when_no_bundle_path():
    # No bundle path means not ready
    s = inf_mod.InferenceService(bundle_path=None)
    assert s.is_ready() is False


def test_inference_service_missing_bundle_file_sets_none(monkeypatch):
    # Missing files should not crash
    def raise_fnf(_path, _device):
        raise FileNotFoundError()

    monkeypatch.setattr(inf_mod, "load_bundle", raise_fnf, raising=True)

    s = inf_mod.InferenceService(bundle_path="x")
    assert s.is_ready() is False


def test_predict_aki_returns_false_when_no_bundle():
    # No bundle gives False
    s = inf_mod.InferenceService(bundle_path=None)
    assert s.predict_aki(mrn="1", test_time_hl7="20250101000000", history_ord_vals=[], demographics=None) is False


def test_predict_aki_defaults_used_when_no_demographics(monkeypatch):
    # Defaults age 0 and sex F
    class FakeBundle:
        threshold = 0.5

    s = inf_mod.InferenceService(bundle_path=None)
    s.bundle = FakeBundle()

    got = {}

    def fake_predict_prob(bundle, *, history_ord_vals, age_years, sex):
        got["bundle"] = bundle
        got["history"] = history_ord_vals
        got["age"] = age_years
        got["sex"] = sex
        return 0.5

    monkeypatch.setattr(inf_mod, "predict_prob", fake_predict_prob, raising=True)

    ok = s.predict_aki(mrn="1", test_time_hl7="20250101000000", history_ord_vals=[(1, 1.0)], demographics=None)

    assert ok is True
    assert got["age"] == 0.0
    assert got["sex"] == "F"


def test_predict_aki_uses_demographics_when_age_parses(monkeypatch):
    # Demographics override defaults
    class FakeBundle:
        threshold = 0.8

    s = inf_mod.InferenceService(bundle_path=None)
    s.bundle = FakeBundle()

    monkeypatch.setattr(inf_mod, "compute_age_years", lambda _dob, _ts: 25.0, raising=True)

    got = {}

    def fake_predict_prob(bundle, *, history_ord_vals, age_years, sex):
        got["age"] = age_years
        got["sex"] = sex
        return 0.79

    monkeypatch.setattr(inf_mod, "predict_prob", fake_predict_prob, raising=True)

    demo = inf_mod.Demographics(dob_yyyymmdd="20000101", sex="M")
    ok = s.predict_aki(mrn="1", test_time_hl7="20250101000000", history_ord_vals=[], demographics=demo)

    assert ok is False
    assert got["age"] == 25.0
    assert got["sex"] == "M"


def test_predict_aki_demographics_age_none_keeps_default_age(monkeypatch):
    # Age None keeps default 0 but sex still used
    class FakeBundle:
        threshold = 0.2

    s = inf_mod.InferenceService(bundle_path=None)
    s.bundle = FakeBundle()

    monkeypatch.setattr(inf_mod, "compute_age_years", lambda _dob, _ts: None, raising=True)

    got = {}

    def fake_predict_prob(bundle, *, history_ord_vals, age_years, sex):
        got["age"] = age_years
        got["sex"] = sex
        return 0.19

    monkeypatch.setattr(inf_mod, "predict_prob", fake_predict_prob, raising=True)

    demo = inf_mod.Demographics(dob_yyyymmdd="20000101", sex="M")
    ok = s.predict_aki(mrn="1", test_time_hl7="20250101000000", history_ord_vals=[], demographics=demo)

    assert ok is False
    assert got["age"] == 0.0
    assert got["sex"] == "M"


def test_predict_aki_rounding_makes_boundary_deterministic(monkeypatch):
    # Rounding to 8 decimals stabilizes boundary
    class FakeBundle:
        threshold = 0.1 + 0.2  # 0.30000000000000004

    s = inf_mod.InferenceService(bundle_path=None)
    s.bundle = FakeBundle()

    # Prob slightly above 0.3 at high precision
    monkeypatch.setattr(inf_mod, "predict_prob", lambda *_a, **_k: 0.3000000049, raising=True)

    ok = s.predict_aki(mrn="1", test_time_hl7="20250101000000", history_ord_vals=[], demographics=None)

    # After rounding prob 0.3 threshold 0.3 so True
    assert ok is True


def test_inference_service_device_parsing():
    # Device string accepted
    s = inf_mod.InferenceService(bundle_path=None, device="cpu")
    assert isinstance(s.device, torch.device)
