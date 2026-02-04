import types
from dataclasses import dataclass
import pytest
import torch

from aki_service.inference import InferenceService, Demographics, compute_age_years

class FakeModel:
    """
    returns a fixed logit so sigmoid(logit) is deterministic
    """
    def __init__(self, logit: float):
        self.logit = float(logit)
        self.calls = 0

    def __call__(self, vals, times, mask, age, sex_t):
        self.calls += 1
        # return scalar tensor logit 
        return torch.tensor(self.logit, dtype=torch.float32)


@dataclass
class FakeBundle:
    model: object
    threshold: float


def fake_build_single_example_tensors(*, history_ord_vals, age_years, sex, device):
    """
    return tensors with the correct shapes expected by InferenceService.predict_aki.
    """

    vals = torch.zeros((1, 1), device=device)
    times = torch.zeros((1, 1), device=device)
    mask = torch.ones((1, 1), device=device)
    age = torch.tensor([age_years], device=device, dtype=torch.float32)
    sex_t = torch.tensor([0.0], device=device, dtype=torch.float32)  # placeholder encoding
    return vals, times, mask, age, sex_t


########## unit tests ##########

def test_compute_age_years_valid():
    age = compute_age_years("20000101", "20200101000000")
    assert age is not None
    assert 19.9 < age < 20.1


def test_compute_age_years_invalid_returns_none():
    # bad DOB format
    assert compute_age_years("2000-01-01", "20200101000000") is None
    # bad timestamp format
    assert compute_age_years("20000101", "not-a-timestamp") is None


def test_inference_service_not_ready_when_bundle_missing(tmp_path):
    # passing nonexistent path triggers FileNotFoundError handling in init
    svc = InferenceService(bundle_path=str(tmp_path / "does_not_exist.pt"), device="cpu")
    assert svc.is_ready() is False


def test_predict_aki_returns_false_when_no_bundle(monkeypatch):
    svc = InferenceService(bundle_path=None, device="cpu")
    assert svc.is_ready() is False

    out = svc.predict_aki(
        mrn="123",
        test_time_hl7="20200101000000",
        history_ord_vals=[(0, 100.0)],
        demographics=None,
    )
    assert out is False


def test_predict_aki_thresholding(monkeypatch):
    """
    prob = sigmoid(logit).item() >= threshold
    """

    svc = InferenceService(bundle_path=None, device="cpu")

    import aki_service.inference as inference_mod
    monkeypatch.setattr(inference_mod, "build_single_example_tensors", fake_build_single_example_tensors)

    # Case 1: logit=0 => sigmoid=0.5
    fake_model = FakeModel(logit=0.0)
    svc.bundle = FakeBundle(model=fake_model, threshold=0.7)  # 0.5 < 0.7 => False

    out = svc.predict_aki(
        mrn="123",
        test_time_hl7="20200101000000",
        history_ord_vals=[(0, 100.0)],
        demographics=None,
    )
    assert out is False
    assert fake_model.calls == 1

    # Case 2: logit=2 => sigmoid~0.88
    fake_model2 = FakeModel(logit=2.0)
    svc.bundle = FakeBundle(model=fake_model2, threshold=0.7)  # ~0.88 >= 0.7 => True

    out2 = svc.predict_aki(
        mrn="123",
        test_time_hl7="20200101000000",
        history_ord_vals=[(0, 100.0)],
        demographics=None,
    )
    assert out2 is True
    assert fake_model2.calls == 1


def test_predict_aki_uses_demographics_when_provided(monkeypatch):
    """
    shows that the code path uses compute_age_years and passes values to the tensor builder
    """
    svc = InferenceService(bundle_path=None, device="cpu")

    import aki_service.inference as inference_mod

    captured = {}

    def capturing_builder(*, history_ord_vals, age_years, sex, device):
        captured["age_years"] = age_years
        captured["sex"] = sex
        return fake_build_single_example_tensors(
            history_ord_vals=history_ord_vals,
            age_years=age_years,
            sex=sex,
            device=device,
        )

    monkeypatch.setattr(inference_mod, "build_single_example_tensors", capturing_builder)

    # choose any threshhold that will return a deterministic value
    svc.bundle = FakeBundle(model=FakeModel(logit=0.0), threshold=0.4)  # sigmoid(0)=0.5 => True

    demo = Demographics(dob_yyyymmdd="20000101", sex="M")
    out = svc.predict_aki(
        mrn="123",
        test_time_hl7="20200101000000",
        history_ord_vals=[(0, 100.0)],
        demographics=demo,
    )

    assert out is True
    assert "age_years" in captured
    assert 19.9 < captured["age_years"] < 20.1
    assert captured["sex"] == "M"


def test_inference_service_init_loads_bundle_when_path_exists(monkeypatch, tmp_path):
    """
    simulate a path that "exists" and patch load_bundle to return a FakeBundle
    """
    bundle_path = tmp_path / "aki_model.pt"
    bundle_path.write_bytes(b"not a real torch file, but path exists")

    import aki_service.inference as inference_mod
    fake = FakeBundle(model=FakeModel(logit=0.0), threshold=0.5)

    def fake_loader(path, device):
        assert str(path) == str(bundle_path)
        return fake

    monkeypatch.setattr(inference_mod, "load_bundle", fake_loader)

    svc = InferenceService(bundle_path=str(bundle_path), device="cpu")
    assert svc.is_ready() is True
    assert svc.bundle is fake
