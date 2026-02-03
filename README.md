# SWEMLS Task 3 (initial commit)

This repo contains:
- The provided simulator scaffolding (simulator.py, messages.mllp, history.csv, aki.csv)
- An initial **AKI inference service** package: `aki_service/`
  - Layer A: MLLP TCP client + framing (`aki_service/mllp.py`)
  - Layer B: HL7 parser + router (`aki_service/hl7.py`, `aki_service/router.py`)
  - Inference service (loads Task-1 exported torch bundle): `aki_service/inference.py`

## Local dev: run simulator
In one terminal:
```bash
python simulator.py
```

It starts:
- MLLP on `localhost:8440`
- Pager on `localhost:8441`

## Local dev: run service (Python)
Put your exported model bundle at `./model/aki_model.pt` (see export instructions below), then:
```bash
export MLLP_ADDRESS=localhost:8440
export PAGER_ADDRESS=localhost:8441
python -m aki_service --history=./history.csv --model-bundle=./model/aki_model.pt
```

If you just want to test parsing/framing without paging:
```bash
python -m aki_service --dry-run-pager --history=./history.csv
```

## Local dev: run service (Docker)
```bash
docker build -t coursework3 .
docker run --rm -v ${PWD}:/data -e MLLP_ADDRESS=host.docker.internal:8440 -e PAGER_ADDRESS=host.docker.internal:8441 coursework3
```

On Linux, use `--network=host` instead of `host.docker.internal`.

## Export model from Task 1
The service expects a torch bundle saved like:
```python
torch.save({
  "version": 1,
  "hidden": 96,
  "state_dict": model.state_dict(),
  "threshold": best_threshold,
}, "model/aki_model.pt")
```

I provide a patch for your Task 1 `model.py` to add `--export-model` in `task1_patch/` (see included folder in the handoff zip).
