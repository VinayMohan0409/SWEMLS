# Streaming AKI Inference Service

An event-driven machine-learning service that consumes HL7v2 hospital messages over MLLP, maintains longitudinal creatinine histories, runs acute kidney injury (AKI) inference, and sends idempotent pager alerts.

The repository focuses on the engineering around a model in production: streaming input, state management, deterministic feature compatibility, failure recovery, protocol handling, and end-to-end evaluation.

## System flow

1. `MLLPClient` incrementally decodes framed messages from a TCP byte stream and reconnects after transport failures.
2. The HL7 parser maps admission, discharge, and creatinine-result messages to typed events.
3. `Router` persists patient state and lab results in SQLite.
4. The service reconstructs the patient's creatinine history up to the event timestamp.
5. `InferenceService` reproduces the training feature order and calls a saved scikit-learn probability model.
6. Predictions above the saved threshold trigger an HTTP pager request.
7. Alert state is persisted to prevent duplicate paging and to retry failed sends after restart.

Malformed messages receive an HL7 `AE` acknowledgement; accepted or safely ignored events receive `AA`.

## Reliability and state management

The service includes:

- incremental MLLP framing for partial and multiple-frame socket reads;
- bounded frame buffers, reconnect backoff, and graceful signal handling;
- SQLite WAL mode and a reusable connection for consistent read-after-write behaviour;
- primary keys for duplicate lab and alert suppression;
- admission-aware paging rules;
- durable alert statuses and bounded startup retries;
- feature construction aligned with the training pipeline;
- explicit handling for absent or invalid model artifacts.

## Verification

The repository contains 91 unit-test functions covering HL7 parsing, MLLP framing and reconnect behaviour, feature compatibility, inference thresholds, routing, history ingestion, pager errors, SQLite state, and retry logic.

An integration harness starts the included simulator and the real service, captures pager events, compares them with `aki.csv`, checks acknowledgement failures, and computes precision, recall, and F3. Separate scripts exercise deterministic predictions across repeated runs.

The suite is designed for local execution and covers both isolated components and the complete simulator-to-alert path.

## Repository layout

| Path | Purpose |
| --- | --- |
| `aki_service/__main__.py` | Service entry point and lifecycle |
| `aki_service/mllp.py` | Streaming MLLP decoder, client, and ACK generation |
| `aki_service/hl7.py` | Defensive HL7 event parsing |
| `aki_service/router.py` | Event routing, inference, and alert orchestration |
| `aki_service/db.py` | SQLite patient, lab, and alert state |
| `aki_service/model_compat.py` | Training/serving feature compatibility and artifact loading |
| `aki_service/integration_test.py` | Simulator-to-pager end-to-end evaluation |
| `tests/` | Unit and determinism tests |
| `simulator.py`, `messages.mllp` | Local event-stream simulator and sample messages |
| `model/` | Included model and threshold artifacts |

## Run locally

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the simulator:

```bash
python simulator.py
```

In a second terminal, start the service with the included history and model artifacts:

```bash
export MLLP_ADDRESS=localhost:8440
export PAGER_ADDRESS=localhost:8441
python -m aki_service \
  --history=./history.csv \
  --model-bundle=./model/model.pt
```

For protocol and inference testing without an HTTP pager:

```bash
python -m aki_service \
  --dry-run-pager \
  --history=./history.csv \
  --model-bundle=./model/model.pt
```

`model_compat.py` expects a saved scikit-learn pipeline in `model.pt` and a sibling `threshold.pt`. It accepts either the model file path or the containing directory.

## Run tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Run the complete simulator/service comparison:

```bash
python -m aki_service.integration_test
```

The integration test can take several minutes while the recorded message stream is replayed.

## Docker

Build the service image:

```bash
docker build -t aki-service .
```

With the simulator running on the host:

```bash
docker run --rm \
  -v "$PWD:/data" \
  -e MLLP_ADDRESS=host.docker.internal:8440 \
  -e PAGER_ADDRESS=host.docker.internal:8441 \
  aki-service
```

On Linux, use host networking and `localhost` addresses instead:

```bash
docker run --rm --network=host \
  -v "$PWD:/data" \
  -e MLLP_ADDRESS=localhost:8440 \
  -e PAGER_ADDRESS=localhost:8441 \
  aki-service
```

## Engineering highlights

The project demonstrates the systems work required to serve an ML model against a streaming clinical event source: protocol boundaries, persistent state, idempotency, recovery, deterministic preprocessing, and end-to-end evaluation.
