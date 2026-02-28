# Federated Hospital Learning System

## Overview

A privacy-preserving distributed learning system where multiple hospitals collaboratively train a shared global model without sharing raw patient data. Each hospital trains locally, sends only model updates, receives an improved global model, and repeats the cycle — all while ensuring fairness, synchronization, security, and time-awareness.

---

## Core Idea

Traditional machine learning in healthcare requires centralizing sensitive patient data, which raises serious privacy, legal, and ethical concerns. This system solves that by keeping data at each hospital while still enabling collaborative model improvement across institutions.

The system is built on three layers: a client layer representing individual hospitals, a secure communication layer handling all data transport, and a server layer that aggregates updates and manages the training lifecycle.

---

## Aggregation: QFedAvg

Instead of simple averaging (FedAvg), this system uses QFedAvg — a fairness-aware aggregation algorithm. Hospitals with higher local loss receive proportionally higher influence over the global model. This prevents large hospitals from dominating training and ensures that hospitals with harder or underrepresented patient populations are not sidelined.

The weight assigned to each hospital's update is:

```
weight = (loss ^ q) * data_size
```

Where `q` controls the degree of fairness enforcement.

---

## Distributed Timeline Management

A timeline manager controls the training lifecycle. It tracks model versions, enforces quorum rules (minimum participation thresholds before aggregation triggers), and handles staleness — the condition where a hospital submits an update based on an outdated version of the global model.

Stale updates are not discarded but down-weighted:

```
adjusted_weight = original_weight * (1 / (1 + staleness))
```

Where staleness is the difference between the current model version and the version the hospital trained on. This allows the system to tolerate real-world variability such as slow or intermittently connected hospitals without destabilizing the global model.

Rounds progress through defined states: waiting for participation, aggregating updates, and completing before the next round begins.

The `RoundManager` (implemented in `server/round_manager.go`) is the concrete realisation of this concept. It tracks:

- `current_round` — the monotonically incrementing round number
- `expected_clients` — the quorum threshold (default: 3)
- `received_clients` — the set of hospital IDs that have submitted in the current round
- `state` — one of `WAITING`, `AGGREGATING`, or `COMPLETE`

Aggregation fires only when `len(received_clients) >= expected_clients`. Duplicate submissions from the same hospital within a round are rejected, as are submissions that reference the wrong round ID.

---

## Secure Communication

All update packets are validated before processing. The system verifies client identity, checks timestamps, prevents replay attacks, and rejects packets referencing invalid or outdated model versions. This makes the system suitable for real healthcare environments where data integrity and authenticity are non-negotiable.

---

## Training Flow

1. The server distributes the current global model to participating hospitals.
2. Each hospital trains a local model on its own data.
3. Each hospital sends a secure update packet containing model weights, loss, data size, round ID, model version, and timestamp.
4. The server validates each packet.
5. The timeline manager tracks participation and waits for quorum.
6. Once quorum is reached, QFedAvg aggregates the updates into a new global model.
7. The updated model is redistributed and the next round begins.

This repeats until the model converges.

---

## Key Properties

**Privacy** — Raw patient data never leaves a hospital.

**Fairness** — QFedAvg ensures smaller or harder hospitals maintain meaningful influence.

**Time-awareness** — Stale updates are tolerated but appropriately scaled down.

**Security** — Signed and validated packets prevent tampering and replay attacks.

**Fault tolerance** — The semi-asynchronous design handles slow or offline hospitals gracefully.

**Scalability** — The architecture supports a growing number of participating institutions.

---

## Significance

This system combines federated optimization, secure distributed communication, timeline synchronization, staleness-aware aggregation, and quorum-based round control into a single coherent infrastructure. It is not simply federated learning — it is a production-oriented system that addresses the practical, security, and fairness challenges of deploying machine learning across real healthcare institutions.

Potential extensions include cross-country hospital collaboration, edge medical devices, government health analytics, and real-time outbreak prediction.

---

## Implementation Status

| Turn | Owner | Component | Status |
|------|-------|-----------|--------|
| 1 — A | Client Foundation | Local training, `UpdatePacket` generation (`step-01/`) | Done |
| 2 — D | Communication Layer | REST server, `/submit_update`, packet validation (`server/`) | Done |
| 3 — B | Aggregation | FedAvg in `aggregateUpdates()`, global model versioning (`server/`) | Done |
| 4 — C | Round Control | `RoundManager`: quorum tracking, round lifecycle, duplicate rejection (`server/round_manager.go`) | Done |

---

## Project Structure

```
hospital_federated_learning/
  Medicaldataset.csv          1 319-row patient dataset (8 features + label)
  client_simulator.go         Standalone script: submits 3 update packets to the server

  step-01/                    Turn 1 — client-side only (no networking)
    main.go                   Runs 3 hospitals locally, prints UpdatePackets
    hospital/
      data.go                 CSV loader + per-partition min-max normalisation
      model.go                Logistic regression (sigmoid + BCE loss)
      trainer.go              Mini-batch SGD training loop
      packet.go               UpdatePacket definition + GenerateUpdatePacket()

  server/                     Turns 2 / 3 / 4 — central server
    main.go                   HTTP server, request handlers, FedAvg aggregation
    round_manager.go          RoundManager: round lifecycle and quorum control
    go.mod
```

---

## How to Run

### Step 01 — local training only (no server needed)

```bash
cd step-01
go run .
```

Prints three `UpdatePacket` JSON blobs with different `loss` values, confirming each hospital trained on a distinct data partition.

### Server + client simulation

Open two terminals from the project root.

**Terminal 1 — start the server**

```bash
cd server
go run .
```

**Terminal 2 — simulate three hospital submissions**

```bash
go run client_simulator.go
```

The simulator submits updates from H1, H2, and H3. After the third submission the `RoundManager` declares quorum, aggregation runs, the global model version increments, and round 1 opens automatically.

---

## Server API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/submit_update` | Hospital submits an `UpdatePacket`; validated and registered with `RoundManager` |
| `GET` | `/global_model` | Returns aggregated weights and current model version |
| `GET` | `/updates_count` | Returns the number of updates buffered for the current round |
| `GET` | `/round_status` | Returns `current_round`, `expected_clients`, `received_clients`, and `state` |