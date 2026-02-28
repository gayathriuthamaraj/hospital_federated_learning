# Step 01 — Running Locally

## Prerequisites

- Go 1.21 or later
- `Medicaldataset.csv` must exist at `../Medicaldataset.csv` relative to this directory
  (i.e., `c:\distributed_project\Medicaldataset.csv`)

## Run

```bash
cd step-01
go run .
```

## What It Does

Simulates three independent hospitals training on their own partitions of the medical
dataset. Each hospital:

1. Loads its assigned CSV partition (no other hospital sees this data)
2. Normalises features using per-partition min-max scaling
3. Trains a logistic regression model via mini-batch SGD
4. Produces an `UpdatePacket` containing model weights and metadata

The dataset (1319 rows, 8 features) is split across hospitals:

| Hospital | Rows      | Count |
|----------|-----------|-------|
| H1       | 0 – 439   | 440   |
| H2       | 440 – 879 | 440   |
| H3       | 880 – end | 439   |

## Expected Output

Three JSON packets printed to stdout, one per hospital. The `loss` value in each
packet's metadata will differ, confirming that the hospitals trained on genuinely
different data distributions.

## Dataset Columns

| Index | Column                  | Notes              |
|-------|-------------------------|--------------------|
| 0     | Age                     | normalised to [0,1] |
| 1     | Gender                  | 0 / 1              |
| 2     | Heart rate              | normalised         |
| 3     | Systolic blood pressure | normalised         |
| 4     | Diastolic blood pressure| normalised         |
| 5     | Blood sugar             | normalised         |
| 6     | CK-MB                   | normalised         |
| 7     | Troponin                | normalised         |
| 8     | Result                  | positive=1, negative=0 (label) |

## Project Layout

```
step-01/
  main.go              entry point — constructs global model, runs 3 hospitals
  hospital/
    data.go            CSV loader + per-partition min-max normalisation
    model.go           logistic regression model (sigmoid + BCE loss)
    trainer.go         mini-batch SGD training loop
    packet.go          UpdatePacket definition + GenerateUpdatePacket()
```

## Next Step
Adding client model receiving (see below).

---

# Step 02 — Client Model Receiving

## Goal

The client simulator now downloads the global model from the server, compares
versions, and replaces its local weights when a newer model is available.
This ensures every submission in round N uses the federated weights produced
by round N-1 aggregation.

## Prerequisites

- Go 1.21 or later
- Server and client live in the same repository root (`c:\distributed_project`)
- Two terminals open in `c:\distributed_project`

## Run

### Terminal 1 — Start the server

```bash
cd server
go run .
```

The server starts on `http://localhost:8080` and waits for round 0 submissions.

### Terminal 2 — Run the client simulator

```bash
cd c:\distributed_project
go run client_simulator.go
```

## What the Client Does (Three Phases)

### Phase 1 — Pre-round model sync
Calls `GET /global_model` before submitting.

- **First run:** server has no model yet → sync is skipped, client falls back to
  `model_version = 0` and hard-coded demo weights.
- **Subsequent runs (or if the server already has a model):** client compares
  `serverModel.ModelVersion` against its local version. If `server > local`,
  local weights and version are replaced and the event is logged.

### Phase 2 — Submit hospital updates
Sends three `POST /submit_update` payloads (H1, H2, H3). Each packet stamps:

| Field            | Value                                   |
|------------------|-----------------------------------------|
| `model_version`  | matches the version downloaded in Phase 1 |
| `round_id`       | same as `model_version` (they are aligned) |
| `weights`        | global weights + small per-hospital delta |

### Phase 3 — Post-aggregation sync
Waits 300 ms for the server to aggregate, then calls `GET /global_model` again.
If the server has produced a newer model, `LocalClientState` is updated and the
following is logged:

```
[model-sync] Model updated to version 1. Client will train on version 1 in the next round.
```

## Expected Output (first run, clean server)

```
=== Federated Client Simulator ===

[Phase 1] Checking for global model before submitting...
[model-sync] Server has no global model yet — skipping sync.

[Phase 2] Submitting round 0 updates (model_version=0)...
[submit] H1 — weights: [10 20] | model_version: 0 | HTTP 200 | ...
[submit] H2 — weights: [20 40] | model_version: 0 | HTTP 200 | ...
[submit] H3 — weights: [30 60] | model_version: 0 | HTTP 200 | ...

[Phase 3] Waiting for server aggregation...
[Phase 3] Pulling updated global model after aggregation...
[model-sync] UPDATE: server version 1 > local version -1 — replacing local weights.
[model-sync] Local model updated to version 1 | weights: [20 40]
[model-sync] Model updated to version 1. Client will train on version 1 in the next round.

=== Final client state: ModelVersion=1 | Weights=[20 40] ===
```

## Checkpoint

- Client receives **version 1** after the first aggregation.
- `LocalClientState.ModelVersion` is set to `1`.
- If the simulator were re-run (or continued to a second round), it would submit
  with `model_version=1` and `round_id=1`, training on the federated weights.

## Version Comparison Logic (key snippet)

```go
if serverModel.ModelVersion > state.ModelVersion {
    state.ModelVersion = serverModel.ModelVersion
    state.Weights      = serverModel.Weights
    // update event logged here
}
```

## Must NOT

- Server code (`server/`) is not modified.
