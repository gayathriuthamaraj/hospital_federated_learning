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
adding FedAvg
