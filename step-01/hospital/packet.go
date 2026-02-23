package hospital

import (
	"encoding/json"
	"fmt"
	"time"
)

// Metadata carries everything the server needs to evaluate and weight
// a hospital's update without seeing any raw patient data.
type Metadata struct {
	HospitalID   string  `json:"hospital_id"`
	DataSize     int     `json:"data_size"`
	Loss         float64 `json:"loss"`
	RoundID      int     `json:"round_id"`
	ModelVersion int     `json:"model_version"`
	Timestamp    string  `json:"timestamp"` // ISO-8601; used by the Timeline Manager in later steps
}

// UpdatePacket is the complete hand-off from a hospital to the server.
// Weights is the flat serialisation produced by Model.FlatWeights().
// Raw patient data is never included.
type UpdatePacket struct {
	Weights  []float64 `json:"weights"`
	Metadata Metadata  `json:"metadata"`
}

// HospitalConfig describes a hospital's identity and its dataset partition.
// StartIdx/EndIdx are 0-based row indices into the CSV (header excluded).
type HospitalConfig struct {
	ID           string
	RoundID      int
	ModelVersion int
	CSVPath      string // absolute or relative path to Medicaldataset.csv
	StartIdx     int    // first row index for this hospital's partition
	EndIdx       int    // one-past-last row index
}

// GenerateUpdatePacket runs a full local training cycle and returns an UpdatePacket.
// Raw patient data never leaves this function.
func GenerateUpdatePacket(globalModel *Model, cfg HospitalConfig) (*UpdatePacket, error) {
	if len(globalModel.Weights) == 0 {
		return nil, fmt.Errorf("hospital %s: global model has no weights", cfg.ID)
	}

	data, err := LoadCSVPartition(cfg.CSVPath, cfg.StartIdx, cfg.EndIdx)
	if err != nil {
		return nil, fmt.Errorf("hospital %s: load data: %w", cfg.ID, err)
	}

	trainedModel, loss := TrainLocalModel(globalModel, data, DefaultTrainConfig())

	packet := &UpdatePacket{
		Weights: trainedModel.FlatWeights(),
		Metadata: Metadata{
			HospitalID:   cfg.ID,
			DataSize:     len(data),
			Loss:         loss,
			RoundID:      cfg.RoundID,
			ModelVersion: cfg.ModelVersion,
			Timestamp:    time.Now().UTC().Format(time.RFC3339),
		},
	}
	return packet, nil
}

// ToJSON serialises the packet to indented JSON for logging and inspection.
func (p *UpdatePacket) ToJSON() (string, error) {
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}
