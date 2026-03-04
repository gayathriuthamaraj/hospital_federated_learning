package hospital

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"
)

// SecretKey is a shared key used for HMAC-style packet signing.
// In production this would be loaded from a secure vault or config.
const SecretKey = "federated_secret_2024"

// Metadata carries everything the server needs to evaluate and weight
// a hospital's update without seeing any raw patient data.
type Metadata struct {
	HospitalID   string  `json:"hospital_id"`
	DataSize     int     `json:"data_size"`
	Loss         float64 `json:"loss"`
	RoundID      int     `json:"round_id"`
	ModelVersion int     `json:"model_version"`
	Timestamp    int64   `json:"timestamp"`
}

// UpdatePacket is the complete hand-off from a hospital to the server.
// Weights is the flat serialisation produced by Model.FlatWeights().
// Raw patient data is never included.
type UpdatePacket struct {
	Weights   []float64 `json:"weights"`
	Metadata  Metadata  `json:"metadata"`
	Signature string    `json:"signature"`
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

// SignPacket computes a SHA256 signature over the metadata and stores it
// in the Signature field.  Hash = SHA256( json(metadata) + SecretKey ).
func (p *UpdatePacket) SignPacket() error {
	metaJSON, err := json.Marshal(p.Metadata)
	if err != nil {
		return fmt.Errorf("sign packet: marshal metadata: %w", err)
	}
	hash := sha256.Sum256(append(metaJSON, []byte(SecretKey)...))
	p.Signature = hex.EncodeToString(hash[:])
	return nil
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
			Timestamp:    time.Now().Unix(),
		},
	}

	// Sign the packet before returning.
	if err := packet.SignPacket(); err != nil {
		return nil, fmt.Errorf("hospital %s: %w", cfg.ID, err)
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
