package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

const SecretKey = "federated_secret_2024"

type Metadata struct {
	HospitalID   string  `json:"hospital_id"`
	DataSize     int     `json:"data_size"`
	Loss         float64 `json:"loss"`
	RoundID      int     `json:"round_id"`
	ModelVersion int     `json:"model_version"`
	Timestamp    int64   `json:"timestamp"`
}

type UpdatePacket struct {
	Weights   []float64 `json:"weights"`
	Metadata  Metadata  `json:"metadata"`
	Signature string    `json:"signature"`
}

func signPacket(p *UpdatePacket) {
	metaJSON, _ := json.Marshal(p.Metadata)
	hash := sha256.Sum256(append(metaJSON, []byte(SecretKey)...))
	p.Signature = hex.EncodeToString(hash[:])
}

func submitUpdate(hospitalID string, roundID int, modelVersion int, weights []float64, dataSize int, loss float64) {
	packet := UpdatePacket{
		Weights: weights,
		Metadata: Metadata{
			HospitalID:   hospitalID,
			DataSize:     dataSize,
			Loss:         loss,
			RoundID:      roundID,
			ModelVersion: modelVersion,
			Timestamp:    time.Now().Unix(),
		},
	}
	signPacket(&packet)

	body, _ := json.Marshal(packet)
	resp, err := http.Post("http://localhost:8080/submit_update", "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Printf("[submit] ERROR: %v", err)
		return
	}
	respBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	fmt.Printf("[submit] %s (Round %d, Ver %d) | HTTP %d | %s", hospitalID, roundID, modelVersion, resp.StatusCode, string(respBody))
}

func main() {
	serverFlag := flag.String("server", "http://localhost:8080", "Server base URL")
	flag.Parse()

	baseURL := *serverFlag
	hospitalID := "H_SLOW_1"
	fmt.Printf("=== Slow-Hospital Client (Connecting to: %s) ===\n", baseURL)
	// Initial Round 0
	submitUpdate("H1", 0, 0, []float64{10, 20}, 100, 0.5)
	submitUpdate("H2", 0, 0, []float64{10, 20}, 100, 0.5)
	submitUpdate("H3", 0, 0, []float64{10, 20}, 100, 0.5)

	fmt.Println("Waiting for aggregation...")
	time.Sleep(1 * time.Second)

	fmt.Println("\n=== Starting Federation Round 1 with a Slow Hospital ===")
	// Round 1
	// H1: Normal client
	submitUpdate("H1", 1, 1, []float64{20, 30}, 100, 0.4)
	
	// H2: Normal client
	submitUpdate("H2", 1, 1, []float64{20, 30}, 100, 0.4)
	
	// H3: Slow client (simulates late submission or training on stale weights)
	// It uses ModelVersion 0 instead of 1. Still submits to RoundID 1 so it's accepted.
	submitUpdate("H3", 1, 0, []float64{15, 25}, 100, 0.4)

	fmt.Println("Waiting for Round 1 aggregation...")
	time.Sleep(1 * time.Second)
	fmt.Println("Done. Check server logs to see the staleness penalty applied to H3.")
}
