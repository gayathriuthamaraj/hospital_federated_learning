package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
)

// Metadata carries everything the server needs to evaluate and weight
// a hospital's update.
type Metadata struct {
	HospitalID   string  `json:"hospital_id"`
	DataSize     int     `json:"data_size"`
	Loss         float64 `json:"loss"`
	RoundID      int     `json:"round_id"`
	ModelVersion int     `json:"model_version"`
}

// UpdatePacket is the complete hand-off from a hospital to the server.
type UpdatePacket struct {
	Weights  []float64 `json:"weights"`
	Metadata Metadata  `json:"metadata"`
}

// In-memory storage for received updates.
var (
	receivedUpdates []UpdatePacket
	mu               sync.Mutex
)

func main() {
	// POST /submit_update
	http.HandleFunc("/submit_update", handleSubmitUpdate)

	// GET /updates_count
	http.HandleFunc("/updates_count", handleUpdatesCount)

	port := ":8080"
	fmt.Printf("Server starting on port %s...\n", port)
	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}

func handleSubmitUpdate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var packet UpdatePacket
	if err := json.NewDecoder(r.Body).Decode(&packet); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	// Validate fields
	if len(packet.Weights) == 0 ||
		packet.Metadata.HospitalID == "" ||
		packet.Metadata.DataSize <= 0 ||
		packet.Metadata.RoundID < 0 ||
		packet.Metadata.ModelVersion < 0 {
		http.Error(w, "Missing or invalid required fields", http.StatusBadRequest)
		return
	}

	// Store valid packet
	mu.Lock()
	receivedUpdates = append(receivedUpdates, packet)
	count := len(receivedUpdates)
	mu.Unlock()

	log.Printf("Received update from %s", packet.Metadata.HospitalID)

	// Return success response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":         "success",
		"total_received": count,
	})
}

func handleUpdatesCount(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	mu.Lock()
	count := len(receivedUpdates)
	mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"count": count,
	})
}
