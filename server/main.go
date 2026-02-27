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
	mu              sync.Mutex

	// Global model state
	globalWeights    []float64
	currentVersion   int
	aggregationMutex sync.Mutex
)

func main() {
	// POST /submit_update
	http.HandleFunc("/submit_update", handleSubmitUpdate)

	// GET /updates_count
	http.HandleFunc("/updates_count", handleUpdatesCount)

	// GET /global_model
	http.HandleFunc("/global_model", handleGetGlobalModel)

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
		packet.Metadata.DataSize <= 0 {
		http.Error(w, "Missing or invalid required fields", http.StatusBadRequest)
		return
	}

	// Store valid packet
	mu.Lock()
	receivedUpdates = append(receivedUpdates, packet)
	count := len(receivedUpdates)
	mu.Unlock()

	log.Printf("Received update from %s (Total: %d/3)", packet.Metadata.HospitalID, count)

	// Trigger aggregation if quorum met (3 updates)
	if count >= 3 {
		go aggregateUpdates()
	}

	// Return success response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":         "success",
		"total_received": count,
	})
}

func aggregateUpdates() {
	mu.Lock()
	defer mu.Unlock()

	if len(receivedUpdates) < 3 {
		return
	}

	log.Println("Quorum met. Starting aggregation...")

	// Initialise with weights from the first packet
	numWeights := len(receivedUpdates[0].Weights)
	sumWeights := make([]float64, numWeights)

	for _, packet := range receivedUpdates {
		for i, w := range packet.Weights {
			sumWeights[i] += w
		}
	}

	// Average weights
	numUpdates := float64(len(receivedUpdates))
	newWeights := make([]float64, numWeights)
	for i, sum := range sumWeights {
		newWeights[i] = sum / numUpdates
	}

	// Update global state
	aggregationMutex.Lock()
	globalWeights = newWeights
	currentVersion++
	aggregationMutex.Unlock()

	// Clear received updates for next round
	receivedUpdates = nil

	log.Printf("Aggregation successful. New Model Version: %d", currentVersion)
}

func handleGetGlobalModel(w http.ResponseWriter, r *http.Request) {
	aggregationMutex.Lock()
	defer aggregationMutex.Unlock()

	if globalWeights == nil {
		http.Error(w, "Global model not yet initialised", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"weights":       globalWeights,
		"model_version": currentVersion,
	})
}

func handleUpdatesCount(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	count := len(receivedUpdates)
	mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"count": count,
	})
}
