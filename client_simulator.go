package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// Metadata carries everything the server needs to evaluate a hospital's update.
type Metadata struct {
	HospitalID   string  `json:"hospital_id"`
	DataSize     int     `json:"data_size"`
	Loss         float64 `json:"loss"`
	RoundID      int     `json:"round_id"`
	ModelVersion int     `json:"model_version"`
}

// UpdatePacket is the complete payload sent from a hospital to the server.
type UpdatePacket struct {
	Weights  []float64 `json:"weights"`
	Metadata Metadata  `json:"metadata"`
}

// GlobalModelResponse is the shape returned by GET /global_model.
type GlobalModelResponse struct {
	Weights      []float64 `json:"weights"`
	ModelVersion int       `json:"model_version"`
}

// LocalClientState tracks the simulated client's current model knowledge.
// ModelVersion -1 means the client has never received a model from the server.
type LocalClientState struct {
	ModelVersion int
	Weights      []float64
}

// fetchGlobalModel calls GET /global_model on the server.
// Returns (response, true) on success, or (zero value, false) when no model is
// available yet (404) or any other error occurs.
func fetchGlobalModel(baseURL string) (GlobalModelResponse, bool) {
	resp, err := http.Get(baseURL + "/global_model")
	if err != nil {
		log.Printf("[model-sync] ERROR: could not reach server: %v", err)
		return GlobalModelResponse{}, false
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Println("[model-sync] Server has no global model yet — skipping sync.")
		return GlobalModelResponse{}, false
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("[model-sync] Unexpected status %d: %s", resp.StatusCode, string(body))
		return GlobalModelResponse{}, false
	}

	var model GlobalModelResponse
	if err := json.NewDecoder(resp.Body).Decode(&model); err != nil {
		log.Printf("[model-sync] ERROR: failed to decode response: %v", err)
		return GlobalModelResponse{}, false
	}

	return model, true
}

// syncModel compares the server's model version against the client's local state.
// If the server version is strictly newer, local weights and version are replaced.
// Returns true if the local model was updated.
func syncModel(baseURL string, state *LocalClientState) bool {
	serverModel, ok := fetchGlobalModel(baseURL)
	if !ok {
		return false
	}

	if serverModel.ModelVersion > state.ModelVersion {
		log.Printf("[model-sync] UPDATE: server version %d > local version %d — replacing local weights.",
			serverModel.ModelVersion, state.ModelVersion)
		state.ModelVersion = serverModel.ModelVersion
		state.Weights = serverModel.Weights
		log.Printf("[model-sync] Local model updated to version %d | weights: %v",
			state.ModelVersion, state.Weights)
		return true
	}

	log.Printf("[model-sync] Local model (version %d) is already up-to-date.", state.ModelVersion)
	return false
}

func main() {
	baseURL := "http://localhost:8080"
	fmt.Println("=== Federated Client Simulator ===")

	// Initialise local state. ModelVersion -1 signals "never synced from server".
	state := &LocalClientState{
		ModelVersion: -1,
		Weights:      nil,
	}

	// ── Phase 1: Pre-round model sync ────────────────────────────────────────
	// Download the global model before submitting so we train on the latest
	// federated weights. On the very first run the server has nothing yet and
	// syncModel will log a skip message.
	fmt.Println("\n[Phase 1] Checking for global model before submitting...")
	syncModel(baseURL, state)

	// Map local model version to the round we will submit to.
	// Convention: round N uses model version N (version 0 = no prior aggregation).
	roundID := state.ModelVersion
	if roundID < 0 {
		roundID = 0
	}

	// ── Phase 2: Submit hospital updates ─────────────────────────────────────
	fmt.Printf("\n[Phase 2] Submitting round %d updates (model_version=%d)...\n", roundID, roundID)

	for i := 1; i <= 3; i++ {
		// Use downloaded global weights as the base if available; otherwise fall
		// back to hard-coded demonstration values so the simulator runs standalone.
		var weights []float64
		if state.Weights != nil {
			// Simulate local gradient updates applied on top of the global model.
			weights = make([]float64, len(state.Weights))
			for j, w := range state.Weights {
				weights[j] = w + float64(i)*0.1
			}
		} else {
			weights = []float64{float64(i * 10), float64(i * 20)}
		}

		packet := UpdatePacket{
			Weights: weights,
			Metadata: Metadata{
				HospitalID:   fmt.Sprintf("H%d", i),
				DataSize:     100 * i,
				Loss:         0.5 / float64(i),
				RoundID:      roundID,
				ModelVersion: roundID,
			},
		}

		body, _ := json.Marshal(packet)
		resp, err := http.Post(baseURL+"/submit_update", "application/json", bytes.NewBuffer(body))
		if err != nil {
			log.Printf("[submit] ERROR reaching server: %v", err)
			return
		}
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		fmt.Printf("[submit] H%d — weights: %v | model_version: %d | HTTP %d | %s",
			i, weights, roundID, resp.StatusCode, string(respBody))
	}

	// ── Phase 3: Wait for aggregation, then pull the new global model ─────────
	fmt.Println("\n[Phase 3] Waiting for server aggregation...")
	time.Sleep(300 * time.Millisecond)

	fmt.Println("[Phase 3] Pulling updated global model after aggregation...")
	if syncModel(baseURL, state) {
		log.Printf("[model-sync] Model updated to version %d. Client will train on version %d in the next round.",
			state.ModelVersion, state.ModelVersion)
	} else {
		log.Println("[model-sync] Could not retrieve updated model — will retry next round.")
	}

	fmt.Printf("\n=== Final client state: ModelVersion=%d | Weights=%v ===\n",
		state.ModelVersion, state.Weights)
}
