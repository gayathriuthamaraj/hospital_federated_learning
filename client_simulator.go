package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Metadata struct {
	HospitalID   string  `json:"hospital_id"`
	DataSize     int     `json:"data_size"`
	Loss         float64 `json:"loss"`
	RoundID      int     `json:"round_id"`
	ModelVersion int     `json:"model_version"`
}

type UpdatePacket struct {
	Weights  []float64 `json:"weights"`
	Metadata Metadata  `json:"metadata"`
}

func main() {
	baseUrl := "http://localhost:8080"
	fmt.Println("Simulating 3 Hospital Submissions...")

	// Submit 3 updates with different weights
	// H1: [10, 20]
	// H2: [20, 40]
	// H3: [30, 60]
	// Average should be: [20, 40]
	
	for i := 1; i <= 3; i++ {
		packet := UpdatePacket{
			Weights: []float64{float64(i * 10), float64(i * 20)},
			Metadata: Metadata{
				HospitalID:   fmt.Sprintf("H%d", i),
				DataSize:     100,
				Loss:         0.5,
				RoundID:      0,
				ModelVersion: 0,
			},
		}

		body, _ := json.Marshal(packet)
		resp, err := http.Post(baseUrl+"/submit_update", "application/json", bytes.NewBuffer(body))
		if err != nil {
			fmt.Printf("Failed to contact server: %v\n", err)
			return
		}
		defer resp.Body.Close()
		fmt.Printf("Submitted H%d (Weights: %v), Status: %d\n", i, packet.Weights, resp.StatusCode)
	}

	// Wait for async aggregation on server
	time.Sleep(200 * time.Millisecond)

	fmt.Println("\nFetching Global Model...")
	resp, err := http.Get(baseUrl + "/global_model")
	if err != nil {
		fmt.Printf("Failed: %v\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		var result map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&result)
		fmt.Printf("Global Model Version: %v\n", result["model_version"])
		fmt.Printf("Aggregated Weights (Average): %v\n", result["weights"])
	} else {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("Error: %d, %s\n", resp.StatusCode, string(body))
	}
}
