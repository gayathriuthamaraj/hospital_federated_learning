package main

import (
	"fmt"
	"log"

	"step01/hospital"
)

// csvPath is relative to the step-01 directory.
// The dataset has 1320 rows (header excluded), split evenly across 3 hospitals.
const csvPath = "../Medicaldataset.csv"

func main() {
	fmt.Println("=== Federated Hospital Learning System — Step 01 ===")
	fmt.Println("Dataset: Medicaldataset.csv | Hospitals: 3 | Round: 0")
	fmt.Println()

	// All hospitals start from the same global model weights.
	// In later steps the server distributes this; here we construct it once.
	globalModel := hospital.NewModel()

	// 1320 rows split into three equal partitions of 440 rows each.
	// Each hospital trains only on its own partition — no data is shared.
	hospitals := []hospital.HospitalConfig{
		{ID: "H1", RoundID: 0, ModelVersion: 0, CSVPath: csvPath, StartIdx: 0, EndIdx: 440},
		{ID: "H2", RoundID: 0, ModelVersion: 0, CSVPath: csvPath, StartIdx: 440, EndIdx: 880},
		{ID: "H3", RoundID: 0, ModelVersion: 0, CSVPath: csvPath, StartIdx: 880, EndIdx: 1320},
	}

	for _, cfg := range hospitals {
		fmt.Printf("--- Hospital %s (rows %d–%d) ---\n", cfg.ID, cfg.StartIdx, cfg.EndIdx-1)

		packet, err := hospital.GenerateUpdatePacket(globalModel, cfg)
		if err != nil {
			log.Fatalf("hospital %s: %v", cfg.ID, err)
		}

		jsonStr, err := packet.ToJSON()
		if err != nil {
			log.Fatalf("hospital %s: serialise: %v", cfg.ID, err)
		}

		fmt.Println(jsonStr)
		fmt.Println()
	}

	fmt.Println("=== Checkpoint passed: 3 hospitals produced update packets ===")
	fmt.Println("Confirm that 'loss' values differ across H1, H2, H3.")
}
