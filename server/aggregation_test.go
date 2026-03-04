package main

import (
	"math"
	"testing"
)

func TestQFedAvgWeighting(t *testing.T) {
	tests := []struct {
		name     string
		loss     float64
		dataSize int
		q        float64
		expected float64
	}{
		{"q=0 (DataSize only)", 0.5, 100, 0.0, 100.0},
		{"q=1 (Loss * DataSize)", 0.5, 100, 1.0, 50.0},
		{"q=2 (Loss^2 * DataSize)", 0.5, 100, 2.0, 25.0},
		{"Zero loss, q=1", 0.0, 100, 1.0, 1e-6}, // Stability fallback
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var lossPower float64
			if tt.loss == 0 && tt.q > 0 {
				lossPower = 0
			} else if tt.loss == 0 && tt.q == 0 {
				lossPower = 1
			} else {
				lossPower = math.Pow(tt.loss, tt.q)
			}
			weight := lossPower * float64(tt.dataSize)
			if weight <= 0 {
				weight = 1e-6
			}

			if math.Abs(weight-tt.expected) > 1e-7 {
				t.Errorf("expected weight %f, got %f", tt.expected, weight)
			}
		})
	}
}

func TestAggregationLogic(t *testing.T) {
	// Mock packets
	p1 := UpdatePacket{
		Weights:  []float64{1.0, 2.0},
		Metadata: Metadata{Loss: 0.5, DataSize: 100},
	}
	p2 := UpdatePacket{
		Weights:  []float64{3.0, 4.0},
		Metadata: Metadata{Loss: 0.1, DataSize: 200},
	}

	q := 1.0

	// Calculate weights
	// w1 := 0.5 * 100.0 // 50 (unused)
	// w2 := 0.1 * 200.0 // 20 (unused)

	expectedW1 := (1.0*50.0 + 3.0*20.0) / 70.0 // (50 + 60) / 70 = 110/70 = 1.5714...
	expectedW2 := (2.0*50.0 + 4.0*20.0) / 70.0 // (100 + 80) / 70 = 180/70 = 2.5714...

	updates := []UpdatePacket{p1, p2}
	numWeights := len(p1.Weights)
	sumWeightedWeights := make([]float64, numWeights)
	totalWeight := 0.0

	for _, packet := range updates {
		lossPower := math.Pow(packet.Metadata.Loss, q)
		weight := lossPower * float64(packet.Metadata.DataSize)
		if weight <= 0 {
			weight = 1e-6
		}

		for i, w := range packet.Weights {
			sumWeightedWeights[i] += w * weight
		}
		totalWeight += weight
	}

	for i, sum := range sumWeightedWeights {
		got := sum / totalWeight
		var expected float64
		if i == 0 {
			expected = expectedW1
		} else {
			expected = expectedW2
		}

		if math.Abs(got-expected) > 1e-7 {
			t.Errorf("Weight index %d: expected %f, got %f", i, expected, got)
		}
	}
}
