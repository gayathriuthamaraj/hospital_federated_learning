package hospital

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
)

// Sample is a single patient record.
// Features are min-max normalised to [0, 1] within the hospital's partition.
// Label: 1.0 = positive (cardiac event), 0.0 = negative.
type Sample struct {
	Features []float64
	Label    float64
}

// CSV column layout (Medicaldataset.csv):
//
//	0 Age | 1 Gender | 2 Heart rate | 3 Systolic BP | 4 Diastolic BP
//	5 Blood sugar | 6 CK-MB | 7 Troponin | 8 Result
const numFeatures = 8

// LoadCSVPartition reads rows [startIdx, endIdx) from the dataset (0-based,
// header excluded), normalises each feature per-partition, and returns samples.
// Raw data never leaves this function — callers receive only []Sample.
func LoadCSVPartition(path string, startIdx, endIdx int) ([]Sample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	if _, err := r.Read(); err != nil { // skip header
		return nil, fmt.Errorf("read header: %w", err)
	}

	var rawFeatures [][]float64
	var labels []float64

	for idx := 0; ; idx++ {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("row %d: %w", idx, err)
		}
		if idx < startIdx {
			continue
		}
		if idx >= endIdx {
			break
		}

		feats := make([]float64, numFeatures)
		for i := 0; i < numFeatures; i++ {
			v, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				return nil, fmt.Errorf("row %d col %d: %w", idx, i, err)
			}
			feats[i] = v
		}

		label := 0.0
		if record[numFeatures] == "positive" {
			label = 1.0
		}
		rawFeatures = append(rawFeatures, feats)
		labels = append(labels, label)
	}

	if len(rawFeatures) == 0 {
		return nil, fmt.Errorf("no rows in range [%d, %d)", startIdx, endIdx)
	}

	// Per-partition min-max normalisation: each hospital scales its own data.
	mins := make([]float64, numFeatures)
	maxs := make([]float64, numFeatures)
	copy(mins, rawFeatures[0])
	copy(maxs, rawFeatures[0])
	for _, row := range rawFeatures[1:] {
		for j, v := range row {
			if v < mins[j] {
				mins[j] = v
			}
			if v > maxs[j] {
				maxs[j] = v
			}
		}
	}

	samples := make([]Sample, len(rawFeatures))
	for i, row := range rawFeatures {
		norm := make([]float64, numFeatures)
		for j, v := range row {
			span := maxs[j] - mins[j]
			if span == 0 {
				norm[j] = 0
			} else {
				norm[j] = (v - mins[j]) / span
			}
		}
		samples[i] = Sample{Features: norm, Label: labels[i]}
	}
	return samples, nil
}
