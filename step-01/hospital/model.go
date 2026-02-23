package hospital

import (
	"math"
	"math/rand"
)

// InputSize must match the number of feature columns in the dataset.
const InputSize = 8

// Model is a logistic regression classifier: sigmoid( dot(Weights, x) + Bias ).
// FlatWeights serialisation format: [w0..w7, bias] — stable across steps.
type Model struct {
	Weights []float64 // length == InputSize
	Bias    float64
}

// NewModel returns a reproducibly initialised model (seed 42).
// All hospitals start from identical weights at round 0.
func NewModel() *Model {
	rng := rand.New(rand.NewSource(42))
	weights := make([]float64, InputSize)
	for i := range weights {
		weights[i] = rng.Float64()*0.1 - 0.05
	}
	return &Model{
		Weights: weights,
		Bias:    0.0,
	}
}

// NewModelFromWeights reconstructs a model from FlatWeights output.
// Convention: last element is the bias term.
func NewModelFromWeights(flat []float64) *Model {
	n := len(flat) - 1
	weights := make([]float64, n)
	copy(weights, flat[:n])
	return &Model{
		Weights: weights,
		Bias:    flat[n],
	}
}

// FlatWeights returns [w0..wN, bias] — the serialisation format used in UpdatePacket.
func (m *Model) FlatWeights() []float64 {
	flat := make([]float64, len(m.Weights)+1)
	copy(flat, m.Weights)
	flat[len(m.Weights)] = m.Bias
	return flat
}

// Forward computes the predicted probability for a single sample.
func (m *Model) Forward(x []float64) float64 {
	z := m.Bias
	for i, xi := range x {
		z += m.Weights[i] * xi
	}
	return sigmoid(z)
}

// BinaryCrossEntropyLoss computes mean BCE loss: -mean( y·log(p) + (1-y)·log(1-p) ).
func (m *Model) BinaryCrossEntropyLoss(data []Sample) float64 {
	total := 0.0
	for _, s := range data {
		p := m.Forward(s.Features)
		p = clamp(p, 1e-9, 1-1e-9)
		total += -(s.Label*math.Log(p) + (1-s.Label)*math.Log(1-p))
	}
	return total / float64(len(data))
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
