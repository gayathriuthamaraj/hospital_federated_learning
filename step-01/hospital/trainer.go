package hospital

// TrainConfig holds local training hyperparameters.
type TrainConfig struct {
	Epochs       int
	LearningRate float64
	BatchSize    int
}

func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Epochs:       50,
		LearningRate: 0.05,
		BatchSize:    32,
	}
}

// TrainLocalModel runs mini-batch SGD and returns the trained model and final BCE loss.
// Gradients: dL/dw_i = (p-y)·x_i, dL/db = (p-y).
// The global model is never mutated — training operates on a deep copy.
func TrainLocalModel(model *Model, data []Sample, cfg TrainConfig) (*Model, float64) {
	// Deep copy so the original global model is unchanged.
	trained := &Model{
		Weights: make([]float64, len(model.Weights)),
		Bias:    model.Bias,
	}
	copy(trained.Weights, model.Weights)

	n := len(data)

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		for start := 0; start < n; start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > n {
				end = n
			}
			batch := data[start:end]

			dw := make([]float64, len(trained.Weights))
			db := 0.0

			for _, s := range batch {
				p := trained.Forward(s.Features)
				err := p - s.Label

				for i, xi := range s.Features {
					dw[i] += err * xi
				}
				db += err
			}

			batchLen := float64(len(batch))
			for i := range trained.Weights {
				trained.Weights[i] -= cfg.LearningRate * (dw[i] / batchLen)
			}
			trained.Bias -= cfg.LearningRate * (db / batchLen)
		}
	}

	finalLoss := trained.BinaryCrossEntropyLoss(data)
	return trained, finalLoss
}
