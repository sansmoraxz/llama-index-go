package llamaindexgo

import (
	"context"
	"fmt"
	"math"
)

type float = float64

type Embedding = []float

type SimilarityMode = int

const (
	COSINE SimilarityMode = iota
	DOT
	EUCLIDEAN
)

type Embedder interface {
	Generator
	// EmbedQuery generates an embedding for the given query
	EmbedQuery(context.Context, string) (Embedding, error)

	// EmbedQueries generates embeddings for the given queries
	EmbedQueries(context.Context, []string) ([]Embedding, error)

	// EmbedText generates an embedding for the given text
	EmbedText(context.Context, string) (Embedding, error)

	// EmbedTexts generates embeddings for the given texts
	EmbedTexts(context.Context, []string) ([]Embedding, error)
}

func cosineSimilarity(x, y []float) (float, error) {
	if len(x) == 0 || len(y) == 0 {
		return 0.0, fmt.Errorf("vectors must not be empty")
	}
	if len(x) != len(y) {
		return 0.0, fmt.Errorf("vectors must be of the same length")
	}
	var dot, nx, ny float

	for i := range x {
		nx += x[i] * x[i]
		ny += y[i] * y[i]
		dot += x[i] * y[i]
	}

	return dot / (math.Sqrt(nx) * math.Sqrt(ny)), nil
}

func dotProduct(x, y []float) (float, error) {
	if len(x) == 0 || len(y) == 0 {
		return 0.0, fmt.Errorf("vectors must not be empty")
	}
	if len(x) != len(y) {
		return 0.0, fmt.Errorf("vectors must be of the same length")
	}
	var dot float

	for i := range x {
		dot += x[i] * y[i]
	}
	return dot, nil
}

func euclideanDistance(x, y []float) (float, error) {
	if len(x) == 0 || len(y) == 0 {
		return 0.0, fmt.Errorf("vectors must not be empty")
	}
	if len(x) != len(y) {
		return 0.0, fmt.Errorf("vectors must be of the same length")
	}
	var sum float

	for i := range x {
		sum += (x[i] - y[i]) * (x[i] - y[i])
	}

	return math.Sqrt(sum), nil
}

// Similarity returns the similarity between two embeddings
func Similarity(ctx context.Context, mode SimilarityMode, e1 Embedding, e2 Embedding) (float, error) {
	ch := make(chan float)
	errCh := make(chan error)

	go func() {
		defer close(ch)
		var s float
		var err error
		switch mode {
		case COSINE:
			s, err = cosineSimilarity(e1, e2)
		case DOT:
			s, err = dotProduct(e1, e2)
		case EUCLIDEAN:
			s, err = euclideanDistance(e1, e2)
		default:
			err = fmt.Errorf("unsupported similarity mode")
		}
		if err != nil {
			errCh <- err
			return
		}
		ch <- s
	}()

	select {
	case <-ctx.Done():
		return 0.0, fmt.Errorf("context cancelled")
	case err := <-errCh:
		return 0.0, err
	case sim := <-ch:
		return sim, nil
	}
}
