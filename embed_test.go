package llamaindexgo

import (
	"context"
	"testing"
)

type testVectorSimilarityType struct {
	mode     SimilarityMode
	vector1  []float
	vector2  []float
	expected float
	isError  bool
}

var testCases = []testVectorSimilarityType{
	// Test cases that pass
	{
		mode:     COSINE,
		vector1:  []float{1, 2, 3},
		vector2:  []float{1, 2, 3},
		expected: 1.0,
		isError:  false,
	},
	{
		mode:     COSINE,
		vector1:  []float{1, 2, 3},
		vector2:  []float{3, 2, 1},
		expected: 0.7142857142857143,
		isError:  false,
	},
	{
		mode:     DOT,
		vector1:  []float{1, 2, 3},
		vector2:  []float{1, 2, 3},
		expected: 14.0,
		isError:  false,
	},
	{
		mode:     DOT,
		vector1:  []float{1, 2, 3},
		vector2:  []float{3, 2, 1},
		expected: 10.0,
		isError:  false,
	},

	{
		mode:     EUCLIDEAN,
		vector1:  []float{1, 2, 3},
		vector2:  []float{1, 2, 3},
		expected: 0.0,
		isError:  false,
	},
	{
		mode:     EUCLIDEAN,
		vector1:  []float{1, 2, 3},
		vector2:  []float{3, 2, 1},
		expected: 2.8284271247461903,
		isError:  false,
	},

	// Empty vectors
	{
		mode:     COSINE,
		vector1:  []float{},
		vector2:  []float{},
		expected: 0.0,
		isError:  true,
	},
	{
		mode:     DOT,
		vector1:  []float{},
		vector2:  []float{},
		expected: 0.0,
		isError:  true,
	},
	{
		mode:     EUCLIDEAN,
		vector1:  []float{},
		vector2:  []float{},
		expected: 0.0,
		isError:  true,
	},

	// Different length
	{
		mode:     COSINE,
		vector1:  []float{1, 2, 3},
		vector2:  []float{1, 2, 3, 4},
		expected: 0.0,
		isError:  true,
	},
	{
		mode:     DOT,
		vector1:  []float{1, 2, 3},
		vector2:  []float{1, 2, 3, 4},
		expected: 0.0,
		isError:  true,
	},
	{
		mode:     EUCLIDEAN,
		vector1:  []float{1, 2, 3},
		vector2:  []float{1, 2, 3, 4},
		expected: 0.0,
		isError:  true,
	},
}

func TestSimilarity(t *testing.T) {
	for _, test := range testCases {
		got, err := Similarity(context.Background(), test.mode, test.vector1, test.vector2)
		if test.isError {
			if err == nil {
				t.Errorf("Expected error but got nil")
			}
		} else {
			if err != nil {
				t.Errorf("Expected nil but got error: %v", err)
			}
			if got != test.expected {
				t.Errorf("Expected %v but got %v", test.expected, got)
			}
		}
	}
}
