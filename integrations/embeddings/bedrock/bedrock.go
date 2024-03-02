package bedrock

import (
	"context"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	llamaindexgo "github.com/sansmoraxz/llama-index-go"
)

const (
	PROVIDER_AMAZON = "amazon"
	PROVIDER_COHERE = "cohere"
)

type Bedrock struct {
	ModelId   string
	BatchSize int
	Client    *bedrockruntime.Client
}

func (b *Bedrock) ModelName() string {
	return b.ModelId
}

func (b *Bedrock) splitModelId() (provider string, modelName string) {
	x := strings.Split(b.ModelId, ".")
	return x[0], x[1]
}


// EmbedQuery implements llamaindexgo.TextEmbedder.
func (b *Bedrock) EmbedQuery(ctx context.Context, query string) ([]float64, error) {
	provider, modelName := b.splitModelId()
	if provider == PROVIDER_COHERE {
		resp, err := FetchCohereTextEmbeddings(ctx, b.Client, modelName, []string{query}, COHERE_INPUT_TYPE_QUERY)
		if err != nil {
			return nil, err
		}
		return resp[0], nil
	} else if provider == PROVIDER_AMAZON {
		return FetchAmazonTextEmbeddings(ctx, b.Client, modelName, query)
	}
	return nil, fmt.Errorf("unknown provider: %s", provider)
}

// EmbedText implements llamaindexgo.TextEmbedder.
func (b *Bedrock) EmbedText(ctx context.Context, text string) ([]float64, error) {
	provider, modelName := b.splitModelId()
	if provider == PROVIDER_COHERE {
		resp, err := FetchCohereTextEmbeddings(ctx, b.Client, modelName, []string{text}, COHERE_INPUT_TYPE_TEXT)
		if err != nil {
			return nil, err
		}
		return resp[0], nil
	} else if provider == PROVIDER_AMAZON {
		return FetchAmazonTextEmbeddings(ctx, b.Client, modelName, text)
	}
	return nil, fmt.Errorf("unknown provider: %s", provider)
}

// EmbedTexts implements llamaindexgo.TextEmbedder.
func (b *Bedrock) EmbedTexts(ctx context.Context, texts []string) ([][]float64, error) {
	provider, modelName := b.splitModelId()
	if provider == PROVIDER_COHERE {
		return FetchCohereTextEmbeddings(ctx, b.Client, modelName, texts, COHERE_INPUT_TYPE_TEXT)
	} else {
		resp := make([][]float64, len(texts))
		for _, text := range texts {
			x, err := b.EmbedText(ctx, text)
			if err != nil {
				return nil, err
			}
			resp = append(resp, x)
		}
		return resp, nil
	}
}

// EmbedQueries implements llamaindexgo.TextEmbedder.
func (b *Bedrock) EmbedQueries(ctx context.Context, queries []string) ([][]float64, error) {
	provider, modelName := b.splitModelId()
	if provider == PROVIDER_COHERE {
		return FetchCohereTextEmbeddings(ctx, b.Client, modelName, queries, COHERE_INPUT_TYPE_QUERY)
	} else {
		resp := make([][]float64, len(queries))
		for _, query := range queries {
			x, err := b.EmbedQuery(ctx, query)
			if err != nil {
				return nil, err
			}
			resp = append(resp, x)
		}
		return resp, nil
	}
}



// check that Bedrock implements llamaindexgo.TextEmbedder
var _ llamaindexgo.TextEmbedder = &Bedrock{}
