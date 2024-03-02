package bedrock

import (
	"context"
	"encoding/json"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const (
	MODEL_TITAN_EMBED_G1 = "amazon.titan-embed-text-v1"
	MODEL_TITAN_MULTIMODAL_EMBED_G1 = "amazon.titan-embed-image-v1"
)

type amazonEmbeddingsInput struct {
	InputText string `json:"inputText"`
}

type amazonMultiModalEmbeddingsInput struct {
	InputText string `json:"inputText"`
	InputImage string `json:"inputImage"`
}

type amazonEmbeddingsOutput struct {
	Embedding          []float64 `json:"embedding"`
	// InputTextTokenCount int       `json:"inputTextTokenCount"`
}

func FetchAmazonTextEmbeddings(ctx context.Context,
	client *bedrockruntime.Client,
	modelId string,
	text string) ([]float64, error) {

	var err error

	bodyStruct := amazonEmbeddingsInput{
		InputText: text,
	}
	body, err := json.Marshal(bodyStruct)
	if err != nil {
		return nil, err
	}
	modelInput := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelId),
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
		Body:        body,
	}

	result, err := client.InvokeModel(ctx, modelInput)
	if err != nil {
		panic(err)
	}
	var response amazonEmbeddingsOutput
	err = json.Unmarshal(result.Body, &response)
	if err != nil {
		return nil, err
	}

	return response.Embedding, nil
}

func FetchAmazonMultiModalEmbeddings(ctx context.Context,
	client *bedrockruntime.Client,
	modelId string,
	text string,
	image string) ([]float64, error) {

	var err error

	bodyStruct := amazonMultiModalEmbeddingsInput{
		InputText: text,
		InputImage: image,
	}
	body, err := json.Marshal(bodyStruct)
	if err != nil {
		return nil, err
	}
	modelInput := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelId),
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
		Body:        body,
	}

	result, err := client.InvokeModel(ctx, modelInput)
	if err != nil {
		panic(err)
	}
	var response amazonEmbeddingsOutput
	err = json.Unmarshal(result.Body, &response)
	if err != nil {
		return nil, err
	}

	return response.Embedding, nil
}
