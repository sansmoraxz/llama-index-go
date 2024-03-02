package bedrock

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const (
	/*
	MODEL_COHERE_EN is the model id for the cohere english embeddings.

	  ModelDimensions := 1024
	  MaxTokens := 512
	  Languages := 1
	*/
	MODEL_COHERE_EN = "cohere.embed-english-v3"

	/*
	MODEL_COHERE_MULTI is the model id for the cohere multilingual embeddings.

	  ModelDimensions := 1024
	  MaxTokens:= 512
	  Languages := 108
	*/
	MODEL_COHERE_MULTI = "cohere.embed-multilingual-v3"
)

const (
	COHERE_INPUT_TYPE_TEXT = "search_document"
	COHERE_INPUT_TYPE_QUERY = "search_query"
)

type cohereTextEmbeddingsInput struct {
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
}

type cohereTextEmbeddingsOutput struct{
	ResponseType string `json:"response_type"`
	Embeddings   [][]float64 `json:"embeddings"`
}

func FetchCohereTextEmbeddings(
	ctx context.Context,
	client *bedrockruntime.Client,
	modelId string,
	inputs []string,
	inputType string) ([][]float64, error) {

	var err error

	bodyStruct := cohereTextEmbeddingsInput{
		Texts:     inputs,
		InputType: inputType,
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
		return nil, err
	}
	meta := result.ResultMetadata
	fmt.Println(meta)
	var response cohereTextEmbeddingsOutput
	fmt.Println(string(result.Body))
	err = json.Unmarshal(result.Body, &response)
	if err != nil {
		return nil, err
	}

	return response.Embeddings, nil
}
