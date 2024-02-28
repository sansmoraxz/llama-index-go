package llamaindexgo

import "time"

type Event string

const (
	// CHUNKING_EVENT logs for the before and after of text splitting.
	CHUNKING_EVENT Event = "chunking"
	// NODE_PARSING_EVENT logs for the documents and the nodes that they are parsed into.
	NODE_PARSING_EVENT Event = "node_parsing"
	// EMBEDDING_EVENT logs for the number of texts embedded.
	EMBEDDING_EVENT Event = "embedding"
	// LLM_EVENT logs for the template and response of LLM calls.
	LLM_EVENT Event = "llm"
	// QUERY_EVENT keeps track of the start and end of each query.
	QUERY_EVENT Event = "query"
	// RETRIEVE_EVENT logs for the nodes retrieved for a query.
	RETRIEVE_EVENT Event = "retrieve"
	// SYNTHESIZE_EVENT logs for the result for synthesize calls.
	SYNTHESIZE_EVENT Event = "synthesize"
	// TREE_EVENT logs for the summary and level of summaries generated.
	TREE_EVENT Event = "tree"
	// SUB_QUESTION_EVENT logs for a generated sub question and answer.
	SUB_QUESTION_EVENT Event = "sub_question"
	// TEMPLATE_EVENT logs for the template and response of LLM calls.
	TEMPLATING_EVENT Event = "templating"
	// FUNCTION_CALL_EVENT logs for the function call and response.
	FUNCTION_CALL_EVENT Event = "function_call"
	// RERANKING_EVENT logs for the reranking and response.
	RERANKING_EVENT Event = "reranking"
	// ERROR_EVENT logs for the error and response.
	ERROR_EVENT Event = "error"
	// AGENT_STEP_EVENT logs for the agent step and response.
	AGENT_STEP_EVENT Event = "agent_step"
)

// CBEvent is a callback event.
type CBEvent struct {
	// Type is the type of event.
	Type Event
	// Payload is the data associated with the event.
	Payload interface{}
	// Time is the time the event occurred.
	Time time.Time
	// ID is the unique identifier for the event.
	ID string
	// ParentID is the unique identifier for the parent event.
	ParentID string
}


// GetLeafEvents returns a list of leaf events.
func GetLeafEvents() []Event {
	return []Event{CHUNKING_EVENT, LLM_EVENT, EMBEDDING_EVENT}
}


// IsLeafEvent returns true if the event is a leaf event.
func IsLeafEvent(event Event) bool {
	switch event {
	case CHUNKING_EVENT, LLM_EVENT, EMBEDDING_EVENT:
		return true
	default:
		return false
	}
}
