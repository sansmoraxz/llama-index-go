package events

import "time"

type Event string

const (
	// CHUNKING logs for the before and after of text splitting.
	CHUNKING Event = "chunking"
	// NODE_PARSING logs for the documents and the nodes that they are parsed into.
	NODE_PARSING Event = "node_parsing"
	// EMBEDDING logs for the number of texts embedded.
	EMBEDDING Event = "embedding"
	// LLM logs for the template and response of LLM calls.
	LLM Event = "llm"
	// QUERY keeps track of the start and end of each query.
	QUERY Event = "query"
	// RETRIEVE logs for the nodes retrieved for a query.
	RETRIEVE Event = "retrieve"
	// SYNTHESIZE logs for the result for synthesize calls.
	SYNTHESIZE Event = "synthesize"
	// TREE logs for the summary and level of summaries generated.
	TREE Event = "tree"
	// SUB_QUESTION logs for a generated sub question and answer.
	SUB_QUESTION Event = "sub_question"
	// TEMPLATE logs for the template and response of LLM calls.
	TEMPLATING Event = "templating"
	// FUNCTION_CALL logs for the function call and response.
	FUNCTION_CALL Event = "function_call"
	// RERANKING logs for the reranking and response.
	RERANKING Event = "reranking"
	// ERROR logs for the error and response.
	ERROR Event = "error"
	// AGENT_STEP logs for the agent step and response.
	AGENT_STEP Event = "agent_step"
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

type CallbackFunc func(CBEvent)

// GetLeafEvents returns a list of leaf events.
func GetLeafEvents() []Event {
	return []Event{CHUNKING, LLM, EMBEDDING}
}

// IsLeafEvent returns true if the event is a leaf event.
func IsLeafEvent(event Event) bool {
	switch event {
	case CHUNKING, LLM, EMBEDDING:
		return true
	default:
		return false
	}
}
