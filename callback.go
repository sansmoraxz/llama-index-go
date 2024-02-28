package llamaindexgo

type CallbackEventHandler interface {
	// OnEventStart runs when an event starts
	OnEventStart(event CBEvent) string
	// OnEventEnd runs when an event ends
	OnEventEnd(event CBEvent) string

	// IgnoredStartEvents returns a list of events that should be ignored when calling OnEventStart
	IgnoredStartEvents() []Event
	// IgnoredEndEvents returns a list of events that should be ignored when calling OnEventEnd
	IgnoredEndEvents() []Event
}

type CallbackTraceHandler interface {
	// StartTrace starts tracing events
	StartTrace(id string)
	// EndTrace ends tracing events
	EndTrace(id string)
}

type CallbackHandler interface {
	CallbackEventHandler
	CallbackTraceHandler
}

type CallbackManager struct {
	// Handlers is a list of callback handlers that will be called when an event starts or ends
	Handlers           []CallbackHandler
	ignoredStartEvents []Event
	ignoredEndEvents   []Event

	// global_stack_trace is a list of strings that represent the stack trace of the current event
	global_stack_trace []string
	// trace_map is a dictionary that maps parent event IDs to a list of child event IDs
	trace_map map[string][]string
}
