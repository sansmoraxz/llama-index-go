package llamaindexgo

// MessageRole is the role associated with a chat message
type MessageRole string

const (
	SYSTEM    MessageRole = "system"
	USER      MessageRole = "user"
	ASSISTANT MessageRole = "assistant"
	FUNCTION  MessageRole = "function"
	TOOL      MessageRole = "tool"
	CHATBOT   MessageRole = "chatbot"
)

// ChatMessage is a message in a chat
type ChatMessage struct {
	// Role of the message
	Role MessageRole
	// Content of the message
	Content string
	// Metadata contains any additional information about the message such as token counts, metrics, etc.
	Metadata map[string]interface{}
}

// String returns the string representation of the chat message
func (c *ChatMessage) String() string {
	return string(c.Role) + ": " + c.Content
}

// ChatMessages is a list of chat messages
type ChatMessages struct {
	// Messages is a list of chat messages
	Messages []ChatMessage
}

// String returns the string representation of the chat messages
func (c ChatMessages) String() string {
	var result string
	for _, message := range c.Messages {
		result += message.String() + "\n"
	}
	return result
}

type ChatResponse = ChatMessage
