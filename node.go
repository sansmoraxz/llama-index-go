package llamaindexgo

import "context"

// Node represents a node in the graph
type Node interface {

	// ID returns the unique identifier of the node
	ID() string

	// Content returns the raw content of the node
	Content() []byte

	// Metadata returns the associated metadata of the node
	Metadata() map[string]any

	// Text returns the text value of the node
	// If the node does not have a text value, it returns an error
	Text() (string, error)

	// Embedding returns the embedding value of the node
	// If the node does not have an embedding, it returns an error
	Embedding() ([]float, error)

	// ParentID returns the unique identifier of the parent node
	// return an empty string if the node does not have a parent
	ParentID() (string, error)

	// ChildrenIDs returns the unique identifiers of the children nodes
	// return an empty slice if the node does not have children
	ChildrenIDs() ([]string, error)
}

type Tokenizer func(ctx context.Context, text string) ([]string, error)


// NodeParser generates Nodes from Documents
type NodeParser interface {

	// GetNodesFromDocuments returns a list of Nodes from the given list of Documents
	GetNodesFromDocuments(ctx context.Context, docs []Node) ([]Node, error)
}

type TextSplitter interface {
	SplitText(ctx context.Context, text string) ([]string, error)
	SplitTexts(ctx context.Context, texts []string) ([]string, error)
}
