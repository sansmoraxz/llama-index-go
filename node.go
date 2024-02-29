package llamaindexgo


type Document interface {
	Node
	ID() string
}

// Node represents a node in the graph
type Node interface {

	// Content returns the raw content of the node
	Content() []byte

	// Metadata returns the associated metadata of the node
	Metadata() map[string]interface{}

	// Text returns the text value of the node
	// If the node does not have a text value, it returns an error
	Text() (*string, error)

	// Embedding returns the embedding value of the node
	// If the node does not have an embedding, it returns an error
	Embedding() ([]float, error)

	// Parent returns the parent of the node
	// If the node is the root, it returns nil
	Parent() (*Node, error)

	// Children returns the children of the node
	// If the node is a leaf, it returns nil
	Children() ([]*Node, error)
}


// NodeParser generates Nodes from Documents
type NodeParser interface {

	// GetNodesFromDocuments returns a channel of Nodes from a list of Documents
	// It returns an error if the parsing fails for any reason
	GetNodesFromDocuments(...Document) (<-chan Node, error)

	// Transform returns a list of Nodes transformed by the given function
	Tranform(func(Node) (Node, error), ...Node) (<-chan Node, error)
}
