package nodes

import (
	llamaindexgo "github.com/sansmoraxz/llama-index-go"
)

type TextNode struct {
	id        string
	content   []byte
	embedding []float64
	metadata  map[string]any
	childrenIds  []string
	parentId    string
}

// ID implements llamaindexgo.Node.
func (t *TextNode) ID() string {
	return t.id
}

// Children implements llamaindexgo.Node.
func (t *TextNode) ChildrenIDs() ([]string, error) {
	if len(t.childrenIds) == 0 {
		return []string{}, nil
	}
	return t.childrenIds, nil
}

// Content implements llamaindexgo.Node.
func (t *TextNode) Content() []byte {
	return t.content
}

// Embedding implements llamaindexgo.Node.
func (t *TextNode) Embedding() ([]float64, error) {
	return t.embedding, nil
}

// Metadata implements llamaindexgo.Node.
func (t *TextNode) Metadata() map[string]any {
	return t.metadata
}

// Parent implements llamaindexgo.Node.
func (t *TextNode) ParentID() (string, error) {
	return t.parentId, nil
}

// Text implements llamaindexgo.Node.
func (t *TextNode) Text() (string, error) {
	return string(t.content), nil
}

var _ llamaindexgo.Node = &TextNode{}
