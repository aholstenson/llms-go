package jsonstream

// SubParser is an interface for content transformers that can process
// string chunks as they arrive. This allows for nested parsing of
// content within JSON string fields (e.g., Markdown within a "text" field).
type SubParser interface {
	// Feed processes a chunk of string content and returns any parsed results.
	// Results should be *TextEvent or *DirectiveEvent, which will be wrapped
	// with path context and emitted as top-level EventText/EventDirective events.
	Feed(chunk string) ([]any, error)

	// Flush signals the end of input and returns any remaining parsed results.
	// This is called when the string value is complete.
	Flush() ([]any, error)

	// Reset prepares the sub-parser for reuse with a new string.
	Reset()
}
