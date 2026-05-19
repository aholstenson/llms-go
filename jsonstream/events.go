package jsonstream

import (
	"fmt"
)

// Event is the interface for all parser events.
// Events are emitted as JSON is incrementally parsed.
type Event interface {
	isEvent()

	// Path returns the JSON path to the field, e.g., "response.text" or "topics[0].name"
	Path() string
}

// EventFieldStart is emitted when a field begins parsing.
type EventFieldStart struct {
	path      string
	FieldType FieldType
}

func (e EventFieldStart) isEvent()     {}
func (e EventFieldStart) Path() string { return e.path }
func (e EventFieldStart) String() string {
	return fmt.Sprintf("FieldStart{path=%q, type=%s}", e.path, e.FieldType)
}

// EventFieldEnd is emitted when a field completes parsing.
type EventFieldEnd struct {
	path string
}

func (e EventFieldEnd) isEvent()     {}
func (e EventFieldEnd) Path() string { return e.path }
func (e EventFieldEnd) String() string {
	return fmt.Sprintf("FieldEnd{path=%q}", e.path)
}

// EventStringChunk is emitted for streaming string content.
type EventStringChunk struct {
	path string

	// Chunk is the string content received in this chunk.
	Chunk string
}

func (e EventStringChunk) isEvent()     {}
func (e EventStringChunk) Path() string { return e.path }
func (e EventStringChunk) String() string {
	return fmt.Sprintf("StringChunk{path=%q, chunk=%q}", e.path, e.Chunk)
}

// EventParsedStringChunk is emitted for streaming string content when a sub-parser
// is active. Each event carries that sub-parser's output for one chunk of the JSON string.
type EventParsedStringChunk struct {
	path string

	// Chunk is the sub-parser output for this chunk (type depends on the sub-parser).
	Chunk any
}

func (e EventParsedStringChunk) isEvent()     {}
func (e EventParsedStringChunk) Path() string { return e.path }
func (e EventParsedStringChunk) String() string {
	return fmt.Sprintf("ParsedStringChunk{path=%q, chunk=%v}", e.path, e.Chunk)
}

// EventStringComplete is emitted when a complete string value is parsed.
// This is used for non-streaming strings or as a final event for streaming strings.
type EventStringComplete struct {
	path string

	// Value is the complete string value.
	Value string
}

func (e EventStringComplete) isEvent()     {}
func (e EventStringComplete) Path() string { return e.path }
func (e EventStringComplete) String() string {
	return fmt.Sprintf("StringComplete{path=%q, value=%q}", e.path, e.Value)
}

// EventNumber is emitted when a number value is parsed.
type EventNumber struct {
	path string

	// Value is the parsed number.
	Value float64
}

func (e EventNumber) isEvent()     {}
func (e EventNumber) Path() string { return e.path }
func (e EventNumber) String() string {
	return fmt.Sprintf("Number{path=%q, value=%v}", e.path, e.Value)
}

// EventBoolean is emitted when a boolean value is parsed.
type EventBoolean struct {
	path string

	// Value is the parsed boolean.
	Value bool
}

func (e EventBoolean) isEvent()     {}
func (e EventBoolean) Path() string { return e.path }
func (e EventBoolean) String() string {
	return fmt.Sprintf("Boolean{path=%q, value=%v}", e.path, e.Value)
}

// EventNull is emitted when a null value is parsed.
type EventNull struct {
	path string
}

func (e EventNull) isEvent()     {}
func (e EventNull) Path() string { return e.path }
func (e EventNull) String() string {
	return fmt.Sprintf("Null{path=%q}", e.path)
}

// EventArrayStart is emitted when an array begins.
type EventArrayStart struct {
	path string
}

func (e EventArrayStart) isEvent()     {}
func (e EventArrayStart) Path() string { return e.path }
func (e EventArrayStart) String() string {
	return fmt.Sprintf("ArrayStart{path=%q}", e.path)
}

// EventArrayItem is emitted when an array item is complete.
type EventArrayItem struct {
	path string

	// Index is the zero-based index of the item in the array.
	Index int

	// Value is the parsed value of the array item.
	Value any
}

func (e EventArrayItem) isEvent()     {}
func (e EventArrayItem) Path() string { return e.path }
func (e EventArrayItem) String() string {
	return fmt.Sprintf("ArrayItem{path=%q, index=%d, value=%v}", e.path, e.Index, e.Value)
}

// EventArrayEnd is emitted when an array ends.
type EventArrayEnd struct {
	path string

	// Count is the number of items in the array.
	Count int
}

func (e EventArrayEnd) isEvent()     {}
func (e EventArrayEnd) Path() string { return e.path }
func (e EventArrayEnd) String() string {
	return fmt.Sprintf("ArrayEnd{path=%q, count=%d}", e.path, e.Count)
}

// EventObjectStart is emitted when an object begins.
type EventObjectStart struct {
	path string
}

func (e EventObjectStart) isEvent()     {}
func (e EventObjectStart) Path() string { return e.path }
func (e EventObjectStart) String() string {
	return fmt.Sprintf("ObjectStart{path=%q}", e.path)
}

// EventObjectComplete is emitted when an object is complete.
type EventObjectComplete struct {
	path string

	// Value is the complete parsed object.
	Value map[string]any
}

func (e EventObjectComplete) isEvent()     {}
func (e EventObjectComplete) Path() string { return e.path }
func (e EventObjectComplete) String() string {
	return fmt.Sprintf("ObjectComplete{path=%q, keys=%d}", e.path, len(e.Value))
}

// EventError is emitted when a parsing error occurs.
type EventError struct {
	path string

	// Err is the error that occurred.
	Err error
}

func (e EventError) isEvent()     {}
func (e EventError) Path() string { return e.path }
func (e EventError) String() string {
	return fmt.Sprintf("Error{path=%q, err=%v}", e.path, e.Err)
}

func (e EventError) Error() string {
	return e.Err.Error()
}
