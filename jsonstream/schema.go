package jsonstream

// FieldType represents the JSON type of a field.
type FieldType int

const (
	TypeString FieldType = iota
	TypeNumber
	TypeBoolean
	TypeArray
	TypeObject
	TypeAny // Accept any type
)

// String returns the string representation of the field type.
func (t FieldType) String() string {
	switch t {
	case TypeString:
		return "string"
	case TypeNumber:
		return "number"
	case TypeBoolean:
		return "boolean"
	case TypeArray:
		return "array"
	case TypeObject:
		return "object"
	case TypeAny:
		return "any"
	default:
		return "unknown"
	}
}

// FieldConfig defines how a field should be parsed and what events to emit.
type FieldConfig struct {
	// Type is the expected JSON type of this field.
	Type FieldType

	// Streaming enables incremental delivery for strings/arrays.
	// When true, EventStringChunk events are emitted as string content arrives,
	// rather than waiting for the complete value.
	Streaming bool

	// SubParser is an optional content transformer for string fields.
	// When set, string chunks are passed through the sub-parser and the
	// resulting events (*TextEvent, *DirectiveEvent) are emitted as top-level
	// EventText/EventDirective events with the appropriate path context.
	SubParser SubParser

	// Children defines the expected fields for object types.
	// Keys are field names, values are their configurations.
	Children map[string]FieldConfig

	// ItemConfig defines the configuration for array items.
	// Only used when Type is TypeArray.
	ItemConfig *FieldConfig
}

// Schema defines the expected structure of the JSON being parsed.
type Schema struct {
	// Root is the configuration for the root element.
	// The root is always expected to be an object.
	Root FieldConfig

	// StrictMode causes parsing to fail on unexpected fields.
	// When false (default), unexpected fields are silently ignored.
	StrictMode bool
}
