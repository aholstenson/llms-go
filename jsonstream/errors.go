package jsonstream

import (
	"fmt"

	"github.com/cockroachdb/errors"
)

var (
	// ErrUnexpectedEOF indicates the input ended unexpectedly.
	// This is normal during streaming and means more data is needed.
	ErrUnexpectedEOF = errors.New("unexpected end of input")

	// ErrInvalidJSON indicates the input is not valid JSON.
	ErrInvalidJSON = errors.New("invalid JSON")

	// ErrTypeMismatch indicates the parsed value doesn't match the schema type.
	ErrTypeMismatch = errors.New("type mismatch")

	// ErrUnexpectedField indicates a field was found that's not in the schema.
	// This is only returned in strict mode.
	ErrUnexpectedField = errors.New("unexpected field")
)

// ParseError represents a parsing error with context.
type ParseError struct {
	Path    string
	Offset  int
	Message string
	Cause   error
}

func (e *ParseError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("jsonstream: %s at path %q (offset %d): %v", e.Message, e.Path, e.Offset, e.Cause)
	}
	return fmt.Sprintf("jsonstream: %s at path %q (offset %d)", e.Message, e.Path, e.Offset)
}

func (e *ParseError) Unwrap() error {
	return e.Cause
}

// newParseError creates a new ParseError.
func newParseError(path string, offset int, message string, cause error) *ParseError {
	return &ParseError{
		Path:    path,
		Offset:  offset,
		Message: message,
		Cause:   cause,
	}
}
