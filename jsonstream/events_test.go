package jsonstream_test

import (
	"errors"

	jsonstream "github.com/aholstenson/llms-go/jsonstream"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ParseError", func() {
	It("formats error with cause", func() {
		err := &jsonstream.ParseError{
			Path:    "response.text",
			Offset:  42,
			Message: "unexpected character",
			Cause:   jsonstream.ErrInvalidJSON,
		}

		Expect(err.Error()).To(ContainSubstring("unexpected character"))
		Expect(err.Error()).To(ContainSubstring("response.text"))
		Expect(err.Error()).To(ContainSubstring("42"))
		Expect(err.Error()).To(ContainSubstring("invalid JSON"))
	})

	It("formats error without cause", func() {
		err := &jsonstream.ParseError{
			Path:    "root",
			Offset:  0,
			Message: "something went wrong",
		}

		Expect(err.Error()).To(ContainSubstring("something went wrong"))
		Expect(err.Error()).To(ContainSubstring("root"))
		Expect(err.Error()).NotTo(ContainSubstring("invalid JSON"))
	})

	It("unwraps to the cause error", func() {
		cause := jsonstream.ErrInvalidJSON
		err := &jsonstream.ParseError{
			Path:    "test",
			Offset:  0,
			Message: "test",
			Cause:   cause,
		}

		Expect(errors.Is(err, jsonstream.ErrInvalidJSON)).To(BeTrue())
		Expect(errors.Unwrap(err)).To(Equal(cause))
	})

	It("unwraps to nil when no cause", func() {
		err := &jsonstream.ParseError{
			Path:    "test",
			Offset:  0,
			Message: "test",
		}

		Expect(errors.Unwrap(err)).To(BeNil())
	})
})

var _ = Describe("FieldType.String", func() {
	It("returns correct string for each type", func() {
		Expect(jsonstream.TypeString.String()).To(Equal("string"))
		Expect(jsonstream.TypeNumber.String()).To(Equal("number"))
		Expect(jsonstream.TypeBoolean.String()).To(Equal("boolean"))
		Expect(jsonstream.TypeArray.String()).To(Equal("array"))
		Expect(jsonstream.TypeObject.String()).To(Equal("object"))
		Expect(jsonstream.TypeAny.String()).To(Equal("any"))
	})

	It("returns unknown for invalid type", func() {
		invalid := jsonstream.FieldType(99)
		Expect(invalid.String()).To(Equal("unknown"))
	})
})

var _ = Describe("Event.String methods", func() {
	It("formats EventFieldStart", func() {
		// Use the parser to generate a real event with a path
		schema := &jsonstream.Schema{
			Root: jsonstream.FieldConfig{
				Type: jsonstream.TypeObject,
				Children: map[string]jsonstream.FieldConfig{
					"name": {Type: jsonstream.TypeString},
				},
			},
		}
		p := jsonstream.New(schema)
		events, err := p.Feed(`{"name": "test"}`)
		Expect(err).NotTo(HaveOccurred())

		for _, e := range events {
			// All events should have a non-empty String() representation
			if stringer, ok := e.(interface{ String() string }); ok {
				str := stringer.String()
				Expect(str).NotTo(BeEmpty())
			}
		}
	})

	It("formats EventError", func() {
		e := jsonstream.EventError{Err: errors.New("test error")}
		Expect(e.String()).To(ContainSubstring("test error"))
		Expect(e.Error()).To(Equal("test error"))
	})
})
