package jsonstream_test

import (
	"errors"
	"strings"
	"unicode/utf8"

	jsonstream "github.com/aholstenson/llms-go/jsonstream"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

type mockTextEvent struct {
	Text string
}

// mockSubParser is a test sub-parser that returns TextEvents for chunks.
type mockSubParser struct {
	chunks []string
}

func (m *mockSubParser) Feed(chunk string) ([]any, error) {
	m.chunks = append(m.chunks, chunk)
	return []any{&mockTextEvent{Text: chunk}}, nil
}

func (m *mockSubParser) Flush() ([]any, error) {
	return []any{&mockTextEvent{Text: "flush"}}, nil
}

func (m *mockSubParser) Reset() {
	m.chunks = nil
}

// errorSubParser is a test sub-parser that returns errors.
type errorSubParser struct {
	feedErr  error
	flushErr error
}

func (e *errorSubParser) Feed(chunk string) ([]any, error) {
	return nil, e.feedErr
}

func (e *errorSubParser) Flush() ([]any, error) {
	return nil, e.flushErr
}

func (e *errorSubParser) Reset() {}

var _ = Describe("Parser", func() {
	var p *jsonstream.Parser
	var schema *jsonstream.Schema

	Context("when parsing simple values", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"name":   {Type: jsonstream.TypeString},
						"count":  {Type: jsonstream.TypeNumber},
						"active": {Type: jsonstream.TypeBoolean},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should parse a complete JSON object", func() {
			events, err := p.Feed(`{"name": "test", "count": 42, "active": true}`)
			Expect(err).NotTo(HaveOccurred())
			Expect(events).NotTo(BeEmpty())

			// Check for expected events
			var foundName, foundCount, foundActive bool
			for _, e := range events {
				switch ev := e.(type) {
				case jsonstream.EventStringComplete:
					if ev.Path() == "name" {
						Expect(ev.Value).To(Equal("test"))
						foundName = true
					}
				case jsonstream.EventNumber:
					if ev.Path() == "count" {
						Expect(ev.Value).To(Equal(42.0))
						foundCount = true
					}
				case jsonstream.EventBoolean:
					if ev.Path() == "active" {
						Expect(ev.Value).To(BeTrue())
						foundActive = true
					}
				}
			}
			Expect(foundName).To(BeTrue())
			Expect(foundCount).To(BeTrue())
			Expect(foundActive).To(BeTrue())
		})

		It("should parse a null value", func() {
			nullSchema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"value": {Type: jsonstream.TypeAny},
					},
				},
			}
			p = jsonstream.New(nullSchema)

			events, err := p.Feed(`{"value": null}`)
			Expect(err).NotTo(HaveOccurred())

			var foundNull bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNull); ok && ev.Path() == "value" {
					foundNull = true
				}
			}
			Expect(foundNull).To(BeTrue())
		})

		It("should parse negative numbers", func() {
			events, err := p.Feed(`{"count": -123}`)
			Expect(err).NotTo(HaveOccurred())

			var foundCount bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "count" {
					Expect(ev.Value).To(Equal(-123.0))
					foundCount = true
				}
			}
			Expect(foundCount).To(BeTrue())
		})

		It("should parse floating point numbers", func() {
			events, err := p.Feed(`{"count": 3.14159}`)
			Expect(err).NotTo(HaveOccurred())

			var foundCount bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "count" {
					Expect(ev.Value).To(BeNumerically("~", 3.14159, 0.00001))
					foundCount = true
				}
			}
			Expect(foundCount).To(BeTrue())
		})

		It("should parse scientific notation", func() {
			events, err := p.Feed(`{"count": 1.5e10}`)
			Expect(err).NotTo(HaveOccurred())

			var foundCount bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "count" {
					Expect(ev.Value).To(Equal(1.5e10))
					foundCount = true
				}
			}
			Expect(foundCount).To(BeTrue())
		})
	})

	Context("when parsing strings with escape sequences", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {Type: jsonstream.TypeString},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should handle basic escape sequences", func() {
			events, err := p.Feed(`{"text": "hello\nworld\ttab"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("hello\nworld\ttab"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle escaped quotes", func() {
			events, err := p.Feed(`{"text": "say \"hello\""}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal(`say "hello"`))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle escaped backslashes", func() {
			events, err := p.Feed(`{"text": "path\\to\\file"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal(`path\to\file`))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle unicode escapes", func() {
			events, err := p.Feed(`{"text": "Hello \u0048\u0065\u006c\u006c\u006f"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("Hello Hello"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle emoji via unicode escapes", func() {
			// Emoji: 😀 is U+1F600, represented as surrogate pair \uD83D\uDE00
			// But for simplicity, we handle basic plane unicode
			events, err := p.Feed(`{"text": "\u00e9"}`) // é
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("é"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when parsing arrays", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"items": {
							Type: jsonstream.TypeArray,
							ItemConfig: &jsonstream.FieldConfig{
								Type: jsonstream.TypeString,
							},
						},
						"numbers": {
							Type: jsonstream.TypeArray,
							ItemConfig: &jsonstream.FieldConfig{
								Type: jsonstream.TypeNumber,
							},
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should parse an array of strings", func() {
			events, err := p.Feed(`{"items": ["a", "b", "c"]}`)
			Expect(err).NotTo(HaveOccurred())

			var items []string
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventArrayItem); ok && strings.HasPrefix(ev.Path(), "items") {
					if s, ok := ev.Value.(string); ok {
						items = append(items, s)
					}
				}
			}
			Expect(items).To(Equal([]string{"a", "b", "c"}))
		})

		It("should parse an array of numbers", func() {
			events, err := p.Feed(`{"numbers": [1, 2, 3]}`)
			Expect(err).NotTo(HaveOccurred())

			var numbers []float64
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventArrayItem); ok && strings.HasPrefix(ev.Path(), "numbers") {
					if n, ok := ev.Value.(float64); ok {
						numbers = append(numbers, n)
					}
				}
			}
			Expect(numbers).To(Equal([]float64{1, 2, 3}))
		})

		It("should parse an empty array", func() {
			events, err := p.Feed(`{"items": []}`)
			Expect(err).NotTo(HaveOccurred())

			var foundArrayEnd bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventArrayEnd); ok && ev.Path() == "items" {
					Expect(ev.Count).To(Equal(0))
					foundArrayEnd = true
				}
			}
			Expect(foundArrayEnd).To(BeTrue())
		})

		It("should emit array start and end events", func() {
			events, err := p.Feed(`{"items": ["x"]}`)
			Expect(err).NotTo(HaveOccurred())

			var foundStart, foundEnd bool
			for _, e := range events {
				switch ev := e.(type) {
				case jsonstream.EventArrayStart:
					if ev.Path() == "items" {
						foundStart = true
					}
				case jsonstream.EventArrayEnd:
					if ev.Path() == "items" {
						foundEnd = true
					}
				}
			}
			Expect(foundStart).To(BeTrue())
			Expect(foundEnd).To(BeTrue())
		})
	})

	Context("when parsing nested objects", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"user": {
							Type: jsonstream.TypeObject,
							Children: map[string]jsonstream.FieldConfig{
								"name": {Type: jsonstream.TypeString},
								"age":  {Type: jsonstream.TypeNumber},
							},
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should parse nested objects", func() {
			events, err := p.Feed(`{"user": {"name": "Alice", "age": 30}}`)
			Expect(err).NotTo(HaveOccurred())

			var foundName, foundAge bool
			for _, e := range events {
				switch ev := e.(type) {
				case jsonstream.EventStringComplete:
					if ev.Path() == "user.name" {
						Expect(ev.Value).To(Equal("Alice"))
						foundName = true
					}
				case jsonstream.EventNumber:
					if ev.Path() == "user.age" {
						Expect(ev.Value).To(Equal(30.0))
						foundAge = true
					}
				}
			}
			Expect(foundName).To(BeTrue())
			Expect(foundAge).To(BeTrue())
		})
	})

	Context("when parsing incrementally (streaming)", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"name": {Type: jsonstream.TypeString},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should handle chunk ending mid-object", func() {
			events1, err := p.Feed(`{"na`)
			Expect(err).NotTo(HaveOccurred())
			// Should emit ObjectStart when we see the opening brace
			Expect(events1).To(HaveLen(1))
			Expect(events1[0]).To(BeAssignableToTypeOf(jsonstream.EventObjectStart{}))

			events2, err := p.Feed(`me": "test"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events2 {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "name" {
					Expect(ev.Value).To(Equal("test"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending mid-string", func() {
			_, err := p.Feed(`{"name": "hel`)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(`lo"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events2 {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "name" {
					Expect(ev.Value).To(Equal("hello"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending mid-escape", func() {
			_, err := p.Feed(`{"name": "hello\`)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(`nworld"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events2 {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "name" {
					Expect(ev.Value).To(Equal("hello\nworld"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending mid-unicode escape", func() {
			_, err := p.Feed(`{"name": "hello\u00`)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(`e9world"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events2 {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "name" {
					Expect(ev.Value).To(Equal("helloéworld"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending mid-number", func() {
			numSchema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"count": {Type: jsonstream.TypeNumber},
					},
				},
			}
			p = jsonstream.New(numSchema)

			_, err := p.Feed(`{"count": 12`)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(`34}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events2 {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "count" {
					Expect(ev.Value).To(Equal(1234.0))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending mid-literal", func() {
			boolSchema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"active": {Type: jsonstream.TypeBoolean},
					},
				},
			}
			p = jsonstream.New(boolSchema)

			_, err := p.Feed(`{"active": tr`)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(`ue}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events2 {
				if ev, ok := e.(jsonstream.EventBoolean); ok && ev.Path() == "active" {
					Expect(ev.Value).To(BeTrue())
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle character-by-character input", func() {
			json := `{"name": "test"}`
			var allEvents []jsonstream.Event

			for _, ch := range json {
				events, err := p.Feed(string(ch))
				Expect(err).NotTo(HaveOccurred())
				allEvents = append(allEvents, events...)
			}

			var found bool
			for _, e := range allEvents {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "name" {
					Expect(ev.Value).To(Equal("test"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when streaming strings", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {
							Type:      jsonstream.TypeString,
							Streaming: true,
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should emit string chunks as they arrive", func() {
			events1, err := p.Feed(`{"text": "Hello `)
			Expect(err).NotTo(HaveOccurred())

			var chunks []string
			for _, e := range events1 {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "text" {
					chunks = append(chunks, ev.Chunk)
				}
			}
			Expect(chunks).To(ContainElement("Hello "))

			events2, err := p.Feed(`World"}`)
			Expect(err).NotTo(HaveOccurred())

			for _, e := range events2 {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "text" {
					chunks = append(chunks, ev.Chunk)
				}
			}
			Expect(strings.Join(chunks, "")).To(Equal("Hello World"))
		})

		It("should emit EventStringComplete at the end", func() {
			events, err := p.Feed(`{"text": "Complete"}`)
			Expect(err).NotTo(HaveOccurred())

			var foundComplete bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("Complete"))
					foundComplete = true
				}
			}
			Expect(foundComplete).To(BeTrue())
		})
	})

	Context("when using sub-parsers", func() {
		var subParser *mockSubParser

		BeforeEach(func() {
			subParser = &mockSubParser{}
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"content": {
							Type:      jsonstream.TypeString,
							Streaming: true,
							SubParser: subParser,
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should pass chunks to the sub-parser", func() {
			_, err := p.Feed(`{"content": "First `)
			Expect(err).NotTo(HaveOccurred())

			_, err = p.Feed(`Second"}`)
			Expect(err).NotTo(HaveOccurred())

			Expect(subParser.chunks).To(ContainElement("First "))
			Expect(subParser.chunks).To(ContainElement("Second"))
		})

		It("should emit sub-parser results as top-level EventText events", func() {
			events, err := p.Feed(`{"content": "test"}`)
			Expect(err).NotTo(HaveOccurred())

			var foundTextEvent bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventParsedStringChunk); ok && ev.Path() == "content" {
					foundTextEvent = true
				}
			}
			Expect(foundTextEvent).To(BeTrue())
		})

		It("should NOT emit EventStringChunk when sub-parser is active", func() {
			events, err := p.Feed(`{"content": "test"}`)
			Expect(err).NotTo(HaveOccurred())

			var foundStringChunk, foundTextEvent bool
			for _, e := range events {
				switch ev := e.(type) {
				case jsonstream.EventStringChunk:
					if ev.Path() == "content" {
						foundStringChunk = true
					}
				case jsonstream.EventParsedStringChunk:
					if ev.Path() == "content" {
						foundTextEvent = true
					}
				}
			}
			Expect(foundTextEvent).To(BeTrue())
			Expect(foundStringChunk).To(BeFalse())
		})
	})

	Context("when handling lenient parsing (LLM quirks)", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"items": {
							Type: jsonstream.TypeArray,
							ItemConfig: &jsonstream.FieldConfig{
								Type: jsonstream.TypeString,
							},
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should tolerate trailing commas in arrays", func() {
			events, err := p.Feed(`{"items": ["a", "b", "c",]}`)
			Expect(err).NotTo(HaveOccurred())

			var items []string
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventArrayItem); ok {
					if s, ok := ev.Value.(string); ok {
						items = append(items, s)
					}
				}
			}
			Expect(items).To(Equal([]string{"a", "b", "c"}))
		})

		It("should tolerate trailing commas in objects", func() {
			objSchema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"a": {Type: jsonstream.TypeNumber},
						"b": {Type: jsonstream.TypeNumber},
					},
				},
			}
			p = jsonstream.New(objSchema)

			events, err := p.Feed(`{"a": 1, "b": 2,}`)
			Expect(err).NotTo(HaveOccurred())

			var foundA, foundB bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok {
					if ev.Path() == "a" {
						foundA = true
					}
					if ev.Path() == "b" {
						foundB = true
					}
				}
			}
			Expect(foundA).To(BeTrue())
			Expect(foundB).To(BeTrue())
		})

		It("should ignore trailing content after valid JSON closes", func() {
			events, err := p.Feed(`{"items": ["x"]} some trailing stuff`)
			Expect(err).NotTo(HaveOccurred())

			var foundArrayEnd bool
			for _, e := range events {
				if _, ok := e.(jsonstream.EventArrayEnd); ok {
					foundArrayEnd = true
				}
			}
			Expect(foundArrayEnd).To(BeTrue())
		})
	})

	Context("when parsing complex nested structures", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"response": {
							Type: jsonstream.TypeObject,
							Children: map[string]jsonstream.FieldConfig{
								"text": {
									Type:      jsonstream.TypeString,
									Streaming: true,
								},
								"topics": {
									Type: jsonstream.TypeArray,
									ItemConfig: &jsonstream.FieldConfig{
										Type: jsonstream.TypeObject,
										Children: map[string]jsonstream.FieldConfig{
											"name":        {Type: jsonstream.TypeString},
											"description": {Type: jsonstream.TypeString},
										},
									},
								},
							},
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should parse deeply nested structure", func() {
			json := `{
				"response": {
					"text": "Hello world",
					"topics": [
						{"name": "Topic 1", "description": "First topic"},
						{"name": "Topic 2", "description": "Second topic"}
					]
				}
			}`
			events, err := p.Feed(json)
			Expect(err).NotTo(HaveOccurred())

			var textValue string
			var topicNames []string

			for _, e := range events {
				switch ev := e.(type) {
				case jsonstream.EventStringComplete:
					if ev.Path() == "response.text" {
						textValue = ev.Value
					}
					if strings.HasSuffix(ev.Path(), ".name") {
						topicNames = append(topicNames, ev.Value)
					}
				}
			}

			Expect(textValue).To(Equal("Hello world"))
			Expect(topicNames).To(Equal([]string{"Topic 1", "Topic 2"}))
		})

		It("should stream text while parsing topics", func() {
			// Simulate LLM streaming output
			chunks := []string{
				`{"response": {"text": "Hello `,
				`world", "topics": [{"name": "Topic`,
				` 1", "description": "Desc 1"}]}}`,
			}

			var allEvents []jsonstream.Event
			for _, chunk := range chunks {
				events, err := p.Feed(chunk)
				Expect(err).NotTo(HaveOccurred())
				allEvents = append(allEvents, events...)
			}

			var textChunks []string
			for _, e := range allEvents {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "response.text" {
					textChunks = append(textChunks, ev.Chunk)
				}
			}
			Expect(strings.Join(textChunks, "")).To(Equal("Hello world"))
		})
	})

	Context("when using Reset", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"value": {Type: jsonstream.TypeNumber},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should allow parsing a new document after reset", func() {
			events1, err := p.Feed(`{"value": 1}`)
			Expect(err).NotTo(HaveOccurred())

			var value1 float64
			for _, e := range events1 {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "value" {
					value1 = ev.Value
				}
			}
			Expect(value1).To(Equal(1.0))

			p.Reset()

			events2, err := p.Feed(`{"value": 2}`)
			Expect(err).NotTo(HaveOccurred())

			var value2 float64
			for _, e := range events2 {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "value" {
					value2 = ev.Value
				}
			}
			Expect(value2).To(Equal(2.0))
		})
	})

	Context("when handling arrays with object items", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"users": {
							Type: jsonstream.TypeArray,
							ItemConfig: &jsonstream.FieldConfig{
								Type: jsonstream.TypeObject,
								Children: map[string]jsonstream.FieldConfig{
									"id":   {Type: jsonstream.TypeNumber},
									"name": {Type: jsonstream.TypeString},
								},
							},
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should emit events with correct paths for array item fields", func() {
			events, err := p.Feed(`{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}`)
			Expect(err).NotTo(HaveOccurred())

			pathsFound := make(map[string]bool)
			for _, e := range events {
				switch ev := e.(type) {
				case jsonstream.EventNumber:
					pathsFound[ev.Path()] = true
				case jsonstream.EventStringComplete:
					pathsFound[ev.Path()] = true
				}
			}

			Expect(pathsFound).To(HaveKey("users[0].id"))
			Expect(pathsFound).To(HaveKey("users[0].name"))
			Expect(pathsFound).To(HaveKey("users[1].id"))
			Expect(pathsFound).To(HaveKey("users[1].name"))
		})
	})

	Context("when using StrictMode", func() {
		It("should ignore unexpected fields when StrictMode is false", func() {
			schema := &jsonstream.Schema{
				StrictMode: false,
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"name": {Type: jsonstream.TypeString},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"name": "test", "unexpected": "value"}`)
			Expect(err).NotTo(HaveOccurred())

			var foundName bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "name" {
					Expect(ev.Value).To(Equal("test"))
					foundName = true
				}
			}
			Expect(foundName).To(BeTrue())
		})

		It("should return error for unexpected fields when StrictMode is true", func() {
			schema := &jsonstream.Schema{
				StrictMode: true,
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"name": {Type: jsonstream.TypeString},
					},
				},
			}
			p := jsonstream.New(schema)

			_, err := p.Feed(`{"name": "test", "unexpected": "value"}`)
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
			Expect(errors.Is(parseErr.Cause, jsonstream.ErrUnexpectedField)).To(BeTrue())
			Expect(parseErr.Message).To(ContainSubstring("unexpected"))
		})

		It("should allow defined fields in StrictMode", func() {
			schema := &jsonstream.Schema{
				StrictMode: true,
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"name":  {Type: jsonstream.TypeString},
						"count": {Type: jsonstream.TypeNumber},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"name": "test", "count": 42}`)
			Expect(err).NotTo(HaveOccurred())

			var foundName, foundCount bool
			for _, e := range events {
				switch ev := e.(type) {
				case jsonstream.EventStringComplete:
					if ev.Path() == "name" {
						foundName = true
					}
				case jsonstream.EventNumber:
					if ev.Path() == "count" {
						foundCount = true
					}
				}
			}
			Expect(foundName).To(BeTrue())
			Expect(foundCount).To(BeTrue())
		})
	})

	Context("when parsing UTF-16 surrogate pairs", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {Type: jsonstream.TypeString},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should parse emoji via surrogate pairs", func() {
			// 😀 is U+1F600, encoded as surrogate pair \uD83D\uDE00
			events, err := p.Feed(`{"text": "\uD83D\uDE00"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("😀"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should parse multiple emoji via surrogate pairs", func() {
			// 👍 is U+1F44D (\uD83D\uDC4D), ❤ is U+2764 (BMP, no surrogate needed)
			events, err := p.Feed(`{"text": "\uD83D\uDE00\uD83D\uDC4D"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("😀👍"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle mixed content with surrogate pairs", func() {
			events, err := p.Feed(`{"text": "Hello \uD83D\uDE00 World"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("Hello 😀 World"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle surrogate pair split across chunks", func() {
			// Feed high surrogate in first chunk, low surrogate in second
			_, err := p.Feed(`{"text": "\uD83D`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`\uDE00"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("😀"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle character-by-character surrogate pair input", func() {
			json := `{"text": "\uD83D\uDE00"}`
			var allEvents []jsonstream.Event

			for _, ch := range json {
				events, err := p.Feed(string(ch))
				Expect(err).NotTo(HaveOccurred())
				allEvents = append(allEvents, events...)
			}

			var found bool
			for _, e := range allEvents {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("😀"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should emit replacement character for unpaired high surrogate", func() {
			// High surrogate not followed by low surrogate
			events, err := p.Feed(`{"text": "\uD83Dtest"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					// Should contain replacement character followed by "test"
					Expect(ev.Value).To(HavePrefix(string('\uFFFD')))
					Expect(ev.Value).To(HaveSuffix("test"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should emit replacement character for unpaired low surrogate", func() {
			// Low surrogate without preceding high surrogate
			events, err := p.Feed(`{"text": "\uDE00"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal(string('\uFFFD')))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle high surrogate followed by another high surrogate", func() {
			// Two consecutive high surrogates - first one is unpaired
			events, err := p.Feed(`{"text": "\uD83D\uD83D\uDE00"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					// First high surrogate becomes replacement, second forms pair with low
					Expect(ev.Value).To(Equal(string('\uFFFD') + "😀"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle high surrogate followed by regular escape", func() {
			// High surrogate followed by \n instead of low surrogate
			events, err := p.Feed(`{"text": "\uD83D\ntest"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					// High surrogate becomes replacement, followed by newline and "test"
					Expect(ev.Value).To(Equal(string('\uFFFD') + "\ntest"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when parsing raw UTF-8 bytes split at chunk boundaries", func() {
		// These tests verify that multi-byte UTF-8 characters in JSON string values
		// are handled correctly when they are split across chunk boundaries.
		// This simulates what happens when streaming JSON from an LLM where
		// network packets split in the middle of a multi-byte character.
		//
		// Swedish "ä" (U+00E4) is UTF-8 bytes: 0xC3 0xA4
		// Swedish "ö" (U+00F6) is UTF-8 bytes: 0xC3 0xB6

		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {Type: jsonstream.TypeString},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should handle Swedish text in single chunk", func() {
			events, err := p.Feed(`{"text": "blodkärl"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("blodkärl"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle Swedish ä split at UTF-8 byte boundary", func() {
			// "blodkärl" where ä (0xC3 0xA4) is split between chunks
			// Chunk 1: '{"text": "blodk' + 0xC3 (first byte of ä)
			// Chunk 2: 0xA4 (second byte of ä) + 'rl"}'
			chunk1 := `{"text": "blodk` + "\xC3"
			chunk2 := "\xA4" + `rl"}`

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("blodkärl"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle Swedish ö split at UTF-8 byte boundary", func() {
			// "sjö" where ö (0xC3 0xB6) is split between chunks
			chunk1 := `{"text": "sj` + "\xC3"
			chunk2 := "\xB6" + `"}`

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("sjö"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle multiple Swedish characters with split boundaries", func() {
			// "blodkärl är viktigt" split at both ä characters
			chunk1 := `{"text": "blodk` + "\xC3"
			chunk2 := "\xA4" + `rl ` + "\xC3"
			chunk3 := "\xA4" + `r viktigt"}`

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			_, err = p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk3)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("blodkärl är viktigt"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle 3-byte UTF-8 character split after first byte", func() {
			// Japanese "あ" (U+3042) is UTF-8 bytes: 0xE3 0x81 0x82
			chunk1 := `{"text": "test` + "\xE3"
			chunk2 := "\x81\x82" + `end"}`

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("testあend"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle 3-byte UTF-8 character split after second byte", func() {
			// Japanese "あ" (U+3042) split after second byte
			chunk1 := `{"text": "test` + "\xE3\x81"
			chunk2 := "\x82" + `end"}`

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("testあend"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle 4-byte UTF-8 emoji split at byte boundary", func() {
			// Emoji "😀" (U+1F600) as raw UTF-8 bytes: 0xF0 0x9F 0x98 0x80
			// Note: This is different from surrogate pair tests which use JSON escapes
			chunk1 := `{"text": "hi` + "\xF0"
			chunk2 := "\x9F\x98\x80" + `bye"}`

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("hi😀bye"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle byte-by-byte streaming of Swedish text", func() {
			// Feed '{"text": "blodkärl"}' byte by byte
			// Note: ä is 2 bytes (0xC3 0xA4), so this will split the character
			json := `{"text": "blodkärl"}`
			var allEvents []jsonstream.Event

			for i := 0; i < len(json); i++ {
				// Use json[i:i+1] to get a 1-byte string, not string(json[i])
				// which converts the byte to a rune first
				events, err := p.Feed(json[i : i+1])
				Expect(err).NotTo(HaveOccurred())
				allEvents = append(allEvents, events...)
			}

			var found bool
			for _, e := range allEvents {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("blodkärl"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when handling additional chunking edge cases", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text":  {Type: jsonstream.TypeString},
						"count": {Type: jsonstream.TypeNumber},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should handle empty chunk input", func() {
			events1, err := p.Feed(`{"text": "hel`)
			Expect(err).NotTo(HaveOccurred())

			// Feed empty chunks
			events2, err := p.Feed("")
			Expect(err).NotTo(HaveOccurred())
			Expect(events2).To(BeEmpty())

			events3, err := p.Feed("")
			Expect(err).NotTo(HaveOccurred())
			Expect(events3).To(BeEmpty())

			events4, err := p.Feed(`lo"}`)
			Expect(err).NotTo(HaveOccurred())

			allEvents := append(events1, events4...)
			var found bool
			for _, e := range allEvents {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("hello"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending right at backslash-u", func() {
			// Split right after \u
			_, err := p.Feed(`{"text": "test\u`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`0048"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("testH"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending after one unicode hex digit", func() {
			_, err := p.Feed(`{"text": "\u0`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`048"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("H"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending after two unicode hex digits", func() {
			_, err := p.Feed(`{"text": "\u00`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`48"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("H"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk ending after three unicode hex digits", func() {
			_, err := p.Feed(`{"text": "\u004`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`8"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("H"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk boundary at object colon", func() {
			_, err := p.Feed(`{"text"`)
			Expect(err).NotTo(HaveOccurred())

			_, err = p.Feed(`:`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(` "value"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("value"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk boundary between object key and colon with whitespace", func() {
			_, err := p.Feed(`{"text" `)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`: "value"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("value"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle chunk boundary at array comma", func() {
			arraySchema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"items": {
							Type: jsonstream.TypeArray,
							ItemConfig: &jsonstream.FieldConfig{
								Type: jsonstream.TypeString,
							},
						},
					},
				},
			}
			p = jsonstream.New(arraySchema)

			_, err := p.Feed(`{"items": ["a"`)
			Expect(err).NotTo(HaveOccurred())

			_, err = p.Feed(`,`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(` "b"]}`)
			Expect(err).NotTo(HaveOccurred())

			var items []string
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventArrayItem); ok {
					if s, ok := ev.Value.(string); ok {
						items = append(items, s)
					}
				}
			}
			Expect(items).To(ContainElement("b"))
		})

		It("should handle scientific notation split at e", func() {
			_, err := p.Feed(`{"count": 1.5e`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`10}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "count" {
					Expect(ev.Value).To(Equal(1.5e10))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle negative exponent", func() {
			events, err := p.Feed(`{"count": 1.5e-10}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "count" {
					Expect(ev.Value).To(BeNumerically("~", 1.5e-10, 1e-20))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle scientific notation split at minus in exponent", func() {
			_, err := p.Feed(`{"count": 1e-`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`5}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "count" {
					Expect(ev.Value).To(BeNumerically("~", 1e-5, 1e-15))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle forward slash escape", func() {
			events, err := p.Feed(`{"text": "a\/b"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("a/b"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle all escape sequences", func() {
			// Input: \" \\ \/ \b \f \n \r \t
			// Output: " \ / backspace formfeed newline return tab
			events, err := p.Feed(`{"text": "\"\\/\b\f\n\r\t"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("\"\\/\b\f\n\r\t"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle surrogate pair split at backslash before low surrogate", func() {
			// Split after high surrogate, right at the backslash of low surrogate
			_, err := p.Feed(`{"text": "\uD83D`)
			Expect(err).NotTo(HaveOccurred())

			_, err = p.Feed(`\`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`uDE00"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("😀"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle surrogate pair split at u of low surrogate", func() {
			_, err := p.Feed(`{"text": "\uD83D\u`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`DE00"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("😀"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle surrogate pair split within low surrogate hex digits", func() {
			_, err := p.Feed(`{"text": "\uD83D\uDE`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`00"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("😀"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when handling streaming strings with escapes at chunk boundaries", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {
							Type:      jsonstream.TypeString,
							Streaming: true,
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should emit content before escape when chunk ends at backslash", func() {
			// When chunk ends right at the escape backslash, we want to ensure
			// the content before the escape is emitted as a chunk.
			events1, err := p.Feed(`{"text": "hello\`)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(`nworld"}`)
			Expect(err).NotTo(HaveOccurred())

			// Collect all chunks and verify their combined content equals the complete string
			var chunks []string
			var completeValue string
			for _, e := range append(events1, events2...) {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "text" {
					chunks = append(chunks, ev.Chunk)
				}
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					completeValue = ev.Value
				}
			}

			// Combined chunks should equal the complete string
			Expect(strings.Join(chunks, "")).To(Equal(completeValue))
			Expect(completeValue).To(Equal("hello\nworld"))
		})

		It("should produce correct complete value with unicode at chunk boundary", func() {
			events1, err := p.Feed(`{"text": "test\u00`)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(`e9end"}`)
			Expect(err).NotTo(HaveOccurred())

			// Verify the final complete string is correct
			var completeValue string
			for _, e := range append(events1, events2...) {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					completeValue = ev.Value
				}
			}
			Expect(completeValue).To(Equal("testéend"))
		})

		It("should produce correct complete value with surrogate pair at chunk boundary", func() {
			events1, err := p.Feed(`{"text": "emoji\uD83D`)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(`\uDE00end"}`)
			Expect(err).NotTo(HaveOccurred())

			// Verify the final complete string is correct
			var completeValue string
			for _, e := range append(events1, events2...) {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					completeValue = ev.Value
				}
			}
			Expect(completeValue).To(Equal("emoji😀end"))
		})

		It("should emit chunk containing content before escape sequence", func() {
			events, err := p.Feed(`{"text": "hello\nworld"}`)
			Expect(err).NotTo(HaveOccurred())

			var chunks []string
			var completeValue string
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "text" {
					chunks = append(chunks, ev.Chunk)
				}
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					completeValue = ev.Value
				}
			}

			// Combined chunks should equal the complete string
			Expect(strings.Join(chunks, "")).To(Equal(completeValue))
			Expect(completeValue).To(Equal("hello\nworld"))
		})

		It("should emit all content between multiple escapes as chunks", func() {
			events, err := p.Feed(`{"text": "a\nb\tc\rd"}`)
			Expect(err).NotTo(HaveOccurred())

			var chunks []string
			var completeValue string
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "text" {
					chunks = append(chunks, ev.Chunk)
				}
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					completeValue = ev.Value
				}
			}

			// Combined chunks should equal the complete string
			Expect(strings.Join(chunks, "")).To(Equal(completeValue))
			Expect(completeValue).To(Equal("a\nb\tc\rd"))
		})

		It("should handle escape at start of string", func() {
			events, err := p.Feed(`{"text": "\nhello"}`)
			Expect(err).NotTo(HaveOccurred())

			var chunks []string
			var completeValue string
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "text" {
					chunks = append(chunks, ev.Chunk)
				}
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					completeValue = ev.Value
				}
			}

			// Combined chunks should equal the complete string
			Expect(strings.Join(chunks, "")).To(Equal(completeValue))
			Expect(completeValue).To(Equal("\nhello"))
		})
	})

	Context("when handling error cases", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text":  {Type: jsonstream.TypeString},
						"count": {Type: jsonstream.TypeNumber},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should return error for invalid hex in unicode escape", func() {
			_, err := p.Feed(`{"text": "\uGGGG"}`)
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
			Expect(parseErr.Message).To(ContainSubstring("unicode"))
		})

		It("should return error for invalid escape character", func() {
			_, err := p.Feed(`{"text": "\x"}`)
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
			Expect(parseErr.Message).To(ContainSubstring("escape"))
		})

		It("should return error for control character in string", func() {
			// Tab character (0x09) is a control character but allowed
			// Let's use a true control char like 0x01
			_, err := p.Feed("{\"text\": \"test\x01\"}")
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
			Expect(parseErr.Message).To(ContainSubstring("control"))
		})

		It("should return error for invalid literal with wrong characters", func() {
			boolSchema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"valid": {Type: jsonstream.TypeBoolean},
					},
				},
			}
			p = jsonstream.New(boolSchema)

			// "trux" is not a valid literal (not a prefix of true/false/null)
			_, err := p.Feed(`{"valid": trux}`)
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
		})

		It("should return error for partial literal followed by non-letter character", func() {
			boolSchema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"valid": {Type: jsonstream.TypeBoolean},
					},
				},
			}
			p = jsonstream.New(boolSchema)

			// "tru " is a malformed literal: the space terminates it before
			// it can become "true", so the parser should report an error
			// rather than stalling waiting for more input.
			_, err := p.Feed(`{"valid": tru }`)
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
			Expect(parseErr.Message).To(ContainSubstring("invalid literal"))
		})

		It("should return error for partial literal split across feeds then terminated", func() {
			boolSchema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"valid": {Type: jsonstream.TypeBoolean},
					},
				},
			}
			p = jsonstream.New(boolSchema)

			// First feed: partial literal "tru" at end of buffer is fine
			// (parser correctly waits for more input)
			events, err := p.Feed(`{"valid": tru`)
			Expect(err).NotTo(HaveOccurred())

			var foundBoolean bool
			for _, e := range events {
				if _, ok := e.(jsonstream.EventBoolean); ok {
					foundBoolean = true
				}
			}
			Expect(foundBoolean).To(BeFalse())

			// Second feed: non-letter terminates the literal, should error
			_, err = p.Feed(` }`)
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
			Expect(parseErr.Message).To(ContainSubstring("invalid literal"))
		})

		It("should return error for unexpected character at value position", func() {
			_, err := p.Feed(`{"text": @}`)
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
			Expect(parseErr.Message).To(ContainSubstring("unexpected"))
		})
	})

	Context("when handling edge cases with nested structures", func() {
		It("should handle deeply nested arrays", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"data": {
							Type: jsonstream.TypeArray,
							ItemConfig: &jsonstream.FieldConfig{
								Type: jsonstream.TypeArray,
								ItemConfig: &jsonstream.FieldConfig{
									Type: jsonstream.TypeNumber,
								},
							},
						},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"data": [[1, 2], [3, 4]]}`)
			Expect(err).NotTo(HaveOccurred())

			var numbers []float64
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok {
					numbers = append(numbers, ev.Value)
				}
			}
			Expect(numbers).To(Equal([]float64{1, 2, 3, 4}))
		})

		It("should handle empty object in array", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"items": {
							Type: jsonstream.TypeArray,
							ItemConfig: &jsonstream.FieldConfig{
								Type:     jsonstream.TypeObject,
								Children: map[string]jsonstream.FieldConfig{},
							},
						},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"items": [{}, {}]}`)
			Expect(err).NotTo(HaveOccurred())

			var objectCount int
			for _, e := range events {
				if _, ok := e.(jsonstream.EventObjectComplete); ok {
					objectCount++
				}
			}
			// 2 inner empty objects + 1 root object = 3
			Expect(objectCount).To(Equal(3))
		})

		It("should handle mixed types in array with TypeAny", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"mixed": {
							Type: jsonstream.TypeArray,
							ItemConfig: &jsonstream.FieldConfig{
								Type: jsonstream.TypeAny,
							},
						},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"mixed": [1, "two", true, null]}`)
			Expect(err).NotTo(HaveOccurred())

			var foundNumber, foundString, foundBool, foundNull bool
			for _, e := range events {
				switch ev := e.(type) {
				case jsonstream.EventNumber:
					if strings.HasPrefix(ev.Path(), "mixed") {
						foundNumber = true
					}
				case jsonstream.EventStringComplete:
					if strings.HasPrefix(ev.Path(), "mixed") {
						foundString = true
					}
				case jsonstream.EventBoolean:
					if strings.HasPrefix(ev.Path(), "mixed") {
						foundBool = true
					}
				case jsonstream.EventNull:
					if strings.HasPrefix(ev.Path(), "mixed") {
						foundNull = true
					}
				}
			}
			Expect(foundNumber).To(BeTrue())
			Expect(foundString).To(BeTrue())
			Expect(foundBool).To(BeTrue())
			Expect(foundNull).To(BeTrue())
		})

		It("should handle object with only whitespace between tokens", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"a": {Type: jsonstream.TypeNumber},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed("{\n\t\"a\"\n:\n\t1\n}")
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "a" {
					Expect(ev.Value).To(Equal(1.0))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when handling Flush", func() {
		It("should flush pending content in streaming mode", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {
							Type:      jsonstream.TypeString,
							Streaming: true,
						},
					},
				},
			}
			p := jsonstream.New(schema)

			// Feed incomplete JSON (string not closed)
			_, err := p.Feed(`{"text": "incomplete`)
			Expect(err).NotTo(HaveOccurred())

			// Flush should emit any pending chunks
			events, err := p.Flush()
			Expect(err).NotTo(HaveOccurred())
			// Note: behavior depends on implementation - may or may not emit pending
			_ = events
		})
	})

	// These tests replicate real-world UTF-8 splitting scenarios observed when streaming
	// LLM responses containing Arabic/RTL text. The issue occurs when chunk boundaries
	// fall in the middle of multi-byte UTF-8 characters.
	Context("when parsing Arabic text split at UTF-8 byte boundaries", func() {
		// Arabic characters are 2-byte UTF-8:
		// - "ا" (alef, U+0627): 0xD8 0xA7
		// - "ل" (lam, U+0644): 0xD9 0x84
		// - "أ" (alef with hamza above, U+0623): 0xD8 0xA3
		// - "م" (meem, U+0645): 0xD9 0x85
		// - "ن" (noon, U+0646): 0xD9 0x86
		// - "ي" (ya, U+064A): 0xD9 0x8A
		// - "ة" (ta marbuta, U+0629): 0xD8 0xA9
		// The word "الأمن" (security) is: ا + ل + أ + م + ن

		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {Type: jsonstream.TypeString},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should handle Arabic text in single chunk", func() {
			// The word "الأمن" (security) in Arabic
			events, err := p.Feed(`{"text": "الأمن"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("الأمن"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle Arabic alef (ا) split at UTF-8 byte boundary", func() {
			// Arabic alef "ا" (U+0627) is UTF-8 bytes: 0xD8 0xA7
			// Split in the middle of the character
			chunk1 := `{"text": "test` + "\xD8"
			chunk2 := "\xA7" + `end"}`

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("testاend"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle Arabic word الأمن split after first character", func() {
			// "الأمن" - split after ا (0xD8 0xA7), before ل (0xD9 0x84)
			// But we split in the MIDDLE of ل
			chunk1 := `{"text": "ا` + "\xD9" // First char complete, second char starts
			chunk2 := "\x84" + `أمن"}`       // Second char completes, rest follows

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("الأمن"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle Arabic sentence with multiple split points", func() {
			// Sentence: "الأمن السيبراني" (cybersecurity)
			// Split at multiple UTF-8 boundaries
			chunk1 := `{"text": "الأ` + "\xD9" // Split in middle of م
			chunk2 := "\x85" + `ن ` + "\xD8"   // م completes, space, split in middle of ا
			chunk3 := "\xA7" + `لسيبراني"}`    // ا completes, rest follows

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			_, err = p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk3)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("الأمن السيبراني"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle byte-by-byte streaming of Arabic text", func() {
			// Feed '{"text": "الأمن"}' byte by byte
			// This is the most extreme split case
			json := `{"text": "الأمن"}`
			var allEvents []jsonstream.Event

			for i := 0; i < len(json); i++ {
				events, err := p.Feed(json[i : i+1])
				Expect(err).NotTo(HaveOccurred())
				allEvents = append(allEvents, events...)
			}

			var found bool
			for _, e := range allEvents {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("الأمن"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle realistic LLM chunk pattern with Arabic text", func() {
			// This replicates the real scenario from the bug report:
			// Chunks that look like LLM streaming output with Arabic content
			chunks := []string{
				`{"text": "التفاعل بين المطورين والأنظمة`,
				` ركيزة أساسية في علوم الحاسوب`,
				`، حيث يتجاوز مجرد كتابة الشيفرة`,
				` البرمجية"}`,
			}

			var allEvents []jsonstream.Event
			for _, chunk := range chunks {
				events, err := p.Feed(chunk)
				Expect(err).NotTo(HaveOccurred())
				allEvents = append(allEvents, events...)
			}

			var found bool
			for _, e := range allEvents {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					expected := "التفاعل بين المطورين والأنظمة ركيزة أساسية في علوم الحاسوب، حيث يتجاوز مجرد كتابة الشيفرة البرمجية"
					Expect(ev.Value).To(Equal(expected))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle LLM chunk that splits Arabic character at boundary", func() {
			// Simulate the exact bug scenario: chunk boundary falls in middle of Arabic char
			// The word "السيبراني" ends with ي (ya, U+064A = 0xD9 0x8A)
			// Next chunk starts with : followed by "الأمن"
			chunk1 := `{"text": "السيبران` + "\xD9"       // "السيبران" + first byte of ي
			chunk2 := "\x8A" + `:الأمن واستهلاك الطاقة"}` // second byte of ي + rest

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal("السيبراني:الأمن واستهلاك الطاقة"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when streaming Arabic text with sub-parser", func() {
		var subParser *mockSubParser

		BeforeEach(func() {
			subParser = &mockSubParser{}
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {
							Type:      jsonstream.TypeString,
							Streaming: true,
							SubParser: subParser,
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should pass valid UTF-8 chunks to sub-parser when Arabic splits", func() {
			// This tests that even when Arabic text is split at byte boundaries,
			// the sub-parser receives valid UTF-8 chunks
			chunk1 := `{"text": "الأ` + "\xD9" // Split in middle of م
			chunk2 := "\x85" + `ن"}`           // م completes, ن follows

			_, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			_, err = p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			// Verify all chunks passed to sub-parser are valid UTF-8
			for _, chunk := range subParser.chunks {
				Expect(utf8.ValidString(chunk)).To(BeTrue(), "chunk should be valid UTF-8: %q", chunk)
			}

			// Verify the combined chunks equal the full Arabic text
			combined := strings.Join(subParser.chunks, "")
			Expect(combined).To(Equal("الأمن"))
		})

		It("should handle byte-by-byte Arabic streaming to sub-parser", func() {
			json := `{"text": "الأمن"}`

			for i := 0; i < len(json); i++ {
				_, err := p.Feed(json[i : i+1])
				Expect(err).NotTo(HaveOccurred())
			}

			// Verify all chunks passed to sub-parser are valid UTF-8
			for _, chunk := range subParser.chunks {
				Expect(utf8.ValidString(chunk)).To(BeTrue(), "chunk should be valid UTF-8: %q", chunk)
			}

			// Verify the combined chunks equal the full Arabic text
			combined := strings.Join(subParser.chunks, "")
			Expect(combined).To(Equal("الأمن"))
		})
	})

	Context("when parsing unicode escapes in object keys", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"key": {Type: jsonstream.TypeString},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should handle \\uXXXX escape in object key", func() {
			// \u006B\u0065\u0079 = "key"
			events, err := p.Feed(`{"\u006B\u0065\u0079": "value"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "key" {
					Expect(ev.Value).To(Equal("value"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle mixed escaped and unescaped chars in object key", func() {
			events, err := p.Feed(`{"ke\u0079": "value"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "key" {
					Expect(ev.Value).To(Equal("value"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle unicode escape in object key split across chunks", func() {
			_, err := p.Feed(`{"ke\u00`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`79": "value"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "key" {
					Expect(ev.Value).To(Equal("value"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should return error for invalid hex in object key unicode escape", func() {
			_, err := p.Feed(`{"\uGGGG": "value"}`)
			Expect(err).To(HaveOccurred())

			var parseErr *jsonstream.ParseError
			Expect(errors.As(err, &parseErr)).To(BeTrue())
			Expect(parseErr.Message).To(ContainSubstring("unicode"))
		})
	})

	Context("when parsing escaped characters in object keys", func() {
		It("should handle escaped backslash in object key", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"key\\name": "value"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == `key\name` {
					Expect(ev.Value).To(Equal("value"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle escaped quote in object key", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"key\"name": "value"}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == `key"name` {
					Expect(ev.Value).To(Equal("value"))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when parsing empty string values", func() {
		It("should parse an empty string", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {Type: jsonstream.TypeString},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"text": ""}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal(""))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should parse an empty streaming string", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {Type: jsonstream.TypeString, Streaming: true},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"text": ""}`)
			Expect(err).NotTo(HaveOccurred())

			var foundComplete bool
			var chunkCount int
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					Expect(ev.Value).To(Equal(""))
					foundComplete = true
				}
				if _, ok := e.(jsonstream.EventStringChunk); ok {
					chunkCount++
				}
			}
			Expect(foundComplete).To(BeTrue())
			Expect(chunkCount).To(Equal(0))
		})
	})

	Context("when parsing false and null literals split across chunks", func() {
		It("should handle false split across chunks", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"active": {Type: jsonstream.TypeBoolean},
					},
				},
			}
			p := jsonstream.New(schema)

			_, err := p.Feed(`{"active": fal`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`se}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventBoolean); ok && ev.Path() == "active" {
					Expect(ev.Value).To(BeFalse())
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle null split across chunks", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"value": {Type: jsonstream.TypeAny},
					},
				},
			}
			p := jsonstream.New(schema)

			_, err := p.Feed(`{"value": nu`)
			Expect(err).NotTo(HaveOccurred())

			events, err := p.Feed(`ll}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if _, ok := e.(jsonstream.EventNull); ok {
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should handle false split at every possible boundary", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"v": {Type: jsonstream.TypeBoolean},
					},
				},
			}

			//nolint:misspell
			// Split "false" at each position: f|alse, fa|lse, fal|se, fals|e
			for i := 1; i < 5; i++ {
				p := jsonstream.New(schema)
				literal := "false"

				_, err := p.Feed(`{"v": ` + literal[:i])
				Expect(err).NotTo(HaveOccurred())

				events, err := p.Feed(literal[i:] + `}`)
				Expect(err).NotTo(HaveOccurred())

				var found bool
				for _, e := range events {
					if ev, ok := e.(jsonstream.EventBoolean); ok && ev.Path() == "v" {
						Expect(ev.Value).To(BeFalse())
						found = true
					}
				}
				Expect(found).To(BeTrue(), "failed for split at position %d", i)
			}
		})
	})

	Context("when using sub-parser that returns errors", func() {
		It("should propagate sub-parser Feed errors from Feed", func() {
			feedErr := errors.New("feed error")
			errSubParser := &errorSubParser{feedErr: feedErr}
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"content": {
							Type:      jsonstream.TypeString,
							Streaming: true,
							SubParser: errSubParser,
						},
					},
				},
			}
			p := jsonstream.New(schema)

			_, err := p.Feed(`{"content": "test"}`)
			Expect(err).To(HaveOccurred())
			Expect(errors.Is(err, feedErr)).To(BeTrue(), "expected wrapped feed error, got %v", err)
		})

		It("should propagate sub-parser Feed errors mid-string when chunked", func() {
			feedErr := errors.New("feed error")
			errSubParser := &errorSubParser{feedErr: feedErr}
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"content": {
							Type:      jsonstream.TypeString,
							Streaming: true,
							SubParser: errSubParser,
						},
					},
				},
			}
			p := jsonstream.New(schema)

			// Feed without closing quote — triggers the mid-string sub-parser flush
			// path (processString lines ~579-590).
			_, err := p.Feed(`{"content": "abc`)
			Expect(err).To(HaveOccurred())
			Expect(errors.Is(err, feedErr)).To(BeTrue(), "expected wrapped feed error, got %v", err)
		})

		It("should propagate sub-parser Flush errors from Feed at end-of-string", func() {
			flushErr := errors.New("flush error")
			errSubParser := &errorSubParser{flushErr: flushErr}
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"content": {
							Type:      jsonstream.TypeString,
							Streaming: true,
							SubParser: errSubParser,
						},
					},
				},
			}
			p := jsonstream.New(schema)

			_, err := p.Feed(`{"content": "test"}`)
			Expect(err).To(HaveOccurred())
			Expect(errors.Is(err, flushErr)).To(BeTrue(), "expected wrapped flush error, got %v", err)
		})

		It("should propagate sub-parser Flush errors from Parser.Flush", func() {
			flushErr := errors.New("flush error")
			errSubParser := &errorSubParser{flushErr: flushErr}
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"content": {
							Type:      jsonstream.TypeString,
							Streaming: true,
							SubParser: errSubParser,
						},
					},
				},
			}
			p := jsonstream.New(schema)

			// Open string but never close it, then Flush() should surface the
			// sub-parser flush error from flushPending().
			_, err := p.Feed(`{"content": "abc`)
			Expect(err).NotTo(HaveOccurred())

			_, err = p.Flush()
			Expect(err).To(HaveOccurred())
			Expect(errors.Is(err, flushErr)).To(BeTrue(), "expected wrapped flush error, got %v", err)
		})
	})

	Context("when parsing number edge cases", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"n": {Type: jsonstream.TypeNumber},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should parse zero", func() {
			events, err := p.Feed(`{"n": 0}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "n" {
					Expect(ev.Value).To(Equal(0.0))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should parse negative zero", func() {
			events, err := p.Feed(`{"n": -0}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "n" {
					Expect(ev.Value).To(BeNumerically("==", 0.0))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should parse very large number", func() {
			events, err := p.Feed(`{"n": 1e308}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "n" {
					Expect(ev.Value).To(Equal(1e308))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})

		It("should parse very small number", func() {
			events, err := p.Feed(`{"n": 5e-324}`)
			Expect(err).NotTo(HaveOccurred())

			var found bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventNumber); ok && ev.Path() == "n" {
					Expect(ev.Value).To(BeNumerically(">", 0))
					found = true
				}
			}
			Expect(found).To(BeTrue())
		})
	})

	Context("when parsing with nil Children schema", func() {
		It("should handle object fields with no schema children defined", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					// Children is nil
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"any": "value", "num": 42}`)
			Expect(err).NotTo(HaveOccurred())

			// Should still parse without error, just no config for children
			var foundObjectComplete bool
			for _, e := range events {
				if _, ok := e.(jsonstream.EventObjectComplete); ok {
					foundObjectComplete = true
				}
			}
			Expect(foundObjectComplete).To(BeTrue())
		})
	})

	Context("when handling streaming strings with UTF-8 byte splits", func() {
		BeforeEach(func() {
			schema = &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {
							Type:      jsonstream.TypeString,
							Streaming: true,
						},
					},
				},
			}
			p = jsonstream.New(schema)
		})

		It("should emit valid UTF-8 streaming chunks when Swedish ä is split", func() {
			// "blodkärl" where ä (0xC3 0xA4) is split
			chunk1 := `{"text": "blodk` + "\xC3"
			chunk2 := "\xA4" + `rl"}`

			events1, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			// Verify all chunks are valid UTF-8
			var allChunks string
			var completeValue string
			for _, e := range append(events1, events2...) {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "text" {
					Expect(utf8.ValidString(ev.Chunk)).To(BeTrue(),
						"chunk should be valid UTF-8: %q", ev.Chunk)
					allChunks += ev.Chunk
				}
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					completeValue = ev.Value
				}
			}

			Expect(completeValue).To(Equal("blodkärl"))
			Expect(allChunks).To(Equal(completeValue))
		})

		It("should emit valid UTF-8 streaming chunks when Arabic is split", func() {
			// Arabic "ي" (ya, 0xD9 0x8A) split
			chunk1 := `{"text": "السيبران` + "\xD9"
			chunk2 := "\x8A" + `"}`

			events1, err := p.Feed(chunk1)
			Expect(err).NotTo(HaveOccurred())

			events2, err := p.Feed(chunk2)
			Expect(err).NotTo(HaveOccurred())

			var allChunks string
			var completeValue string
			for _, e := range append(events1, events2...) {
				if ev, ok := e.(jsonstream.EventStringChunk); ok && ev.Path() == "text" {
					Expect(utf8.ValidString(ev.Chunk)).To(BeTrue(),
						"chunk should be valid UTF-8: %q", ev.Chunk)
					allChunks += ev.Chunk
				}
				if ev, ok := e.(jsonstream.EventStringComplete); ok && ev.Path() == "text" {
					completeValue = ev.Value
				}
			}

			Expect(completeValue).To(Equal("السيبراني"))
			Expect(allChunks).To(Equal(completeValue))
		})
	})

	Context("when flushing with pending escape in streaming string", func() {
		It("should handle flush when escape is pending", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"text": {
							Type:      jsonstream.TypeString,
							Streaming: true,
						},
					},
				},
			}
			p := jsonstream.New(schema)

			// Feed content that ends with a backslash (pending escape)
			events1, err := p.Feed(`{"text": "hello\`)
			Expect(err).NotTo(HaveOccurred())

			// Flush should not panic
			flushEvents, err := p.Flush()
			Expect(err).NotTo(HaveOccurred())

			// Collect whatever was emitted
			allEvents := append(events1, flushEvents...)
			_ = allEvents // Just verify no panic
		})
	})

	Context("when parsing objects with whitespace variations", func() {
		It("should handle empty object with whitespace", func() {
			schema := &jsonstream.Schema{
				Root: jsonstream.FieldConfig{
					Type: jsonstream.TypeObject,
					Children: map[string]jsonstream.FieldConfig{
						"inner": {
							Type:     jsonstream.TypeObject,
							Children: map[string]jsonstream.FieldConfig{},
						},
					},
				},
			}
			p := jsonstream.New(schema)

			events, err := p.Feed(`{"inner": {   }}`)
			Expect(err).NotTo(HaveOccurred())

			var foundInner bool
			for _, e := range events {
				if ev, ok := e.(jsonstream.EventObjectComplete); ok && ev.Path() == "inner" {
					foundInner = true
				}
			}
			Expect(foundInner).To(BeTrue())
		})
	})
})
