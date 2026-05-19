package llms

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"time"

	"github.com/cockroachdb/errors"

	jsonstream "github.com/aholstenson/llms-go/jsonstream"
)

var ErrRefusal = errors.New("llm refused to generate content")

// Result is returned by GenerateContent.
type Result interface {
	isResult()
}

// TextResult contains plain text response.
type TextResult struct {
	Text string
}

func (TextResult) isResult() {}

// StructuredResult contains parsed JSON response.
type StructuredResult[T any] struct {
	Data T
	Raw  string // Original JSON for debugging
}

func (StructuredResult[T]) isResult() {}

// ResponseSchema contains the JSON schema for structured output.
type ResponseSchema struct {
	Name      string
	Schema    any                       // *jsonschema.Schema
	ParseInto func([]byte) (any, error) // For parsing into typed result
}

// StructuredStreamingFunc is called with jsonstream events during structured streaming.
type StructuredStreamingFunc func(ctx context.Context, event jsonstream.Event) error

type StreamingEvent interface {
	isEvent()
}

type StreamingEventTextChunk struct {
	Text string
}

func (StreamingEventTextChunk) isEvent() {}

type StreamingEventToolUse struct {
	ID        string
	ToolID    string
	Arguments any
}

func (StreamingEventToolUse) isEvent() {}

type StreamingEventToolResult struct {
	ID     string
	ToolID string
	Result any
}

func (StreamingEventToolResult) isEvent() {}

type StreamingEventCitation struct {
	Title string
	URL   string
}

func (StreamingEventCitation) isEvent() {}

// StreamingEventMessageStart signals the beginning of a new LLM response message.
type StreamingEventMessageStart struct{}

func (StreamingEventMessageStart) isEvent() {}

// StreamingEventMessageEnd signals the end of an LLM response message.
// Final is true when this is the last message in an agentic loop (no more
// tool calls will follow). Intermediate replies have Final set to false.
type StreamingEventMessageEnd struct {
	Final bool
}

func (StreamingEventMessageEnd) isEvent() {}

// StreamingEventThinking streams thinking/reasoning token chunks.
type StreamingEventThinking struct {
	Text string
}

func (StreamingEventThinking) isEvent() {}

// StreamingFunc is a function that will be called when the LLM is streaming
// content.
type StreamingFunc func(ctx context.Context, event StreamingEvent) error

// structuredStreamingSchemaBuilder builds a jsonstream schema using the
// provided sub-parser registry. The generic type T is captured in a closure
// so it survives past the initial WithStructuredStreaming call.
type structuredStreamingSchemaBuilder func(registry map[string]SubParserConfig) *jsonstream.Schema

type generateContentOptions struct {
	MaxTokens         int
	MaxThinkingTokens int
	Temperature       float64
	Tools             []ToolDef
	MaxSteps          int
	SystemPrompt      string
	Messages          []*Message
	StreamingFunc     StreamingFunc
	WebSearch         bool
	ToolCallTimeout   time.Duration
	// Structured output options
	ResponseSchema                      *ResponseSchema
	StructuredStreamingFunc             StructuredStreamingFunc
	StructuredStreamingSchema           *jsonstream.Schema // Optional custom jsonstream schema
	StructuredStreamingSchemaAuto       bool               // True if schema was auto-generated
	StructuredStreamingSchemaBuilder    structuredStreamingSchemaBuilder
}

// DefaultToolCallTimeout is the timeout applied to a single tool call when no
// timeout is configured via WithToolCallTimeout.
const DefaultToolCallTimeout = 5 * time.Minute

type GenerateOption func(*generateContentOptions)

// resolveGenerateContentOptions applies GenerateOptions and resolves any
// deferred schema builder using the model's sub-parser registry.
func resolveGenerateContentOptions(registry map[string]SubParserConfig, opts ...GenerateOption) *generateContentOptions {
	o := &generateContentOptions{}
	for _, opt := range opts {
		opt(o)
	}
	if o.ToolCallTimeout == 0 {
		o.ToolCallTimeout = DefaultToolCallTimeout
	}
	if o.StructuredStreamingSchemaBuilder != nil && o.StructuredStreamingSchema == nil {
		o.StructuredStreamingSchema = o.StructuredStreamingSchemaBuilder(registry)
	}
	return o
}

// Model represents an LLM API that can be used to generate content.
type Model interface {
	GenerateContent(ctx context.Context, opts ...GenerateOption) (Result, error)
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(maxTokens int) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.MaxTokens = maxTokens
	}
}

// WithMaxThinkingTokens sets the maximum number of tokens to use for thinking.
func WithMaxThinkingTokens(maxThinkingTokens int) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.MaxThinkingTokens = maxThinkingTokens
	}
}

// WithTemperature sets the temperature of the LLM. A higher temperature will
// result in more creative and varied responses.
func WithTemperature(temperature float64) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.Temperature = temperature
	}
}

// WithTools sets the tools that the LLM can use.
func WithTools(tools ...ToolDef) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.Tools = tools
	}
}

// WithMaxSteps sets the maximum number of steps, such as tool calls, to make.
func WithMaxSteps(maxSteps int) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.MaxSteps = maxSteps
	}
}

// WithToolCallTimeout sets the maximum duration a single tool call may run
// before its context is cancelled. If not set, DefaultToolCallTimeout is used.
// A negative duration disables the timeout entirely.
func WithToolCallTimeout(timeout time.Duration) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.ToolCallTimeout = timeout
	}
}

// WithSystemPrompt sets the system prompt for the LLM.
func WithSystemPrompt(systemPrompt string) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.SystemPrompt = systemPrompt
	}
}

// WithMessages sets the messages that the LLM will use to generate content.
func WithMessages(messages ...*Message) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.Messages = messages
	}
}

// WithStreamingFunc enables streaming of the LLM's response. The given function
// will be called with each chunk of the response.
func WithStreamingFunc(streaming StreamingFunc) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.StreamingFunc = streaming
	}
}

// WithWebSearch enables web search tool calls.
func WithWebSearch(webSearch bool) GenerateOption {
	return func(opts *generateContentOptions) {
		opts.WebSearch = webSearch
	}
}

// WithResponseSchema enables structured output with a JSON schema derived from type T.
// The schema name is auto-derived from the type name (e.g., "TopicSummary" -> "topic_summary").
func WithResponseSchema[T any]() GenerateOption {
	schema := jsonSchemaReflector.Reflect(new(T))
	name := deriveSchemaName[T]()
	return func(opts *generateContentOptions) {
		opts.ResponseSchema = &ResponseSchema{
			Name:   name,
			Schema: schema,
			ParseInto: func(data []byte) (any, error) {
				var result T
				if err := json.Unmarshal(data, &result); err != nil {
					return nil, err
				}
				return StructuredResult[T]{Data: result, Raw: string(data)}, nil
			},
		}
	}
}

// WithStructuredStreaming enables structured streaming output with auto-generated jsonstream schema.
// Events are emitted as the JSON is parsed incrementally.
//
// All string fields automatically have streaming enabled. Use struct tags or options to configure
// sub-parsers for specific fields. Tag values are resolved via the Manager's sub-parser registry.
//
// Struct tag example:
//
//	type Response struct {
//	    Topic string `json:"topic"`
//	    Text  string `json:"text" jsonstream:"markdown"`  // Resolved via Manager.RegisterSubParser("markdown", ...)
//	}
//
// Programmatic example (overrides tags):
//
//	llms.WithStructuredStreaming[Response](handler,
//	    llms.ConfigureSubParser("text", mySubParserConfig),
//	)
func WithStructuredStreaming[T any](handler StructuredStreamingFunc, streamOpts ...StructuredStreamingOption) GenerateOption {
	schema := jsonSchemaReflector.Reflect(new(T))
	name := deriveSchemaName[T]()
	return func(opts *generateContentOptions) {
		opts.ResponseSchema = &ResponseSchema{
			Name:   name,
			Schema: schema,
			ParseInto: func(data []byte) (any, error) {
				var result T
				if err := json.Unmarshal(data, &result); err != nil {
					return nil, err
				}
				return StructuredResult[T]{Data: result, Raw: string(data)}, nil
			},
		}
		opts.StructuredStreamingFunc = handler
		opts.StructuredStreamingSchemaAuto = true
		opts.StructuredStreamingSchemaBuilder = func(registry map[string]SubParserConfig) *jsonstream.Schema {
			return ConvertToJsonstreamSchemaFromType[T](registry, streamOpts...)
		}
	}
}

// WithStructuredStreamingCustom enables structured streaming with a custom jsonstream schema.
// Use this for advanced control over which fields emit streaming events.
func WithStructuredStreamingCustom[T any](jsSchema *jsonstream.Schema, handler StructuredStreamingFunc) GenerateOption {
	schema := jsonSchemaReflector.Reflect(new(T))
	name := deriveSchemaName[T]()
	return func(opts *generateContentOptions) {
		opts.ResponseSchema = &ResponseSchema{
			Name:   name,
			Schema: schema,
			ParseInto: func(data []byte) (any, error) {
				var result T
				if err := json.Unmarshal(data, &result); err != nil {
					return nil, err
				}
				return StructuredResult[T]{Data: result, Raw: string(data)}, nil
			},
		}
		opts.StructuredStreamingFunc = handler
		opts.StructuredStreamingSchema = jsSchema
		opts.StructuredStreamingSchemaAuto = false
	}
}

// deriveSchemaName derives a snake_case schema name from a type name.
// e.g., "TopicSummary" -> "topic_summary"
func deriveSchemaName[T any]() string {
	t := reflect.TypeOf((*T)(nil)).Elem()
	name := t.Name()
	return toSnakeCase(name)
}

// toSnakeCase converts a CamelCase string to snake_case.
func toSnakeCase(s string) string {
	var result strings.Builder
	for i, r := range s {
		if i > 0 && r >= 'A' && r <= 'Z' {
			result.WriteByte('_')
		}
		if r >= 'A' && r <= 'Z' {
			result.WriteRune(r + 32) // Convert to lowercase
		} else {
			result.WriteRune(r)
		}
	}
	return result.String()
}
