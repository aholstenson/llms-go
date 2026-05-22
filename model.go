package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"

	jsonstream "github.com/aholstenson/llms-go/jsonstream"
)

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

// StreamingEventToolError is the terminal event for a tool call that failed,
// emitted in place of StreamingEventToolResult. It always follows the matching
// StreamingEventToolUse (paired by ID), so consumers tracking the tool-call
// lifecycle can move the call out of a running state on failure. Error carries
// the failure (including tool panics, surfaced as a VisibleToolError).
//
// Because Go type switches are not exhaustiveness-checked, a consumer that does
// not handle this event will leave a failed call pinned to its running state;
// handle it alongside StreamingEventToolResult.
type StreamingEventToolError struct {
	ID     string
	ToolID string
	Error  error
}

func (StreamingEventToolError) isEvent() {}

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
	MaxOutputTokens   int
	MaxThinkingTokens int
	Temperature       float64
	Tools             []ToolDef
	MaxSteps          int
	SystemPrompt      string
	Messages          []*Message
	StreamingFunc     StreamingFunc
	WebSearch         bool
	ToolCallTimeout   time.Duration
	ParentExecution   ExecutionContext
	// Retry options
	MaxRetries    int
	RetryAfterCap time.Duration
	RetryBackoff  BackoffPolicy
	// Structured output options
	ResponseSchema                   *ResponseSchema
	StructuredStreamingFunc          StructuredStreamingFunc
	StructuredStreamingSchema        *jsonstream.Schema // Optional custom jsonstream schema
	StructuredStreamingSchemaAuto    bool               // True if schema was auto-generated
	StructuredStreamingSchemaBuilder structuredStreamingSchemaBuilder
}

// DefaultMaxRetries is the default number of retries applied to failing
// requests. Matches the Anthropic and OpenAI SDK defaults so behavior is
// uniform across providers out of the box.
const DefaultMaxRetries = 2

// DefaultRetryAfterCap clamps server-supplied Retry-After hints so a
// pathological hint cannot stall a request for minutes.
const DefaultRetryAfterCap = 60 * time.Second

// DefaultToolCallTimeout is the timeout applied to a single tool call when no
// timeout is configured via WithToolCallTimeout.
const DefaultToolCallTimeout = 5 * time.Minute

type GenerateOption func(*generateContentOptions) error

// defaultGenerateContentOptions returns a generateContentOptions populated
// with library defaults. Options applied on top of this overwrite the
// defaults; an explicit zero from an option therefore means zero, not
// "use the default."
func defaultGenerateContentOptions() *generateContentOptions {
	return &generateContentOptions{
		ToolCallTimeout: DefaultToolCallTimeout,
		MaxRetries:      DefaultMaxRetries,
		RetryAfterCap:   DefaultRetryAfterCap,
		RetryBackoff:    defaultBackoffPolicy(),
	}
}

// resolveGenerateContentOptions applies GenerateOptions and resolves any
// deferred schema builder using the model's sub-parser registry.
func resolveGenerateContentOptions(registry map[string]SubParserConfig, opts ...GenerateOption) (*generateContentOptions, error) {
	o := defaultGenerateContentOptions()
	for _, opt := range opts {
		if err := opt(o); err != nil {
			return nil, err
		}
	}
	if o.StructuredStreamingSchemaBuilder != nil && o.StructuredStreamingSchema == nil {
		o.StructuredStreamingSchema = o.StructuredStreamingSchemaBuilder(registry)
	}
	return o, nil
}

// Model represents an LLM API that can be used to generate content.
type Model interface {
	GenerateContent(ctx context.Context, opts ...GenerateOption) (Result, error)
}

// WithMaxOutputTokens sets the maximum number of output tokens the model may
// produce in a single response. This cap covers visible output only; any
// thinking budget configured via WithMaxThinkingTokens is added on top by
// providers that count thinking against the output limit.
func WithMaxOutputTokens(maxOutputTokens int) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.MaxOutputTokens = maxOutputTokens
		return nil
	}
}

// WithMaxThinkingTokens sets the maximum number of tokens to use for thinking.
func WithMaxThinkingTokens(maxThinkingTokens int) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.MaxThinkingTokens = maxThinkingTokens
		return nil
	}
}

// WithTemperature sets the temperature of the LLM. A higher temperature will
// result in more creative and varied responses.
func WithTemperature(temperature float64) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.Temperature = temperature
		return nil
	}
}

// WithTools sets the tools that the LLM can use.
func WithTools(tools ...ToolDef) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.Tools = tools
		return nil
	}
}

// WithMaxSteps sets the maximum number of steps, such as tool calls, to make.
func WithMaxSteps(maxSteps int) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.MaxSteps = maxSteps
		return nil
	}
}

// WithToolCallTimeout sets the maximum duration a single tool call may run
// before its context is cancelled. Pass 0 to disable the timeout. If the
// option is not set at all, DefaultToolCallTimeout (5 minutes) applies.
func WithToolCallTimeout(timeout time.Duration) GenerateOption {
	return func(opts *generateContentOptions) error {
		if timeout < 0 {
			return fmt.Errorf("llms.WithToolCallTimeout: timeout must be >= 0, got %s", timeout)
		}
		opts.ToolCallTimeout = timeout
		return nil
	}
}

// WithParentExecution rolls this generation's token and tool-call totals up
// to a parent ExecutionContext. Use it for the lightweight sub-agent pattern:
// a tool's Execute calls another model's GenerateContent with
// WithParentExecution(GetExecutionContext(ctx)) so child spend accrues to the
// parent tracker.
//
// Step budgets are NOT rolled up: the child runs its own independent
// WithMaxSteps budget and does not consume the parent's remaining steps. Only
// token and tool-call accounting accrues to the parent.
func WithParentExecution(parent ExecutionContext) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.ParentExecution = parent
		return nil
	}
}

// WithMaxRetries sets the maximum number of retry attempts after the initial
// request. The total number of attempts is n+1. Default is DefaultMaxRetries
// (2). Passing 0 disables retries; n < 0 returns an error.
//
// For Anthropic and OpenAI this is forwarded to the SDK, which runs its own
// retry loop (with its own backoff policy — see WithRetryBackoff). For
// Google and OpenRouter the retry loop is owned by llms-go and obeys
// WithRetryBackoff.
//
// Streaming generations are never retried at the body level by any
// provider — once a stream has begun emitting events, mid-stream failures
// surface to the caller with ErrStreamingPartialOutput.
func WithMaxRetries(n int) GenerateOption {
	return func(opts *generateContentOptions) error {
		if n < 0 {
			return fmt.Errorf("llms.WithMaxRetries: n must be >= 0, got %d", n)
		}
		opts.MaxRetries = n
		return nil
	}
}

// WithRetryAfterCap clamps server-supplied Retry-After hints to the given
// upper bound. Default is DefaultRetryAfterCap (60s). Pass 0 to disable the
// cap entirely. The cap is applied to both the internal sleep duration and
// to the value surfaced on UnavailableError.RetryAfter.
func WithRetryAfterCap(d time.Duration) GenerateOption {
	return func(opts *generateContentOptions) error {
		if d < 0 {
			return fmt.Errorf("llms.WithRetryAfterCap: duration must be >= 0, got %s", d)
		}
		opts.RetryAfterCap = d
		return nil
	}
}

// WithRetryBackoff overrides the BackoffPolicy used by the llms-go-owned
// retry loop (Google and OpenRouter, non-streaming only).
//
// NOTE: This option does not affect Anthropic or OpenAI. Those SDKs run
// their own internal backoff and llms-go cannot inject a policy into them;
// only WithMaxRetries is plumbed through. The default ExponentialBackoff
// parameters are chosen to match the SDK defaults so behavior is uniform
// across all four providers out of the box. If you need identical custom
// backoff for all providers, install a custom HTTP transport on the
// Anthropic/OpenAI clients yourself, or set MaxRetries to 0 there and own
// retries via an outer loop.
func WithRetryBackoff(p BackoffPolicy) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.RetryBackoff = p
		return nil
	}
}

// WithSystemPrompt sets the system prompt for the LLM.
func WithSystemPrompt(systemPrompt string) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.SystemPrompt = systemPrompt
		return nil
	}
}

// WithMessages sets the messages that the LLM will use to generate content.
func WithMessages(messages ...*Message) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.Messages = messages
		return nil
	}
}

// WithStreamingFunc enables streaming of the LLM's response. The given function
// will be called with each chunk of the response.
func WithStreamingFunc(streaming StreamingFunc) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.StreamingFunc = streaming
		return nil
	}
}

// WithWebSearch enables web search tool calls.
func WithWebSearch(webSearch bool) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.WebSearch = webSearch
		return nil
	}
}

// WithResponseSchema enables structured output with a JSON schema derived from type T.
// The schema name is auto-derived from the type name (e.g., "TopicSummary" -> "topic_summary").
func WithResponseSchema[T any]() GenerateOption {
	schema := jsonSchemaReflector.Reflect(new(T))
	name := deriveSchemaName[T]()
	return func(opts *generateContentOptions) error {
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
		return nil
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
	return func(opts *generateContentOptions) error {
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
		return nil
	}
}

// WithStructuredStreamingCustom enables structured streaming with a custom jsonstream schema.
// Use this for advanced control over which fields emit streaming events.
func WithStructuredStreamingCustom[T any](jsSchema *jsonstream.Schema, handler StructuredStreamingFunc) GenerateOption {
	schema := jsonSchemaReflector.Reflect(new(T))
	name := deriveSchemaName[T]()
	return func(opts *generateContentOptions) error {
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
		return nil
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
