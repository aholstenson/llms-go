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
	ReasoningEffort   Effort
	ReasoningMode     ReasoningMode
	Temperature       float64
	Tools             []ToolDef
	MaxSteps          int
	SystemPrompt      string
	Messages          []*Message
	StreamingFunc     StreamingFunc
	WebSearch         bool
	ToolCallTimeout   time.Duration
	ParentExecution   ExecutionContext
	// restoreSnapshot, when set via WithSnapshot, seeds a restored Session's
	// transcript, step budget, phase, and cumulative usage.
	restoreSnapshot *SessionSnapshot
	// Retry options
	MaxRetries    int
	RetryAfterCap time.Duration
	RetryBackoff  BackoffPolicy
	RetryNotify   RetryNotifyFunc
	// Liveness options
	StallTimeout   time.Duration
	RequestTimeout time.Duration
	LivenessNotify LivenessNotifyFunc
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
	if err := resolveSnapshot(o); err != nil {
		return nil, err
	}
	return o, nil
}

// resolveSnapshot validates a restore snapshot and seeds the transcript from
// it. It is the single point where WithSnapshot becomes the request's
// Messages, so restore flows uniformly through every provider's
// convertMessages without per-provider changes.
func resolveSnapshot(o *generateContentOptions) error {
	snap := o.restoreSnapshot
	if snap == nil {
		return nil
	}
	if o.Messages != nil {
		return fmt.Errorf("WithSnapshot and WithMessages are mutually exclusive")
	}
	if len(snap.Transcript) == 0 {
		return fmt.Errorf("WithSnapshot: snapshot has an empty transcript")
	}
	switch snap.Phase {
	case SessionPhaseReady:
	case SessionPhaseAwaitingTools:
		if len(snap.PendingCalls) == 0 {
			return fmt.Errorf("WithSnapshot: %s snapshot has no pending tool calls", snap.Phase)
		}
	default:
		return fmt.Errorf("WithSnapshot: unknown snapshot phase %q", snap.Phase)
	}
	o.Messages = snap.Transcript
	return nil
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

// Effort is a portable reasoning-effort level: the primary, provider-
// independent knob for how much a model reasons before answering. It is
// translated per-provider (OpenAI reasoning effort, Anthropic output-config
// effort / adaptive thinking, Gemini thinking level, OpenRouter effort) using
// per-model metadata, and converted to a token budget for budget-style models
// (older Anthropic models, Gemini 2.5).
//
// Not every model accepts every tier. A tier the model does not accept is
// moved to the nearest tier it does accept (with a warning), matching how
// temperature is gated — portability over strictness.
//
// The zero value is the empty string, meaning "no reasoning option given",
// which leaves reasoning at the provider's default (see ReasoningDefault).
type Effort string

const (
	// EffortNone turns reasoning off, the same as WithReasoning(ReasoningOff).
	EffortNone Effort = "none"
	// EffortMinimal requests the least reasoning a model can do without
	// turning it off.
	EffortMinimal Effort = "minimal"
	// EffortLow requests little reasoning.
	EffortLow Effort = "low"
	// EffortMedium requests a moderate amount of reasoning.
	EffortMedium Effort = "medium"
	// EffortHigh requests extensive reasoning.
	EffortHigh Effort = "high"
	// EffortXHigh requests more reasoning than EffortHigh.
	EffortXHigh Effort = "xhigh"
	// EffortMax requests the most reasoning the model can do.
	EffortMax Effort = "max"
)

// ReasoningMode selects whether a model reasons (thinks) before answering.
type ReasoningMode int

const (
	// ReasoningDefault sends no reasoning setting, so the provider's default
	// for the model applies. Some models reason by default (for example
	// Claude Opus 5 and later, Gemini 2.5 and 3, most OpenAI reasoning
	// models) and some do not.
	ReasoningDefault ReasoningMode = iota
	// ReasoningOff turns reasoning off. Models that cannot turn reasoning
	// off use their lowest effort tier instead, with a warning.
	ReasoningOff
	// ReasoningOn turns reasoning on. Without WithReasoningEffort the
	// provider's default depth applies, except for OpenAI and OpenRouter,
	// which use EffortMedium.
	ReasoningOn
)

// WithReasoning sets whether the model reasons. ReasoningOff and
// ReasoningDefault clear an effort set with WithReasoningEffort; ReasoningOn
// keeps it.
func WithReasoning(mode ReasoningMode) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.ReasoningMode = mode
		if mode != ReasoningOn {
			opts.ReasoningEffort = ""
		}
		return nil
	}
}

// WithReasoningEffort turns reasoning on at the given effort. This is the
// recommended way to control reasoning/thinking across providers: the effort
// is mapped to each provider's native control, or to a token budget for
// budget-style models. EffortNone turns reasoning off.
//
// Without a reasoning option the provider's default applies; see
// ReasoningDefault. An effort given after WithReasoning(ReasoningOff) turns
// reasoning back on: the last reasoning option wins.
func WithReasoningEffort(e Effort) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.ReasoningEffort = e
		return nil
	}
}

// WithMaxThinkingTokens sets the maximum number of tokens to use for thinking.
//
// Deprecated: prefer WithReasoningEffort, the portable reasoning knob. This
// option remains as an advanced escape hatch for budget-style models (older
// Anthropic models, Gemini 2.5) and OpenRouter, where it turns reasoning on
// and takes precedence over WithReasoningEffort. On effort-only models
// (OpenAI, newer Anthropic and Gemini) it does not control reasoning and is
// ignored except for reserving output headroom; use WithReasoningEffort there
// instead.
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
// The retry loop is owned by llms-go for every provider, so WithMaxRetries,
// WithRetryBackoff and WithRetryNotify behave the same everywhere.
//
// A stream is never retried after its first event: mid-stream failures
// surface to the caller with ErrStreamingPartialOutput. Anthropic and
// OpenAI stream every generation and retry a failure to open the stream;
// Google and OpenRouter retry their non-streaming requests.
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

// WithRetryBackoff overrides the BackoffPolicy that decides how long to wait
// between attempts. It applies to every provider. The default is an
// ExponentialBackoff with 500ms base, an 8s cap and 25% jitter, which honors
// a server Retry-After hint when there is one.
func WithRetryBackoff(p BackoffPolicy) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.RetryBackoff = p
		return nil
	}
}

// WithRetryNotify registers a callback that runs after a failed attempt,
// just before the library waits to make the next attempt. Use it to show
// retry progress to the user:
//
//	llms.WithRetryNotify(func(_ context.Context, n llms.RetryNotice) {
//		log.Printf("Failed attempt %d of %d, waiting %s", n.Attempt, n.MaxAttempts, n.Delay)
//	})
//
// The callback runs on the goroutine that made the request, so keep it
// short. It is not called for an error that is not retryable, and not called
// for the last attempt, because no wait follows it — use the returned
// UnavailableError to report the final failure. Pass nil to remove a
// callback.
func WithRetryNotify(fn RetryNotifyFunc) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.RetryNotify = fn
		return nil
	}
}

// WithStallTimeout fails a model call that has already started responding and
// then goes quiet for longer than d. Pass 0 (the default) to disable it.
//
// The budget measures bytes from the provider, not the events the library
// hands you. Keepalives and provider ping events are bytes, so a model that
// spends minutes reasoning keeps resetting the budget; a connection that died
// mid-response does not. This is the setting that turns a silent hang into a
// visible failure.
//
// A stall is retryable. It is reported through WithRetryNotify like a rate
// limit, retried while nothing has reached your streaming callbacks, and
// surfaces as an UnavailableError wrapping a *StallError when the attempts run
// out. Because the budget applies per attempt, and only to the model call, it
// does not cut short a slow tool call (see WithToolCallTimeout) or a long
// agentic loop.
//
// Pick a value well above the provider's keepalive interval — a minute or two
// is generous for every supported provider:
//
//	llms.WithStallTimeout(90 * time.Second)
func WithStallTimeout(d time.Duration) GenerateOption {
	return func(opts *generateContentOptions) error {
		if d < 0 {
			return fmt.Errorf("llms.WithStallTimeout: duration must be >= 0, got %s", d)
		}
		opts.StallTimeout = d
		return nil
	}
}

// WithRequestTimeout bounds how long one attempt may wait for the provider to
// start responding. Pass 0 (the default) to disable it.
//
// What that covers depends on the provider. Anthropic and OpenAI stream, so it
// is the time to open the stream and the rest is governed by WithStallTimeout.
// Google and OpenRouter make non-streaming requests when no streaming callback
// is set, and those send nothing until the answer is complete — so for them
// this is effectively a deadline on the whole generation. Size it accordingly,
// or leave it off and rely on WithStallTimeout for the streaming paths.
//
// The budget applies per attempt, so exceeding it is a retryable *StallError
// rather than the end of the run.
func WithRequestTimeout(d time.Duration) GenerateOption {
	return func(opts *generateContentOptions) error {
		if d < 0 {
			return fmt.Errorf("llms.WithRequestTimeout: duration must be >= 0, got %s", d)
		}
		opts.RequestTimeout = d
		return nil
	}
}

// WithLivenessNotify registers a callback that runs when data arrives from the
// provider after a quiet gap of at least LivenessQuietThreshold. Use it to
// tell "the model is thinking" apart from "the connection is dead" while a
// turn produces no tokens:
//
//	llms.WithLivenessNotify(func(_ context.Context, n llms.LivenessNotice) {
//		log.Printf("%s alive, quiet for %s", n.Provider, n.Idle)
//	})
//
// Notices stop arriving the moment the provider goes silent, so a watcher can
// show the growing gap instead of guessing. Gaps shorter than the threshold
// are not reported: while tokens flow you already have the stream.
//
// The callback runs on the goroutine reading the response, so keep it short.
// Pass nil to remove a callback.
func WithLivenessNotify(fn LivenessNotifyFunc) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.LivenessNotify = fn
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

// WithSnapshot restores a Session from a SessionSnapshot captured earlier with
// Session.Snapshot, reconstructing the transcript, the consumed step budget,
// the plan/observe phase, and cumulative token usage. Use it with NewSession,
// re-supplying tools and other options (notably the same WithMaxSteps) as for
// the original session:
//
//	s, err := llms.NewSession(ctx, model,
//		llms.WithSnapshot(&snap),
//		llms.WithTools(tools...),
//		llms.WithMaxSteps(8),
//	)
//
// When the snapshot's phase is SessionPhaseAwaitingTools the restored session
// is primed for StepObserve with the outcomes of snap.PendingCalls (no model
// call); otherwise the next call is StepPlan/Step. WithSnapshot is mutually
// exclusive with WithMessages — it supplies the transcript itself.
func WithSnapshot(snap *SessionSnapshot) GenerateOption {
	return func(opts *generateContentOptions) error {
		opts.restoreSnapshot = snap
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
