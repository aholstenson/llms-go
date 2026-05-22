package llms

import (
	"context"
	"errors"
	"fmt"
)

// StopReason describes why a model turn (or the loop) stopped.
type StopReason string

const (
	StopReasonEndTurn   StopReason = "end_turn"
	StopReasonToolUse   StopReason = "tool_use"
	StopReasonMaxTokens StopReason = "max_tokens"
	StopReasonRefusal   StopReason = "refusal"
	// StopReasonMaxSteps is loop-level: it is set by Session when the
	// configured step budget is exhausted, never by a Turn.
	StopReasonMaxSteps StopReason = "max_steps"
)

// ToolCall is a single tool invocation requested by the model. Arguments is
// the raw JSON arguments string as produced by the provider.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// ToolOutcome is the result of executing a ToolCall. Exactly one of Text or
// Error is meaningful: Error is non-nil when the call failed, otherwise
// Text holds the rendered tool result.
//
// The full error is preserved in Error for callers and logs. The message
// sent to the model is produced by ModelError, which redacts internal
// detail unless the error chain contains a VisibleToolError.
type ToolOutcome struct {
	ID    string
	Name  string
	Text  string
	Error error
}

// ModelError returns the message that should be sent to the model for this
// outcome. It returns "" when the call succeeded. When Error is non-nil but
// the chain contains no VisibleToolError, a generic placeholder is returned
// so internal details (paths, secrets, stack traces) are not leaked.
func (o ToolOutcome) ModelError() string {
	if o.Error == nil {
		return ""
	}
	var v *VisibleToolError
	if errors.As(o.Error, &v) {
		return v.Message
	}
	return "tool execution failed"
}

// VisibleToolError marks an error as safe to surface to the model in the
// tool-result block. Tools that want the model to read a specific message
// (for self-correction, etc.) should return a VisibleToolError directly or
// wrap one in a chain via fmt.Errorf("...: %w", ...).
//
// Any error that does not contain a VisibleToolError in its chain is
// reduced to a generic placeholder when rendered for the model; the full
// error remains in ToolOutcome.Error for callers and logs.
type VisibleToolError struct {
	Message string
}

func (e *VisibleToolError) Error() string { return e.Message }

// NewVisibleToolError returns a VisibleToolError with a formatted message.
func NewVisibleToolError(format string, args ...any) *VisibleToolError {
	return &VisibleToolError{Message: fmt.Sprintf(format, args...)}
}

// ErrToolNotFound is set on ToolOutcome.Error when a ToolCall references a
// tool name not present in the session's tool registry. It is a
// VisibleToolError, so the model sees the message and can recover by
// calling a different tool.
var ErrToolNotFound = &VisibleToolError{Message: "requested tool not found"}

// TurnUsage carries per-turn token accounting in neutral form.
//
// The prompt-token buckets are disjoint: every prompt token is counted in
// exactly one of InputTokens, CachedReadTokens, or CachedWriteTokens. So the
// total number of prompt tokens processed for a turn is:
//
//	InputTokens + CachedReadTokens + CachedWriteTokens
//
// This mirrors Anthropic's native usage accounting. Providers whose APIs
// instead report a single prompt total with the cached tokens nested inside it
// (Google, OpenAI, OpenRouter) are normalized to this disjoint form before
// TurnUsage is populated, so InputTokens is always *uncached* input regardless
// of provider. Keeping the buckets disjoint is what makes cost correct: cached
// tokens are billed once at the (cheaper) cache rate, never double-counted at
// the full input rate as well.
type TurnUsage struct {
	// InputTokens is the count of fresh (uncached) prompt tokens — prompt
	// tokens that were neither read from nor written to the cache.
	InputTokens int64
	// OutputTokens is the count of generated output tokens.
	OutputTokens int64
	// CachedReadTokens is the count of prompt tokens served from the cache,
	// billed at the cache-read rate. Disjoint from InputTokens.
	CachedReadTokens int64
	// CachedWriteTokens is the count of prompt tokens written to the cache when
	// creating a new entry, billed at the cache-write rate. Reported only by
	// providers with explicit caching (Anthropic); zero for providers whose
	// caching is implicit. Disjoint from InputTokens.
	CachedWriteTokens int64
	// ThinkingTokens is the count of reasoning/thinking tokens when the provider
	// reports them separately.
	ThinkingTokens int64
}

// uncachedInputTokens returns the number of fresh input tokens, i.e. prompt
// tokens not accounted to the cache. Providers whose API reports a prompt total
// with the cached tokens nested inside (Google, OpenAI, OpenRouter) call this to
// derive the disjoint InputTokens that TurnUsage, pricing, and metrics expect.
// cached must be the sum of all cache-accounted prompt tokens for the turn —
// cache reads plus cache writes (OpenRouter is the only such provider that
// reports writes). Anthropic's API already reports a disjoint input_tokens and
// does not use this helper.
//
// The result is clamped at zero to defend against a provider ever reporting a
// cached count larger than the prompt total.
func uncachedInputTokens(promptTotal, cached int64) int64 {
	if fresh := promptTotal - cached; fresh > 0 {
		return fresh
	}
	return 0
}

// ThinkingBlock is one neutral extended-thinking/reasoning block produced by
// the model. Signature is the provider-opaque cryptographic signature
// (Anthropic) that must be replayed verbatim; Data carries the encrypted
// payload of a redacted thinking block when Redacted is true.
type ThinkingBlock struct {
	Text      string
	Signature string
	Data      string
	Redacted  bool
}

// TurnOutput is the neutral result of exactly one model API call.
type TurnOutput struct {
	Text       string
	Thinking   []ThinkingBlock
	ToolCalls  []ToolCall
	StopReason StopReason
	Usage      TurnUsage
}

// Turn is the provider-specific seam: exactly one model API call. It is
// stateful and owns its provider-native message history, so provider
// bookkeeping (Anthropic cache-control, Google Part ordering, thinking
// signatures) never round-trips through neutral types.
type Turn interface {
	// Next performs exactly one model API call, emitting streaming events
	// (chunk/thinking/tooluse) internally, and returns the neutral output.
	Next(ctx context.Context) (TurnOutput, error)

	// Observe appends the assistant message and the tool results from this
	// turn to the provider-native history, ready for the next Next.
	Observe(ctx context.Context, out TurnOutput, outcomes []ToolOutcome) error

	// ObserveToolResults folds outcomes into native history WITHOUT appending
	// an assistant message — used when the assistant turn is already in native
	// history (a Session reconstructed via WithMessages where Next never ran
	// this process). calls carries the originating tool calls so providers
	// that pair results to calls can do so when no stashed response exists.
	ObserveToolResults(ctx context.Context, calls []ToolCall, outcomes []ToolOutcome) error

	// Inject queues caller-supplied messages to be appended before the next
	// Next call.
	Inject(msgs ...*Message)

	// FinalText returns the assistant text for the terminal turn, or the
	// accumulated jsonstream buffer when structured streaming is active.
	FinalText() string
}

// LoopResult is the trajectory recovered from a step-limited run via
// MaxStepsError.
type LoopResult struct {
	StopReason StopReason
	Steps      int
	FinalText  string
	Messages   []*Message
	Usage      TurnUsage
}

// MaxStepsError is returned by GenerateContent / Session.Result when the
// loop hits its step budget. Callers can errors.As it to recover the
// trajectory accumulated before the budget ran out.
type MaxStepsError struct {
	Result *LoopResult
}

func (e *MaxStepsError) Error() string { return "max steps reached" }

// MaxTokensError is returned by GenerateContent / Session.Result when the
// final turn stopped because the model hit the output-token cap mid-generation.
// Callers can errors.As it to recover the partial text and usage that was
// produced before truncation.
type MaxTokensError struct {
	Result *LoopResult
}

func (e *MaxTokensError) Error() string { return "max tokens reached" }

// RefusalError is returned by GenerateContent / Session.Result when the model
// declined to produce content. errors.As recovers any partial trajectory the
// model emitted before refusing.
type RefusalError struct {
	Result *LoopResult
}

func (e *RefusalError) Error() string { return "llm refused to generate content" }

// StreamScope identifies which agent/run a stream of events belongs to. It is
// carried out-of-band on the context so sub-agents can be distinguished
// without changing the StreamingEvent types.
type StreamScope struct {
	AgentID string
	RunID   string
}

type streamScopeKey struct{}

// WithStreamScope attaches a StreamScope to the context.
func WithStreamScope(ctx context.Context, s StreamScope) context.Context {
	return context.WithValue(ctx, streamScopeKey{}, s)
}

// GetStreamScope retrieves the StreamScope from the context, if present.
func GetStreamScope(ctx context.Context) (StreamScope, bool) {
	s, ok := ctx.Value(streamScopeKey{}).(StreamScope)
	return s, ok
}
