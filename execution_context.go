package llms

import (
	"context"
	"sync"
)

// ExecutionContext provides read access to the current execution state during
// an LLM generation loop. Tools can use this to adapt their behavior based on
// how many times they've been called or how many tokens have been consumed.
type ExecutionContext interface {
	// CurrentStep returns the current step number (1-indexed).
	CurrentStep() int

	// MaxSteps returns the maximum number of steps allowed.
	MaxSteps() int

	// RemainingSteps returns how many steps are left.
	RemainingSteps() int

	// ToolCallCount returns the number of times a specific tool has been called.
	ToolCallCount(toolName string) int

	// TotalToolCalls returns the total number of tool calls across all tools.
	TotalToolCalls() int

	// InputTokens returns cumulative input tokens across all steps.
	InputTokens() int64

	// OutputTokens returns cumulative output tokens across all steps.
	OutputTokens() int64

	// CachedTokens returns cumulative cached tokens across all steps.
	CachedTokens() int64

	// TotalTokens returns the sum of input and output tokens.
	TotalTokens() int64
}

// ExecutionTracker is the mutable implementation of ExecutionContext used by
// LLM providers to track execution state.
type ExecutionTracker struct {
	mu sync.RWMutex

	// Step tracking
	currentStep int
	maxSteps    int

	// Tool call tracking
	toolCalls      map[string]int
	totalToolCalls int

	// Token tracking (cumulative across all steps)
	inputTokens  int64
	outputTokens int64
	cachedTokens int64
}

// NewExecutionTracker creates a new execution tracker with the given max steps.
func NewExecutionTracker(maxSteps int) *ExecutionTracker {
	return &ExecutionTracker{
		maxSteps:  maxSteps,
		toolCalls: make(map[string]int),
	}
}

// Read methods (implement ExecutionContext)

func (t *ExecutionTracker) CurrentStep() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.currentStep
}

func (t *ExecutionTracker) MaxSteps() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.maxSteps
}

func (t *ExecutionTracker) RemainingSteps() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.maxSteps - t.currentStep
}

func (t *ExecutionTracker) ToolCallCount(toolName string) int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.toolCalls[toolName]
}

func (t *ExecutionTracker) TotalToolCalls() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.totalToolCalls
}

func (t *ExecutionTracker) InputTokens() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.inputTokens
}

func (t *ExecutionTracker) OutputTokens() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.outputTokens
}

func (t *ExecutionTracker) CachedTokens() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.cachedTokens
}

func (t *ExecutionTracker) TotalTokens() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.inputTokens + t.outputTokens
}

// Write methods (used only by LLM providers)

// IncrementStep increments the current step counter.
func (t *ExecutionTracker) IncrementStep() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.currentStep++
}

// RecordToolCall records that a tool was called.
func (t *ExecutionTracker) RecordToolCall(toolName string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.toolCalls[toolName]++
	t.totalToolCalls++
}

// AddTokens adds token counts to the cumulative totals.
func (t *ExecutionTracker) AddTokens(input, output, cached int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.inputTokens += input
	t.outputTokens += output
	t.cachedTokens += cached
}

// Ensure ExecutionTracker implements ExecutionContext
var _ ExecutionContext = (*ExecutionTracker)(nil)

type executionContextKey struct{}

// WithExecutionContext adds an ExecutionContext to the context.
func WithExecutionContext(ctx context.Context, ec ExecutionContext) context.Context {
	return context.WithValue(ctx, executionContextKey{}, ec)
}

// GetExecutionContext retrieves the ExecutionContext from the context, or nil if not present.
func GetExecutionContext(ctx context.Context) ExecutionContext {
	if ec, ok := ctx.Value(executionContextKey{}).(ExecutionContext); ok {
		return ec
	}
	return nil
}
