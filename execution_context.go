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

// executionTracker is the mutable implementation of ExecutionContext used
// internally to track execution state. It is intentionally unexported:
// external code only ever reads through the ExecutionContext interface.
type executionTracker struct {
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

	// parent, when set, receives token and tool-call rollups so a sub-agent's
	// spend accrues to its caller's budget.
	parent *executionTracker
}

// newExecutionTracker creates a new execution tracker with the given max steps.
func newExecutionTracker(maxSteps int) *executionTracker {
	return &executionTracker{
		maxSteps:  maxSteps,
		toolCalls: make(map[string]int),
	}
}

// newChildTracker creates an execution tracker whose token and tool-call
// totals also roll up to the given parent. Step counts are deliberately not
// rolled up: the child keeps its own independent step budget. If parent is
// not an *executionTracker, the child behaves like a plain tracker.
func newChildTracker(maxSteps int, parent ExecutionContext) *executionTracker {
	t := newExecutionTracker(maxSteps)
	if p, ok := parent.(*executionTracker); ok {
		t.parent = p
	}
	return t
}

// Read methods (implement ExecutionContext)

func (t *executionTracker) CurrentStep() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.currentStep
}

func (t *executionTracker) MaxSteps() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.maxSteps
}

func (t *executionTracker) RemainingSteps() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.maxSteps - t.currentStep
}

func (t *executionTracker) ToolCallCount(toolName string) int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.toolCalls[toolName]
}

func (t *executionTracker) TotalToolCalls() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.totalToolCalls
}

func (t *executionTracker) InputTokens() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.inputTokens
}

func (t *executionTracker) OutputTokens() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.outputTokens
}

func (t *executionTracker) CachedTokens() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.cachedTokens
}

func (t *executionTracker) TotalTokens() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.inputTokens + t.outputTokens
}

// Write methods (used only by the session loop)

func (t *executionTracker) IncrementStep() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.currentStep++
}

func (t *executionTracker) RecordToolCall(toolName string) {
	t.mu.Lock()
	t.toolCalls[toolName]++
	t.totalToolCalls++
	parent := t.parent
	t.mu.Unlock()
	if parent != nil {
		parent.RecordToolCall(toolName)
	}
}

func (t *executionTracker) AddTokens(input, output, cached int64) {
	t.mu.Lock()
	t.inputTokens += input
	t.outputTokens += output
	t.cachedTokens += cached
	parent := t.parent
	t.mu.Unlock()
	if parent != nil {
		parent.AddTokens(input, output, cached)
	}
}

var _ ExecutionContext = (*executionTracker)(nil)

type executionContextKey struct{}

// withExecutionContext attaches the session's tracker to ctx so downstream
// tool calls can record into it.
func withExecutionContext(ctx context.Context, tracker *executionTracker) context.Context {
	return context.WithValue(ctx, executionContextKey{}, tracker)
}

// GetExecutionContext retrieves the ExecutionContext from the context, or nil if not present.
func GetExecutionContext(ctx context.Context) ExecutionContext {
	if t, ok := ctx.Value(executionContextKey{}).(*executionTracker); ok {
		return t
	}
	return nil
}
