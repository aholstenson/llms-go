package llms

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
)

// defaultMaxSteps is used when WithMaxSteps was not set (or set to 0).
const defaultMaxSteps = 10

// stepPhase tracks where a Session is in the plan/observe cycle of a single
// model turn. A Session is phaseReady between turns and phaseAwaitingObserve
// after StepPlan (or ResumeForToolResults) yielded tool calls that have not
// yet been folded back via StepObserve.
type stepPhase int

const (
	phaseReady stepPhase = iota
	phaseAwaitingObserve
)

// StepInfo describes the result of a single Session.Step.
type StepInfo struct {
	// Step is the 1-indexed model turn this info describes.
	Step int
	// Output is what the model produced this turn.
	Output TurnOutput
	// Outcomes are the results of this turn's tool calls (nil if none).
	Outcomes []ToolOutcome
	// Exec is the live execution tracker.
	Exec ExecutionContext
}

// Session is a driveable agentic loop. Each call to Step advances exactly one
// model turn; the caller drives the loop and may interpose between turns
// (inspect/approve tool calls, inject a message, fork a sub-agent).
// GenerateContent is a Session run to completion internally.
//
// A Session is not safe for concurrent use. All methods must be called from a
// single goroutine. To inject messages from another goroutine (e.g. an
// operator-steering UI), send them over a channel and have the driving
// goroutine call Inject between Steps.
type Session struct {
	turn    Turn
	tracker *executionTracker
	toolMap map[string]ToolDef
	opts    *generateContentOptions
	logger  *slog.Logger

	maxSteps int

	step       int
	done       bool
	lastErr    error
	stopReason StopReason
	messages   []*Message

	// phase / plan-observe bookkeeping. planOutput stashes the StepPlan
	// TurnOutput so StepObserve can feed it to the live Turn.Observe;
	// reconstructedObserve routes StepObserve to ObserveToolResults (no
	// stashed provider response) and pendingCalls carries the originating
	// calls for that path.
	phase                stepPhase
	planOutput           TurnOutput
	reconstructedObserve bool
	pendingCalls         []ToolCall
}

// PlanInfo is the result of the model-call phase of a step.
type PlanInfo struct {
	// Step is the 1-indexed model turn this info describes.
	Step int
	// Output is what the model produced this turn.
	Output TurnOutput
	// ToolCalls == Output.ToolCalls, surfaced for convenience.
	ToolCalls []ToolCall
	// NeedsTools is true when the caller must run tools then StepObserve.
	NeedsTools bool
	// Exec is the live execution tracker.
	Exec ExecutionContext
}

// newSession is the internal constructor used by providers (and tests). The
// provider builds its native Turn, an execution tracker and tool map, then
// hands them here. Token/step rollup and streaming-event emission live here.
func newSession(turn Turn, tracker *executionTracker, toolMap map[string]ToolDef, opts *generateContentOptions, logger *slog.Logger) *Session {
	maxSteps := opts.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}

	// Seed the neutral transcript with the caller's input messages.
	messages := make([]*Message, 0, len(opts.Messages)+4)
	messages = append(messages, opts.Messages...)

	return &Session{
		turn:     turn,
		tracker:  tracker,
		toolMap:  toolMap,
		opts:     opts,
		logger:   logger,
		maxSteps: maxSteps,
		messages: messages,
	}
}

// newTracker builds the execution tracker for a generation, rolling up to a
// parent when WithParentExecution was supplied (the sub-agent pattern).
func newTracker(opts *generateContentOptions) *executionTracker {
	maxSteps := opts.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}
	if opts.ParentExecution != nil {
		return newChildTracker(maxSteps, opts.ParentExecution)
	}
	return newExecutionTracker(maxSteps)
}

// sessionModel is implemented by provider models that can construct a
// driveable Session. It is the seam used by the public NewSession.
type sessionModel interface {
	newSession(ctx context.Context, options ...GenerateOption) (*Session, error)
}

// NewSession creates a driveable Session for the given model. The model must
// be one of the built-in providers (Anthropic, OpenAI/OpenRouter, Google).
// The ctx is used for session-creation work such as ConditionalTool
// availability checks; it is not retained for later turns.
func NewSession(ctx context.Context, m Model, options ...GenerateOption) (*Session, error) {
	sm, ok := m.(sessionModel)
	if !ok {
		return nil, fmt.Errorf("model %T does not support driveable sessions", m)
	}
	return sm.newSession(ctx, options...)
}

// Step advances one model turn: Turn.Next -> (if tool calls) parallel
// doToolCall fan-out -> Turn.Observe. It returns done=true when the model
// ends its turn with no tool calls, the step budget is exhausted, or ctx is
// cancelled. A non-nil error is returned for genuine failures (provider
// error, context cancellation); step-limit termination is reported via
// done=true with Result() returning a *MaxStepsError.
func (s *Session) Step(ctx context.Context) (StepInfo, bool, error) {
	plan, done, err := s.StepPlan(ctx)
	if err != nil {
		return StepInfo{Step: plan.Step, Output: plan.Output, Exec: plan.Exec}, true, err
	}
	if done {
		return StepInfo{Step: plan.Step, Output: plan.Output, Exec: plan.Exec}, true, nil
	}

	outcomes, err := s.RunTools(ctx, plan.ToolCalls)
	if err != nil {
		s.fail(err)
		return StepInfo{Step: s.step, Output: plan.Output, Exec: s.tracker}, true, err
	}

	return s.StepObserve(ctx, outcomes)
}

// StepPlan runs exactly one model turn (Turn.Next) WITHOUT executing tools.
// NeedsTools == len(ToolCalls)>0 && !done. When NeedsTools is false the
// session is done (terminal/budget/ctx) and StepObserve MUST NOT be called.
// Conversely, when NeedsTools is true the caller MUST run the tools and then
// call StepObserve exactly once.
func (s *Session) StepPlan(ctx context.Context) (PlanInfo, bool, error) {
	if s.done {
		return PlanInfo{Step: s.step, Exec: s.tracker}, true, s.lastErr
	}

	if s.phase == phaseAwaitingObserve {
		return PlanInfo{Step: s.step, Exec: s.tracker}, false,
			errors.New("StepPlan called before StepObserve")
	}

	if ctx.Err() != nil {
		s.fail(ctx.Err())
		return PlanInfo{Step: s.step, Exec: s.tracker}, true, ctx.Err()
	}

	// Make the execution tracker visible to tools (and sub-agents).
	ctx = withExecutionContext(ctx, s.tracker)

	s.step++
	// WithMaxSteps(N) permits exactly N model calls.
	if s.step > s.maxSteps {
		s.step--
		s.done = true
		s.stopReason = StopReasonMaxSteps
		return PlanInfo{Step: s.step, Exec: s.tracker}, true, nil
	}

	s.tracker.IncrementStep()

	if s.opts.StreamingFunc != nil {
		if err := s.opts.StreamingFunc(ctx, StreamingEventMessageStart{}); err != nil {
			s.fail(err)
			return PlanInfo{Step: s.step, Exec: s.tracker}, true, err
		}
	}

	out, err := s.turn.Next(ctx)
	if err != nil {
		s.fail(err)
		return PlanInfo{Step: s.step, Exec: s.tracker}, true, err
	}

	s.stopReason = out.StopReason
	s.tracker.AddTokens(out.Usage.InputTokens, out.Usage.OutputTokens, out.Usage.CachedReadTokens, out.Usage.CachedWriteTokens)
	if am := assistantMessage(out); am != nil {
		s.messages = append(s.messages, am)
	}

	if len(out.ToolCalls) == 0 {
		if s.opts.StreamingFunc != nil {
			if err := s.opts.StreamingFunc(ctx, StreamingEventMessageEnd{Final: true}); err != nil {
				s.fail(err)
				return PlanInfo{Step: s.step, Output: out, Exec: s.tracker}, true, err
			}
		}
		s.done = true
		return PlanInfo{Step: s.step, Output: out, Exec: s.tracker}, true, nil
	}

	// Tool turn. The non-final MessageEnd means "model finished talking" and
	// is emitted here, before tools run, so it stays strictly before the
	// tooluse/toolresult events that originate in tool execution.
	if s.opts.StreamingFunc != nil {
		if err := s.opts.StreamingFunc(ctx, StreamingEventMessageEnd{Final: false}); err != nil {
			s.fail(err)
			return PlanInfo{Step: s.step, Output: out, Exec: s.tracker}, true, err
		}
	}

	s.phase = phaseAwaitingObserve
	s.planOutput = out
	return PlanInfo{
		Step:       s.step,
		Output:     out,
		ToolCalls:  out.ToolCalls,
		NeedsTools: true,
		Exec:       s.tracker,
	}, false, nil
}

// StepObserve completes a step: it appends the tool-outcome message to the
// neutral transcript and folds the outcomes into the provider-native history
// (Turn.Observe for a live plan, Turn.ObserveToolResults for a Session
// reconstructed via ResumeForToolResults). Call exactly once after a
// StepPlan / ResumeForToolResults that yielded tool calls. It emits no
// streaming events. StepObserve is not a model call and does not consume
// step budget.
func (s *Session) StepObserve(ctx context.Context, outcomes []ToolOutcome) (StepInfo, bool, error) {
	if s.done {
		return StepInfo{Step: s.step, Exec: s.tracker}, true, s.lastErr
	}

	if s.phase != phaseAwaitingObserve {
		return StepInfo{Step: s.step, Exec: s.tracker}, false,
			errors.New("StepObserve without pending plan")
	}

	ctx = withExecutionContext(ctx, s.tracker)

	s.messages = append(s.messages, toolOutcomeMessage(outcomes))

	out := s.planOutput
	var err error
	if s.reconstructedObserve {
		err = s.turn.ObserveToolResults(ctx, s.pendingCalls, outcomes)
	} else {
		err = s.turn.Observe(ctx, out, outcomes)
	}
	if err != nil {
		s.fail(err)
		return StepInfo{Step: s.step, Output: out, Outcomes: outcomes, Exec: s.tracker}, true, err
	}

	s.phase = phaseReady
	s.reconstructedObserve = false
	s.pendingCalls = nil
	s.planOutput = TurnOutput{}
	return StepInfo{Step: s.step, Output: out, Outcomes: outcomes, Exec: s.tracker}, false, nil
}

// ResumeForToolResults puts a Session reconstructed from a persisted
// transcript (replayed via WithMessages) into the awaiting-observe state for
// the assistant turn's tool calls, so StepObserve can feed outcomes back
// without a model call in this process. It does not consume step budget and
// must be called on a ready, non-done session.
func (s *Session) ResumeForToolResults(calls []ToolCall) error {
	if s.done {
		if s.lastErr != nil {
			return s.lastErr
		}
		return errors.New("ResumeForToolResults on a finished session")
	}
	if s.phase != phaseReady {
		return errors.New("ResumeForToolResults requires a ready session")
	}
	s.phase = phaseAwaitingObserve
	s.reconstructedObserve = true
	s.pendingCalls = calls
	return nil
}

// RunTools executes the given tool calls in parallel using the session's
// tool registry, preserving call order in the returned slice. Cancellation
// mid-fan-in surfaces ctx.Err(). The session's ExecutionContext is attached
// to ctx so tools (and any sub-agents they spawn) can see it.
//
// RunTools is the standard tool-execution path used by Step. Call it
// directly when you want to interpose between StepPlan and StepObserve —
// for example, to gate or audit calls before they run — while still using
// the session's tool dispatch and parallelism.
func (s *Session) RunTools(ctx context.Context, calls []ToolCall) ([]ToolOutcome, error) {
	ctx = withExecutionContext(ctx, s.tracker)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	outcomes := make([]ToolOutcome, len(calls))

	type indexed struct {
		i       int
		outcome ToolOutcome
	}
	ch := make(chan indexed, len(calls))

	pending := 0
	for i, call := range calls {
		tool, ok := s.toolMap[call.Name]
		if !ok {
			outcomes[i] = ToolOutcome{ID: call.ID, Name: call.Name, Error: ErrToolNotFound}
			continue
		}

		pending++
		go func(i int, call ToolCall, tool ToolDef) {
			outcome := ToolOutcome{ID: call.ID, Name: call.Name}
			result, err := doToolCall(ctx, s.logger, s.opts.StreamingFunc, s.opts.ToolCallTimeout, call.ID, tool, call.Arguments)
			if err != nil {
				outcome.Error = err
			} else {
				outcome.Text = result
			}
			ch <- indexed{i: i, outcome: outcome}
		}(i, call, tool)
	}

	for k := 0; k < pending; k++ {
		select {
		case r := <-ch:
			outcomes[r.i] = r.outcome
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	return outcomes, nil
}

// assistantMessage renders a turn's output as a neutral assistant message:
// the model text (if any) followed by a ToolCallPart per requested call. It
// returns nil when the turn produced neither text nor tool calls, so empty
// messages never enter the transcript.
func assistantMessage(out TurnOutput) *Message {
	parts := make([]MessagePart, 0, len(out.Thinking)+1+len(out.ToolCalls))
	// Thinking blocks come first: Anthropic requires thinking before text and
	// tool_use in the assistant turn, and convertMessages relies on it.
	for _, tb := range out.Thinking {
		parts = append(parts, &ThinkingPart{
			Text:      tb.Text,
			Signature: tb.Signature,
			Redacted:  tb.Redacted,
			Data:      tb.Data,
		})
	}
	if out.Text != "" {
		parts = append(parts, NewTextPart(out.Text))
	}
	for _, c := range out.ToolCalls {
		parts = append(parts, &ToolCallPart{ID: c.ID, Name: c.Name, Arguments: c.Arguments})
	}
	if len(parts) == 0 {
		return nil
	}
	return NewMessage(RoleAssistant, parts...)
}

// toolOutcomeMessage renders tool outcomes into a neutral user message of
// ToolResultParts, each paired to its ToolCallPart by ID. This keeps the
// accumulated transcript faithful and replayable via WithMessages, even
// though providers keep their own native history as the request's source of
// truth.
func toolOutcomeMessage(outcomes []ToolOutcome) *Message {
	parts := make([]MessagePart, 0, len(outcomes))
	for _, o := range outcomes {
		parts = append(parts, &ToolResultPart{ID: o.ID, Name: o.Name, Text: o.Text, Error: o.ModelError()})
	}
	return NewMessage(RoleUser, parts...)
}

func (s *Session) fail(err error) {
	s.done = true
	s.lastErr = err
	// Keep a failed session's phase consistent so a later StepPlan hits the
	// done-guard rather than the awaiting-observe ordering error.
	s.phase = phaseReady
}

// Inject appends caller messages as a user turn before the next Step.
func (s *Session) Inject(msgs ...*Message) {
	s.turn.Inject(msgs...)
	s.messages = append(s.messages, msgs...)
}

// Result returns the terminal Result for the session. It returns a
// *MaxStepsError when the run was step-limited, a *MaxTokensError when the
// final turn was truncated by the output-token cap, a *RefusalError when the
// model declined to produce content, or the underlying error if the run
// failed.
func (s *Session) Result() (Result, error) {
	if s.lastErr != nil {
		return nil, s.lastErr
	}

	switch s.stopReason {
	case StopReasonMaxSteps:
		return nil, &MaxStepsError{Result: s.loopResult()}
	case StopReasonMaxTokens:
		return nil, &MaxTokensError{Result: s.loopResult()}
	case StopReasonRefusal:
		return nil, &RefusalError{Result: s.loopResult()}
	}

	text := s.turn.FinalText()
	if s.opts.ResponseSchema != nil {
		result, err := s.opts.ResponseSchema.ParseInto([]byte(text))
		if err != nil {
			return nil, fmt.Errorf("structured output parsing failed (raw: %s): %w", text, err)
		}
		return result.(Result), nil
	}

	return TextResult{Text: text}, nil
}

func (s *Session) loopResult() *LoopResult {
	return &LoopResult{
		StopReason: s.stopReason,
		Steps:      s.step,
		FinalText:  s.turn.FinalText(),
		Messages:   s.Messages(),
		Usage: TurnUsage{
			InputTokens:       s.tracker.InputTokens(),
			OutputTokens:      s.tracker.OutputTokens(),
			CachedReadTokens:  s.tracker.CachedTokens(),
			CachedWriteTokens: s.tracker.CachedWriteTokens(),
		},
	}
}

// Messages returns the neutral accumulated transcript.
func (s *Session) Messages() []*Message {
	out := make([]*Message, len(s.messages))
	copy(out, s.messages)
	return out
}

// StopReason returns the most recent stop reason observed by the session.
func (s *Session) StopReason() StopReason {
	return s.stopReason
}

// runSession drives a Session to completion. It is the shared body of every
// provider's GenerateContent.
func runSession(ctx context.Context, s *Session) (Result, error) {
	for {
		_, done, err := s.Step(ctx)
		if err != nil {
			return nil, err
		}
		if done {
			break
		}
	}
	return s.Result()
}
