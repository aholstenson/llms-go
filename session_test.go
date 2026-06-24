package llms

import (
	"context"
	"errors"
	"log/slog"
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// fakeTurn is a scripted Turn used to exercise Session loop semantics without
// any provider. Each Next pops the next scripted TurnOutput; emit, when set,
// is called from Next so streaming-event ordering can be asserted.
type fakeTurn struct {
	mu       sync.Mutex
	script   []TurnOutput
	nextIdx  int
	nextCnt  int
	observed              [][]ToolOutcome
	reconstructedObserved [][]ToolOutcome
	injected              []*Message
	finalTxt string
	emit     func(ctx context.Context) error
}

func (f *fakeTurn) Next(ctx context.Context) (TurnOutput, error) {
	f.mu.Lock()
	f.nextCnt++
	var out TurnOutput
	if f.nextIdx < len(f.script) {
		out = f.script[f.nextIdx]
		f.nextIdx++
	} else {
		// Past the script: keep requesting a tool so a step-limited run
		// never naturally terminates.
		out = TurnOutput{
			Text:       "loop",
			ToolCalls:  []ToolCall{{ID: "loop", Name: "echo", Arguments: `{"v":"loop"}`}},
			StopReason: StopReasonToolUse,
		}
	}
	f.finalTxt = out.Text
	emit := f.emit
	f.mu.Unlock()

	if emit != nil {
		if err := emit(ctx); err != nil {
			return TurnOutput{}, err
		}
	}
	return out, nil
}

func (f *fakeTurn) Observe(_ context.Context, _ TurnOutput, outcomes []ToolOutcome) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.observed = append(f.observed, outcomes)
	return nil
}

func (f *fakeTurn) ObserveToolResults(_ context.Context, _ []ToolCall, outcomes []ToolOutcome) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.reconstructedObserved = append(f.reconstructedObserved, outcomes)
	return nil
}

func (f *fakeTurn) Inject(msgs ...*Message) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.injected = append(f.injected, msgs...)
}

func (f *fakeTurn) FinalText() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.finalTxt
}

func (f *fakeTurn) calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.nextCnt
}

// sessArgs / sessTool is a trivial tool returning a configurable value.
type sessArgs struct {
	V string `json:"v"`
}

type sessTool struct {
	name string
	fn   func(ctx context.Context, in sessArgs) (string, error)
}

func (t sessTool) Name() string        { return t.name }
func (t sessTool) Description() string { return "echo tool" }
func (t sessTool) Schema() *sessArgs   { return &sessArgs{} }
func (t sessTool) ToString(s string) string {
	return s
}
func (t sessTool) Execute(ctx context.Context, in *sessArgs) (string, error) {
	if t.fn != nil {
		return t.fn(ctx, *in)
	}
	return in.V, nil
}

func newTestSession(turn Turn, tools []ToolDef, options ...GenerateOption) (*Session, *executionTracker) {
	opts, err := resolveGenerateContentOptions(nil, options...)
	if err != nil {
		panic(err)
	}
	toolMap := make(map[string]ToolDef, len(tools))
	for _, t := range tools {
		toolMap[t.Name()] = t
	}
	// Mirror production: newTracker applies the maxSteps fallback and seeds
	// from a restore snapshot when WithSnapshot was supplied.
	tracker := newTracker(opts)
	return newSession(turn, tracker, toolMap, opts, slog.Default()), tracker
}

var _ = Describe("Session", func() {
	echo := NewToolDef[*sessArgs, string](sessTool{name: "echo"})

	Describe("step counting and termination", func() {
		It("runs tool turns then ends, accumulating usage", func() {
			turn := &fakeTurn{script: []TurnOutput{
				{
					Text:       "thinking",
					ToolCalls:  []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"hi"}`}},
					StopReason: StopReasonToolUse,
					Usage:      TurnUsage{InputTokens: 10, OutputTokens: 5, CachedReadTokens: 2},
				},
				{
					Text:       "all done",
					StopReason: StopReasonEndTurn,
					Usage:      TurnUsage{InputTokens: 7, OutputTokens: 3},
				},
			}}
			s, tracker := newTestSession(turn, []ToolDef{echo})

			res, err := runSession(context.Background(), s)
			Expect(err).NotTo(HaveOccurred())
			Expect(res).To(Equal(TextResult{Text: "all done"}))
			Expect(turn.calls()).To(Equal(2))
			Expect(s.StopReason()).To(Equal(StopReasonEndTurn))
			Expect(tracker.InputTokens()).To(Equal(int64(17)))
			Expect(tracker.OutputTokens()).To(Equal(int64(8)))
			Expect(tracker.CachedTokens()).To(Equal(int64(2)))
			Expect(turn.observed).To(HaveLen(1))
			Expect(turn.observed[0]).To(Equal([]ToolOutcome{{ID: "1", Name: "echo", Text: "hi"}}))
		})
	})

	Describe("fixed max-steps semantics", func() {
		It("permits exactly N model calls for WithMaxSteps(N)", func() {
			turn := &fakeTurn{} // empty script => always requests a tool
			s, _ := newTestSession(turn, []ToolDef{echo}, WithMaxSteps(3))

			res, err := runSession(context.Background(), s)
			Expect(res).To(BeNil())
			Expect(turn.calls()).To(Equal(3))

			var mse *MaxStepsError
			Expect(errors.As(err, &mse)).To(BeTrue())
			Expect(err.Error()).To(Equal("max steps reached"))
			Expect(mse.Result.StopReason).To(Equal(StopReasonMaxSteps))
			Expect(mse.Result.Steps).To(Equal(3))
			Expect(s.StopReason()).To(Equal(StopReasonMaxSteps))
		})
	})

	Describe("tool fan-out/fan-in ordering", func() {
		It("preserves tool-call order in outcomes regardless of completion order", func() {
			release := make(chan struct{})
			slow := NewToolDef[*sessArgs, string](sessTool{name: "slow", fn: func(ctx context.Context, in sessArgs) (string, error) {
				<-release // completes last
				return in.V, nil
			}})
			fast := NewToolDef[*sessArgs, string](sessTool{name: "fast"})

			turn := &fakeTurn{script: []TurnOutput{
				{
					ToolCalls: []ToolCall{
						{ID: "a", Name: "slow", Arguments: `{"v":"A"}`},
						{ID: "b", Name: "fast", Arguments: `{"v":"B"}`},
						{ID: "c", Name: "missing", Arguments: `{}`},
					},
					StopReason: StopReasonToolUse,
				},
				{Text: "fin", StopReason: StopReasonEndTurn},
			}}
			s, _ := newTestSession(turn, []ToolDef{slow, fast})

			done := make(chan struct{})
			go func() {
				defer close(done)
				_, _ = runSession(context.Background(), s)
			}()
			// Let the fast tool finish, then release the slow one.
			close(release)
			<-done

			Expect(turn.observed).To(HaveLen(1))
			Expect(turn.observed[0]).To(Equal([]ToolOutcome{
				{ID: "a", Name: "slow", Text: "A"},
				{ID: "b", Name: "fast", Text: "B"},
				{ID: "c", Name: "missing", Error: ErrToolNotFound},
			}))
		})
	})

	Describe("Inject", func() {
		It("queues messages on the turn and records them in the transcript", func() {
			turn := &fakeTurn{script: []TurnOutput{{Text: "ok", StopReason: StopReasonEndTurn}}}
			s, _ := newTestSession(turn, nil, WithMessages(NewMessage(RoleUser, NewTextPart("start"))))

			s.Inject(NewMessage(RoleUser, NewTextPart("steer")))
			_, err := runSession(context.Background(), s)
			Expect(err).NotTo(HaveOccurred())

			Expect(turn.injected).To(HaveLen(1))
			Expect(turn.injected[0].Parts[0].(*TextPart).Text).To(Equal("steer"))

			msgs := s.Messages()
			Expect(msgs[0].Parts[0].(*TextPart).Text).To(Equal("start"))
			Expect(msgs[1].Parts[0].(*TextPart).Text).To(Equal("steer"))
		})
	})

	Describe("faithful transcript", func() {
		It("records tool calls and results as paired parts, skipping empty turns", func() {
			turn := &fakeTurn{script: []TurnOutput{
				{
					Text:       "thinking",
					ToolCalls:  []ToolCall{{ID: "t1", Name: "echo", Arguments: `{"v":"hi"}`}},
					StopReason: StopReasonToolUse,
				},
				// Tool-only turn: no text => no empty assistant message.
				{
					ToolCalls:  []ToolCall{{ID: "t2", Name: "echo", Arguments: `{"v":"again"}`}},
					StopReason: StopReasonToolUse,
				},
				{Text: "done", StopReason: StopReasonEndTurn},
			}}
			s, _ := newTestSession(turn, []ToolDef{echo},
				WithMessages(NewMessage(RoleUser, NewTextPart("go"))))

			_, err := runSession(context.Background(), s)
			Expect(err).NotTo(HaveOccurred())

			msgs := s.Messages()
			// user "go"
			Expect(msgs[0].Role).To(Equal(RoleUser))
			Expect(msgs[0].Parts[0].(*TextPart).Text).To(Equal("go"))
			// assistant: text + tool call
			Expect(msgs[1].Role).To(Equal(RoleAssistant))
			Expect(msgs[1].Parts[0].(*TextPart).Text).To(Equal("thinking"))
			Expect(msgs[1].Parts[1]).To(Equal(&ToolCallPart{ID: "t1", Name: "echo", Arguments: `{"v":"hi"}`}))
			// tool result paired by ID
			Expect(msgs[2].Role).To(Equal(RoleUser))
			Expect(msgs[2].Parts[0]).To(Equal(&ToolResultPart{ID: "t1", Name: "echo", Text: "hi"}))
			// tool-only turn: assistant message is just the tool call (no empty TextPart)
			Expect(msgs[3].Role).To(Equal(RoleAssistant))
			Expect(msgs[3].Parts).To(HaveLen(1))
			Expect(msgs[3].Parts[0]).To(Equal(&ToolCallPart{ID: "t2", Name: "echo", Arguments: `{"v":"again"}`}))
			Expect(msgs[4].Parts[0]).To(Equal(&ToolResultPart{ID: "t2", Name: "echo", Text: "again"}))
			// terminal text turn
			Expect(msgs[5].Parts[0].(*TextPart).Text).To(Equal("done"))
			Expect(msgs).To(HaveLen(6))
		})
	})

	Describe("context cancellation mid-tool", func() {
		It("returns ctx.Err() while a tool is in flight", func() {
			started := make(chan struct{})
			block := make(chan struct{})
			hang := NewToolDef[*sessArgs, string](sessTool{name: "echo", fn: func(ctx context.Context, in sessArgs) (string, error) {
				close(started)
				select {
				case <-block:
				case <-ctx.Done():
				}
				return in.V, nil
			}})
			turn := &fakeTurn{script: []TurnOutput{{
				ToolCalls:  []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"x"}`}},
				StopReason: StopReasonToolUse,
			}}}
			s, _ := newTestSession(turn, []ToolDef{hang})

			ctx, cancel := context.WithCancel(context.Background())
			var resErr error
			fin := make(chan struct{})
			go func() {
				defer close(fin)
				_, resErr = runSession(ctx, s)
			}()

			<-started
			cancel()
			<-fin
			close(block)

			Expect(resErr).To(MatchError(context.Canceled))
			_, err := s.Result()
			Expect(err).To(MatchError(context.Canceled))
		})
	})

	Describe("sub-agent budget rollup", func() {
		It("rolls child token and tool spend up to the parent tracker", func() {
			parent := newExecutionTracker(10)

			child := &fakeTurn{script: []TurnOutput{
				{
					ToolCalls:  []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"q"}`}},
					StopReason: StopReasonToolUse,
					Usage:      TurnUsage{InputTokens: 4, OutputTokens: 2, CachedReadTokens: 1},
				},
				{Text: "child answer", StopReason: StopReasonEndTurn, Usage: TurnUsage{InputTokens: 3, OutputTokens: 1}},
			}}
			opts, err := resolveGenerateContentOptions(nil)
			Expect(err).NotTo(HaveOccurred())
			toolMap := map[string]ToolDef{"echo": echo}
			childTracker := newChildTracker(10, parent)
			s := newSession(child, childTracker, toolMap, opts, slog.Default())

			_, err = runSession(context.Background(), s)
			Expect(err).NotTo(HaveOccurred())

			Expect(parent.InputTokens()).To(Equal(int64(7)))
			Expect(parent.OutputTokens()).To(Equal(int64(3)))
			Expect(parent.CachedTokens()).To(Equal(int64(1)))
			Expect(parent.ToolCallCount("echo")).To(Equal(1))
			Expect(parent.TotalToolCalls()).To(Equal(1))
		})
	})

	Describe("streaming event ordering", func() {
		It("emits MessageStart/End around each turn with chunk events between", func() {
			var (
				mu     sync.Mutex
				events []string
			)
			record := func(_ context.Context, e StreamingEvent) error {
				mu.Lock()
				defer mu.Unlock()
				switch ev := e.(type) {
				case StreamingEventMessageStart:
					events = append(events, "start")
				case StreamingEventMessageEnd:
					if ev.Final {
						events = append(events, "end-final")
					} else {
						events = append(events, "end")
					}
				case StreamingEventTextChunk:
					events = append(events, "chunk:"+ev.Text)
				case StreamingEventToolUse:
					events = append(events, "tooluse")
				case StreamingEventToolResult:
					events = append(events, "toolresult")
				}
				return nil
			}

			turn := &fakeTurn{script: []TurnOutput{
				{Text: "a", ToolCalls: []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"v"}`}}, StopReason: StopReasonToolUse},
				{Text: "done", StopReason: StopReasonEndTurn},
			}}
			calls := 0
			turn.emit = func(ctx context.Context) error {
				calls++
				if calls == 1 {
					return record(ctx, StreamingEventTextChunk{Text: "a"})
				}
				return record(ctx, StreamingEventTextChunk{Text: "done"})
			}

			s, _ := newTestSession(turn, []ToolDef{echo}, WithStreamingFunc(record))
			_, err := runSession(context.Background(), s)
			Expect(err).NotTo(HaveOccurred())

			Expect(events).To(Equal([]string{
				"start", "chunk:a", "end",
				"tooluse", "toolresult",
				"start", "chunk:done", "end-final",
			}))
		})

		It("emits ToolError in place of ToolResult when a tool returns an error", func() {
			var (
				mu     sync.Mutex
				events []string
				gotErr error
			)
			record := func(_ context.Context, e StreamingEvent) error {
				mu.Lock()
				defer mu.Unlock()
				switch ev := e.(type) {
				case StreamingEventToolUse:
					events = append(events, "tooluse")
				case StreamingEventToolResult:
					events = append(events, "toolresult")
				case StreamingEventToolError:
					events = append(events, "toolerror")
					gotErr = ev.Error
				}
				return nil
			}

			boom := NewToolDef[*sessArgs, string](sessTool{name: "echo", fn: func(_ context.Context, _ sessArgs) (string, error) {
				return "", errors.New("boom")
			}})
			turn := &fakeTurn{script: []TurnOutput{
				{ToolCalls: []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"v"}`}}, StopReason: StopReasonToolUse},
				{Text: "done", StopReason: StopReasonEndTurn},
			}}

			s, _ := newTestSession(turn, []ToolDef{boom}, WithStreamingFunc(record))
			_, err := runSession(context.Background(), s)
			Expect(err).NotTo(HaveOccurred())

			Expect(events).To(Equal([]string{"tooluse", "toolerror"}))
			Expect(gotErr).To(MatchError(ContainSubstring("boom")))
		})

		It("emits ToolError when a tool panics", func() {
			var (
				mu     sync.Mutex
				events []string
				gotErr error
			)
			record := func(_ context.Context, e StreamingEvent) error {
				mu.Lock()
				defer mu.Unlock()
				switch ev := e.(type) {
				case StreamingEventToolUse:
					events = append(events, "tooluse")
				case StreamingEventToolResult:
					events = append(events, "toolresult")
				case StreamingEventToolError:
					events = append(events, "toolerror")
					gotErr = ev.Error
				}
				return nil
			}

			boom := NewToolDef[*sessArgs, string](sessTool{name: "echo", fn: func(_ context.Context, _ sessArgs) (string, error) {
				panic("kaboom")
			}})
			turn := &fakeTurn{script: []TurnOutput{
				{ToolCalls: []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"v"}`}}, StopReason: StopReasonToolUse},
				{Text: "done", StopReason: StopReasonEndTurn},
			}}

			s, _ := newTestSession(turn, []ToolDef{boom}, WithStreamingFunc(record))
			_, err := runSession(context.Background(), s)
			Expect(err).NotTo(HaveOccurred())

			Expect(events).To(Equal([]string{"tooluse", "toolerror"}))
			var visible *VisibleToolError
			Expect(errors.As(gotErr, &visible)).To(BeTrue())
			Expect(gotErr).To(MatchError(ContainSubstring("kaboom")))
		})
	})
})
