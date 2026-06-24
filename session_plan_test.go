package llms

import (
	"context"
	"encoding/json"
	"errors"
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// drive runs a Session to completion using the split StepPlan / RunTools /
// StepObserve API, mirroring what runSession does via Step.
func drive(ctx context.Context, s *Session) (Result, error) {
	for {
		plan, done, err := s.StepPlan(ctx)
		if err != nil {
			return nil, err
		}
		if done {
			break
		}
		outcomes, err := s.RunTools(ctx, plan.ToolCalls)
		if err != nil {
			s.fail(err)
			return nil, err
		}
		if _, _, err := s.StepObserve(ctx, outcomes); err != nil {
			return nil, err
		}
	}
	return s.Result()
}

var _ = Describe("Session plan/observe", func() {
	echo := NewToolDef[*sessArgs, string](sessTool{name: "echo"})

	script := func() []TurnOutput {
		return []TurnOutput{
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
		}
	}

	Describe("phase guards", func() {
		It("rejects StepObserve without a pending plan", func() {
			s, _ := newTestSession(&fakeTurn{script: script()}, []ToolDef{echo})
			_, _, err := s.StepObserve(context.Background(), nil)
			Expect(err).To(MatchError("StepObserve without pending plan"))
		})

		It("rejects a second StepPlan before StepObserve", func() {
			s, _ := newTestSession(&fakeTurn{script: script()}, []ToolDef{echo})
			_, done, err := s.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeFalse())

			_, _, err = s.StepPlan(context.Background())
			Expect(err).To(MatchError("StepPlan called before StepObserve"))
		})

		It("returns the done-guard once the session is terminal", func() {
			turn := &fakeTurn{script: []TurnOutput{{Text: "fin", StopReason: StopReasonEndTurn}}}
			s, _ := newTestSession(turn, []ToolDef{echo})

			_, done, err := s.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeTrue())

			plan, done, err := s.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeTrue())
			Expect(plan.NeedsTools).To(BeFalse())
		})
	})

	Describe("NeedsTools", func() {
		It("is true for a tool turn and false for a terminal turn", func() {
			s, _ := newTestSession(&fakeTurn{script: script()}, []ToolDef{echo})

			plan, done, err := s.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeFalse())
			Expect(plan.NeedsTools).To(BeTrue())
			Expect(plan.ToolCalls).To(HaveLen(1))

			_, _, err = s.StepObserve(context.Background(),
				[]ToolOutcome{{ID: "1", Name: "echo", Text: "hi"}})
			Expect(err).NotTo(HaveOccurred())

			plan, done, err = s.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeTrue())
			Expect(plan.NeedsTools).To(BeFalse())
		})

		It("is false/done with a *MaxStepsError when the budget is exhausted", func() {
			s, _ := newTestSession(&fakeTurn{}, []ToolDef{echo}, WithMaxSteps(1))

			plan, done, err := s.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeFalse())
			Expect(plan.NeedsTools).To(BeTrue())
			_, _, err = s.StepObserve(context.Background(),
				[]ToolOutcome{{ID: "loop", Name: "echo", Text: "loop"}})
			Expect(err).NotTo(HaveOccurred())

			_, done, err = s.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeTrue())

			_, rerr := s.Result()
			var mse *MaxStepsError
			Expect(errors.As(rerr, &mse)).To(BeTrue())
			Expect(mse.Result.StopReason).To(Equal(StopReasonMaxSteps))
		})
	})

	Describe("parity with runSession", func() {
		It("produces identical transcript, result and token rollup", func() {
			a, at := newTestSession(&fakeTurn{script: script()}, []ToolDef{echo},
				WithMessages(NewMessage(RoleUser, NewTextPart("go"))))
			ra, ea := runSession(context.Background(), a)

			b, bt := newTestSession(&fakeTurn{script: script()}, []ToolDef{echo},
				WithMessages(NewMessage(RoleUser, NewTextPart("go"))))
			rb, eb := drive(context.Background(), b)

			Expect(ea).To(BeNil())
			Expect(eb).To(BeNil())
			Expect(rb).To(Equal(ra))
			Expect(b.Messages()).To(Equal(a.Messages()))
			Expect(bt.InputTokens()).To(Equal(at.InputTokens()))
			Expect(bt.OutputTokens()).To(Equal(at.OutputTokens()))
			Expect(bt.CachedTokens()).To(Equal(at.CachedTokens()))
		})
	})

	Describe("streaming ordering via the split API", func() {
		It("matches the Step ordering", func() {
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
			_, err := drive(context.Background(), s)
			Expect(err).NotTo(HaveOccurred())

			Expect(events).To(Equal([]string{
				"start", "chunk:a", "end",
				"tooluse", "toolresult",
				"start", "chunk:done", "end-final",
			}))
		})
	})

	Describe("reconstructed session", func() {
		It("feeds tool results back without a model call or duplicated turn", func() {
			calls := []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"hi"}`}}

			a, _ := newTestSession(&fakeTurn{script: script()}, []ToolDef{echo},
				WithMessages(NewMessage(RoleUser, NewTextPart("go"))))
			plan, done, err := a.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeFalse())
			Expect(plan.ToolCalls).To(Equal(calls))

			// Reconstruct B purely from A's neutral transcript.
			bTurn := &fakeTurn{script: []TurnOutput{
				{Text: "all done", StopReason: StopReasonEndTurn},
			}}
			b, _ := newTestSession(bTurn, []ToolDef{echo}, WithMessages(a.Messages()...))

			Expect(b.ResumeForToolResults(calls)).To(Succeed())
			_, _, err = b.StepObserve(context.Background(),
				[]ToolOutcome{{ID: "1", Name: "echo", Text: "hi"}})
			Expect(err).NotTo(HaveOccurred())

			_, done, err = b.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeTrue())

			// Reconstructed dispatch went through ObserveToolResults, not Observe.
			Expect(bTurn.reconstructedObserved).To(HaveLen(1))
			Expect(bTurn.observed).To(BeEmpty())

			msgs := b.Messages()
			// user "go", assistant text+toolcall, tool result, assistant "all done"
			Expect(msgs).To(HaveLen(4))
			Expect(msgs[0].Parts[0].(*TextPart).Text).To(Equal("go"))
			Expect(msgs[1].Role).To(Equal(RoleAssistant))
			Expect(msgs[1].Parts[1]).To(Equal(&ToolCallPart{ID: "1", Name: "echo", Arguments: `{"v":"hi"}`}))
			Expect(msgs[2].Parts[0]).To(Equal(&ToolResultPart{ID: "1", Name: "echo", Text: "hi"}))
			Expect(msgs[3].Parts[0].(*TextPart).Text).To(Equal("all done"))
		})

		It("rejects ResumeForToolResults on a non-ready session", func() {
			s, _ := newTestSession(&fakeTurn{script: script()}, []ToolDef{echo})
			_, _, err := s.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(s.ResumeForToolResults(nil)).To(MatchError("ResumeForToolResults requires a ready session"))
		})
	})

	Describe("snapshot / restore", func() {
		It("round-trips an awaiting-tools snapshot and resumes without a model call", func() {
			calls := []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"hi"}`}}

			a, _ := newTestSession(&fakeTurn{script: script()}, []ToolDef{echo},
				WithMessages(NewMessage(RoleUser, NewTextPart("go"))))
			_, done, err := a.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeFalse())

			snap := a.Snapshot()
			Expect(snap.Phase).To(Equal(SessionPhaseAwaitingTools))
			Expect(snap.PendingCalls).To(Equal(calls))
			Expect(snap.Step).To(Equal(1))
			// Usage reflects the first scripted turn.
			Expect(snap.Usage.InputTokens).To(Equal(int64(10)))
			Expect(snap.Usage.OutputTokens).To(Equal(int64(5)))
			Expect(snap.Usage.CachedReadTokens).To(Equal(int64(2)))

			// Persist and reload as another process would.
			raw, err := json.Marshal(snap)
			Expect(err).NotTo(HaveOccurred())
			var restored SessionSnapshot
			Expect(json.Unmarshal(raw, &restored)).To(Succeed())

			bTurn := &fakeTurn{script: []TurnOutput{
				{Text: "all done", StopReason: StopReasonEndTurn},
			}}
			b, bt := newTestSession(bTurn, []ToolDef{echo}, WithSnapshot(&restored))

			// Cumulative usage and step survive the restore.
			Expect(bt.InputTokens()).To(Equal(int64(10)))
			Expect(bt.OutputTokens()).To(Equal(int64(5)))
			Expect(bt.CachedTokens()).To(Equal(int64(2)))

			_, _, err = b.StepObserve(context.Background(),
				[]ToolOutcome{{ID: "1", Name: "echo", Text: "hi"}})
			Expect(err).NotTo(HaveOccurred())

			plan, done, err := b.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeTrue())
			// Step numbering continues from the snapshot (the "all done" turn is 2).
			Expect(plan.Step).To(Equal(2))

			// Resume used the no-model-call fold, not Observe.
			Expect(bTurn.reconstructedObserved).To(HaveLen(1))
			Expect(bTurn.observed).To(BeEmpty())

			msgs := b.Messages()
			Expect(msgs).To(HaveLen(4))
			Expect(msgs[2].Parts[0]).To(Equal(&ToolResultPart{ID: "1", Name: "echo", Text: "hi"}))
			Expect(msgs[3].Parts[0].(*TextPart).Text).To(Equal("all done"))
		})

		It("continues the step budget across a ready-phase restore instead of resetting it", func() {
			// Empty script => every turn requests a tool, so the run only ever
			// ends by exhausting the step budget.
			a, _ := newTestSession(&fakeTurn{}, []ToolDef{echo}, WithMaxSteps(2))
			_, done, err := a.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeFalse())
			_, _, err = a.StepObserve(context.Background(),
				[]ToolOutcome{{ID: "loop", Name: "echo", Text: "loop"}})
			Expect(err).NotTo(HaveOccurred())

			snap := a.Snapshot()
			Expect(snap.Phase).To(Equal(SessionPhaseReady))
			Expect(snap.Step).To(Equal(1))

			b, _ := newTestSession(&fakeTurn{}, []ToolDef{echo},
				WithSnapshot(&snap), WithMaxSteps(2))

			// First restored plan is step 2, not step 1: the budget did not reset.
			plan, done, err := b.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeFalse())
			Expect(plan.Step).To(Equal(2))
			_, _, err = b.StepObserve(context.Background(),
				[]ToolOutcome{{ID: "loop", Name: "echo", Text: "loop"}})
			Expect(err).NotTo(HaveOccurred())

			// One more plan exhausts the original budget of 2.
			_, done, err = b.StepPlan(context.Background())
			Expect(err).NotTo(HaveOccurred())
			Expect(done).To(BeTrue())

			_, rerr := b.Result()
			var mse *MaxStepsError
			Expect(errors.As(rerr, &mse)).To(BeTrue())
			Expect(mse.Result.StopReason).To(Equal(StopReasonMaxSteps))
		})

		It("rejects invalid or conflicting snapshots", func() {
			valid := SessionSnapshot{
				Transcript: []*Message{NewMessage(RoleUser, NewTextPart("go"))},
				Phase:      SessionPhaseReady,
			}

			_, err := resolveGenerateContentOptions(nil,
				WithSnapshot(&SessionSnapshot{Phase: SessionPhaseReady}))
			Expect(err).To(MatchError(ContainSubstring("empty transcript")))

			_, err = resolveGenerateContentOptions(nil, WithSnapshot(&SessionSnapshot{
				Transcript: valid.Transcript,
				Phase:      SessionPhaseAwaitingTools,
			}))
			Expect(err).To(MatchError(ContainSubstring("no pending tool calls")))

			_, err = resolveGenerateContentOptions(nil, WithSnapshot(&SessionSnapshot{
				Transcript: valid.Transcript,
				Phase:      "bogus",
			}))
			Expect(err).To(MatchError(ContainSubstring("unknown snapshot phase")))

			_, err = resolveGenerateContentOptions(nil,
				WithMessages(NewMessage(RoleUser, NewTextPart("go"))),
				WithSnapshot(&valid))
			Expect(err).To(MatchError(ContainSubstring("mutually exclusive")))
		})

		It("survives a JSON round-trip with every part type and pending calls", func() {
			snap := SessionSnapshot{
				Transcript: []*Message{
					NewMessage(RoleUser, NewTextPart("go")),
					NewMessage(RoleAssistant,
						&ThinkingPart{Text: "ponder", Signature: "sig"},
						NewTextPart("calling"),
						NewToolCallPart("1", "echo", `{"v":"hi"}`),
					),
					NewMessage(RoleUser, &ToolResultPart{
						ID:   "1",
						Name: "echo",
						Text: "rendered",
						Attachments: []MessagePart{
							NewImagePart("https://example.com/a.png"),
							NewBinaryPart("image/png", []byte{4, 5, 6}),
						},
					}),
				},
				Step:         1,
				Phase:        SessionPhaseAwaitingTools,
				PendingCalls: []ToolCall{{ID: "1", Name: "echo", Arguments: `{"v":"hi"}`}},
				Usage:        TurnUsage{InputTokens: 10, OutputTokens: 5, CachedReadTokens: 2},
			}

			raw, err := json.Marshal(snap)
			Expect(err).NotTo(HaveOccurred())
			var back SessionSnapshot
			Expect(json.Unmarshal(raw, &back)).To(Succeed())
			Expect(back).To(Equal(snap))
		})
	})

	Describe("Message JSON codec", func() {
		It("round-trips every part type including signed/redacted thinking", func() {
			msg := NewMessage(RoleAssistant,
				&ThinkingPart{Text: "ponder", Signature: "sig-abc"},
				&ThinkingPart{Redacted: true, Data: "enc-payload"},
				NewTextPart("hello"),
				NewToolCallPart("t1", "echo", `{"v":"x"}`),
			)
			user := NewMessage(RoleUser,
				NewImagePart("https://example.com/i.png"),
				NewBinaryPart("application/pdf", []byte{1, 2, 3}),
				NewToolResultPart("t1", "echo", "x", ""),
				&ToolResultPart{
					ID:   "t2",
					Name: "screenshot",
					Text: "captured",
					Attachments: []MessagePart{
						NewImagePart("https://example.com/shot.png"),
						NewBinaryPart("image/png", []byte{9, 8, 7}),
					},
				},
			).WithCache(true)

			for _, original := range []*Message{msg, user} {
				raw, err := json.Marshal(original)
				Expect(err).NotTo(HaveOccurred())
				var back Message
				Expect(json.Unmarshal(raw, &back)).To(Succeed())
				Expect(&back).To(Equal(original))
			}
		})
	})

	Describe("Anthropic convertMessages thinking", func() {
		It("emits a signed BetaThinkingBlockParam before tool_use", func() {
			m := &anthropicModel{}
			msgs := []*Message{
				NewMessage(RoleUser, NewTextPart("go")),
				NewMessage(RoleAssistant,
					&ThinkingPart{Text: "reason", Signature: "sig-xyz"},
					NewTextPart("calling"),
					NewToolCallPart("t1", "echo", `{"v":"x"}`),
				),
			}
			out, _, err := m.convertMessages("", msgs)
			Expect(err).NotTo(HaveOccurred())

			assistant := out[1]
			Expect(assistant.Content[0].OfThinking).NotTo(BeNil())
			Expect(assistant.Content[0].OfThinking.Signature).To(Equal("sig-xyz"))
			Expect(assistant.Content[0].OfThinking.Thinking).To(Equal("reason"))
			Expect(assistant.Content[1].OfText).NotTo(BeNil())
			Expect(assistant.Content[2].OfToolUse).NotTo(BeNil())
		})
	})
})
