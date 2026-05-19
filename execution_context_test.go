package llms_test

import (
	"context"
	"sync"

	"github.com/aholstenson/llms-go"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ExecutionTracker", func() {
	var tracker *llms.ExecutionTracker

	BeforeEach(func() {
		tracker = llms.NewExecutionTracker(10)
	})

	Describe("Step tracking", func() {
		It("should initialize with step 0", func() {
			Expect(tracker.CurrentStep()).To(Equal(0))
		})

		It("should track max steps", func() {
			Expect(tracker.MaxSteps()).To(Equal(10))
		})

		It("should calculate remaining steps", func() {
			Expect(tracker.RemainingSteps()).To(Equal(10))

			tracker.IncrementStep()
			Expect(tracker.RemainingSteps()).To(Equal(9))

			tracker.IncrementStep()
			Expect(tracker.RemainingSteps()).To(Equal(8))
		})

		It("should increment step counter", func() {
			tracker.IncrementStep()
			Expect(tracker.CurrentStep()).To(Equal(1))

			tracker.IncrementStep()
			Expect(tracker.CurrentStep()).To(Equal(2))
		})
	})

	Describe("Tool call tracking", func() {
		It("should start with zero tool calls", func() {
			Expect(tracker.ToolCallCount("websearch")).To(Equal(0))
			Expect(tracker.TotalToolCalls()).To(Equal(0))
		})

		It("should track tool calls per tool", func() {
			tracker.RecordToolCall("websearch")
			Expect(tracker.ToolCallCount("websearch")).To(Equal(1))
			Expect(tracker.TotalToolCalls()).To(Equal(1))

			tracker.RecordToolCall("websearch")
			Expect(tracker.ToolCallCount("websearch")).To(Equal(2))
			Expect(tracker.TotalToolCalls()).To(Equal(2))

			tracker.RecordToolCall("get_website_content")
			Expect(tracker.ToolCallCount("websearch")).To(Equal(2))
			Expect(tracker.ToolCallCount("get_website_content")).To(Equal(1))
			Expect(tracker.TotalToolCalls()).To(Equal(3))
		})

		It("should handle unknown tools", func() {
			Expect(tracker.ToolCallCount("unknown")).To(Equal(0))
		})
	})

	Describe("Token tracking", func() {
		It("should start with zero tokens", func() {
			Expect(tracker.InputTokens()).To(Equal(int64(0)))
			Expect(tracker.OutputTokens()).To(Equal(int64(0)))
			Expect(tracker.CachedTokens()).To(Equal(int64(0)))
			Expect(tracker.TotalTokens()).To(Equal(int64(0)))
		})

		It("should accumulate token counts", func() {
			tracker.AddTokens(100, 50, 10)
			Expect(tracker.InputTokens()).To(Equal(int64(100)))
			Expect(tracker.OutputTokens()).To(Equal(int64(50)))
			Expect(tracker.CachedTokens()).To(Equal(int64(10)))
			Expect(tracker.TotalTokens()).To(Equal(int64(150)))

			tracker.AddTokens(200, 75, 20)
			Expect(tracker.InputTokens()).To(Equal(int64(300)))
			Expect(tracker.OutputTokens()).To(Equal(int64(125)))
			Expect(tracker.CachedTokens()).To(Equal(int64(30)))
			Expect(tracker.TotalTokens()).To(Equal(int64(425)))
		})
	})

	Describe("Thread safety", func() {
		It("should handle concurrent reads and writes", func() {
			var wg sync.WaitGroup
			iterations := 100

			// Concurrent step increments
			for range iterations {
				wg.Add(1)
				go func() {
					defer wg.Done()
					tracker.IncrementStep()
				}()
			}

			// Concurrent tool calls
			for i := range iterations {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					toolName := "tool" + string(rune('A'+i%3))
					tracker.RecordToolCall(toolName)
				}(i)
			}

			// Concurrent token additions
			for range iterations {
				wg.Add(1)
				go func() {
					defer wg.Done()
					tracker.AddTokens(10, 5, 2)
				}()
			}

			// Concurrent reads
			for range iterations {
				wg.Add(1)
				go func() {
					defer wg.Done()
					_ = tracker.CurrentStep()
					_ = tracker.TotalToolCalls()
					_ = tracker.TotalTokens()
				}()
			}

			wg.Wait()

			Expect(tracker.CurrentStep()).To(Equal(iterations))
			Expect(tracker.TotalToolCalls()).To(Equal(iterations))
			Expect(tracker.InputTokens()).To(Equal(int64(iterations * 10)))
			Expect(tracker.OutputTokens()).To(Equal(int64(iterations * 5)))
			Expect(tracker.CachedTokens()).To(Equal(int64(iterations * 2)))
		})
	})
})

var _ = Describe("ExecutionContext context helpers", func() {
	var ctx context.Context
	var tracker *llms.ExecutionTracker

	BeforeEach(func() {
		ctx = context.Background()
		tracker = llms.NewExecutionTracker(5)
	})

	It("should store and retrieve ExecutionContext", func() {
		ctx = llms.WithExecutionContext(ctx, tracker)
		retrieved := llms.GetExecutionContext(ctx)

		Expect(retrieved).NotTo(BeNil())
		Expect(retrieved.MaxSteps()).To(Equal(5))
	})

	It("should return nil when context not set", func() {
		retrieved := llms.GetExecutionContext(ctx)
		Expect(retrieved).To(BeNil())
	})

	It("should allow reading through interface", func() {
		tracker.IncrementStep()
		tracker.RecordToolCall("test")
		tracker.AddTokens(100, 50, 10)

		ctx = llms.WithExecutionContext(ctx, tracker)
		retrieved := llms.GetExecutionContext(ctx)

		Expect(retrieved.CurrentStep()).To(Equal(1))
		Expect(retrieved.ToolCallCount("test")).To(Equal(1))
		Expect(retrieved.InputTokens()).To(Equal(int64(100)))
	})
})
