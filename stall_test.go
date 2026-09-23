package llms

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// stallClient builds an http.Client wired the way every provider's client is,
// so the transport reads its budgets from the request context.
func stallClient(provider string) *http.Client {
	return &http.Client{Transport: newStallTransport(http.DefaultTransport, provider)}
}

// stallCtx attaches the budgets a GenerateOption would have produced.
func stallCtx(stall, request time.Duration) context.Context {
	return withLiveness(context.Background(), "m", &generateContentOptions{
		StallTimeout:   stall,
		RequestTimeout: request,
	})
}

var _ = Describe("Stall detection", func() {
	It("fails a response that goes quiet mid-body", func() {
		release := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, "event: ping\n\n")
			// Go silent until the test is over.
			<-release
		}))
		DeferCleanup(func() {
			close(release)
			srv.Close()
		})

		req, err := http.NewRequestWithContext(stallCtx(50*time.Millisecond, 0), http.MethodGet, srv.URL, nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := stallClient("anthropic").Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		_, err = io.ReadAll(resp.Body)
		var se *StallError
		Expect(errors.As(err, &se)).To(BeTrue())
		Expect(se.Phase).To(Equal(StallPhaseStream))
		Expect(se.Provider).To(Equal("anthropic"))
		Expect(se.Model).To(Equal("m"))
		Expect(se.Timeout).To(Equal(50 * time.Millisecond))
		Expect(errors.Is(err, ErrStall)).To(BeTrue())
	})

	It("fails a request that never starts responding", func() {
		release := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			<-release
		}))
		DeferCleanup(func() {
			close(release)
			srv.Close()
		})

		req, err := http.NewRequestWithContext(stallCtx(0, 50*time.Millisecond), http.MethodGet, srv.URL, nil)
		Expect(err).NotTo(HaveOccurred())

		_, err = stallClient("openai").Do(req)
		var se *StallError
		Expect(errors.As(err, &se)).To(BeTrue())
		Expect(se.Phase).To(Equal(StallPhaseResponse))
		Expect(se.Timeout).To(Equal(50 * time.Millisecond))
	})

	It("treats keepalives as liveness, so a quiet turn is not a stall", func() {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			// Ten keepalives at half the budget: no single gap trips it, but
			// the whole response takes twice the budget.
			for i := 0; i < 10; i++ {
				writeSSE(w, "event: ping\n\n")
				time.Sleep(20 * time.Millisecond)
			}
			writeSSE(w, "done")
		}))
		DeferCleanup(srv.Close)

		req, err := http.NewRequestWithContext(stallCtx(100*time.Millisecond, 0), http.MethodGet, srv.URL, nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := stallClient("anthropic").Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(HaveSuffix("done"))
	})

	It("stays out of the way when no budget is set", func() {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			time.Sleep(60 * time.Millisecond)
			writeSSE(w, "ok")
		}))
		DeferCleanup(srv.Close)

		req, err := http.NewRequestWithContext(stallCtx(0, 0), http.MethodGet, srv.URL, nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := stallClient("anthropic").Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())
		Expect(string(body)).To(Equal("ok"))
	})

	It("reports a caller's cancellation as cancellation, not as a stall", func() {
		release := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
			<-release
		}))
		DeferCleanup(func() {
			close(release)
			srv.Close()
		})

		ctx, cancel := context.WithCancel(stallCtx(0, time.Minute))
		go func() {
			time.Sleep(30 * time.Millisecond)
			cancel()
		}()
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
		Expect(err).NotTo(HaveOccurred())

		_, err = stallClient("anthropic").Do(req)
		Expect(err).To(HaveOccurred())
		Expect(isStallError(err)).To(BeFalse())
		Expect(errors.Is(err, context.Canceled)).To(BeTrue())
	})

	It("reports data that arrives after a quiet gap", func() {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, "event: ping\n\n")
			time.Sleep(LivenessQuietThreshold + 100*time.Millisecond)
			writeSSE(w, "done")
		}))
		DeferCleanup(srv.Close)

		var notices []LivenessNotice
		ctx := withLiveness(context.Background(), "m", &generateContentOptions{
			StallTimeout: 10 * time.Second,
			LivenessNotify: func(_ context.Context, n LivenessNotice) {
				notices = append(notices, n)
			},
		})

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
		Expect(err).NotTo(HaveOccurred())

		resp, err := stallClient("anthropic").Do(req)
		Expect(err).NotTo(HaveOccurred())
		defer resp.Body.Close()

		_, err = io.ReadAll(resp.Body)
		Expect(err).NotTo(HaveOccurred())

		Expect(notices).NotTo(BeEmpty())
		last := notices[len(notices)-1]
		Expect(last.Provider).To(Equal("anthropic"))
		Expect(last.Model).To(Equal("m"))
		Expect(last.Phase).To(Equal(StallPhaseStream))
		Expect(last.Idle).To(BeNumerically(">=", LivenessQuietThreshold))
		Expect(last.StallTimeout).To(Equal(10 * time.Second))
	})
})

var _ = Describe("IsRetryable", func() {
	It("is false for a nil error and for an ordinary failure", func() {
		Expect(IsRetryable(nil)).To(BeFalse())
		Expect(IsRetryable(errors.New("boom"))).To(BeFalse())
		Expect(IsRetryable(context.Canceled)).To(BeFalse())
	})

	It("is true for a stall and for an unavailable provider", func() {
		Expect(IsRetryable(&StallError{Provider: "anthropic"})).To(BeTrue())
		Expect(IsRetryable(&UnavailableError{StatusCode: 529})).To(BeTrue())
	})

	It("is false once output has reached the caller", func() {
		ue := &UnavailableError{StatusCode: 529, PartialOutput: true}
		Expect(IsRetryable(ue)).To(BeFalse())
		Expect(IsRetryable(errors.Join(ue, ErrStreamingPartialOutput))).To(BeFalse())
	})
})

// flakyTurn fails a scripted number of Next calls with err, then succeeds. It
// stands in for a provider whose turn stalls and then comes back.
type flakyTurn struct {
	failures int
	err      error
	calls    int
}

func (f *flakyTurn) Next(context.Context) (TurnOutput, error) {
	f.calls++
	if f.calls <= f.failures {
		return TurnOutput{}, f.err
	}
	return TurnOutput{Text: "done", StopReason: StopReasonEndTurn}, nil
}

func (f *flakyTurn) Observe(context.Context, TurnOutput, []ToolOutcome) error { return nil }

func (f *flakyTurn) ObserveToolResults(context.Context, []ToolCall, []ToolOutcome) error {
	return nil
}

func (f *flakyTurn) Inject(...*Message) {}

func (f *flakyTurn) FinalText() string { return "done" }

var _ = Describe("Session recovery", func() {
	It("survives a stalled turn and retries it against the same transcript", func() {
		turn := &flakyTurn{failures: 1, err: &UnavailableError{
			Provider: "anthropic",
			Model:    "m",
			Attempts: 3,
			Cause:    &StallError{Provider: "anthropic", Model: "m", Phase: StallPhaseStream},
		}}
		s, tracker := newTestSession(turn, nil, WithMaxSteps(2))

		_, done, err := s.Step(context.Background())
		Expect(err).To(HaveOccurred())
		Expect(done).To(BeFalse())
		Expect(s.Retryable()).To(BeTrue())
		// The failed turn gave its step back, so the retry is not charged
		// against the budget.
		Expect(tracker.CurrentStep()).To(Equal(0))

		_, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(s.Retryable()).To(BeFalse())
		Expect(turn.calls).To(Equal(2))
		Expect(tracker.CurrentStep()).To(Equal(1))

		result, err := s.Result()
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "done"}))
	})

	It("ends the run for a failure that cannot be attempted again", func() {
		turn := &flakyTurn{failures: 1, err: errors.New("boom")}
		s, _ := newTestSession(turn, nil, WithMaxSteps(2))

		_, done, err := s.Step(context.Background())
		Expect(err).To(MatchError("boom"))
		Expect(done).To(BeTrue())
		Expect(s.Retryable()).To(BeFalse())

		_, done, err = s.Step(context.Background())
		Expect(err).To(MatchError("boom"))
		Expect(done).To(BeTrue())
		Expect(turn.calls).To(Equal(1))
	})
})

var _ = Describe("Anthropic stalls", func() {
	const successSSE = "event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\n" +
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\n" +
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\n" +
		"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\n" +
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n" +
		"event: message_stop\n" +
		"data: {\"type\":\"message_stop\"}\n\n"

	// openingSSE starts a response without finishing it: enough for the SDK to
	// hand back an open stream, not enough to produce any content.
	const openingSSE = "event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n"

	// newModel points an anthropicModel at handler, through the same stall
	// transport every real provider client is built with.
	newModel := func(handler http.HandlerFunc) *anthropicModel {
		srv := httptest.NewServer(handler)
		DeferCleanup(srv.Close)
		return &anthropicModel{
			logger:  discardLogger(),
			metrics: NewNoopMetrics(),
			client: anthropic.NewClient(
				option.WithBaseURL(srv.URL+"/"),
				option.WithAPIKey("test"),
				option.WithHTTPClient(stallClient("anthropic")),
			),
			model:      "m",
			statsModel: "anthropic/m",
			info:       ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}},
		}
	}

	It("retries a stream that opens and then goes silent", func() {
		var requests int32
		release := make(chan struct{})

		m := newModel(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if atomic.AddInt32(&requests, 1) == 1 {
				writeSSE(w, openingSSE)
				<-release
				return
			}
			writeSSE(w, successSSE)
		})
		// Registered after newModel so it runs before the server is closed:
		// Ginkgo runs cleanups in reverse order, and Close waits for handlers.
		DeferCleanup(func() { close(release) })

		var notices []RetryNotice
		result, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStallTimeout(80*time.Millisecond),
			WithRetryNotify(func(_ context.Context, n RetryNotice) {
				notices = append(notices, n)
			}),
		)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
		Expect(atomic.LoadInt32(&requests)).To(Equal(int32(2)))

		Expect(notices).To(HaveLen(1))
		Expect(notices[0].Attempt).To(Equal(1))
		Expect(notices[0].StatusCode).To(Equal(0))
		Expect(isStallError(notices[0].Err)).To(BeTrue())
	})

	It("reports a stall as an unavailable model once the retries run out", func() {
		var requests int32
		release := make(chan struct{})

		m := newModel(func(w http.ResponseWriter, _ *http.Request) {
			atomic.AddInt32(&requests, 1)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, openingSSE)
			<-release
		})
		// Registered after newModel so it runs before the server is closed:
		// Ginkgo runs cleanups in reverse order, and Close waits for handlers.
		DeferCleanup(func() { close(release) })

		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStallTimeout(50*time.Millisecond),
			WithMaxRetries(1),
		)
		Expect(err).To(HaveOccurred())
		Expect(atomic.LoadInt32(&requests)).To(Equal(int32(2)))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.Provider).To(Equal("anthropic"))
		Expect(ue.Model).To(Equal("m"))
		Expect(ue.Attempts).To(Equal(2))
		Expect(ue.PartialOutput).To(BeFalse())
		// The input tokens Anthropic reported on message_start are preserved.
		Expect(ue.PartialUsage).NotTo(BeNil())
		Expect(ue.PartialUsage.InputTokens).To(Equal(int64(5)))

		Expect(errors.Is(err, ErrStall)).To(BeTrue())
		Expect(IsRetryable(err)).To(BeTrue())
	})

	It("does not replay a stream that already reached the caller", func() {
		var requests int32
		release := make(chan struct{})

		m := newModel(func(w http.ResponseWriter, _ *http.Request) {
			atomic.AddInt32(&requests, 1)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			// Deliver a token, then go silent.
			writeSSE(w, openingSSE+
				"event: content_block_start\n"+
				"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"+
				"event: content_block_delta\n"+
				"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n")
			<-release
		})
		// Registered after newModel so it runs before the server is closed:
		// Ginkgo runs cleanups in reverse order, and Close waits for handlers.
		DeferCleanup(func() { close(release) })

		var chunks []string
		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStallTimeout(50*time.Millisecond),
			WithStreamingFunc(func(_ context.Context, ev StreamingEvent) error {
				if c, ok := ev.(StreamingEventTextChunk); ok {
					chunks = append(chunks, c.Text)
				}
				return nil
			}),
		)
		Expect(err).To(HaveOccurred())
		Expect(atomic.LoadInt32(&requests)).To(Equal(int32(1)))
		Expect(chunks).To(Equal([]string{"hi"}))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.PartialOutput).To(BeTrue())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())
		Expect(IsRetryable(err)).To(BeFalse())
	})
})
