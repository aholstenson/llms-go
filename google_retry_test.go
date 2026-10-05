package llms

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"google.golang.org/genai"
)

var _ = Describe("Google streaming retries", func() {
	// googleSuccessSSE is the smallest stream that yields one text answer.
	const googleSuccessSSE = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}]," +
		"\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2}}\n\n"

	// fastBackoff keeps the tests fast. The test model has no header
	// capture, so there is no Retry-After hint to use.
	fastBackoff := WithRetryBackoff(BackoffFunc(func(int, time.Duration, bool) time.Duration {
		return time.Millisecond
	}))

	// newModel points a googleModel at handler.
	newModel := func(handler http.HandlerFunc) *googleModel {
		srv := httptest.NewServer(handler)
		DeferCleanup(srv.Close)
		client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
			APIKey:      "test",
			Backend:     genai.BackendGeminiAPI,
			HTTPClient:  srv.Client(),
			HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
		})
		Expect(err).NotTo(HaveOccurred())
		return &googleModel{
			logger:     discardLogger(),
			metrics:    NewNoopMetrics(),
			client:     client,
			statsModel: "google/m",
			model:      "m",
			info:       fixedModelInfo(ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}}),
		}
	}

	It("retries a rate-limited stream before the first chunk", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, r *http.Request) {
			if atomic.AddInt32(&requests, 1) <= 2 {
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, googleSuccessSSE)
		})

		var streamed string
		var notices []RetryNotice
		result, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStreamingFunc(func(_ context.Context, evt StreamingEvent) error {
				if chunk, ok := evt.(StreamingEventTextChunk); ok {
					streamed += chunk.Text
				}
				return nil
			}),
			fastBackoff,
			WithRetryNotify(func(_ context.Context, n RetryNotice) {
				notices = append(notices, n)
			}),
		)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
		Expect(streamed).To(Equal("hi"))
		Expect(requests).To(Equal(int32(3)))

		Expect(notices).To(HaveLen(2))
		Expect(notices[0].Provider).To(Equal("google"))
		Expect(notices[0].Attempt).To(Equal(1))
		Expect(notices[0].MaxAttempts).To(Equal(3))
		Expect(notices[0].StatusCode).To(Equal(429))
		Expect(notices[1].Attempt).To(Equal(2))
	})

	It("stops after MaxRetries and reports the attempts made", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&requests, 1)
			w.WriteHeader(http.StatusServiceUnavailable)
		})

		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStreamingFunc(func(context.Context, StreamingEvent) error { return nil }),
			WithMaxRetries(1),
			fastBackoff,
		)
		Expect(err).To(HaveOccurred())
		Expect(requests).To(Equal(int32(2)))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.StatusCode).To(Equal(503))
		Expect(ue.Attempts).To(Equal(2))
		Expect(ue.PartialOutput).To(BeFalse())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeFalse())
	})

	It("does not retry a stream that fails after the first chunk", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&requests, 1)
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hi\"}]}}]}\n\n")
			writeSSE(w, "{\"error\":{\"code\":503,\"message\":\"overloaded\",\"status\":\"UNAVAILABLE\"}}\n\n")
		})

		var notices int
		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStreamingFunc(func(context.Context, StreamingEvent) error { return nil }),
			fastBackoff,
			WithRetryNotify(func(context.Context, RetryNotice) { notices++ }),
		)
		Expect(err).To(HaveOccurred())
		Expect(requests).To(Equal(int32(1)))
		Expect(notices).To(BeZero())

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.Attempts).To(Equal(1))
		Expect(ue.PartialOutput).To(BeTrue())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())
	})
})
