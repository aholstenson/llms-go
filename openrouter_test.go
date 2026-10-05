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
	"github.com/revrost/go-openrouter"
)

var _ = Describe("OpenRouter streaming retries", func() {
	// openrouterSuccessSSE is the smallest stream that yields one text answer.
	const openrouterSuccessSSE = "data: {\"id\":\"gen_1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]," +
		"\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n" +
		"data: [DONE]\n\n"

	// fastBackoff keeps the tests fast. The test model has no header
	// capture, so there is no Retry-After hint to use.
	fastBackoff := WithRetryBackoff(BackoffFunc(func(int, time.Duration, bool) time.Duration {
		return time.Millisecond
	}))

	// newModel points an openrouterModel at handler.
	newModel := func(handler http.HandlerFunc) *openrouterModel {
		srv := httptest.NewServer(handler)
		DeferCleanup(srv.Close)
		return &openrouterModel{
			logger:  discardLogger(),
			metrics: NewNoopMetrics(),
			client: openrouter.NewClient("test", func(c *openrouter.ClientConfig) {
				c.BaseURL = srv.URL
			}),
			model:      "m",
			statsModel: "openrouter/m",
			info:       fixedModelInfo(ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}}),
		}
	}

	streamTo := func(streamed *string) GenerateOption {
		return WithStreamingFunc(func(_ context.Context, evt StreamingEvent) error {
			if chunk, ok := evt.(StreamingEventTextChunk); ok {
				*streamed += chunk.Text
			}
			return nil
		})
	}

	It("retries a rate-limited stream before it opens", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, r *http.Request) {
			if atomic.AddInt32(&requests, 1) <= 2 {
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, openrouterSuccessSSE)
		})

		var streamed string
		var notices []RetryNotice
		result, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			streamTo(&streamed),
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
		Expect(notices[0].Provider).To(Equal("openrouter"))
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

		var streamed string
		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			streamTo(&streamed),
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
})
