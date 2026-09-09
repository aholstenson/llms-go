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
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

var _ = Describe("OpenAI retries", func() {
	// openaiSuccessSSE is the smallest stream that yields one text answer.
	const openaiSuccessSSE = "event: response.completed\n" +
		"data: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":5,\"output_tokens\":2,\"total_tokens\":7}}}\n\n"

	// newModel points an openaiModel at handler.
	newModel := func(handler http.HandlerFunc) *openaiModel {
		srv := httptest.NewServer(handler)
		DeferCleanup(srv.Close)
		return &openaiModel{
			logger:  discardLogger(),
			metrics: NewNoopMetrics(),
			client: openai.NewClient(
				option.WithBaseURL(srv.URL+"/"),
				option.WithAPIKey("test"),
			),
			model:      "m",
			statsModel: "openai/m",
			info:       ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}},
		}
	}

	It("retries a rate-limited request and reports every wait", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, r *http.Request) {
			if atomic.AddInt32(&requests, 1) <= 2 {
				w.Header().Set("Retry-After-Ms", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, openaiSuccessSSE)
		})

		var notices []RetryNotice
		result, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithRetryNotify(func(_ context.Context, n RetryNotice) {
				notices = append(notices, n)
			}),
		)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
		Expect(requests).To(Equal(int32(3)))

		Expect(notices).To(HaveLen(2))
		Expect(notices[0].Provider).To(Equal("openai"))
		Expect(notices[0].Model).To(Equal("m"))
		Expect(notices[0].Attempt).To(Equal(1))
		Expect(notices[0].MaxAttempts).To(Equal(3))
		Expect(notices[0].StatusCode).To(Equal(429))
		Expect(notices[0].Delay).To(Equal(1 * time.Millisecond))
		Expect(notices[1].Attempt).To(Equal(2))
	})

	It("stops after MaxRetries and reports the attempts made", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&requests, 1)
			w.Header().Set("Retry-After-Ms", "1")
			w.WriteHeader(http.StatusServiceUnavailable)
		})

		var notices int
		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithMaxRetries(1),
			WithRetryNotify(func(context.Context, RetryNotice) { notices++ }),
		)
		Expect(err).To(HaveOccurred())
		Expect(requests).To(Equal(int32(2)))
		Expect(notices).To(Equal(1))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.StatusCode).To(Equal(503))
		Expect(ue.Attempts).To(Equal(2))
		Expect(ue.PartialOutput).To(BeFalse())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeFalse())
	})

	It("makes a single request when retries are disabled", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, r *http.Request) {
			atomic.AddInt32(&requests, 1)
			w.WriteHeader(http.StatusServiceUnavailable)
		})

		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithMaxRetries(0),
		)
		Expect(err).To(HaveOccurred())
		Expect(requests).To(Equal(int32(1)))
	})
})
