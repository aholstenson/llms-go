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
			info:       fixedModelInfo(ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}}),
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

	Context("errors sent inside an open stream", func() {
		const created = "event: response.created\n" +
			"data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}\n\n"
		const delta = "event: response.output_text.delta\n" +
			"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hi\",\"sequence_number\":1,\"logprobs\":[]}\n\n"
		// overloaded is the form that the SDK changes into a plain error,
		// because the event has an "error" field.
		const overloaded = "data: {\"type\":\"error\",\"error\":{\"type\":\"service_unavailable_error\",\"code\":\"server_is_overloaded\",\"message\":\"Our servers are currently overloaded.\",\"param\":null}}\n\n"
		// overloadedEvent is the form that the SDK gives as a ResponseErrorEvent.
		const overloadedEvent = "event: error\n" +
			"data: {\"type\":\"error\",\"code\":\"server_is_overloaded\",\"message\":\"Our servers are currently overloaded.\",\"param\":null,\"sequence_number\":1}\n\n"

		fastBackoff := WithRetryBackoff(BackoffFunc(func(int, time.Duration, bool) time.Duration {
			return time.Millisecond
		}))

		// sendFirst sends failure for the first n requests, and a full
		// answer after that.
		sendFirst := func(requests *int32, n int32, failure string) http.HandlerFunc {
			return func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				if atomic.AddInt32(requests, 1) <= n {
					writeSSE(w, failure)
					return
				}
				writeSSE(w, openaiSuccessSSE)
			}
		}

		It("retries an overload error that comes before any output", func() {
			var requests int32
			m := newModel(sendFirst(&requests, 1, created+overloaded))

			var notices []RetryNotice
			result, err := m.GenerateContent(context.Background(),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
				fastBackoff,
				WithRetryNotify(func(_ context.Context, n RetryNotice) {
					notices = append(notices, n)
				}),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(TextResult{Text: "hi"}))
			Expect(requests).To(Equal(int32(2)))
			Expect(notices).To(HaveLen(1))
			Expect(notices[0].StatusCode).To(BeZero())
			Expect(notices[0].Err).To(MatchError(ContainSubstring("server_is_overloaded")))
		})

		It("retries an overload error event that comes before any output", func() {
			var requests int32
			m := newModel(sendFirst(&requests, 1, created+overloadedEvent))

			result, err := m.GenerateContent(context.Background(),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
				fastBackoff,
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(TextResult{Text: "hi"}))
			Expect(requests).To(Equal(int32(2)))
		})

		It("reports the attempts made when the overload continues", func() {
			var requests int32
			m := newModel(sendFirst(&requests, 100, created+overloaded))

			_, err := m.GenerateContent(context.Background(),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
				WithMaxRetries(1),
				fastBackoff,
			)
			Expect(requests).To(Equal(int32(2)))

			var ue *UnavailableError
			Expect(errors.As(err, &ue)).To(BeTrue())
			Expect(ue.Attempts).To(Equal(2))
			Expect(ue.StatusCode).To(BeZero())
			Expect(ue.PartialOutput).To(BeFalse())
			Expect(err).To(MatchError(ContainSubstring("server_is_overloaded")))
		})

		It("does not retry an overload error that comes after output", func() {
			var requests int32
			m := newModel(sendFirst(&requests, 100, created+delta+overloaded))

			var streamed string
			_, err := m.GenerateContent(context.Background(),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
				WithStreamingFunc(func(_ context.Context, evt StreamingEvent) error {
					if chunk, ok := evt.(StreamingEventTextChunk); ok {
						streamed += chunk.Text
					}
					return nil
				}),
				fastBackoff,
			)
			Expect(requests).To(Equal(int32(1)))
			Expect(streamed).To(Equal("hi"))
			Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())

			var ue *UnavailableError
			Expect(errors.As(err, &ue)).To(BeTrue())
			Expect(ue.Attempts).To(Equal(1))
			Expect(ue.PartialOutput).To(BeTrue())
		})

		It("does not retry other errors", func() {
			var requests int32
			m := newModel(sendFirst(&requests, 100, created+
				"data: {\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"code\":\"invalid_prompt\",\"message\":\"bad\",\"param\":null}}\n\n"))

			_, err := m.GenerateContent(context.Background(),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
				fastBackoff,
			)
			Expect(err).To(MatchError(ContainSubstring("invalid_prompt")))
			Expect(requests).To(Equal(int32(1)))

			var ue *UnavailableError
			Expect(errors.As(err, &ue)).To(BeFalse())
		})
	})
})
