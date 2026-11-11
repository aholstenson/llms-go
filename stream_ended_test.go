package llms

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go/v2"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/v2/option"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/revrost/go-openrouter"
	"google.golang.org/genai"
)

// sseScript is a fake SSE server. Request n gets bodies[n]. When there are
// more requests than bodies, the last body is sent again. It records each
// request body, so a test can compare what two attempts sent.
type sseScript struct {
	bodies []string

	mu       sync.Mutex
	requests []string
}

func (s *sseScript) handler(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	s.mu.Lock()
	n := len(s.requests)
	s.requests = append(s.requests, string(body))
	s.mu.Unlock()

	if n >= len(s.bodies) {
		n = len(s.bodies) - 1
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	writeSSE(w, s.bodies[n])
}

func (s *sseScript) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

func (s *sseScript) request(n int) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.requests[n]
}

// fastRetry keeps the wait between attempts short in tests.
var fastRetry = WithRetryBackoff(BackoffFunc(func(int, time.Duration, bool) time.Duration {
	return time.Millisecond
}))

var userHi = WithMessages(NewMessage(RoleUser, NewTextPart("hi")))

// discardStream is a streaming callback that accepts each event.
var discardStream = WithStreamingFunc(func(context.Context, StreamingEvent) error { return nil })

var _ = Describe("Stream ended error", func() {
	It("gives the event count, the last event and the response ID", func() {
		err := &streamEndedError{events: 214, lastEvent: "response.reasoning_summary_text.delta", responseID: "resp_1"}
		Expect(err.Error()).To(Equal("stream ended after 214 events without a final response (last event: response.reasoning_summary_text.delta, response ID: resp_1)"))
		Expect(errors.Is(err, ErrStreamEnded)).To(BeTrue())
	})

	It("leaves out the details that are not known", func() {
		err := &streamEndedError{}
		Expect(err.Error()).To(Equal("stream ended after 0 events without a final response"))
	})

	It("is retryable and unavailable inside an UnavailableError", func() {
		err := &UnavailableError{Provider: "openai", Attempts: 3, Cause: &streamEndedError{events: 1}}
		Expect(IsRetryable(err)).To(BeTrue())
		Expect(IsUnavailable(err)).To(BeTrue())
		Expect(errors.Is(err, ErrStreamEnded)).To(BeTrue())
		Expect(transientErrorType(err)).To(Equal(GenAIErrorTypeStreamEnded))
	})

	It("is unavailable but not retryable after partial output", func() {
		ue := &UnavailableError{Provider: "openai", Attempts: 1, PartialOutput: true, Cause: &streamEndedError{events: 1}}
		err := errors.Join(ue, ErrStreamingPartialOutput)
		Expect(IsRetryable(err)).To(BeFalse())
		Expect(IsUnavailable(err)).To(BeTrue())
	})

	It("is not unavailable for other errors", func() {
		Expect(IsUnavailable(nil)).To(BeFalse())
		Expect(IsUnavailable(errors.New("x"))).To(BeFalse())
		Expect(IsUnavailable(&ContentPolicyError{})).To(BeFalse())
	})
})

var _ = Describe("OpenAI stream that ends without a final event", func() {
	const created = "event: response.created\n" +
		"data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}\n\n"
	const delta = "event: response.output_text.delta\n" +
		"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hi\",\"sequence_number\":1,\"logprobs\":[]}\n\n"
	const completed = "event: response.completed\n" +
		"data: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":5,\"output_tokens\":2,\"total_tokens\":7}}}\n\n"
	const failedServerError = "event: response.failed\n" +
		"data: {\"type\":\"response.failed\",\"sequence_number\":2,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"boom\"},\"output\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":0,\"total_tokens\":5}}}\n\n"

	newModel := func(script *sseScript) *openaiModel {
		srv := httptest.NewServer(http.HandlerFunc(script.handler))
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

	It("retries when no output reached the caller", func() {
		script := &sseScript{bodies: []string{created + delta, created + delta + completed}}
		m := newModel(script)

		var notices []RetryNotice
		result, err := m.GenerateContent(context.Background(), userHi, fastRetry,
			WithRetryNotify(func(_ context.Context, n RetryNotice) { notices = append(notices, n) }),
		)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
		Expect(script.count()).To(Equal(2))
		Expect(notices).To(HaveLen(1))
		Expect(errors.Is(notices[0].Err, ErrStreamEnded)).To(BeTrue())
	})

	It("returns a retryable UnavailableError when the retries run out", func() {
		script := &sseScript{bodies: []string{created + delta}}
		m := newModel(script)

		_, err := m.GenerateContent(context.Background(), userHi, fastRetry, WithMaxRetries(1))
		Expect(script.count()).To(Equal(2))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.Attempts).To(Equal(2))
		Expect(ue.PartialOutput).To(BeFalse())
		Expect(errors.Is(err, ErrStreamEnded)).To(BeTrue())
		Expect(IsRetryable(err)).To(BeTrue())
	})

	It("does not retry after output and reports partial output", func() {
		script := &sseScript{bodies: []string{created + delta, created + delta + completed}}
		m := newModel(script)

		_, err := m.GenerateContent(context.Background(), userHi, fastRetry, discardStream)
		Expect(script.count()).To(Equal(1))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.Provider).To(Equal("openai"))
		Expect(ue.Attempts).To(Equal(1))
		Expect(ue.PartialOutput).To(BeTrue())
		Expect(errors.Is(err, ErrStreamEnded)).To(BeTrue())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())
		Expect(IsUnavailable(err)).To(BeTrue())
		Expect(IsRetryable(err)).To(BeFalse())
		Expect(err.Error()).To(ContainSubstring("stream ended after 2 events without a final response"))
		Expect(err.Error()).To(ContainSubstring("last event: response.output_text.delta"))
		Expect(err.Error()).To(ContainSubstring("response ID: resp_1"))
	})

	It("keeps the session usable and sends the same transcript again", func() {
		script := &sseScript{bodies: []string{created + delta, created + delta + completed}}
		m := newModel(script)

		var streamed string
		s, err := NewSession(context.Background(), m, userHi, fastRetry,
			WithStreamingFunc(func(_ context.Context, evt StreamingEvent) error {
				if chunk, ok := evt.(StreamingEventTextChunk); ok {
					streamed += chunk.Text
				}
				return nil
			}),
		)
		Expect(err).NotTo(HaveOccurred())

		info, done, err := s.Step(context.Background())
		Expect(err).To(HaveOccurred())
		Expect(done).To(BeFalse())
		Expect(IsUnavailable(err)).To(BeTrue())
		Expect(errors.Is(err, ErrStreamEnded)).To(BeTrue())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())
		Expect(s.Retryable()).To(BeFalse())
		Expect(info.Step).To(Equal(0))
		Expect(info.Exec.CurrentStep()).To(Equal(0))
		Expect(s.Snapshot().Step).To(Equal(0))
		Expect(s.Messages()).To(HaveLen(1))
		Expect(script.count()).To(Equal(1))

		info, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(info.Step).To(Equal(1))
		Expect(script.count()).To(Equal(2))
		Expect(script.request(1)).To(Equal(script.request(0)))
		Expect(s.Retryable()).To(BeFalse())
		// Each attempt streamed "hi". The caller discards the first one.
		Expect(streamed).To(Equal("hihi"))

		result, err := s.Result()
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
	})

	It("still ends the session on a response that failed with a non-transient code", func() {
		script := &sseScript{bodies: []string{created + failedServerError, created + completed}}
		m := newModel(script)

		s, err := NewSession(context.Background(), m, userHi, fastRetry)
		Expect(err).NotTo(HaveOccurred())

		_, done, err := s.Step(context.Background())
		Expect(err).To(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(err.Error()).To(ContainSubstring("server_error"))
		Expect(IsUnavailable(err)).To(BeFalse())
		Expect(IsRetryable(err)).To(BeFalse())
		Expect(errors.Is(err, ErrStreamEnded)).To(BeFalse())
		Expect(script.count()).To(Equal(1))
	})
})

var _ = Describe("Anthropic stream that ends without message_stop", func() {
	const start = "event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\n" +
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\n" +
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
	const end = "event: content_block_stop\n" +
		"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\n" +
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n" +
		"event: message_stop\n" +
		"data: {\"type\":\"message_stop\"}\n\n"

	newModel := func(script *sseScript) *anthropicModel {
		srv := httptest.NewServer(http.HandlerFunc(script.handler))
		DeferCleanup(srv.Close)
		return &anthropicModel{
			logger:  discardLogger(),
			metrics: NewNoopMetrics(),
			client: anthropic.NewClient(
				anthropicoption.WithBaseURL(srv.URL+"/"),
				anthropicoption.WithAPIKey("test"),
			),
			model:      "m",
			statsModel: "anthropic/m",
			info:       fixedModelInfo(ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}}),
		}
	}

	It("retries when no output reached the caller", func() {
		script := &sseScript{bodies: []string{start, start + end}}
		m := newModel(script)

		result, err := m.GenerateContent(context.Background(), userHi, fastRetry)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
		Expect(script.count()).To(Equal(2))
	})

	It("does not retry after output and reports partial output", func() {
		script := &sseScript{bodies: []string{start, start + end}}
		m := newModel(script)

		_, err := m.GenerateContent(context.Background(), userHi, fastRetry, discardStream)
		Expect(script.count()).To(Equal(1))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.Provider).To(Equal("anthropic"))
		Expect(ue.PartialOutput).To(BeTrue())
		Expect(ue.PartialUsage).NotTo(BeNil())
		Expect(ue.PartialUsage.InputTokens).To(Equal(int64(5)))
		Expect(errors.Is(err, ErrStreamEnded)).To(BeTrue())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())
		Expect(err.Error()).To(ContainSubstring("stream ended after 3 events"))
		Expect(err.Error()).To(ContainSubstring("last event: content_block_delta"))
		Expect(err.Error()).To(ContainSubstring("response ID: msg_1"))
	})

	It("keeps the session usable and sends the same transcript again", func() {
		script := &sseScript{bodies: []string{start, start + end}}
		m := newModel(script)

		s, err := NewSession(context.Background(), m, userHi, fastRetry, discardStream)
		Expect(err).NotTo(HaveOccurred())

		_, done, err := s.Step(context.Background())
		Expect(IsUnavailable(err)).To(BeTrue())
		Expect(done).To(BeFalse())

		_, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(script.request(1)).To(Equal(script.request(0)))
	})
})

var _ = Describe("Google stream that ends without a finish reason", func() {
	const usageOnly = `data: {"usageMetadata":{"promptTokenCount":5},"responseId":"r_1"}` + "\n\n"
	const text = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}],"usageMetadata":{"promptTokenCount":5},"responseId":"r_1"}` + "\n\n"
	const stop = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1},"responseId":"r_1"}` + "\n\n"

	newModel := func(script *sseScript) *googleModel {
		srv := httptest.NewServer(http.HandlerFunc(script.handler))
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

	It("retries when no output reached the caller", func() {
		script := &sseScript{bodies: []string{usageOnly, stop}}
		m := newModel(script)

		result, err := m.GenerateContent(context.Background(), userHi, fastRetry, discardStream)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
		Expect(script.count()).To(Equal(2))
	})

	It("does not retry after output and reports partial output", func() {
		script := &sseScript{bodies: []string{text, stop}}
		m := newModel(script)

		_, err := m.GenerateContent(context.Background(), userHi, fastRetry, discardStream)
		Expect(script.count()).To(Equal(1))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.Provider).To(Equal("google"))
		Expect(ue.PartialOutput).To(BeTrue())
		Expect(ue.PartialUsage).NotTo(BeNil())
		Expect(errors.Is(err, ErrStreamEnded)).To(BeTrue())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())
		Expect(err.Error()).To(ContainSubstring("stream ended after 1 events"))
		Expect(err.Error()).To(ContainSubstring("last event: text"))
		Expect(err.Error()).To(ContainSubstring("response ID: r_1"))
	})

	It("keeps the session usable and sends the same transcript again", func() {
		script := &sseScript{bodies: []string{text, stop}}
		m := newModel(script)

		s, err := NewSession(context.Background(), m, userHi, fastRetry, discardStream)
		Expect(err).NotTo(HaveOccurred())

		_, done, err := s.Step(context.Background())
		Expect(IsUnavailable(err)).To(BeTrue())
		Expect(done).To(BeFalse())

		_, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(script.request(1)).To(Equal(script.request(0)))
	})
})

var _ = Describe("OpenRouter stream that ends without a finish reason", func() {
	const role = `data: {"id":"gen_1","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}` + "\n\n"
	const content = `data: {"id":"gen_1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}` + "\n\n"
	const finish = `data: {"id":"gen_1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}` + "\n\n" +
		"data: [DONE]\n\n"

	newModel := func(script *sseScript) *openrouterModel {
		srv := httptest.NewServer(http.HandlerFunc(script.handler))
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

	It("retries when no output reached the caller", func() {
		script := &sseScript{bodies: []string{role, role + content + finish}}
		m := newModel(script)

		result, err := m.GenerateContent(context.Background(), userHi, fastRetry, discardStream)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
		Expect(script.count()).To(Equal(2))
	})

	It("does not retry after output and reports partial output", func() {
		script := &sseScript{bodies: []string{role + content, role + content + finish}}
		m := newModel(script)

		_, err := m.GenerateContent(context.Background(), userHi, fastRetry, discardStream)
		Expect(script.count()).To(Equal(1))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.Provider).To(Equal("openrouter"))
		Expect(ue.PartialOutput).To(BeTrue())
		Expect(errors.Is(err, ErrStreamEnded)).To(BeTrue())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())
		Expect(err.Error()).To(ContainSubstring("stream ended after 2 events"))
		Expect(err.Error()).To(ContainSubstring("last event: content"))
		Expect(err.Error()).To(ContainSubstring("response ID: gen_1"))
	})

	It("keeps the session usable and sends the same transcript again", func() {
		script := &sseScript{bodies: []string{role + content, role + content + finish}}
		m := newModel(script)

		s, err := NewSession(context.Background(), m, userHi, fastRetry, discardStream)
		Expect(err).NotTo(HaveOccurred())

		_, done, err := s.Step(context.Background())
		Expect(IsUnavailable(err)).To(BeTrue())
		Expect(done).To(BeFalse())

		_, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(script.request(1)).To(Equal(script.request(0)))
	})
})
