package llms

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/revrost/go-openrouter"
	"google.golang.org/genai"
)

var _ = Describe("ContentPolicyError", func() {
	It("names the provider, the model and the code", func() {
		err := &ContentPolicyError{
			Provider:   "openai",
			Model:      "m",
			StatusCode: 400,
			Code:       "cyber_policy",
			Message:    "flagged",
		}
		Expect(err.Error()).To(Equal("blocked by content policy (openai/m) (status 400): cyber_policy: flagged"))
	})

	It("is found in a wrapped error chain", func() {
		pe := &ContentPolicyError{Code: "cyber_policy"}
		Expect(IsContentPolicy(errors.Join(pe, ErrStreamingPartialOutput))).To(BeTrue())
		Expect(IsContentPolicy(errors.New("boom"))).To(BeFalse())
		Expect(IsContentPolicy(nil)).To(BeFalse())
	})

	It("is not retryable", func() {
		Expect(IsRetryable(&ContentPolicyError{Code: "cyber_policy"})).To(BeFalse())
	})
})

var _ = Describe("Session after a content policy block", func() {
	It("stays usable and lets the caller try the turn again", func() {
		turn := &flakyTurn{failures: 1, err: &ContentPolicyError{
			Provider: "openai",
			Model:    "m",
			Code:     "cyber_policy",
		}}
		s, tracker := newTestSession(turn, nil, WithMaxSteps(2))

		_, done, err := s.Step(context.Background())
		Expect(IsContentPolicy(err)).To(BeTrue())
		Expect(done).To(BeFalse())
		// The block is not transient, so the session does not call it
		// retryable, but it gives the step back.
		Expect(s.Retryable()).To(BeFalse())
		Expect(tracker.CurrentStep()).To(Equal(0))

		_, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(turn.calls).To(Equal(2))
		Expect(tracker.CurrentStep()).To(Equal(1))

		result, err := s.Result()
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "done"}))
	})

	It("stays usable when output reached the caller before the block", func() {
		turn := &flakyTurn{failures: 1, err: errors.Join(
			&ContentPolicyError{Code: "cyber_policy", PartialOutput: true},
			ErrStreamingPartialOutput,
		)}
		s, _ := newTestSession(turn, nil, WithMaxSteps(2))

		_, done, err := s.Step(context.Background())
		Expect(IsContentPolicy(err)).To(BeTrue())
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())
		Expect(done).To(BeFalse())

		_, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
	})

	It("reports the block from GenerateContent", func() {
		turn := &flakyTurn{failures: 1, err: &ContentPolicyError{Code: "cyber_policy"}}
		s, _ := newTestSession(turn, nil, WithMaxSteps(2))

		_, err := runSession(context.Background(), s)
		Expect(IsContentPolicy(err)).To(BeTrue())
		Expect(turn.calls).To(Equal(1))
	})
})

var _ = Describe("OpenAI content policy blocks", func() {
	const successSSE = "event: response.completed\n" +
		"data: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":5,\"output_tokens\":2,\"total_tokens\":7}}}\n\n"
	const created = "event: response.created\n" +
		"data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}\n\n"
	const delta = "event: response.output_text.delta\n" +
		"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hi\",\"sequence_number\":1,\"logprobs\":[]}\n\n"
	// cyberPolicy is the form that the SDK changes into a plain error,
	// because the event has an "error" field.
	const cyberPolicy = "data: {\"type\":\"invalid_request\",\"error\":{\"type\":\"invalid_request\",\"code\":\"cyber_policy\",\"message\":\"This content was flagged for possible cybersecurity risk.\",\"param\":null}}\n\n"
	// cyberPolicyEvent is the form that the SDK gives as a
	// ResponseErrorEvent.
	const cyberPolicyEvent = "event: error\n" +
		"data: {\"type\":\"error\",\"code\":\"cyber_policy\",\"message\":\"This content was flagged for possible cybersecurity risk.\",\"param\":null,\"sequence_number\":1}\n\n"
	// failedInvalidPrompt is a response that fails with a policy code.
	const failedInvalidPrompt = "event: response.failed\n" +
		"data: {\"type\":\"response.failed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"failed\",\"error\":{\"code\":\"invalid_prompt\",\"message\":\"Your prompt was flagged.\"},\"output\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":0,\"total_tokens\":5}}}\n\n"

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

	// sendFirst sends failure for the first n requests, and a full answer
	// after that.
	sendFirst := func(requests *int32, n int32, failure string) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if atomic.AddInt32(requests, 1) <= n {
				writeSSE(w, failure)
				return
			}
			writeSSE(w, successSSE)
		}
	}

	generate := func(m *openaiModel, options ...GenerateOption) error {
		options = append([]GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}, options...)
		_, err := m.GenerateContent(context.Background(), options...)
		return err
	}

	It("reports an error field in the stream without a retry", func() {
		var requests int32
		m := newModel(sendFirst(&requests, 1, created+cyberPolicy))

		err := generate(m)
		Expect(requests).To(Equal(int32(1)))
		Expect(IsRetryable(err)).To(BeFalse())

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Provider).To(Equal("openai"))
		Expect(pe.Model).To(Equal("m"))
		Expect(pe.StatusCode).To(BeZero())
		Expect(pe.Code).To(Equal("cyber_policy"))
		Expect(pe.Message).To(ContainSubstring("cybersecurity risk"))
		Expect(pe.PartialOutput).To(BeFalse())

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeFalse())
	})

	It("reports an error event in the stream", func() {
		var requests int32
		m := newModel(sendFirst(&requests, 1, created+cyberPolicyEvent))

		err := generate(m)
		Expect(requests).To(Equal(int32(1)))

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Code).To(Equal("cyber_policy"))
	})

	It("reports a failed response and keeps its usage", func() {
		var requests int32
		m := newModel(sendFirst(&requests, 1, created+failedInvalidPrompt))

		err := generate(m)
		Expect(requests).To(Equal(int32(1)))

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Code).To(Equal("invalid_prompt"))
		Expect(pe.Message).To(Equal("Your prompt was flagged."))
		Expect(pe.PartialUsage).NotTo(BeNil())
		Expect(pe.PartialUsage.InputTokens).To(Equal(int64(5)))
	})

	It("reports an HTTP error with its status", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, _ *http.Request) {
			atomic.AddInt32(&requests, 1)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"type":"invalid_request_error","code":"cyber_policy","message":"flagged","param":null}}`))
		})

		err := generate(m)
		Expect(requests).To(Equal(int32(1)))

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.StatusCode).To(Equal(400))
		Expect(pe.Code).To(Equal("cyber_policy"))
	})

	It("marks a block that comes after output", func() {
		var requests int32
		m := newModel(sendFirst(&requests, 1, created+delta+cyberPolicy))

		err := generate(m, WithStreamingFunc(func(context.Context, StreamingEvent) error { return nil }))
		Expect(requests).To(Equal(int32(1)))
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.PartialOutput).To(BeTrue())
	})

	It("reports a response that a content filter stopped", func() {
		const incomplete = "event: response.incomplete\n" +
			"data: {\"type\":\"response.incomplete\",\"sequence_number\":2,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"},\"output\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":1,\"total_tokens\":6}}}\n\n"
		var requests int32
		m := newModel(sendFirst(&requests, 1, created+incomplete))

		err := generate(m)
		Expect(requests).To(Equal(int32(1)))

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Code).To(Equal("content_filter"))
		Expect(pe.PartialUsage).NotTo(BeNil())
	})

	It("lets a session try the turn again after a block", func() {
		var requests int32
		m := newModel(sendFirst(&requests, 1, created+delta+cyberPolicy))

		var streamed string
		s, err := NewSession(context.Background(), m,
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStreamingFunc(func(_ context.Context, evt StreamingEvent) error {
				if chunk, ok := evt.(StreamingEventTextChunk); ok {
					streamed += chunk.Text
				}
				return nil
			}),
		)
		Expect(err).NotTo(HaveOccurred())

		_, done, err := s.Step(context.Background())
		Expect(IsContentPolicy(err)).To(BeTrue())
		Expect(done).To(BeFalse())

		_, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())
		Expect(requests).To(Equal(int32(2)))
		// The blocked attempt streamed "hi" before the block. The
		// successful attempt streams nothing in this fixture.
		Expect(streamed).To(Equal("hi"))

		result, err := s.Result()
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
	})
})

var _ = Describe("Anthropic content policy blocks", func() {
	const start = "event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\n" +
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\n" +
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n" +
		"event: content_block_stop\n" +
		"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
	end := func(stopReason string) string {
		return "event: message_delta\n" +
			"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"" + stopReason + "\"},\"usage\":{\"output_tokens\":2}}\n\n" +
			"event: message_stop\n" +
			"data: {\"type\":\"message_stop\"}\n\n"
	}

	newModel := func(requests *int32, failures int32) *anthropicModel {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if atomic.AddInt32(requests, 1) <= failures {
				writeSSE(w, start+end("refusal"))
				return
			}
			writeSSE(w, start+end("end_turn"))
		}))
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

	It("reports the stop reason refusal as a block with its usage", func() {
		var requests int32
		m := newModel(&requests, 1)

		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStreamingFunc(func(context.Context, StreamingEvent) error { return nil }),
		)
		Expect(requests).To(Equal(int32(1)))
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Provider).To(Equal("anthropic"))
		Expect(pe.Code).To(Equal("refusal"))
		Expect(pe.PartialOutput).To(BeTrue())
		Expect(pe.PartialUsage).NotTo(BeNil())
		Expect(pe.PartialUsage.InputTokens).To(Equal(int64(5)))
	})

	It("lets a session try the turn again after a refusal", func() {
		var requests int32
		m := newModel(&requests, 1)

		s, err := NewSession(context.Background(), m, WithMessages(NewMessage(RoleUser, NewTextPart("hi"))))
		Expect(err).NotTo(HaveOccurred())

		_, done, err := s.Step(context.Background())
		Expect(IsContentPolicy(err)).To(BeTrue())
		Expect(done).To(BeFalse())

		_, done, err = s.Step(context.Background())
		Expect(err).NotTo(HaveOccurred())
		Expect(done).To(BeTrue())

		result, err := s.Result()
		Expect(err).NotTo(HaveOccurred())
		Expect(result).To(Equal(TextResult{Text: "hi"}))
	})
})

var _ = Describe("Google content policy blocks", func() {
	const promptBlocked = `{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"},"usageMetadata":{"promptTokenCount":5}}`
	const safetyStop = `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"SAFETY"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1}}`

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

	sendJSON := func(body string) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(body))
		}
	}
	sendSSE := func(body string) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, "data: "+body+"\n\n")
		}
	}

	It("reports a blocked prompt instead of an empty answer", func() {
		m := newModel(sendJSON(promptBlocked))

		_, err := m.GenerateContent(context.Background(), WithMessages(NewMessage(RoleUser, NewTextPart("hi"))))

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Provider).To(Equal("google"))
		Expect(pe.Code).To(Equal("PROHIBITED_CONTENT"))
		Expect(pe.PartialOutput).To(BeFalse())
		Expect(pe.PartialUsage).NotTo(BeNil())
	})

	It("reports a blocked prompt in a stream", func() {
		m := newModel(sendSSE(promptBlocked))

		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStreamingFunc(func(context.Context, StreamingEvent) error { return nil }),
		)

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Code).To(Equal("PROHIBITED_CONTENT"))
	})

	It("reports a response that a safety filter stopped", func() {
		m := newModel(sendSSE(safetyStop))

		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithStreamingFunc(func(context.Context, StreamingEvent) error { return nil }),
		)
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Code).To(Equal("SAFETY"))
		// The SDK drops finishMessage for the Gemini API backend, so the
		// message is empty here.
		Expect(pe.PartialOutput).To(BeTrue())
	})
})

var _ = Describe("OpenRouter content policy blocks", func() {
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

	sendSSE := func(body string) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, body)
		}
	}
	sendError := func(status int, body string) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			_, _ = w.Write([]byte(body))
		}
	}
	finish := func(reason, native string) string {
		return "data: {\"id\":\"gen_1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"" + reason + "\",\"native_finish_reason\":\"" + native + "\"}]," +
			"\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}\n\n" +
			"data: [DONE]\n\n"
	}
	stream := WithStreamingFunc(func(context.Context, StreamingEvent) error { return nil })
	generate := func(m *openrouterModel, options ...GenerateOption) error {
		options = append([]GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}, options...)
		_, err := m.GenerateContent(context.Background(), options...)
		return err
	}

	It("reports the finish reason content_filter", func() {
		err := generate(newModel(sendSSE(finish("content_filter", "SAFETY"))), stream)
		Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Provider).To(Equal("openrouter"))
		Expect(pe.Code).To(Equal("SAFETY"))
		Expect(pe.PartialOutput).To(BeTrue())
		Expect(pe.PartialUsage).NotTo(BeNil())
	})

	It("reports the finish reason error with a policy reason from upstream", func() {
		err := generate(newModel(sendSSE(finish("error", "refusal"))), stream)

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.Code).To(Equal("refusal"))
	})

	It("does not report other errors as blocks", func() {
		err := generate(newModel(sendSSE(finish("error", "error"))), stream)
		Expect(err).To(HaveOccurred())
		Expect(IsContentPolicy(err)).To(BeFalse())
	})

	It("reports an OpenRouter moderation error", func() {
		err := generate(newModel(sendError(http.StatusForbidden,
			`{"error":{"code":403,"message":"Input was flagged","metadata":{"reasons":["violence"],"flagged_input":"hi","provider_name":"p","model_slug":"m"}}}`)))

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.StatusCode).To(Equal(403))
		Expect(pe.Code).To(Equal("moderation"))
		Expect(pe.Message).To(Equal("Input was flagged"))
	})

	It("reports a policy error from the upstream provider", func() {
		err := generate(newModel(sendError(http.StatusBadRequest,
			`{"error":{"code":400,"message":"Provider returned error","metadata":{"provider_name":"OpenAI","raw":"{\"error\":{\"type\":\"invalid_request\",\"code\":\"cyber_policy\",\"message\":\"flagged\"}}"}}}`)))

		var pe *ContentPolicyError
		Expect(errors.As(err, &pe)).To(BeTrue())
		Expect(pe.StatusCode).To(Equal(400))
		Expect(pe.Code).To(Equal("cyber_policy"))
		Expect(pe.Message).To(Equal("flagged"))
	})

	It("does not report other upstream errors as blocks", func() {
		err := generate(newModel(sendError(http.StatusBadRequest,
			`{"error":{"code":400,"message":"Provider returned error","metadata":{"provider_name":"OpenAI","raw":"{\"error\":{\"code\":\"unsupported_parameter\",\"message\":\"bad\"}}"}}}`)))
		Expect(err).To(HaveOccurred())
		Expect(IsContentPolicy(err)).To(BeFalse())
	})
})
