package llms

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func toolResultBlock(id string, cached bool) anthropic.BetaContentBlockParamUnion {
	cc := anthropic.BetaCacheControlEphemeralParam{}
	if cached {
		cc = anthropic.NewBetaCacheControlEphemeralParam()
	}
	return anthropic.BetaContentBlockParamUnion{
		OfToolResult: &anthropic.BetaToolResultBlockParam{
			ToolUseID:    id,
			Content:      []anthropic.BetaToolResultBlockParamContentUnion{{OfText: &anthropic.BetaTextBlockParam{Text: "old"}}},
			CacheControl: cc,
		},
	}
}

func countCacheControl(p anthropic.BetaMessageNewParams) int {
	n := 0
	for _, s := range p.System {
		if string(s.CacheControl.Type) != "" {
			n++
		}
	}
	for _, tu := range p.Tools {
		if tu.OfTool != nil && string(tu.OfTool.CacheControl.Type) != "" {
			n++
		}
	}
	for _, msg := range p.Messages {
		for _, b := range msg.Content {
			if b.OfText != nil && string(b.OfText.CacheControl.Type) != "" {
				n++
			}
			if b.OfToolResult != nil && string(b.OfToolResult.CacheControl.Type) != "" {
				n++
			}
		}
	}
	return n
}

var _ = Describe("anthropicTurn cache-control budget", func() {
	It("keeps total cache_control blocks within Anthropic's limit of 4 across Inject", func() {
		m := &anthropicModel{}

		msgs, _, err := m.convertMessages("", []*Message{
			NewMessage(RoleUser, NewTextPart("hello")),
		})
		Expect(err).NotTo(HaveOccurred())

		tools, _ := m.convertTools([]ToolDef{
			NewToolDef[*sessArgs, string](sessTool{name: "a"}),
			NewToolDef[*sessArgs, string](sessTool{name: "b"}),
		})

		turn := &anthropicTurn{
			m:        m,
			response: &anthropic.BetaMessage{Role: "assistant"},
			params: anthropic.BetaMessageNewParams{
				System: []anthropic.BetaTextBlockParam{
					{Text: "sys", CacheControl: anthropic.NewBetaCacheControlEphemeralParam()},
				},
				Tools:    tools,
				Messages: msgs,
			},
		}

		// system(1) + last tool(1) + last user text(1) = 3
		Expect(countCacheControl(turn.params)).To(Equal(3))

		// A tool-result turn adds exactly one cached block (the last result).
		Expect(turn.Observe(context.Background(), TurnOutput{}, []ToolOutcome{
			{ID: "1", Name: "a", Text: "r"},
		})).To(Succeed())
		Expect(countCacheControl(turn.params)).To(Equal(4))

		// Injecting an operator message must not add a 5th breakpoint.
		turn.Inject(NewMessage(RoleUser, NewTextPart("operator: stop")))
		conv, _, err := m.convertMessages("", turn.pending)
		Expect(err).NotTo(HaveOccurred())
		clearAnthropicCacheControl(conv)
		turn.params.Messages = append(turn.params.Messages, conv...)

		Expect(countCacheControl(turn.params)).To(Equal(4))
	})
})

var _ = Describe("anthropicTurn tool-result attachments", func() {
	It("appends an OfImage content block to the tool_result for an attachment", func() {
		m := &anthropicModel{}
		turn := &anthropicTurn{
			m:        m,
			response: &anthropic.BetaMessage{Role: "assistant"},
			params:   anthropic.BetaMessageNewParams{},
		}

		Expect(turn.Observe(context.Background(), TurnOutput{}, []ToolOutcome{
			{
				ID:   "1",
				Name: "shot",
				Text: "captured",
				Attachments: []MessagePart{
					NewBinaryPart("image/png", []byte{1, 2, 3}),
				},
			},
		})).To(Succeed())

		// The last message is the tool-result user message.
		last := turn.params.Messages[len(turn.params.Messages)-1]
		Expect(last.Content).To(HaveLen(1))
		tr := last.Content[0].OfToolResult
		Expect(tr).NotTo(BeNil())
		Expect(tr.Content).To(HaveLen(2))
		Expect(tr.Content[0].OfText).NotTo(BeNil())
		Expect(tr.Content[0].OfText.Text).To(Equal("captured"))
		Expect(tr.Content[1].OfImage).NotTo(BeNil())
		Expect(tr.Content[1].OfImage.Source.OfBase64).NotTo(BeNil())
	})

	It("carries attachments through convertMessages tool-result replay", func() {
		m := &anthropicModel{}
		msgs := []*Message{
			NewMessage(RoleUser, &ToolResultPart{
				ID:   "1",
				Name: "shot",
				Text: "captured",
				Attachments: []MessagePart{
					NewImagePart("https://example.com/a.png"),
				},
			}),
		}
		out, _, err := m.convertMessages("", msgs)
		Expect(err).NotTo(HaveOccurred())
		tr := out[0].Content[0].OfToolResult
		Expect(tr).NotTo(BeNil())
		Expect(tr.Content).To(HaveLen(2))
		Expect(tr.Content[1].OfImage).NotTo(BeNil())
		Expect(tr.Content[1].OfImage.Source.OfURL).NotTo(BeNil())
		Expect(tr.Content[1].OfImage.Source.OfURL.URL).To(Equal("https://example.com/a.png"))
	})
})

var _ = Describe("anthropicModel.convertTools cache-control", func() {
	It("caches the last deduped tool when duplicates appear at the tail", func() {
		m := &anthropicModel{}

		tools, _ := m.convertTools([]ToolDef{
			NewToolDef[*sessArgs, string](sessTool{name: "a"}),
			NewToolDef[*sessArgs, string](sessTool{name: "b"}),
			NewToolDef[*sessArgs, string](sessTool{name: "a"}),
		})

		Expect(tools).To(HaveLen(2))
		Expect(tools[0].OfTool.Name).To(Equal("a"))
		Expect(tools[1].OfTool.Name).To(Equal("b"))
		Expect(string(tools[0].OfTool.CacheControl.Type)).To(Equal(""))
		Expect(string(tools[1].OfTool.CacheControl.Type)).To(Equal("ephemeral"))
	})

	It("caches the last tool when there are no duplicates", func() {
		m := &anthropicModel{}

		tools, _ := m.convertTools([]ToolDef{
			NewToolDef[*sessArgs, string](sessTool{name: "a"}),
			NewToolDef[*sessArgs, string](sessTool{name: "b"}),
		})

		Expect(tools).To(HaveLen(2))
		Expect(string(tools[0].OfTool.CacheControl.Type)).To(Equal(""))
		Expect(string(tools[1].OfTool.CacheControl.Type)).To(Equal("ephemeral"))
	})

	It("returns nil for empty input", func() {
		m := &anthropicModel{}
		tools, toolMap := m.convertTools(nil)
		Expect(tools).To(BeNil())
		Expect(toolMap).To(BeEmpty())
	})
})

var _ = Describe("anthropicTurn.Observe cache-control", func() {
	It("clears prior tool_result cache-control and caches only the last new block", func() {
		turn := &anthropicTurn{
			response: &anthropic.BetaMessage{Role: "assistant"},
			params: anthropic.BetaMessageNewParams{
				Messages: []anthropic.BetaMessageParam{
					// A prior tool-result turn that currently carries cache-control.
					newBetaUserMessage(toolResultBlock("prior", true)),
				},
			},
		}

		err := turn.Observe(context.Background(), TurnOutput{}, []ToolOutcome{
			{ID: "a", Name: "x", Text: "ra"},
			{ID: "b", Name: "y", Text: "rb"},
			{ID: "c", Name: "z", Error: NewVisibleToolError("boom")},
		})
		Expect(err).NotTo(HaveOccurred())

		msgs := turn.params.Messages
		// prior(0), assistant(1, appended by Observe), new tool results(2)
		Expect(msgs).To(HaveLen(3))

		// The prior tool_result block must have had its cache-control cleared.
		prior := msgs[0].Content[0].OfToolResult
		Expect(prior).NotTo(BeNil())
		Expect(string(prior.CacheControl.Type)).To(Equal(""))

		// The newly appended tool-results message: only the last block cached.
		newBlocks := msgs[2].Content
		Expect(newBlocks).To(HaveLen(3))
		Expect(string(newBlocks[0].OfToolResult.CacheControl.Type)).To(Equal(""))
		Expect(string(newBlocks[1].OfToolResult.CacheControl.Type)).To(Equal(""))
		Expect(string(newBlocks[2].OfToolResult.CacheControl.Type)).To(Equal("ephemeral"))

		// Error outcome is rendered into the text content.
		Expect(newBlocks[2].OfToolResult.Content[0].OfText.Text).To(Equal("boom"))
	})
})

var _ = Describe("Anthropic retries", func() {
	// anthropicSuccessSSE is the smallest stream that yields one text block.
	const anthropicSuccessSSE = "event: message_start\n" +
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

	// newModel points an anthropicModel at handler.
	newModel := func(handler http.HandlerFunc) *anthropicModel {
		srv := httptest.NewServer(handler)
		DeferCleanup(srv.Close)
		return &anthropicModel{
			logger:  discardLogger(),
			metrics: NewNoopMetrics(),
			client: anthropic.NewClient(
				option.WithBaseURL(srv.URL+"/"),
				option.WithAPIKey("test"),
			),
			model:      "m",
			statsModel: "anthropic/m",
			info:       ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}},
		}
	}

	It("retries an overloaded request and reports every wait", func() {
		var requests int32
		m := newModel(func(w http.ResponseWriter, r *http.Request) {
			if atomic.AddInt32(&requests, 1) <= 2 {
				w.Header().Set("Retry-After-Ms", "1")
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			writeSSE(w, anthropicSuccessSSE)
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
		Expect(notices[0].Provider).To(Equal("anthropic"))
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
