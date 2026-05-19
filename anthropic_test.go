package llms

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
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
			{ID: "c", Name: "z", Error: "boom"},
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
