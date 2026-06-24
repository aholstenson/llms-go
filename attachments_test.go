package llms

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	openrouter "github.com/revrost/go-openrouter"
	"google.golang.org/genai"
)

// toolResultWithImage is a replayed tool-result user message carrying a single
// image attachment, shared by the provider trailing-message-fallback tests.
func toolResultWithImage() []*Message {
	return []*Message{
		NewMessage(RoleUser, &ToolResultPart{
			ID:   "call_1",
			Name: "shot",
			Text: "captured",
			Attachments: []MessagePart{
				NewBinaryPart("image/png", []byte{1, 2, 3}),
			},
		}),
	}
}

var _ = Describe("tool-result attachment fallback", func() {
	Describe("OpenAI convertMessages", func() {
		It("keeps the tool-result item text-only and appends a labeled trailing user message", func() {
			m := &openaiModel{}
			out, err := m.convertMessages(toolResultWithImage())
			Expect(err).NotTo(HaveOccurred())
			Expect(out).To(HaveLen(2))

			Expect(out[0].OfFunctionCallOutput).NotTo(BeNil())
			Expect(out[0].OfFunctionCallOutput.Output).To(Equal("captured"))

			msg := out[1].OfInputMessage
			Expect(msg).NotTo(BeNil())
			Expect(msg.Role).To(Equal("user"))
			// Label text part referencing the call, then the image.
			Expect(msg.Content).To(HaveLen(2))
			Expect(msg.Content[0].OfInputText).NotTo(BeNil())
			Expect(msg.Content[0].OfInputText.Text).To(And(ContainSubstring("shot"), ContainSubstring("call_1")))
			Expect(msg.Content[1].OfInputImage).NotTo(BeNil())
		})
	})

	Describe("OpenAI ObserveToolResults", func() {
		It("labels each call's attachments in the trailing message for parallel calls", func() {
			turn := &openaiTurn{m: &openaiModel{}}
			Expect(turn.ObserveToolResults(context.Background(), nil, []ToolOutcome{
				{ID: "call_a", Name: "chart", Text: "a", Attachments: []MessagePart{
					NewBinaryPart("image/png", []byte{1}),
				}},
				{ID: "call_b", Name: "shot", Text: "b", Attachments: []MessagePart{
					NewBinaryPart("image/png", []byte{2}),
				}},
			})).To(Succeed())

			// Two function-call outputs, then one trailing user message.
			Expect(turn.inputItems).To(HaveLen(3))
			Expect(turn.inputItems[0].OfFunctionCallOutput).NotTo(BeNil())
			Expect(turn.inputItems[1].OfFunctionCallOutput).NotTo(BeNil())

			msg := turn.inputItems[2].OfInputMessage
			Expect(msg).NotTo(BeNil())
			// label_a, image_a, label_b, image_b
			Expect(msg.Content).To(HaveLen(4))
			Expect(msg.Content[0].OfInputText.Text).To(And(ContainSubstring("chart"), ContainSubstring("call_a")))
			Expect(msg.Content[1].OfInputImage).NotTo(BeNil())
			Expect(msg.Content[2].OfInputText.Text).To(And(ContainSubstring("shot"), ContainSubstring("call_b")))
			Expect(msg.Content[3].OfInputImage).NotTo(BeNil())
		})
	})

	Describe("Google convertMessages", func() {
		It("keeps the function_response text-only and appends a labeled trailing user message", func() {
			m := &googleModel{}
			out, err := m.convertMessages(toolResultWithImage())
			Expect(err).NotTo(HaveOccurred())
			Expect(out).To(HaveLen(2))

			Expect(out[0].Role).To(Equal(genai.RoleUser))
			Expect(out[0].Parts[0].FunctionResponse).NotTo(BeNil())

			Expect(out[1].Role).To(Equal(genai.RoleUser))
			Expect(out[1].Parts).To(HaveLen(2))
			Expect(out[1].Parts[0].Text).To(And(ContainSubstring("shot"), ContainSubstring("call_1")))
			Expect(out[1].Parts[1].InlineData).NotTo(BeNil())
			Expect(out[1].Parts[1].InlineData.MIMEType).To(Equal("image/png"))
		})
	})

	Describe("Google ObserveToolResults", func() {
		It("appends a labeled trailing user message carrying the attachment", func() {
			turn := &googleTurn{m: &googleModel{}}
			Expect(turn.ObserveToolResults(context.Background(), nil, []ToolOutcome{
				{ID: "call_1", Name: "shot", Text: "captured", Attachments: []MessagePart{
					NewBinaryPart("image/png", []byte{1, 2, 3}),
				}},
			})).To(Succeed())
			Expect(turn.messages).To(HaveLen(2))
			Expect(turn.messages[0].Parts[0].FunctionResponse).NotTo(BeNil())
			Expect(turn.messages[1].Parts).To(HaveLen(2))
			Expect(turn.messages[1].Parts[0].Text).To(ContainSubstring("shot"))
			Expect(turn.messages[1].Parts[1].InlineData).NotTo(BeNil())
		})
	})

	Describe("OpenRouter convertMessages", func() {
		It("keeps the tool message text-only and appends a labeled trailing user message", func() {
			m := &openrouterModel{}
			out, err := m.convertMessages("", toolResultWithImage())
			Expect(err).NotTo(HaveOccurred())
			Expect(out).To(HaveLen(2))

			Expect(out[0].Role).To(Equal(openrouter.ChatMessageRoleTool))

			Expect(out[1].Role).To(Equal(openrouter.ChatMessageRoleUser))
			Expect(out[1].Content.Multi).To(HaveLen(2))
			Expect(out[1].Content.Multi[0].Type).To(Equal(openrouter.ChatMessagePartTypeText))
			Expect(out[1].Content.Multi[0].Text).To(And(ContainSubstring("shot"), ContainSubstring("call_1")))
			Expect(out[1].Content.Multi[1].Type).To(Equal(openrouter.ChatMessagePartTypeImageURL))
		})
	})

	Describe("OpenRouter ObserveToolResults", func() {
		It("appends a labeled trailing user message carrying the attachment", func() {
			turn := &openrouterTurn{m: &openrouterModel{}}
			Expect(turn.ObserveToolResults(context.Background(), nil, []ToolOutcome{
				{ID: "call_1", Name: "shot", Text: "captured", Attachments: []MessagePart{
					NewBinaryPart("image/png", []byte{1, 2, 3}),
				}},
			})).To(Succeed())
			Expect(turn.params.Messages).To(HaveLen(2))
			Expect(turn.params.Messages[0].Role).To(Equal(openrouter.ChatMessageRoleTool))
			Expect(turn.params.Messages[1].Role).To(Equal(openrouter.ChatMessageRoleUser))
			multi := turn.params.Messages[1].Content.Multi
			Expect(multi).To(HaveLen(2))
			Expect(multi[0].Type).To(Equal(openrouter.ChatMessagePartTypeText))
			Expect(multi[1].Type).To(Equal(openrouter.ChatMessagePartTypeImageURL))
		})
	})
})
