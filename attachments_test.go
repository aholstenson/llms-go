package llms

import (
	"context"
	"encoding/json"

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
		It("puts the attachments inside the function_call_output item", func() {
			m := &openaiModel{}
			out, err := m.convertMessages(toolResultWithImage())
			Expect(err).NotTo(HaveOccurred())
			Expect(out).To(HaveLen(1))

			fco := out[0].OfFunctionCallOutput
			Expect(fco).NotTo(BeNil())
			Expect(fco.CallID.Value).To(Equal("call_1"))
			Expect(fco.Output.OfString.Valid()).To(BeFalse())

			items := fco.Output.OfResponseFunctionCallOutputItemArray
			Expect(items).To(HaveLen(2))
			Expect(items[0].OfInputText).NotTo(BeNil())
			Expect(items[0].OfInputText.Text).To(Equal("captured"))
			Expect(items[1].OfInputImage).NotTo(BeNil())
			Expect(items[1].OfInputImage.ImageURL.Value).To(Equal("data:image/png;base64,AQID"))
		})

		It("sends the output as a list of typed content items", func() {
			m := &openaiModel{}
			out, err := m.convertMessages(toolResultWithImage())
			Expect(err).NotTo(HaveOccurred())

			raw, err := json.Marshal(out[0])
			Expect(err).NotTo(HaveOccurred())
			var wire struct {
				Type   string `json:"type"`
				CallID string `json:"call_id"`
				Output []struct {
					Type     string `json:"type"`
					Text     string `json:"text"`
					ImageURL string `json:"image_url"`
				} `json:"output"`
			}
			Expect(json.Unmarshal(raw, &wire)).To(Succeed())
			Expect(wire.Type).To(Equal("function_call_output"))
			Expect(wire.CallID).To(Equal("call_1"))
			Expect(wire.Output).To(HaveLen(2))
			Expect(wire.Output[0].Type).To(Equal("input_text"))
			Expect(wire.Output[0].Text).To(Equal("captured"))
			Expect(wire.Output[1].Type).To(Equal("input_image"))
			Expect(wire.Output[1].ImageURL).To(HavePrefix("data:image/png;base64,"))
		})

		It("keeps a result without attachments as a plain string", func() {
			m := &openaiModel{}
			out, err := m.convertMessages([]*Message{
				NewMessage(RoleUser, &ToolResultPart{ID: "call_1", Name: "echo", Text: "ok"}),
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(out).To(HaveLen(1))
			Expect(out[0].OfFunctionCallOutput.Output.OfString.Value).To(Equal("ok"))
			Expect(out[0].OfFunctionCallOutput.Output.OfResponseFunctionCallOutputItemArray).To(BeEmpty())
		})

		It("sends only the error text for a failed result", func() {
			m := &openaiModel{}
			out, err := m.convertMessages([]*Message{
				NewMessage(RoleUser, &ToolResultPart{
					ID:          "call_1",
					Name:        "shot",
					Error:       "camera offline",
					Attachments: []MessagePart{NewBinaryPart("image/png", []byte{1})},
				}),
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(out).To(HaveLen(1))
			Expect(out[0].OfFunctionCallOutput.Output.OfString.Value).To(Equal("camera offline"))
			Expect(out[0].OfFunctionCallOutput.Output.OfResponseFunctionCallOutputItemArray).To(BeEmpty())
		})

		It("sends a non-image attachment as a file item", func() {
			m := &openaiModel{}
			out, err := m.convertMessages([]*Message{
				NewMessage(RoleUser, &ToolResultPart{
					ID:          "call_1",
					Name:        "report",
					Attachments: []MessagePart{NewBinaryPart("application/pdf", []byte{1, 2, 3})},
				}),
			})
			Expect(err).NotTo(HaveOccurred())

			// An empty text is left out of the list.
			items := out[0].OfFunctionCallOutput.Output.OfResponseFunctionCallOutputItemArray
			Expect(items).To(HaveLen(1))
			Expect(items[0].OfInputFile).NotTo(BeNil())
			Expect(items[0].OfInputFile.FileData.Value).To(Equal("data:application/pdf;base64,AQID"))
			Expect(items[0].OfInputFile.Filename.Value).To(Equal("file.pdf"))
		})
	})

	Describe("OpenAI ObserveToolResults", func() {
		It("keeps each call's attachments in its own output for parallel calls", func() {
			turn := &openaiTurn{m: &openaiModel{}}
			Expect(turn.ObserveToolResults(context.Background(), nil, []ToolOutcome{
				{ID: "call_a", Name: "chart", Text: "a", Attachments: []MessagePart{
					NewBinaryPart("image/png", []byte{1}),
				}},
				{ID: "call_b", Name: "shot", Text: "b", Attachments: []MessagePart{
					NewImagePart("https://example.com/b.png"),
				}},
			})).To(Succeed())

			// One function-call output per call, and no trailing user message.
			Expect(turn.inputItems).To(HaveLen(2))

			a := turn.inputItems[0].OfFunctionCallOutput
			Expect(a).NotTo(BeNil())
			Expect(a.CallID.Value).To(Equal("call_a"))
			Expect(a.Output.OfResponseFunctionCallOutputItemArray).To(HaveLen(2))
			Expect(a.Output.OfResponseFunctionCallOutputItemArray[0].OfInputText.Text).To(Equal("a"))
			Expect(a.Output.OfResponseFunctionCallOutputItemArray[1].OfInputImage).NotTo(BeNil())

			b := turn.inputItems[1].OfFunctionCallOutput
			Expect(b).NotTo(BeNil())
			Expect(b.CallID.Value).To(Equal("call_b"))
			Expect(b.Output.OfResponseFunctionCallOutputItemArray).To(HaveLen(2))
			Expect(b.Output.OfResponseFunctionCallOutputItemArray[1].OfInputImage.ImageURL.Value).To(Equal("https://example.com/b.png"))
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
