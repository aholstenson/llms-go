package llms

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"io"
	"log/slog"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	openrouter "github.com/revrost/go-openrouter"
)

// pngOfSize returns a valid PNG image with the given pixel size.
func pngOfSize(w, h int) []byte {
	var buf bytes.Buffer
	Expect(png.Encode(&buf, image.NewGray(image.Rect(0, 0, w, h)))).To(Succeed())
	return buf.Bytes()
}

// webpHeader returns the first 30 bytes of a WebP file with the given chunk
// type and chunk payload, padded with zeros.
func webpHeader(chunk string, payload []byte) []byte {
	data := make([]byte, 30)
	copy(data[0:], "RIFF")
	copy(data[8:], "WEBP")
	copy(data[12:], chunk)
	copy(data[20:], payload)
	return data
}

// inputError returns the *InputError in err, failing the test when there is
// none.
func inputError(err error) *InputError {
	var ie *InputError
	ExpectWithOffset(1, errors.As(err, &ie)).To(BeTrue(), "expected an *InputError, got %v", err)
	return ie
}

var _ = Describe("CheckMessages", func() {
	var logger = slog.New(slog.NewTextHandler(io.Discard, nil))

	newModel := func(info ModelInfo) Model {
		return newOpenAIModel(logger, NewNoopMetrics(), StaticCredentials("test-key"), "m", nil, fixedModelInfo(info))
	}
	user := func(parts ...MessagePart) []*Message {
		return []*Message{NewMessage(RoleUser, parts...)}
	}

	It("accepts parts the model supports", func() {
		m := newModel(knownInfo(Capabilities{}, "text", "image", "pdf"))
		Expect(CheckMessages(m, user(
			NewTextPart("hi"),
			NewBinaryPart("image/png", pngOfSize(2, 2)),
			NewBinaryPart("application/pdf", []byte("%PDF")),
			NewBinaryPart("text/markdown", []byte("# spec")).WithName("spec.md"),
		))).To(Succeed())
	})

	It("reports the failing part with ErrUnsupportedModality", func() {
		m := newModel(knownInfo(Capabilities{}, "text", "image"))
		err := CheckMessages(m, []*Message{
			NewMessage(RoleUser, NewTextPart("hi")),
			NewMessage(RoleUser, NewTextPart("see"), NewBinaryPart("application/pdf", []byte("%PDF")).WithName("a.pdf")),
		})
		Expect(err).To(MatchError(ErrUnsupportedModality))
		ie := inputError(err)
		Expect(ie.Model).To(Equal("openai/m"))
		Expect(ie.Message).To(Equal(1))
		Expect(ie.Part).To(Equal(1))
		Expect(ie.Name).To(Equal("a.pdf"))
		Expect(ie.MediaType).To(Equal("application/pdf"))
		Expect(ie.Modality).To(Equal("pdf"))
		Expect(err.Error()).To(ContainSubstring("does not support pdf input"))
	})

	It("is the same check that GenerateContent runs", func() {
		m := newModel(knownInfo(Capabilities{}, "text"))
		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewBinaryPart("image/png", []byte{1}))))
		Expect(err).To(MatchError(ErrUnsupportedModality))
		Expect(inputError(err).Part).To(Equal(0))
	})

	It("accepts any input for an unknown model", func() {
		m := newModel(ModelInfo{})
		Expect(CheckMessages(m, user(NewBinaryPart("video/mp4", []byte{1})))).To(Succeed())
	})

	It("accepts any input for a Model from another package", func() {
		Expect(CheckMessages(&testModel{}, user(NewBinaryPart("video/mp4", []byte{1})))).To(Succeed())
	})

	It("rejects a text file that is not valid UTF-8", func() {
		m := newModel(knownInfo(Capabilities{}, "text"))
		err := CheckMessages(m, user(NewBinaryPart("text/plain", []byte{0xff, 0xfe})))
		Expect(err).To(MatchError(ErrInvalidAttachment))
		Expect(inputError(err).Modality).To(Equal("text"))
	})

	Describe("limits", func() {
		info := func(l Limits) ModelInfo {
			mi := knownInfo(Capabilities{}, "text", "image", "pdf")
			mi.Limits = l
			return mi
		}

		It("rejects an image above MaxBytes", func() {
			m := newModel(info(Limits{Image: ImageLimits{MaxBytes: 10}}))
			err := CheckMessages(m, user(NewBinaryPart("image/png", make([]byte, 11))))
			Expect(err).To(MatchError(ErrAttachmentTooLarge))
			Expect(CheckMessages(m, user(NewBinaryPart("image/png", make([]byte, 10))))).To(Succeed())
		})

		It("rejects an image above the pixel size", func() {
			m := newModel(info(Limits{Image: ImageLimits{MaxWidth: 8, MaxHeight: 8}}))
			Expect(CheckMessages(m, user(NewBinaryPart("image/png", pngOfSize(8, 8))))).To(Succeed())
			err := CheckMessages(m, user(NewBinaryPart("image/png", pngOfSize(9, 2))))
			Expect(err).To(MatchError(ErrAttachmentTooLarge))
			Expect(err.Error()).To(ContainSubstring("9x2 pixels"))
		})

		It("uses the lower pixel size when the request has many images", func() {
			m := newModel(info(Limits{Image: ImageLimits{
				MaxWidth: 8, MaxHeight: 8, ManyImages: 1, ManyMaxWidth: 4, ManyMaxHeight: 4,
			}}))
			img := pngOfSize(6, 6)
			Expect(CheckMessages(m, user(NewBinaryPart("image/png", img)))).To(Succeed())
			err := CheckMessages(m, user(NewBinaryPart("image/png", img), NewBinaryPart("image/png", img)))
			Expect(err).To(MatchError(ErrAttachmentTooLarge))
		})

		It("rejects more images than MaxCount", func() {
			m := newModel(info(Limits{Image: ImageLimits{MaxCount: 1}}))
			err := CheckMessages(m, user(NewImagePart("https://x/a.png"), NewTextPart("and"), NewImagePart("https://x/b.png")))
			Expect(err).To(MatchError(ErrAttachmentTooLarge))
			Expect(inputError(err).Part).To(Equal(2))
		})

		It("skips the pixel check for an image it cannot read", func() {
			m := newModel(info(Limits{Image: ImageLimits{MaxWidth: 1, MaxHeight: 1}}))
			Expect(CheckMessages(m, user(NewBinaryPart("image/png", []byte("not a png"))))).To(Succeed())
		})

		It("rejects an image media type the provider does not list", func() {
			m := newModel(info(Limits{Image: ImageLimits{MediaTypes: []string{"image/png"}}}))
			err := CheckMessages(m, user(NewBinaryPart("image/bmp", []byte{1})))
			Expect(err).To(MatchError(ErrUnsupportedMediaType))
			Expect(CheckMessages(m, user(NewBinaryPart("image/png; x=y", []byte{1})))).To(Succeed())
		})

		It("rejects a PDF above MaxBytes", func() {
			m := newModel(info(Limits{PDF: PDFLimits{MaxBytes: 3}}))
			err := CheckMessages(m, user(NewBinaryPart("application/pdf", []byte("%PDF"))))
			Expect(err).To(MatchError(ErrAttachmentTooLarge))
			Expect(inputError(err).Modality).To(Equal("pdf"))
		})
	})

	Describe("ModelInfoOf", func() {
		It("returns the gating ModelInfo", func() {
			mi := knownInfo(Capabilities{}, "text", "image")
			got, ok := ModelInfoOf(newModel(mi))
			Expect(ok).To(BeTrue())
			Expect(got.Modalities).To(Equal([]string{"text", "image"}))
		})

		It("reports an unknown model", func() {
			_, ok := ModelInfoOf(newModel(ModelInfo{}))
			Expect(ok).To(BeFalse())
			_, ok = ModelInfoOf(&testModel{})
			Expect(ok).To(BeFalse())
		})
	})
})

var _ = Describe("imageDimensions", func() {
	It("reads PNG", func() {
		w, h, ok := imageDimensions("image/png", pngOfSize(3, 5))
		Expect(ok).To(BeTrue())
		Expect([]int{w, h}).To(Equal([]int{3, 5}))
	})

	It("reads extended WebP", func() {
		// Canvas width-1 = 799, height-1 = 599 as 24-bit values at offset 24.
		payload := make([]byte, 10)
		payload[4], payload[5] = 0x1f, 0x03
		payload[7], payload[8] = 0x57, 0x02
		w, h, ok := imageDimensions("image/webp", webpHeader("VP8X", payload))
		Expect(ok).To(BeTrue())
		Expect([]int{w, h}).To(Equal([]int{800, 600}))
	})

	It("reads lossless WebP", func() {
		bits := uint32(99) | uint32(49)<<14
		payload := make([]byte, 5)
		payload[0] = 0x2f
		binary.LittleEndian.PutUint32(payload[1:], bits)
		w, h, ok := imageDimensions("image/webp", webpHeader("VP8L", payload))
		Expect(ok).To(BeTrue())
		Expect([]int{w, h}).To(Equal([]int{100, 50}))
	})

	It("reads lossy WebP", func() {
		payload := make([]byte, 10)
		copy(payload[3:], []byte{0x9d, 0x01, 0x2a})
		binary.LittleEndian.PutUint16(payload[6:], 640)
		binary.LittleEndian.PutUint16(payload[8:], 480)
		w, h, ok := imageDimensions("image/webp", webpHeader("VP8 ", payload))
		Expect(ok).To(BeTrue())
		Expect([]int{w, h}).To(Equal([]int{640, 480}))
	})

	It("does not read unknown formats", func() {
		_, _, ok := imageDimensions("image/bmp", []byte("BM"))
		Expect(ok).To(BeFalse())
	})
})

var _ = Describe("text files", func() {
	It("puts the content in a file element with the name", func() {
		p := NewBinaryPart("text/markdown; charset=utf-8", []byte("# Spec")).WithName(`a "b".md`)
		Expect(textFileContent(p)).To(Equal("<file name=\"a &#34;b&#34;.md\" media_type=\"text/markdown\">\n# Spec\n</file>"))
	})

	It("leaves out the name when there is none", func() {
		p := NewBinaryPart("text/plain", []byte("x\n"))
		Expect(textFileContent(p)).To(Equal("<file media_type=\"text/plain\">\nx\n</file>"))
	})
})

var _ = Describe("BinaryPart JSON", func() {
	It("keeps Name in a round trip", func() {
		msg := NewMessage(RoleUser, NewBinaryPart("text/plain", []byte("x")).WithName("a.txt"))
		raw, err := json.Marshal(msg)
		Expect(err).NotTo(HaveOccurred())
		var got Message
		Expect(json.Unmarshal(raw, &got)).To(Succeed())
		Expect(got.Parts[0].(*BinaryPart).Name).To(Equal("a.txt"))
	})

	It("leaves Name out when it is empty", func() {
		raw, err := json.Marshal(NewMessage(RoleUser, NewBinaryPart("text/plain", []byte("x"))))
		Expect(err).NotTo(HaveOccurred())
		Expect(string(raw)).NotTo(ContainSubstring("Name"))
	})
})

var _ = Describe("attachment conversion", func() {
	named := func() []*Message {
		return []*Message{NewMessage(RoleUser,
			NewTextPart("Read these."),
			NewBinaryPart("application/pdf", []byte("%PDF")).WithName("report.pdf"),
			NewBinaryPart("text/markdown", []byte("# Spec")).WithName("spec.md"),
			NewBinaryPart("image/png", []byte{1}).WithName("chart.png"),
		)}
	}

	It("sends Anthropic documents with titles", func() {
		m := &anthropicModel{}
		out, _, err := m.convertMessages("", named())
		Expect(err).NotTo(HaveOccurred())
		blocks := out[0].OfUser.Content
		Expect(blocks).To(HaveLen(5))

		pdf := blocks[1].OfDocument
		Expect(pdf).NotTo(BeNil())
		Expect(pdf.Title.Value).To(Equal("report.pdf"))
		Expect(pdf.Source.OfBase64).NotTo(BeNil())
		Expect(pdf.Source.OfBase64.Data).To(Equal("JVBERg=="))

		text := blocks[2].OfDocument
		Expect(text).NotTo(BeNil())
		Expect(text.Title.Value).To(Equal("spec.md"))
		Expect(text.Source.OfText.Data).To(Equal("# Spec"))

		Expect(blocks[3].OfText.Text).To(Equal(`File "chart.png":`))
		Expect(blocks[4].OfImage.Source.OfBase64.MediaType).To(BeEquivalentTo("image/png"))
	})

	It("sends Anthropic tool-result PDFs as documents", func() {
		m := &anthropicModel{}
		out, _, err := m.convertMessages("", []*Message{NewMessage(RoleUser, &ToolResultPart{
			ID: "call_1", Name: "fetch", Text: "ok",
			Attachments: []MessagePart{NewBinaryPart("application/pdf", []byte("%PDF"))},
		})})
		Expect(err).NotTo(HaveOccurred())
		content := out[0].OfUser.Content[0].OfToolResult.Content
		Expect(content).To(HaveLen(2))
		Expect(content[1].OfDocument).NotTo(BeNil())
	})

	It("rejects an unknown binary type for Anthropic before the request", func() {
		m := &anthropicModel{}
		_, _, err := m.convertMessages("", []*Message{NewMessage(RoleUser, NewBinaryPart("application/zip", []byte{1}))})
		Expect(err).To(MatchError(ErrUnsupportedMediaType))
	})

	It("sends OpenAI files with their names and text files as text", func() {
		m := &openaiModel{}
		out, err := m.convertMessages(named())
		Expect(err).NotTo(HaveOccurred())
		content := out[0].OfInputMessage.Content
		Expect(content).To(HaveLen(5))
		Expect(content[1].OfInputFile.Filename.Value).To(Equal("report.pdf"))
		Expect(content[2].OfInputText.Text).To(HavePrefix(`<file name="spec.md" media_type="text/markdown">`))
		Expect(content[3].OfInputText.Text).To(Equal(`File "chart.png":`))
		Expect(content[4].OfInputImage).NotTo(BeNil())
	})

	It("sends Google files with labels and text files as text", func() {
		m := &googleModel{}
		out, err := m.convertMessages(named())
		Expect(err).NotTo(HaveOccurred())
		parts := out[0].Parts
		Expect(parts).To(HaveLen(6))
		Expect(parts[1].Text).To(Equal(`File "report.pdf":`))
		Expect(parts[2].InlineData.MIMEType).To(Equal("application/pdf"))
		Expect(parts[3].Text).To(HavePrefix(`<file name="spec.md"`))
		Expect(parts[4].Text).To(Equal(`File "chart.png":`))
		Expect(parts[5].InlineData.MIMEType).To(Equal("image/png"))
	})

	It("sends OpenRouter PDFs as file parts with their names", func() {
		m := &openrouterModel{}
		out, err := m.convertMessages("", named())
		Expect(err).NotTo(HaveOccurred())
		multi := out[0].Content.Multi
		Expect(multi).To(HaveLen(5))
		Expect(multi[1].Type).To(Equal(openrouter.ChatMessagePartTypeFile))
		Expect(multi[1].File.Filename).To(Equal("report.pdf"))
		Expect(multi[1].File.FileData).To(Equal("data:application/pdf;base64,JVBERg=="))
		Expect(multi[2].Type).To(Equal(openrouter.ChatMessagePartTypeText))
		Expect(multi[2].Text).To(HavePrefix(`<file name="spec.md"`))
		Expect(multi[3].Text).To(Equal(`File "chart.png":`))
		Expect(multi[4].Type).To(Equal(openrouter.ChatMessagePartTypeImageURL))
	})

	It("uses a synthetic file name when the part has no name", func() {
		m := &openaiModel{}
		out, err := m.convertMessages([]*Message{NewMessage(RoleUser, NewBinaryPart("application/pdf", []byte{1}))})
		Expect(err).NotTo(HaveOccurred())
		Expect(out[0].OfInputMessage.Content[0].OfInputFile.Filename.Value).To(Equal("file.pdf"))
	})
})

var _ = Describe("defaultInputLimits", func() {
	It("uses the lower Anthropic limits below a 1M context window", func() {
		img, pdf := defaultInputLimits("anthropic", 200_000)
		Expect(img.MaxCount).To(Equal(100))
		Expect(pdf.MaxPages).To(Equal(100))
		img, pdf = defaultInputLimits("anthropic", 1_000_000)
		Expect(img.MaxCount).To(Equal(600))
		Expect(pdf.MaxPages).To(Equal(600))
	})

	It("measures the Anthropic image limit on the base64 string", func() {
		img, _ := defaultInputLimits("anthropic", 0)
		Expect(int64(base64.StdEncoding.EncodedLen(int(img.MaxBytes)))).To(BeNumerically("<=", 10_000_000))
	})

	It("gives OpenRouter no limits", func() {
		img, pdf := defaultInputLimits("openrouter", 0)
		Expect(img).To(BeZero())
		Expect(pdf).To(BeZero())
	})

	It("is in the models.dev data", func() {
		info, ok := LookupModelInfo("anthropic/claude-haiku-4-5")
		Expect(ok).To(BeTrue())
		Expect(info.Limits.Image.MaxWidth).To(Equal(8000))
		Expect(info.Limits.PDF.MaxPages).To(Equal(100))
	})
})
