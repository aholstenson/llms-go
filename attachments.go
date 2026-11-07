package llms

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"html"
	"image/gif"
	"image/jpeg"
	"image/png"
	"mime"
	"slices"
	"strings"
	"unicode/utf8"
)

// Sentinel errors for input that a model cannot accept. Each one is wrapped
// in an *InputError that tells which part failed. They are configuration
// errors: sending the same input again fails the same way, so do not retry.
var (
	// ErrUnsupportedModality is returned when a part needs an input
	// modality (image, pdf, audio, video) that the model does not accept.
	ErrUnsupportedModality = errors.New("unsupported input modality")
	// ErrUnsupportedMediaType is returned when the model accepts the
	// modality but not the media type, for example image/bmp.
	ErrUnsupportedMediaType = errors.New("unsupported media type")
	// ErrAttachmentTooLarge is returned when a part is larger than a known
	// provider limit (see ImageLimits and PDFLimits).
	ErrAttachmentTooLarge = errors.New("attachment exceeds model input limit")
	// ErrInvalidAttachment is returned when a part is malformed, for
	// example a text file that is not valid UTF-8.
	ErrInvalidAttachment = errors.New("invalid attachment")
)

// InputError tells which message part a model cannot accept. Err is one of
// ErrUnsupportedModality, ErrUnsupportedMediaType, ErrAttachmentTooLarge or
// ErrInvalidAttachment, so errors.Is works on the wrapped chain:
//
//	if errors.Is(err, llms.ErrUnsupportedModality) { ... }
//
// Use errors.As to get the location of the part.
type InputError struct {
	// Model is the qualified model name, for example "anthropic/claude-opus-4-5".
	Model string
	// Message is the index of the message in the list that was checked.
	Message int
	// Part is the index of the part in Message.Parts.
	Part int
	// Name is BinaryPart.Name, or "" when the part has no name.
	Name string
	// MediaType is the media type of the part, or "" for an *ImagePart.
	MediaType string
	// Modality is the input modality of the part: "image", "pdf", "audio",
	// "video", "text", or "" when it is not known.
	Modality string
	// Detail is a human-readable description of the problem.
	Detail string
	// Err is the sentinel error for the problem.
	Err error
}

func (e *InputError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "model %s: message %d part %d", e.Model, e.Message, e.Part)
	if e.Name != "" {
		fmt.Fprintf(&b, " (%q)", e.Name)
	}
	b.WriteString(": ")
	b.WriteString(e.Detail)
	return b.String()
}

func (e *InputError) Unwrap() error {
	return e.Err
}

// CheckMessages reports whether model can accept every part of messages. It
// runs the same input check that GenerateContent and Session run before
// they send a request, and it makes no network call. Use it to reject a bad
// input early.
//
// It returns nil or an error that wraps an *InputError. The check uses the
// model's ModelInfo (see ModelInfoOf). Unknown models, and limits that are
// not known, are treated permissively. The check does not count PDF pages
// and does not look at the attachments of tool results.
//
// For a Model that this package did not create, CheckMessages returns nil.
func CheckMessages(model Model, messages []*Message) error {
	src, ok := model.(modelInfoSource)
	if !ok {
		return nil
	}
	name, info := src.gateInfo()
	return checkMessages(name, info, messages)
}

// ModelInfoOf returns the ModelInfo that model uses to gate requests,
// without a network call. For Anthropic models it includes the Models API
// data when an earlier request got it. The boolean is false when the model
// is not known or when this package did not create the Model.
func ModelInfoOf(model Model) (ModelInfo, bool) {
	src, ok := model.(modelInfoSource)
	if !ok {
		return ModelInfo{}, false
	}
	_, info := src.gateInfo()
	return info, !info.isUnknown()
}

// modelInfoSource is implemented by the provider models. gateInfo returns
// the qualified model name and the ModelInfo that gates requests. It must
// not make a network call.
type modelInfoSource interface {
	gateInfo() (string, ModelInfo)
}

// checkMessages runs the input check on every top-level part of messages.
func checkMessages(model string, info ModelInfo, messages []*Message) error {
	images := 0
	for _, msg := range messages {
		for _, part := range msg.Parts {
			if partModality(part) == "image" {
				images++
			}
		}
	}
	seen := 0
	for i, msg := range messages {
		for j, part := range msg.Parts {
			err := checkPart(info, part, images)
			if err == nil && partModality(part) == "image" {
				seen++
				if limit := info.Limits.Image.MaxCount; limit > 0 && seen > limit {
					err = &InputError{
						Modality: "image",
						Detail:   fmt.Sprintf("request has %d images, limit is %d", images, limit),
						Err:      ErrAttachmentTooLarge,
					}
					if bp, ok := part.(*BinaryPart); ok {
						err.Name, err.MediaType = bp.Name, bp.MediaType
					}
				}
			}
			if err != nil {
				err.Model = model
				err.Message = i
				err.Part = j
				return err
			}
		}
	}
	return nil
}

// checkPart checks one part against the model's modalities and limits.
// images is the number of image parts in the request. The returned error
// has no location; checkMessages adds it.
func checkPart(info ModelInfo, part MessagePart, images int) *InputError {
	modality := partModality(part)
	bp, isBinary := part.(*BinaryPart)
	fail := func(sentinel error, format string, args ...any) *InputError {
		e := &InputError{Modality: modality, Err: sentinel, Detail: fmt.Sprintf(format, args...)}
		if isBinary {
			e.Name = bp.Name
			e.MediaType = bp.MediaType
		}
		return e
	}

	if modality != "" && modality != "text" && !info.allowsModality(modality) {
		return fail(ErrUnsupportedModality, "model does not support %s input", modality)
	}
	if !isBinary {
		return nil
	}

	size := int64(len(bp.Data))
	mediaType := baseMediaType(bp.MediaType)
	switch modality {
	case "text":
		if !utf8.Valid(bp.Data) {
			return fail(ErrInvalidAttachment, "%s content is not valid UTF-8", mediaType)
		}
	case "image":
		lim := info.Limits.Image
		if len(lim.MediaTypes) > 0 && !slices.Contains(lim.MediaTypes, mediaType) {
			return fail(ErrUnsupportedMediaType, "model does not support %s images (supported: %s)",
				mediaType, strings.Join(lim.MediaTypes, ", "))
		}
		if lim.MaxBytes > 0 && size > lim.MaxBytes {
			return fail(ErrAttachmentTooLarge, "image is %d bytes, limit is %d", size, lim.MaxBytes)
		}
		maxW, maxH := lim.maxDimensions(images)
		if maxW > 0 || maxH > 0 {
			if w, h, ok := imageDimensions(mediaType, bp.Data); ok &&
				((maxW > 0 && w > maxW) || (maxH > 0 && h > maxH)) {
				return fail(ErrAttachmentTooLarge, "image is %dx%d pixels, limit is %dx%d", w, h, maxW, maxH)
			}
		}
	case "pdf":
		if lim := info.Limits.PDF; lim.MaxBytes > 0 && size > lim.MaxBytes {
			return fail(ErrAttachmentTooLarge, "PDF is %d bytes, limit is %d", size, lim.MaxBytes)
		}
	}
	return nil
}

// partModality returns the models.dev input modality of a part ("image",
// "audio", "pdf", "video"), "text" for a BinaryPart with a text media type
// (see isTextMediaType), or "" for parts that do not gate on modality (text,
// tool calls, thinking, unknown binary types).
func partModality(part MessagePart) string {
	switch p := part.(type) {
	case *ImagePart:
		return "image"
	case *BinaryPart:
		mediaType := baseMediaType(p.MediaType)
		switch {
		case strings.HasPrefix(mediaType, "image/"):
			return "image"
		case strings.HasPrefix(mediaType, "audio/"):
			return "audio"
		case strings.HasPrefix(mediaType, "video/"):
			return "video"
		case mediaType == "application/pdf":
			return "pdf"
		case isTextMediaType(mediaType):
			return "text"
		}
	}
	return ""
}

// baseMediaType returns the media type without parameters, in lower case.
// "text/plain; charset=utf-8" becomes "text/plain".
func baseMediaType(mediaType string) string {
	if mt, _, err := mime.ParseMediaType(mediaType); err == nil {
		return mt
	}
	mt, _, _ := strings.Cut(mediaType, ";")
	return strings.ToLower(strings.TrimSpace(mt))
}

// isTextMediaType reports whether a base media type holds text that is sent
// to the model as text: text/*, JSON, XML and YAML.
func isTextMediaType(mediaType string) bool {
	if strings.HasPrefix(mediaType, "text/") {
		return true
	}
	switch mediaType {
	case "application/json", "application/xml", "application/yaml", "application/x-yaml":
		return true
	}
	return strings.HasSuffix(mediaType, "+json") ||
		strings.HasSuffix(mediaType, "+xml") ||
		strings.HasSuffix(mediaType, "+yaml")
}

// isTextPart reports whether part is a BinaryPart with a text media type.
func isTextPart(part MessagePart) (*BinaryPart, bool) {
	bp, ok := part.(*BinaryPart)
	if !ok || !isTextMediaType(baseMediaType(bp.MediaType)) {
		return nil, false
	}
	return bp, true
}

// textFileContent returns the text that providers without a native text
// document get for a text BinaryPart. The content is put in a <file> element
// with the name and media type as attributes, so the model can refer to the
// file by name and see where it ends:
//
//	<file name="spec.md" media_type="text/markdown">
//	...
//	</file>
func textFileContent(p *BinaryPart) string {
	var b strings.Builder
	b.Grow(len(p.Data) + len(p.Name) + 64)
	b.WriteString("<file")
	if p.Name != "" {
		b.WriteString(` name="`)
		b.WriteString(html.EscapeString(p.Name))
		b.WriteByte('"')
	}
	b.WriteString(` media_type="`)
	b.WriteString(html.EscapeString(baseMediaType(p.MediaType)))
	b.WriteString("\">\n")
	b.Write(p.Data)
	if len(p.Data) > 0 && p.Data[len(p.Data)-1] != '\n' {
		b.WriteByte('\n')
	}
	b.WriteString("</file>")
	return b.String()
}

// fileLabel returns the text part that comes before a named image or file
// when the provider has no name field for it.
func fileLabel(name string) *TextPart {
	return NewTextPart(fmt.Sprintf("File %q:", name))
}

// syntheticFilename returns a file name for a provider that requires one
// when the part has no Name. The extension comes from the media type, with
// ".bin" for unknown types.
func syntheticFilename(mediaType string) string {
	exts, _ := mime.ExtensionsByType(baseMediaType(mediaType))
	if len(exts) > 0 {
		return "file" + exts[0]
	}
	return "file.bin"
}

// partFilename returns the name of a BinaryPart, or a synthetic name when it
// has none.
func partFilename(p *BinaryPart) string {
	if p.Name != "" {
		return p.Name
	}
	return syntheticFilename(p.MediaType)
}

// imageDimensions returns the pixel width and height from the header of a
// PNG, JPEG, GIF or WebP image. ok is false for other formats and for data
// that cannot be read.
func imageDimensions(mediaType string, data []byte) (width, height int, ok bool) {
	rd := bytes.NewReader(data)
	switch mediaType {
	case "image/png":
		if cfg, err := png.DecodeConfig(rd); err == nil {
			return cfg.Width, cfg.Height, true
		}
	case "image/jpeg", "image/jpg":
		if cfg, err := jpeg.DecodeConfig(rd); err == nil {
			return cfg.Width, cfg.Height, true
		}
	case "image/gif":
		if cfg, err := gif.DecodeConfig(rd); err == nil {
			return cfg.Width, cfg.Height, true
		}
	case "image/webp":
		return webpDimensions(data)
	}
	return 0, 0, false
}

// webpDimensions reads the canvas size from a WebP header. It supports the
// lossy (VP8), lossless (VP8L) and extended (VP8X) formats.
func webpDimensions(data []byte) (width, height int, ok bool) {
	if len(data) < 30 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		return 0, 0, false
	}
	switch string(data[12:16]) {
	case "VP8X":
		// 24-bit little-endian canvas width-1 and height-1.
		w := int(data[24]) | int(data[25])<<8 | int(data[26])<<16
		h := int(data[27]) | int(data[28])<<8 | int(data[29])<<16
		return w + 1, h + 1, true
	case "VP8L":
		// Signature byte 0x2f, then 14-bit width-1 and height-1.
		if data[20] != 0x2f {
			return 0, 0, false
		}
		bits := binary.LittleEndian.Uint32(data[21:25])
		return int(bits&0x3fff) + 1, int((bits>>14)&0x3fff) + 1, true
	case "VP8 ":
		// Key frame start code, then 14-bit width and height.
		if data[23] != 0x9d || data[24] != 0x01 || data[25] != 0x2a {
			return 0, 0, false
		}
		w := int(binary.LittleEndian.Uint16(data[26:28]) & 0x3fff)
		h := int(binary.LittleEndian.Uint16(data[28:30]) & 0x3fff)
		return w, h, true
	}
	return 0, 0, false
}

// expandFileParts rewrites parts for a provider that has no native text
// document. A text BinaryPart becomes a TextPart (see textFileContent). A
// named image gets a fileLabel before it, because no provider has a name
// field for images. When fileNames is false, the provider has no name field
// for other files either, so a named non-image BinaryPart also gets a label.
func expandFileParts(parts []MessagePart, fileNames bool) []MessagePart {
	out := make([]MessagePart, 0, len(parts))
	for _, part := range parts {
		bp, ok := part.(*BinaryPart)
		if !ok {
			out = append(out, part)
			continue
		}
		switch modality := partModality(bp); {
		case modality == "text":
			out = append(out, NewTextPart(textFileContent(bp)))
		case bp.Name != "" && (modality == "image" || !fileNames):
			out = append(out, fileLabel(bp.Name), bp)
		default:
			out = append(out, bp)
		}
	}
	return out
}

// Byte sizes used in the provider limits.
const (
	megabyte = 1_000_000
)

// base64Limit returns the largest raw size whose base64 encoding fits in
// limit bytes. Some providers measure their limits on the encoded string.
func base64Limit(limit int64) int64 {
	return limit / 4 * 3
}

// defaultInputLimits returns the documented attachment limits of the
// provider's own API for a model with the given context window. It returns
// zero limits for providers that do not document them (OpenRouter, where
// the limits depend on the upstream provider).
//
// Where the documentation does not say whether a limit counts raw or base64
// bytes, or where two pages give different values, the lower value is used.
// Sources (checked 2026-11):
//   - Anthropic: platform.claude.com/docs/en/build-with-claude/vision and
//     pdf-support, and the API overview for the request size.
//   - OpenAI: developers.openai.com/api/docs/guides/images-vision and
//     file-inputs.
//   - Google: aistudio.google.com/docs/image-understanding and
//     document-processing.
func defaultInputLimits(provider string, context int) (ImageLimits, PDFLimits) {
	switch provider {
	case "anthropic":
		// Models with a context window under 1M tokens take fewer images
		// and PDF pages.
		maxImages, maxPages := 600, 600
		if context > 0 && context < 1_000_000 {
			maxImages, maxPages = 100, 100
		}
		return ImageLimits{
			MediaTypes: []string{"image/jpeg", "image/png", "image/gif", "image/webp"},
			// 10 MB, measured on the base64 string.
			MaxBytes:  base64Limit(10 * megabyte),
			MaxWidth:  8000,
			MaxHeight: 8000,
			MaxCount:  maxImages,
			// Requests with more than 20 images have a lower pixel
			// limit. The documentation does not give it; 2000 px is
			// the value it gave before.
			ManyImages:    20,
			ManyMaxWidth:  2000,
			ManyMaxHeight: 2000,
		}, PDFLimits{
			// No PDF byte limit is documented, only the 128 MB request
			// limit, which counts the base64 string.
			MaxBytes: base64Limit(128 * megabyte),
			MaxPages: maxPages,
		}
	case "openai":
		// No per-image byte or pixel limit: large images are scaled down.
		return ImageLimits{
			MediaTypes: []string{"image/png", "image/jpeg", "image/webp", "image/gif"},
			MaxCount:   1500,
		}, PDFLimits{
			// Each file must be under 50 MB. No page limit is documented.
			MaxBytes: 50 * megabyte,
		}
	case "google":
		// No per-image byte or pixel limit: large images are tiled. The
		// inline request limit is 20 MB on one page and 100 MB on another,
		// so it is not set here.
		return ImageLimits{
			MediaTypes: []string{
				"image/png", "image/jpeg", "image/webp", "image/heic",
				"image/heif", "image/gif", "image/avif",
			},
			MaxCount: 3600,
		}, PDFLimits{
			MaxBytes: 50 * megabyte,
			MaxPages: 1000,
		}
	}
	return ImageLimits{}, PDFLimits{}
}
