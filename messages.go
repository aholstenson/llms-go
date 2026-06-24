package llms

import (
	"encoding/json"
	"fmt"
)

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type Message struct {
	Role  Role
	Parts []MessagePart
	Cache bool
}

func NewMessage(role Role, parts ...MessagePart) *Message {
	return &Message{Role: role, Parts: parts}
}

func (m *Message) WithCache(cache bool) *Message {
	m.Cache = cache
	return m
}

type MessagePart interface {
	isPart()
}

type TextPart struct {
	Text string
}

func NewTextPart(text string) *TextPart {
	return &TextPart{Text: text}
}

func (TextPart) isPart() {}

type ImagePart struct {
	URL string
}

func NewImagePart(url string) *ImagePart {
	return &ImagePart{URL: url}
}

func (ImagePart) isPart() {}

type BinaryPart struct {
	MediaType string
	Data      []byte
}

func NewBinaryPart(mediaType string, data []byte) *BinaryPart {
	return &BinaryPart{MediaType: mediaType, Data: data}
}

func (BinaryPart) isPart() {}

// ToolCallPart is a tool invocation requested by the model. It appears in an
// assistant message and is paired with a ToolResultPart by ID. Arguments is
// the raw JSON arguments string as produced by the provider.
type ToolCallPart struct {
	ID        string
	Name      string
	Arguments string
}

func NewToolCallPart(id, name, arguments string) *ToolCallPart {
	return &ToolCallPart{ID: id, Name: name, Arguments: arguments}
}

func (ToolCallPart) isPart() {}

// ToolResultPart is the result of a tool invocation. It appears in a user
// message and is paired with its ToolCallPart by ID. Exactly one of Text or
// Error is meaningful: Error is non-empty when the call failed, otherwise
// Text holds the rendered tool result.
type ToolResultPart struct {
	ID    string
	Name  string
	Text  string
	Error string
	// Attachments are optional rich-content parts (*ImagePart / *BinaryPart)
	// produced by a tool alongside its textual result. They are delivered to
	// the model either natively (Anthropic) or via a trailing synthetic user
	// message (OpenAI / Google / OpenRouter).
	Attachments []MessagePart
}

func NewToolResultPart(id, name, text, errText string) *ToolResultPart {
	return &ToolResultPart{ID: id, Name: name, Text: text, Error: errText}
}

func (ToolResultPart) isPart() {}

// ThinkingPart carries a model extended-thinking/reasoning block in neutral
// form. Signature is the provider-opaque cryptographic signature (Anthropic)
// that must be replayed verbatim alongside tool use; Redacted/Data cover an
// encrypted redacted_thinking block. It appears in an assistant message,
// before any text or tool-call parts.
type ThinkingPart struct {
	Text      string
	Signature string
	Redacted  bool
	Data      string
}

func NewThinkingPart(text, signature string) *ThinkingPart {
	return &ThinkingPart{Text: text, Signature: signature}
}

func (ThinkingPart) isPart() {}

// --- JSON codec ------------------------------------------------------------
//
// MessagePart is a closed interface, so a Message cannot round-trip through
// encoding/json without a discriminator. MarshalJSON tags each part with a
// "type" field; UnmarshalJSON dispatches on it. This is the only supported
// serialization format for the neutral transcript (e.g. for durable jobs).

type messageJSON struct {
	Role  Role              `json:"role"`
	Parts []json.RawMessage `json:"parts"`
	Cache bool              `json:"cache,omitempty"`
}

type partEnvelope struct {
	Type string `json:"type"`
}

func partTypeName(p MessagePart) (string, error) {
	switch p.(type) {
	case *TextPart:
		return "text", nil
	case *ImagePart:
		return "image", nil
	case *BinaryPart:
		return "binary", nil
	case *ToolCallPart:
		return "tool_call", nil
	case *ToolResultPart:
		return "tool_result", nil
	case *ThinkingPart:
		return "thinking", nil
	default:
		return "", fmt.Errorf("cannot marshal unknown message part type %T", p)
	}
}

// marshalParts encodes a slice of MessageParts into discriminated raw JSON
// objects, splicing a "type" field into each part. It is shared by the
// Message codec and by ToolResultPart's attachment encoding so nested parts
// round-trip identically to top-level ones.
func marshalParts(parts []MessagePart) ([]json.RawMessage, error) {
	out := make([]json.RawMessage, 0, len(parts))
	for _, p := range parts {
		typeName, err := partTypeName(p)
		if err != nil {
			return nil, err
		}
		body, err := json.Marshal(p)
		if err != nil {
			return nil, err
		}
		// Splice the discriminator into the part object.
		merged := append([]byte(`{"type":`), mustJSON(typeName)...)
		if len(body) > 2 {
			merged = append(merged, ',')
			merged = append(merged, body[1:]...)
		} else {
			merged = append(merged, '}')
		}
		out = append(out, merged)
	}
	return out, nil
}

// unmarshalParts decodes discriminated raw JSON objects back into MessageParts,
// dispatching on the "type" field. Shared by the Message codec and
// ToolResultPart's attachment decoding.
func unmarshalParts(raw []json.RawMessage) ([]MessagePart, error) {
	parts := make([]MessagePart, 0, len(raw))
	for _, rp := range raw {
		var env partEnvelope
		if err := json.Unmarshal(rp, &env); err != nil {
			return nil, err
		}
		var part MessagePart
		switch env.Type {
		case "text":
			part = new(TextPart)
		case "image":
			part = new(ImagePart)
		case "binary":
			part = new(BinaryPart)
		case "tool_call":
			part = new(ToolCallPart)
		case "tool_result":
			part = new(ToolResultPart)
		case "thinking":
			part = new(ThinkingPart)
		default:
			return nil, fmt.Errorf("cannot unmarshal unknown message part type %q", env.Type)
		}
		if err := json.Unmarshal(rp, part); err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}
	return parts, nil
}

// marshalAttachments encodes tool-result attachments, rejecting any part that
// is not an image or binary so the rich-result boundary stays narrow.
func marshalAttachments(parts []MessagePart) ([]json.RawMessage, error) {
	for _, p := range parts {
		switch p.(type) {
		case *ImagePart, *BinaryPart:
		default:
			return nil, fmt.Errorf("tool-result attachment must be *ImagePart or *BinaryPart, got %T", p)
		}
	}
	return marshalParts(parts)
}

// unmarshalAttachments decodes tool-result attachments and enforces the same
// image/binary restriction as marshalAttachments.
func unmarshalAttachments(raw []json.RawMessage) ([]MessagePart, error) {
	parts, err := unmarshalParts(raw)
	if err != nil {
		return nil, err
	}
	for _, p := range parts {
		switch p.(type) {
		case *ImagePart, *BinaryPart:
		default:
			return nil, fmt.Errorf("tool-result attachment must be image or binary, got %T", p)
		}
	}
	return parts, nil
}

func (m *Message) MarshalJSON() ([]byte, error) {
	parts, err := marshalParts(m.Parts)
	if err != nil {
		return nil, err
	}
	return json.Marshal(messageJSON{Role: m.Role, Parts: parts, Cache: m.Cache})
}

func mustJSON(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var raw messageJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role = raw.Role
	m.Cache = raw.Cache
	parts, err := unmarshalParts(raw.Parts)
	if err != nil {
		return err
	}
	m.Parts = parts
	return nil
}

// toolResultPartJSON is the on-disk shape of a ToolResultPart. The scalar
// fields keep their default (capitalized) names for backward compatibility;
// attachments are encoded as discriminated parts via the shared helpers so
// they survive the snapshot round-trip even though they nest under a part.
type toolResultPartJSON struct {
	ID          string            `json:"ID"`
	Name        string            `json:"Name"`
	Text        string            `json:"Text"`
	Error       string            `json:"Error"`
	Attachments []json.RawMessage `json:"Attachments,omitempty"`
}

func (p *ToolResultPart) MarshalJSON() ([]byte, error) {
	out := toolResultPartJSON{ID: p.ID, Name: p.Name, Text: p.Text, Error: p.Error}
	if len(p.Attachments) > 0 {
		atts, err := marshalAttachments(p.Attachments)
		if err != nil {
			return nil, err
		}
		out.Attachments = atts
	}
	return json.Marshal(out)
}

func (p *ToolResultPart) UnmarshalJSON(data []byte) error {
	var raw toolResultPartJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	p.ID = raw.ID
	p.Name = raw.Name
	p.Text = raw.Text
	p.Error = raw.Error
	if len(raw.Attachments) > 0 {
		atts, err := unmarshalAttachments(raw.Attachments)
		if err != nil {
			return err
		}
		p.Attachments = atts
	}
	return nil
}
