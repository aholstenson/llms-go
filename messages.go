package llms

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
