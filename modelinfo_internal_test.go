package llms

import (
	"context"
	"io"
	"log/slog"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/metric/noop"
)

// knownInfo returns a non-zero ModelInfo with the given capabilities so the
// gate helpers treat it as a known model rather than permissively.
func knownInfo(caps Capabilities, modalities ...string) ModelInfo {
	return ModelInfo{
		Cost:       Cost{Input: 1, Output: 1},
		Caps:       caps,
		Modalities: modalities,
		Family:     "test",
	}
}

func TestModelInfoIsUnknown(t *testing.T) {
	if !(ModelInfo{}).isUnknown() {
		t.Error("zero ModelInfo should be unknown")
	}
	if knownInfo(Capabilities{}).isUnknown() {
		t.Error("ModelInfo with metadata should not be unknown")
	}
}

func TestModelInfoGatesPermissiveWhenUnknown(t *testing.T) {
	var zero ModelInfo
	if !zero.allowsTemperature() || !zero.allowsReasoning() || !zero.allowsToolCall() {
		t.Error("unknown model should permit temperature/reasoning/tool-call")
	}
	if !zero.allowsModality("image") {
		t.Error("unknown model should permit any modality")
	}
	if v, clamped := zero.clampMaxTokens(1_000_000); clamped || v != 1_000_000 {
		t.Errorf("unknown model should not clamp max tokens, got %d clamped=%v", v, clamped)
	}
}

func TestModelInfoGatesRespectKnownCaps(t *testing.T) {
	info := knownInfo(Capabilities{Temperature: false, Reasoning: false, ToolCall: false}, "text")

	if info.allowsTemperature() {
		t.Error("Caps.Temperature=false should disallow temperature")
	}
	if info.allowsReasoning() {
		t.Error("Caps.Reasoning=false should disallow reasoning")
	}
	if info.allowsToolCall() {
		t.Error("Caps.ToolCall=false should disallow tool calling")
	}
	if info.allowsModality("image") {
		t.Error("modalities=[text] should disallow image input")
	}
	if !info.allowsModality("text") {
		t.Error("modalities=[text] should allow text input")
	}

	// A known model with no declared modalities is permissive.
	noMods := knownInfo(Capabilities{})
	noMods.Modalities = nil
	if !noMods.allowsModality("image") {
		t.Error("known model with no modalities should permit any modality")
	}
}

func TestModelInfoClampMaxTokens(t *testing.T) {
	info := knownInfo(Capabilities{})
	info.Limits = Limits{Output: 4096}

	if v, clamped := info.clampMaxTokens(8000); !clamped || v != 4096 {
		t.Errorf("expected clamp to 4096, got %d clamped=%v", v, clamped)
	}
	if v, clamped := info.clampMaxTokens(1000); clamped || v != 1000 {
		t.Errorf("expected no clamp, got %d clamped=%v", v, clamped)
	}
	if v, clamped := info.clampMaxTokens(0); clamped || v != 0 {
		t.Errorf("expected pass-through of 0, got %d clamped=%v", v, clamped)
	}
}

func TestMessagesContainImages(t *testing.T) {
	textOnly := []*Message{NewMessage(RoleUser, NewTextPart("hi"))}
	if messagesContainImages(textOnly) {
		t.Error("text-only messages should not be detected as containing images")
	}

	withImage := []*Message{NewMessage(RoleUser, NewTextPart("look"), NewImagePart("http://x/y.png"))}
	if !messagesContainImages(withImage) {
		t.Error("messages with an ImagePart should be detected")
	}

	withBinary := []*Message{NewMessage(RoleUser, NewBinaryPart("image/png", []byte{1}))}
	if !messagesContainImages(withBinary) {
		t.Error("messages with a BinaryPart should be detected")
	}
}

// echoTool is a minimal ToolDef for exercising the tool-call gate.
type echoTool struct{}

type echoInput struct {
	Text string `json:"text"`
}

func (echoTool) Name() string                                             { return "echo" }
func (echoTool) Description() string                                      { return "echoes input" }
func (echoTool) Schema() *echoInput                                       { return &echoInput{} }
func (echoTool) Execute(_ context.Context, in *echoInput) (string, error) { return in.Text, nil }
func (echoTool) ToString(s string) string                                 { return s }

func testMetrics(t *testing.T) *Metrics {
	t.Helper()
	m, err := NewMetrics(noop.NewMeterProvider().Meter("test"))
	if err != nil {
		t.Fatalf("NewMetrics: %v", err)
	}
	return m
}

// TestToolCallGateBlocksBeforeAPI verifies a known model that does not support
// tool calling fails fast, before any network call.
func TestToolCallGateBlocksBeforeAPI(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := NewOpenAIModel(logger, testMetrics(t), "test-key", "no-tools-model", nil,
		knownInfo(Capabilities{ToolCall: false}, "text"))

	_, err := m.GenerateContent(context.Background(),
		WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
		WithTools(NewToolDef[*echoInput, string](echoTool{})),
	)
	if err == nil || !strings.Contains(err.Error(), "does not support tool calling") {
		t.Fatalf("expected tool-call gate error, got %v", err)
	}
}

// TestImageModalityGateBlocksBeforeAPI verifies a known text-only model
// rejects image input before any network call.
func TestImageModalityGateBlocksBeforeAPI(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := NewOpenAIModel(logger, testMetrics(t), "test-key", "text-only-model", nil,
		knownInfo(Capabilities{ToolCall: true}, "text"))

	_, err := m.GenerateContent(context.Background(),
		WithMessages(NewMessage(RoleUser, NewImagePart("http://x/y.png"))),
	)
	if err == nil || !strings.Contains(err.Error(), "does not support image input") {
		t.Fatalf("expected image-modality gate error, got %v", err)
	}
}
