package llms

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicModelJSON builds a Models API response. adaptive and enabled set
// the thinking types; efforts lists the supported effort tiers.
func anthropicModelJSON(id string, adaptive, enabled bool, efforts ...string) string {
	effort := map[string]any{"supported": len(efforts) > 0}
	for _, e := range []string{"low", "medium", "high", "xhigh", "max"} {
		effort[e] = map[string]bool{"supported": slices.Contains(efforts, e)}
	}
	doc := map[string]any{
		"id": id, "type": "model", "display_name": id, "created_at": "2026-09-22T00:00:00Z",
		"max_input_tokens": 1000000, "max_tokens": 128000,
		"capabilities": map[string]any{
			"batch":              map[string]bool{"supported": true},
			"citations":          map[string]bool{"supported": true},
			"code_execution":     map[string]bool{"supported": true},
			"context_management": map[string]any{"supported": true},
			"effort":             effort,
			"image_input":        map[string]bool{"supported": true},
			"pdf_input":          map[string]bool{"supported": true},
			"structured_outputs": map[string]bool{"supported": true},
			"thinking": map[string]any{"supported": adaptive || enabled, "types": map[string]any{
				"adaptive": map[string]bool{"supported": adaptive},
				"enabled":  map[string]bool{"supported": enabled},
			}},
		},
	}
	b, _ := json.Marshal(doc)
	return string(b)
}

func decodeAnthropicModel(t *testing.T, doc string) *anthropic.ModelInfo {
	t.Helper()
	var info anthropic.ModelInfo
	if err := json.Unmarshal([]byte(doc), &info); err != nil {
		t.Fatalf("decoding model info: %v", err)
	}
	return &info
}

func TestApplyAnthropicCapabilitiesToUnknownModel(t *testing.T) {
	api := decodeAnthropicModel(t, anthropicModelJSON("claude-opus-5-5", true, false, "low", "medium", "high", "xhigh", "max"))
	info := applyAnthropicCapabilities(ModelInfo{}, "claude-opus-5-5", api)

	if !info.Caps.AdaptiveThinking || info.Caps.ReasoningBudget != nil {
		t.Errorf("expected adaptive thinking without a budget, got %+v", info.Caps)
	}
	if want := []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}; !slices.Equal(info.Caps.ReasoningEfforts, want) {
		t.Errorf("efforts = %v, want %v (xhigh read from raw JSON)", info.Caps.ReasoningEfforts, want)
	}
	if !info.Caps.ReasoningMandatory {
		t.Error("claude-opus-5-5 cannot turn thinking off")
	}
	if !info.Caps.ToolCall || info.Caps.Temperature {
		t.Errorf("unknown adaptive-only model: expected tools and no temperature, got %+v", info.Caps)
	}
	if !slices.Equal(info.Modalities, []string{"text", "image", "pdf"}) || info.Limits.Output != 128000 {
		t.Errorf("unexpected modalities %v or limits %+v", info.Modalities, info.Limits)
	}
}

func TestApplyAnthropicCapabilitiesSwitchesKnownModelToAdaptive(t *testing.T) {
	known := ModelInfo{
		Cost:   Cost{Input: 3, Output: 15},
		Family: "claude-sonnet",
		Caps: Capabilities{
			Temperature:      true,
			Reasoning:        true,
			ToolCall:         true,
			ReasoningBudget:  &TokenRange{Min: 1024},
			ReasoningEfforts: []Effort{EffortLow, EffortMedium, EffortHigh, EffortMax},
		},
	}
	api := decodeAnthropicModel(t, anthropicModelJSON("claude-sonnet-4-6", true, true, "low", "medium", "high", "max"))
	info := applyAnthropicCapabilities(known, "claude-sonnet-4-6", api)

	if !info.Caps.AdaptiveThinking || info.Caps.ReasoningBudget == nil {
		t.Errorf("expected adaptive thinking and a budget, got %+v", info.Caps)
	}
	if !info.Caps.Temperature || info.Cost.Input != 3 {
		t.Errorf("known data outside the API answer must be kept, got %+v", info)
	}
}

func TestApplyAnthropicCapabilitiesBudgetOnlyModel(t *testing.T) {
	api := decodeAnthropicModel(t, anthropicModelJSON("claude-haiku-4-5", false, true))
	info := applyAnthropicCapabilities(ModelInfo{}, "claude-haiku-4-5", api)

	if info.Caps.AdaptiveThinking || len(info.Caps.ReasoningEfforts) != 0 {
		t.Errorf("expected no adaptive thinking and no efforts, got %+v", info.Caps)
	}
	if info.Caps.ReasoningBudget == nil || info.Caps.ReasoningBudget.Min != anthropicMinThinkingBudget {
		t.Errorf("expected a budget from 1024, got %+v", info.Caps.ReasoningBudget)
	}
	if !info.Caps.Temperature {
		t.Error("budget-thinking models accept temperature")
	}
}

const anthropicTestSSE = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
	"event: content_block_start\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n" +
	"event: content_block_stop\n" +
	`data: {"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}` + "\n\n" +
	"event: message_stop\n" +
	`data: {"type":"message_stop"}` + "\n\n"

// anthropicCapabilityServer serves the Models API with modelsHandler and
// answers message requests with a short stream, recording the last body.
func anthropicCapabilityServer(t *testing.T, modelsHandler http.HandlerFunc) (*httptest.Server, *atomic.Int32, *atomic.Pointer[map[string]any]) {
	t.Helper()
	var lookups atomic.Int32
	var lastBody atomic.Pointer[map[string]any]
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/models/") {
			lookups.Add(1)
			modelsHandler(w, r)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		lastBody.Store(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, anthropicTestSSE)
	}))
	t.Cleanup(srv.Close)
	return srv, &lookups, &lastBody
}

func TestAnthropicModelUsesModelsAPICapabilities(t *testing.T) {
	srv, lookups, lastBody := anthropicCapabilityServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, anthropicModelJSON("claude-new", false, true))
	})

	// The model is unknown to the model data, so without the Models API it
	// would be treated as adaptive. The API says it only takes a budget.
	m := newAnthropicModel(discardLogger(), NewNoopMetrics(), StaticCredentials("key"), "claude-new", nil,
		fixedModelInfo(ModelInfo{}), option.WithBaseURL(srv.URL))

	for range 2 {
		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithReasoningEffort(EffortHigh))
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	if got := lookups.Load(); got != 1 {
		t.Errorf("expected one Models API lookup for two requests, got %d", got)
	}
	thinking, _ := (*lastBody.Load())["thinking"].(map[string]any)
	if thinking["type"] != "enabled" {
		t.Errorf("expected budget thinking from the Models API answer, got %v", thinking)
	}
	if _, ok := (*lastBody.Load())["output_config"]; ok {
		t.Error("a model without effort support must not get an output_config effort")
	}
}

func TestAnthropicModelContinuesWhenModelsAPIFails(t *testing.T) {
	srv, lookups, lastBody := anthropicCapabilityServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})

	m := newAnthropicModel(discardLogger(), NewNoopMetrics(), StaticCredentials("key"), "claude-new", nil,
		fixedModelInfo(ModelInfo{}), option.WithBaseURL(srv.URL))

	for range 2 {
		_, err := m.GenerateContent(context.Background(),
			WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
			WithReasoningEffort(EffortHigh))
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	if got := lookups.Load(); got != 1 {
		t.Errorf("a failed lookup should wait before retrying, got %d lookups", got)
	}
	thinking, _ := (*lastBody.Load())["thinking"].(map[string]any)
	if thinking["type"] != "adaptive" {
		t.Errorf("expected the unknown-model default (adaptive), got %v", thinking)
	}
}

func TestAnthropicModelKeepsRegisteredModelInfo(t *testing.T) {
	isolateModelInfos(t)
	RegisterModelInfo("anthropic/claude-registered", ModelInfo{
		Caps: Capabilities{Reasoning: true, ToolCall: true, AdaptiveThinking: true, ReasoningEfforts: []Effort{EffortLow, EffortHigh}},
	})

	srv, lookups, lastBody := anthropicCapabilityServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, anthropicModelJSON("claude-registered", false, true))
	})
	m := newAnthropicModel(discardLogger(), NewNoopMetrics(), StaticCredentials("key"), "claude-registered", nil,
		registeredModelInfo("anthropic/claude-registered"), option.WithBaseURL(srv.URL))

	_, err := m.GenerateContent(context.Background(),
		WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
		WithReasoningEffort(EffortHigh))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}

	if got := lookups.Load(); got != 0 {
		t.Errorf("a registered model must not be looked up, got %d lookups", got)
	}
	thinking, _ := (*lastBody.Load())["thinking"].(map[string]any)
	if thinking["type"] != "adaptive" {
		t.Errorf("expected the registered adaptive thinking, got %v", thinking)
	}
}

func TestAnthropicLookupIgnoresCallerCancellation(t *testing.T) {
	srv, lookups, _ := anthropicCapabilityServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, anthropicModelJSON("claude-new", true, false, "low", "high"))
	})
	m := newAnthropicModel(discardLogger(), NewNoopMetrics(), StaticCredentials("key"), "claude-new", nil,
		fixedModelInfo(ModelInfo{}), option.WithBaseURL(srv.URL)).(*anthropicModel)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	info := m.modelInfo(ctx)

	if got := lookups.Load(); got != 1 || !info.Caps.AdaptiveThinking {
		t.Errorf("expected a completed lookup despite the cancelled caller, got %d lookups and %+v", got, info.Caps)
	}
}
