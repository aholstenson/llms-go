package modelsdev_test

import (
	"encoding/json"
	"os"
	"testing"

	llms "github.com/aholstenson/llms-go"
	"github.com/aholstenson/llms-go/internal/modelsdev"
)

func loadSample(t *testing.T) map[string]llms.ModelInfo {
	t.Helper()

	data, err := os.ReadFile("testdata/modelsdev_sample.json")
	if err != nil {
		t.Fatalf("reading fixture: %v", err)
	}

	var raw modelsdev.RawData
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("parsing fixture: %v", err)
	}

	return modelsdev.Transform(raw)
}

func TestTransformKeysAndProviderFilter(t *testing.T) {
	out := loadSample(t)

	want := []string{
		"anthropic/claude-haiku-4-5",
		"openai/gpt-5",
		"openrouter/google/gemini-2.5-flash-lite",
	}
	for _, k := range want {
		if _, ok := out[k]; !ok {
			t.Errorf("expected key %q in transformed output", k)
		}
	}

	// deepseek is not in the allowlist and must be dropped entirely.
	if len(out) != len(want) {
		t.Errorf("expected %d models, got %d: %v", len(want), len(out), keys(out))
	}
	for k := range out {
		if k == "deepseek/deepseek-chat" {
			t.Errorf("non-allowlisted provider leaked into output: %q", k)
		}
	}
}

func TestTransformCapabilitiesAndCost(t *testing.T) {
	out := loadSample(t)

	// gpt-5 style: temperature unsupported, reasoning supported.
	gpt5 := out["openai/gpt-5"]
	if gpt5.Caps.Temperature {
		t.Error("gpt-5: expected Caps.Temperature=false")
	}
	if !gpt5.Caps.Reasoning {
		t.Error("gpt-5: expected Caps.Reasoning=true")
	}
	if gpt5.Cost.CacheWrite != 0 {
		t.Errorf("gpt-5: expected zero CacheWrite, got %v", gpt5.Cost.CacheWrite)
	}

	// anthropic model with full cache costs.
	haiku := out["anthropic/claude-haiku-4-5"]
	if haiku.Cost.Input != 1 || haiku.Cost.Output != 5 {
		t.Errorf("haiku: unexpected base cost %+v", haiku.Cost)
	}
	if haiku.Cost.CacheRead != 0.1 || haiku.Cost.CacheWrite != 1.25 {
		t.Errorf("haiku: unexpected cache cost %+v", haiku.Cost)
	}
	if haiku.Limits.Context != 200000 || haiku.Limits.Output != 64000 {
		t.Errorf("haiku: unexpected limits %+v", haiku.Limits)
	}
	if haiku.Family != "claude-haiku" || haiku.Knowledge != "2025-02" {
		t.Errorf("haiku: unexpected metadata family=%q knowledge=%q", haiku.Family, haiku.Knowledge)
	}

	// model missing cache fields: cache costs default to zero.
	or := out["openrouter/google/gemini-2.5-flash-lite"]
	if or.Cost.CacheRead != 0 || or.Cost.CacheWrite != 0 {
		t.Errorf("openrouter gemini: expected zero cache cost, got %+v", or.Cost)
	}
	if len(or.Modalities) != 3 || or.Modalities[1] != "image" {
		t.Errorf("openrouter gemini: unexpected modalities %v", or.Modalities)
	}
}

func TestTransformOmitsEmptyCacheInJSON(t *testing.T) {
	out := loadSample(t)

	encoded, err := json.Marshal(out["openrouter/google/gemini-2.5-flash-lite"])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var generic map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &generic); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	cost := generic["c"]
	var costMap map[string]json.RawMessage
	if err := json.Unmarshal(cost, &costMap); err != nil {
		t.Fatalf("unmarshal cost: %v", err)
	}
	if _, ok := costMap["r"]; ok {
		t.Error("expected cache_read (c.r) to be omitted when zero")
	}
	if _, ok := costMap["w"]; ok {
		t.Error("expected cache_write (c.w) to be omitted when zero")
	}
}

func keys(m map[string]llms.ModelInfo) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
