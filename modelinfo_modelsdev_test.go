package llms

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/aholstenson/llms-go/internal/modelsdev"
)

// sampleModelsDev has reasoning_options and costs in the shapes models.dev
// publishes for these models.
const sampleModelsDev = `{
  "anthropic": {"id": "anthropic", "models": {
    "claude-opus-5-5": {"reasoning": true, "temperature": false,
      "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}]},
    "claude-opus-5": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}]},
    "claude-fable-5-1": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}]},
    "claude-opus-4-7": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}]},
    "claude-sonnet-4-6": {"reasoning": true, "temperature": true,
      "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high", "max"]}, {"type": "budget_tokens", "min": 1024}]},
    "claude-haiku-4-5": {"reasoning": true,
      "reasoning_options": [{"type": "budget_tokens", "min": 1024}]}
  }},
  "openai": {"id": "openai", "models": {
    "gpt-5.6-sol": {"reasoning": true, "structured_output": true,
      "reasoning_options": [{"type": "effort", "values": ["none", "low", "medium", "high", "xhigh", "max"]}],
      "cost": {"input": 4, "output": 20, "tiers": [{"input": 8, "output": 30, "tier": {"type": "context", "size": 272000}}]}},
    "gpt-5": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["minimal", "low", "medium", "high"]}]},
    "o3": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high"]}]},
    "gpt-4.1": {"reasoning": false, "temperature": true, "structured_output": true, "status": "deprecated"},
    "gpt-image-x": {"reasoning": false, "structured_output": false}
  }},
  "google": {"id": "google", "models": {
    "gemini-3.8-flash": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high"]}]},
    "gemini-3-flash": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["minimal", "low", "medium", "high"]}]},
    "gemini-2.5-pro": {"reasoning": true,
      "reasoning_options": [{"type": "budget_tokens", "min": 128, "max": 32768}],
      "cost": {"input": 1.25, "output": 10, "tiers": [{"input": 2.5, "output": 15, "tier": {"type": "context", "size": 200000}}]}},
    "gemini-2.5-flash": {"reasoning": true,
      "reasoning_options": [{"type": "toggle"}, {"type": "budget_tokens", "min": 0, "max": 24576}]}
  }},
  "openrouter": {"id": "openrouter", "models": {
    "anthropic/claude-opus-4.7": {"reasoning": true,
      "reasoning_options": [{"type": "toggle"}, {"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}]},
    "anthropic/claude-opus-5.5": {"reasoning": true,
      "reasoning_options": [{"type": "toggle"}, {"type": "effort", "values": ["low", "medium", "high", "xhigh", "max"]}]},
    "openai/o3": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["low", "medium", "high"]}]},
    "qwen/qwen3-max": {"reasoning": true,
      "reasoning_options": [{"type": "effort", "values": ["low", "high"]}]}
  }}
}`

func sampleModelInfo(t *testing.T) map[string]ModelInfo {
	t.Helper()
	var raw modelsdev.RawData
	if err := json.Unmarshal([]byte(sampleModelsDev), &raw); err != nil {
		t.Fatalf("parsing sample: %v", err)
	}
	return modelInfoFromModelsDev(raw)
}

func TestModelsDevReasoningControls(t *testing.T) {
	infos := sampleModelInfo(t)
	lowToMax := []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}

	cases := []struct {
		key       string
		efforts   []Effort
		budget    *TokenRange
		adaptive  bool
		mandatory bool
	}{
		// Anthropic: effort without budget means adaptive thinking. Only the
		// models in anthropicAlwaysReasons reject turning thinking off.
		{"anthropic/claude-opus-5-5", lowToMax, nil, true, true},
		{"anthropic/claude-opus-5", lowToMax, nil, true, false},
		{"anthropic/claude-fable-5-1", lowToMax, nil, true, true},
		{"anthropic/claude-opus-4-7", lowToMax, nil, true, false},
		// Models that accept both keep budget thinking until the Models API
		// says otherwise.
		{"anthropic/claude-sonnet-4-6", []Effort{EffortLow, EffortMedium, EffortHigh, EffortMax}, &TokenRange{Min: 1024}, false, false},
		// Haiku 4.5 takes only a budget: no effort may be sent.
		{"anthropic/claude-haiku-4-5", nil, &TokenRange{Min: 1024}, false, false},
		// OpenAI: a "none" tier means reasoning can be turned off.
		{"openai/gpt-5.6-sol", lowToMax, nil, false, false},
		{"openai/gpt-5", []Effort{EffortMinimal, EffortLow, EffortMedium, EffortHigh}, nil, false, true},
		{"openai/o3", []Effort{EffortLow, EffortMedium, EffortHigh}, nil, false, true},
		// Gemini: the minimal level or a zero budget turns thinking off.
		{"google/gemini-3.8-flash", []Effort{EffortLow, EffortMedium, EffortHigh}, nil, false, true},
		{"google/gemini-3-flash", []Effort{EffortMinimal, EffortLow, EffortMedium, EffortHigh}, nil, false, false},
		{"google/gemini-2.5-pro", nil, &TokenRange{Min: 128, Max: 32768}, false, true},
		{"google/gemini-2.5-flash", nil, &TokenRange{Min: 0, Max: 24576}, false, false},
		// OpenRouter follows the rules of the model's vendor.
		{"openrouter/anthropic/claude-opus-4.7", lowToMax, nil, false, false},
		{"openrouter/anthropic/claude-opus-5.5", lowToMax, nil, false, true},
		{"openrouter/openai/o3", []Effort{EffortLow, EffortMedium, EffortHigh}, nil, false, true},
		{"openrouter/qwen/qwen3-max", []Effort{EffortLow, EffortHigh}, nil, false, false},
	}
	for _, tc := range cases {
		c := infos[tc.key].Caps
		if !slices.Equal(c.ReasoningEfforts, tc.efforts) {
			t.Errorf("%s: ReasoningEfforts = %v, want %v", tc.key, c.ReasoningEfforts, tc.efforts)
		}
		switch {
		case (c.ReasoningBudget == nil) != (tc.budget == nil):
			t.Errorf("%s: ReasoningBudget = %v, want %v", tc.key, c.ReasoningBudget, tc.budget)
		case tc.budget != nil && *c.ReasoningBudget != *tc.budget:
			t.Errorf("%s: ReasoningBudget = %+v, want %+v", tc.key, *c.ReasoningBudget, *tc.budget)
		}
		if c.AdaptiveThinking != tc.adaptive {
			t.Errorf("%s: AdaptiveThinking = %v, want %v", tc.key, c.AdaptiveThinking, tc.adaptive)
		}
		if c.ReasoningMandatory != tc.mandatory {
			t.Errorf("%s: ReasoningMandatory = %v, want %v", tc.key, c.ReasoningMandatory, tc.mandatory)
		}
	}

	if gpt41 := infos["openai/gpt-4.1"].Caps; gpt41.Reasoning || len(gpt41.ReasoningEfforts) > 0 || gpt41.ReasoningMandatory {
		t.Errorf("gpt-4.1: expected no reasoning metadata, got %+v", gpt41)
	}
}

func TestModelsDevStructuredOutputAndStatus(t *testing.T) {
	infos := sampleModelInfo(t)

	if !infos["openai/gpt-4.1"].Caps.StructuredOutput {
		t.Error("gpt-4.1: expected structured output")
	}
	if infos["openai/gpt-image-x"].Caps.StructuredOutput {
		t.Error("gpt-image-x: expected no structured output when models.dev says false")
	}
	if !infos["openai/o3"].Caps.StructuredOutput {
		t.Error("o3: expected structured output when models.dev does not say")
	}
	if infos["openai/gpt-4.1"].Status != ModelStatusDeprecated {
		t.Errorf("gpt-4.1: Status = %q, want deprecated", infos["openai/gpt-4.1"].Status)
	}
}

func TestModelsDevCostTiers(t *testing.T) {
	infos := sampleModelInfo(t)

	cost := infos["openai/gpt-5.6-sol"].Cost
	want := []CostTier{{ContextOver: 272000, Input: 8, Output: 30}}
	if !slices.Equal(cost.Tiers, want) {
		t.Errorf("gpt-5.6-sol tiers = %+v, want %+v", cost.Tiers, want)
	}
	if tier := cost.tierFor(272000); tier != nil {
		t.Errorf("a prompt of exactly the tier size should use the base price, got %+v", tier)
	}
	if tier := cost.tierFor(272001); tier == nil || tier.ContextOver != 272000 {
		t.Errorf("a prompt over the tier size should use the tier, got %+v", tier)
	}
}

func TestEmbeddedModelInfo(t *testing.T) {
	raw, err := modelsdev.Parse(embeddedModelsDev)
	if err != nil {
		t.Fatalf("embedded data does not parse: %v", err)
	}
	infos := modelInfoFromModelsDev(raw)
	if len(infos) < 100 {
		t.Fatalf("expected the embedded data to hold many models, got %d", len(infos))
	}

	opus := infos["anthropic/claude-opus-5-5"]
	if !opus.Caps.AdaptiveThinking || !opus.Caps.ReasoningMandatory {
		t.Errorf("claude-opus-5-5: expected adaptive thinking that cannot be turned off, got %+v", opus.Caps)
	}
	haiku := infos["anthropic/claude-haiku-4-5"]
	if len(haiku.Caps.ReasoningEfforts) != 0 || haiku.Caps.ReasoningBudget == nil {
		t.Errorf("claude-haiku-4-5: expected budget thinking without effort, got %+v", haiku.Caps)
	}
}

func TestCostUnmarshalAcceptsBothKeyStyles(t *testing.T) {
	var short, long Cost
	if err := json.Unmarshal([]byte(`{"i": 1, "o": 2, "r": 0.1, "w": 0.2}`), &short); err != nil {
		t.Fatalf("short keys: %v", err)
	}
	if err := json.Unmarshal([]byte(`{"input": 1, "output": 2, "cache_read": 0.1, "cache_write": 0.2}`), &long); err != nil {
		t.Fatalf("models.dev keys: %v", err)
	}
	want := Cost{Input: 1, Output: 2, CacheRead: 0.1, CacheWrite: 0.2}
	if !costEqual(short, want) || !costEqual(long, want) {
		t.Errorf("short=%+v long=%+v, want %+v", short, long, want)
	}
}

func costEqual(a, b Cost) bool {
	return a.Input == b.Input && a.Output == b.Output && a.CacheRead == b.CacheRead &&
		a.CacheWrite == b.CacheWrite && a.Reasoning == b.Reasoning && slices.Equal(a.Tiers, b.Tiers)
}
