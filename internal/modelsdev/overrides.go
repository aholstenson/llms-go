package modelsdev

import (
	"strings"

	llms "github.com/aholstenson/llms-go"
)

// reasoningOverride is a hand-maintained correction to the per-model reasoning
// metadata that models.dev does not provide (it only exposes reasoning:bool).
// Matches are evaluated most-specific-first; the first match wins.
//
// Match is a substring test against a normalized "<provider>/<id>" key in which
// "." is rewritten to "-", so a single dash-form pattern (e.g.
// "claude-opus-4-7") covers both the Anthropic id ("claude-opus-4-7") and the
// OpenRouter id ("anthropic/claude-opus-4.7", incl. "-fast" variants).
type reasoningOverride struct {
	Match     string // substring on the normalized "provider/id" key
	Style     string // -> Caps.ReasoningStyle (empty: keep provider default)
	MaxEffort string // -> Caps.MaxEffort (empty: keep provider default)
	Mandatory bool   // -> Caps.ReasoningMandatory
}

// reasoningOverrides is the ordered, most-specific-first override table.
var reasoningOverrides = []reasoningOverride{
	// Anthropic Opus 4.7: adaptive-only thinking, always reasons. (It also
	// rejects temperature, but models.dev already reports temperature:false for
	// it, so the Caps.Temperature gate handles that without an override.)
	{Match: "claude-opus-4-7", Style: "adaptive", MaxEffort: "max", Mandatory: true},

	// Anthropic 4.6: adaptive thinking (budget_tokens rejected; effort tier
	// optional alongside adaptive).
	{Match: "claude-opus-4-6", Style: "adaptive"},
	{Match: "claude-sonnet-4-6", Style: "adaptive"},

	// Anthropic 4.5: effort tier via output_config (no adaptive).
	{Match: "claude-opus-4-5", Style: "effort"},
	{Match: "claude-sonnet-4-5", Style: "effort"},
	{Match: "claude-haiku-4-5", Style: "effort"},

	// Gemini 3+: thinking_level (thinkingBudget discouraged). Matches 3, 3.1,
	// 3.5, ... via the "gemini-3" stem on the normalized key.
	{Match: "gemini-3", Style: "level"},

	// OpenAI o-series: reasoning cannot be disabled (rejects effort "none").
	// Style stays the OpenAI effort default.
	{Match: "openai/o1", Mandatory: true},
	{Match: "openai/o3", Mandatory: true},
	{Match: "openai/o4", Mandatory: true},

	// Pre-4.5 Claude and Gemini 2.5 intentionally have no entry: they keep the
	// provider-default budget style and remain disableable.
}

// ApplyReasoningMetadata fills in the per-model reasoning metadata for a single
// transformed ModelInfo, keyed by its "<provider>/<id>" key. It first applies
// the provider-default reasoning style (only for reasoning-capable models),
// then the first matching override. Temperature support is taken from
// models.dev as-is and is not adjusted here.
//
// It is shared by the gen-time Transform and by tooling that re-derives the
// metadata over an already-transformed artifact, so both stay consistent.
func ApplyReasoningMetadata(key string, mi *llms.ModelInfo) {
	provider := key
	if i := strings.IndexByte(key, '/'); i >= 0 {
		provider = key[:i]
	}

	if mi.Caps.Reasoning {
		switch provider {
		case "openai", "openrouter":
			mi.Caps.ReasoningStyle = "effort"
			mi.Caps.MaxEffort = "high"
		case "google", "anthropic":
			mi.Caps.ReasoningStyle = "budget"
		}
	}

	norm := strings.ReplaceAll(key, ".", "-")
	for _, ov := range reasoningOverrides {
		if !strings.Contains(norm, ov.Match) {
			continue
		}
		if mi.Caps.Reasoning {
			if ov.Style != "" {
				mi.Caps.ReasoningStyle = ov.Style
			}
			if ov.MaxEffort != "" {
				mi.Caps.MaxEffort = ov.MaxEffort
			}
			if ov.Mandatory {
				mi.Caps.ReasoningMandatory = true
			}
		}
		break // first (most-specific) match wins
	}
}
