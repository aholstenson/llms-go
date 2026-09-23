package llms

import (
	"slices"
	"strings"

	"github.com/aholstenson/llms-go/internal/modelsdev"
)

// anthropicAlwaysReasons lists Anthropic model id prefixes whose thinking
// cannot be turned off: they reject {type: "disabled"}. Neither models.dev nor
// the Anthropic Models API exposes this, so it is kept by hand. Ids are
// compared after "." is replaced by "-", so one entry covers both the
// Anthropic id ("claude-opus-5-5") and the OpenRouter id
// ("anthropic/claude-opus-5.5").
var anthropicAlwaysReasons = []string{
	"claude-fable-5",
	"claude-mythos",
	"claude-opus-5-5",
}

// anthropicMinThinkingBudget is the smallest budget_tokens value Anthropic
// accepts.
const anthropicMinThinkingBudget = 1024

// modelInfoFromModelsDev converts models.dev data into ModelInfo keyed by
// "<provider>/<models.dev model id>", which matches the qualified model names
// used by Manager and the stats keys (including nested ids such as
// "openrouter/google/gemini-2.5-flash-lite").
func modelInfoFromModelsDev(raw modelsdev.RawData) map[string]ModelInfo {
	out := make(map[string]ModelInfo)
	for _, provider := range modelsdev.Providers {
		for id, model := range raw[provider].Models {
			out[provider+"/"+id] = convertModelsDevModel(provider, id, model)
		}
	}
	return out
}

func convertModelsDevModel(provider, id string, m modelsdev.RawModel) ModelInfo {
	mi := ModelInfo{
		Cost: convertModelsDevCost(m.Cost),
		Limits: Limits{
			Context: m.Limit.Context,
			Input:   m.Limit.Input,
			Output:  m.Limit.Output,
		},
		Caps: Capabilities{
			Temperature: m.Temperature,
			Reasoning:   m.Reasoning,
			ToolCall:    m.ToolCall,
			Attachment:  m.Attachment,
			// Most models support structured output; only an explicit false
			// from models.dev turns the gate on.
			StructuredOutput: m.StructuredOutput == nil || *m.StructuredOutput,
		},
		Modalities: m.Modalities.Input,
		Family:     m.Family,
		Knowledge:  m.Knowledge,
		Released:   m.ReleaseDate,
		Status:     m.Status,
	}
	if m.Reasoning {
		applyReasoningOptions(provider, id, m.ReasoningOptions, &mi.Caps)
	}
	return mi
}

func convertModelsDevCost(c modelsdev.RawCost) Cost {
	cost := Cost{
		Input:      c.Input,
		Output:     c.Output,
		CacheRead:  c.CacheRead,
		CacheWrite: c.CacheWrite,
		Reasoning:  c.Reasoning,
	}
	for _, t := range c.Tiers {
		if t.Tier.Type != "context" || t.Tier.Size <= 0 {
			continue
		}
		cost.Tiers = append(cost.Tiers, CostTier{
			ContextOver: t.Tier.Size,
			Input:       t.Input,
			Output:      t.Output,
			CacheRead:   t.CacheRead,
			CacheWrite:  t.CacheWrite,
			Reasoning:   t.Reasoning,
		})
	}
	slices.SortFunc(cost.Tiers, func(a, b CostTier) int { return a.ContextOver - b.ContextOver })
	return cost
}

// applyReasoningOptions fills the reasoning controls of a reasoning model from
// its models.dev reasoning_options.
func applyReasoningOptions(provider, id string, opts []modelsdev.RawReasoningOption, caps *Capabilities) {
	var toggle, effortNone bool
	for _, o := range opts {
		switch o.Type {
		case modelsdev.ReasoningOptionEffort:
			for _, v := range o.Values {
				e := Effort(v)
				switch {
				case e == EffortNone:
					effortNone = true
				case effortRank(e) >= 0 && !slices.Contains(caps.ReasoningEfforts, e):
					caps.ReasoningEfforts = append(caps.ReasoningEfforts, e)
				}
			}
		case modelsdev.ReasoningOptionBudget:
			budget := &TokenRange{}
			if o.Min != nil {
				budget.Min = *o.Min
			}
			if o.Max != nil {
				budget.Max = *o.Max
			}
			caps.ReasoningBudget = budget
		case modelsdev.ReasoningOptionToggle:
			toggle = true
		}
	}
	slices.SortFunc(caps.ReasoningEfforts, func(a, b Effort) int { return effortRank(a) - effortRank(b) })

	// Anthropic models that take an effort but no budget use adaptive
	// thinking (Opus 4.7 and later). Models that take both (Opus 4.5 and
	// 4.6, Sonnet 4.6) keep budget thinking here, which they all accept; the
	// Models API lookup at runtime switches the 4.6 models to adaptive.
	if provider == "anthropic" && len(caps.ReasoningEfforts) > 0 && caps.ReasoningBudget == nil {
		caps.AdaptiveThinking = true
	}

	caps.ReasoningMandatory = !canDisableReasoning(provider, id, len(opts) > 0, toggle, effortNone, caps)
}

// canDisableReasoning reports whether a reasoning model accepts a request to
// turn reasoning off. The signal differs per vendor: OpenAI lists a "none"
// effort, Gemini accepts a zero budget or the minimal level, and Anthropic is
// covered by anthropicAlwaysReasons.
func canDisableReasoning(provider, id string, hasOptions, toggle, effortNone bool, caps *Capabilities) bool {
	vendor, modelID := provider, id
	if provider == "openrouter" {
		vendor, modelID, _ = strings.Cut(id, "/")
	}

	if vendor == "anthropic" {
		return !isAnthropicAlwaysReasons(modelID)
	}
	if toggle || effortNone || !hasOptions {
		return true
	}

	switch vendor {
	case "openai":
		return false
	case "google":
		if caps.ReasoningBudget != nil && caps.ReasoningBudget.Min == 0 {
			return true
		}
		return slices.Contains(caps.ReasoningEfforts, EffortMinimal)
	default:
		return true
	}
}

// isAnthropicAlwaysReasons reports whether an Anthropic model id (without a
// provider prefix) is in anthropicAlwaysReasons.
func isAnthropicAlwaysReasons(id string) bool {
	norm := strings.ReplaceAll(id, ".", "-")
	for _, prefix := range anthropicAlwaysReasons {
		if strings.HasPrefix(norm, prefix) {
			return true
		}
	}
	return false
}
