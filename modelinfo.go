package llms

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

//go:generate go run ./cmd/genmodelinfo

// ModelInfo holds the metadata for a model: pricing, token limits, and the
// capabilities that gate what providers send to the API.
//
// It is built from models.dev data (embedded at build time, or loaded with
// LoadModelInfo, LLM_MODELS_FILE or RefreshModelInfo), from RegisterModelInfo,
// and for Anthropic models also from the Anthropic Models API at runtime.
type ModelInfo struct {
	Cost       Cost         `json:"cost"`
	Limits     Limits       `json:"limits"`
	Caps       Capabilities `json:"capabilities"`
	Modalities []string     `json:"modalities,omitempty"` // input modalities: text/image/audio/pdf/video
	Family     string       `json:"family,omitempty"`
	Knowledge  string       `json:"knowledge,omitempty"` // training cutoff
	Released   string       `json:"released,omitempty"`  // release date
	// Status is empty for generally available models, otherwise one of
	// ModelStatusAlpha, ModelStatusBeta or ModelStatusDeprecated.
	Status string `json:"status,omitempty"`
}

// Model status values used in ModelInfo.Status.
const (
	ModelStatusAlpha      = "alpha"
	ModelStatusBeta       = "beta"
	ModelStatusDeprecated = "deprecated"
)

// Cost is the per-model token pricing, in USD per 1 million tokens.
// A zero value means the dimension is unsupported or free.
type Cost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read,omitempty"`
	CacheWrite float64 `json:"cache_write,omitempty"`
	// Reasoning is the rate for thinking tokens that a provider reports
	// apart from output tokens. Zero means thinking tokens use the Output
	// rate.
	Reasoning float64 `json:"reasoning,omitempty"`
	// Tiers are higher prices for large requests, ordered by ContextOver.
	Tiers []CostTier `json:"tiers,omitempty"`
}

// CostTier is the pricing for requests whose prompt is larger than
// ContextOver tokens. The prompt size is the input tokens plus the cached
// (read and written) tokens of one request. A zero rate keeps the base rate.
type CostTier struct {
	ContextOver int     `json:"context_over"`
	Input       float64 `json:"input,omitempty"`
	Output      float64 `json:"output,omitempty"`
	CacheRead   float64 `json:"cache_read,omitempty"`
	CacheWrite  float64 `json:"cache_write,omitempty"`
	Reasoning   float64 `json:"reasoning,omitempty"`
}

// UnmarshalJSON decodes a Cost from the models.dev key names. It also accepts
// the short keys "i", "o", "r" and "w" that earlier pricing files use.
func (c *Cost) UnmarshalJSON(data []byte) error {
	type plain Cost
	var v struct {
		plain
		ShortInput      *float64 `json:"i"`
		ShortOutput     *float64 `json:"o"`
		ShortCacheRead  *float64 `json:"r"`
		ShortCacheWrite *float64 `json:"w"`
	}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	*c = Cost(v.plain)
	for _, f := range []struct {
		short *float64
		dst   *float64
	}{
		{v.ShortInput, &c.Input},
		{v.ShortOutput, &c.Output},
		{v.ShortCacheRead, &c.CacheRead},
		{v.ShortCacheWrite, &c.CacheWrite},
	} {
		if f.short != nil {
			*f.dst = *f.short
		}
	}
	return nil
}

// Limits describes the token limits for a model.
type Limits struct {
	Context int `json:"context,omitempty"`
	// Input is the maximum prompt size when it is lower than Context.
	Input  int `json:"input,omitempty"`
	Output int `json:"output,omitempty"`
}

// TokenRange is an inclusive token count range. A zero Max means no upper
// limit is known.
type TokenRange struct {
	Min int `json:"min,omitempty"`
	Max int `json:"max,omitempty"`
}

// Capabilities describes which behaviors a model supports. These flags gate
// what providers send to the API.
type Capabilities struct {
	Temperature      bool `json:"temperature"`
	Reasoning        bool `json:"reasoning"`
	ToolCall         bool `json:"tool_call"`
	Attachment       bool `json:"attachment"`
	StructuredOutput bool `json:"structured_output"`
	// ReasoningEfforts lists the effort tiers the model accepts, lowest
	// first. Empty means the model has no effort control or that it is not
	// known. WithReasoningEffort values the model does not accept are moved to
	// the nearest tier in this list.
	ReasoningEfforts []Effort `json:"reasoning_efforts,omitempty"`
	// ReasoningBudget is the thinking-token budget range the model accepts.
	// Nil means the model does not take a thinking-token budget.
	ReasoningBudget *TokenRange `json:"reasoning_budget,omitempty"`
	// AdaptiveThinking marks Anthropic models that accept adaptive thinking,
	// where the model decides how much to think.
	AdaptiveThinking bool `json:"adaptive_thinking,omitempty"`
	// ReasoningMandatory marks models that always reason and reject every
	// form of turning reasoning off.
	ReasoningMandatory bool `json:"reasoning_mandatory,omitempty"`
}

// isUnknown reports whether this is the zero ModelInfo, i.e. the model was
// not found in any model data. Unknown models are treated permissively by
// the behavior gates so a model missing from models.dev never silently breaks.
func (mi ModelInfo) isUnknown() bool {
	// Empty slices count as zero, so an entry decoded from "modalities": []
	// is not taken as known.
	if len(mi.Modalities) == 0 {
		mi.Modalities = nil
	}
	if len(mi.Caps.ReasoningEfforts) == 0 {
		mi.Caps.ReasoningEfforts = nil
	}
	if len(mi.Cost.Tiers) == 0 {
		mi.Cost.Tiers = nil
	}
	return reflect.ValueOf(mi).IsZero()
}

// allowsStructuredOutput reports whether a response JSON schema may be sent.
func (mi ModelInfo) allowsStructuredOutput() bool {
	return mi.isUnknown() || mi.Caps.StructuredOutput
}

// minEffort returns the lowest effort tier the model accepts, or "" when the
// model has no known effort tiers.
func (mi ModelInfo) minEffort() Effort {
	if len(mi.Caps.ReasoningEfforts) == 0 {
		return ""
	}
	return mi.Caps.ReasoningEfforts[0]
}

// acceptsEffort reports whether the model lists e as an accepted effort tier.
func (mi ModelInfo) acceptsEffort(e Effort) bool {
	return slices.Contains(mi.Caps.ReasoningEfforts, e)
}

// allowsTemperature reports whether a temperature parameter may be sent.
func (mi ModelInfo) allowsTemperature() bool {
	return mi.isUnknown() || mi.Caps.Temperature
}

// allowsReasoning reports whether thinking/reasoning params may be sent.
func (mi ModelInfo) allowsReasoning() bool {
	return mi.isUnknown() || mi.Caps.Reasoning
}

// allowsToolCall reports whether tool/function calling may be requested.
func (mi ModelInfo) allowsToolCall() bool {
	return mi.isUnknown() || mi.Caps.ToolCall
}

// allowsModality reports whether the model accepts the given input modality
// (e.g. "image"). Unknown models, and models with no declared modalities,
// are treated permissively.
func (mi ModelInfo) allowsModality(modality string) bool {
	if mi.isUnknown() || len(mi.Modalities) == 0 {
		return true
	}
	for _, m := range mi.Modalities {
		if m == modality {
			return true
		}
	}
	return false
}

// clampMaxOutputTokens clamps a requested max output token count to the model's
// declared output limit. It returns the (possibly reduced) value; a value of
// 0 (caller default) and unknown models are passed through unchanged.
func (mi ModelInfo) clampMaxOutputTokens(requested int) (int, bool) {
	if mi.isUnknown() || requested == 0 || mi.Limits.Output == 0 {
		return requested, false
	}
	if requested > mi.Limits.Output {
		return mi.Limits.Output, true
	}
	return requested, false
}

// resolveMaxOutputTokens returns the effective max output tokens to send to
// the provider. A nonzero requested value is returned as-is. Otherwise the
// model's declared output limit is used. If neither is known, fallback is
// returned (pass 0 to signal "leave unset on the wire").
func (mi ModelInfo) resolveMaxOutputTokens(requested, fallback int) int {
	if requested != 0 {
		return requested
	}
	if mi.Limits.Output > 0 {
		return mi.Limits.Output
	}
	return fallback
}

// partModality returns the models.dev input modality required by a part
// ("image", "audio", "pdf", "video"), or "" for parts that don't gate on
// modality (text, tool calls, thinking, etc.).
func partModality(part MessagePart) string {
	switch p := part.(type) {
	case *ImagePart:
		return "image"
	case *BinaryPart:
		switch {
		case strings.HasPrefix(p.MediaType, "image/"):
			return "image"
		case strings.HasPrefix(p.MediaType, "audio/"):
			return "audio"
		case strings.HasPrefix(p.MediaType, "video/"):
			return "video"
		case p.MediaType == "application/pdf":
			return "pdf"
		}
	}
	return ""
}

// firstUnsupportedModality scans messages for the first part whose required
// input modality is not allowed by the model. It returns "" if every part is
// acceptable.
func firstUnsupportedModality(messages []*Message, info ModelInfo) string {
	for _, msg := range messages {
		for _, part := range msg.Parts {
			modality := partModality(part)
			if modality == "" {
				continue
			}
			if !info.allowsModality(modality) {
				return modality
			}
		}
	}
	return ""
}

// checkRequestCapabilities rejects a request that uses a capability the model
// is known not to have (tools, an input modality, or a response schema), so it
// fails before any network call.
func checkRequestCapabilities(statsModel string, info ModelInfo, opts *generateContentOptions) error {
	if len(opts.Tools) > 0 && !info.allowsToolCall() {
		return fmt.Errorf("model %s does not support tool calling", statsModel)
	}
	if modality := firstUnsupportedModality(opts.Messages, info); modality != "" {
		return fmt.Errorf("model %s does not support %s input", statsModel, modality)
	}
	if opts.ResponseSchema != nil && !info.allowsStructuredOutput() {
		return fmt.Errorf("model %s does not support structured output", statsModel)
	}
	return nil
}
