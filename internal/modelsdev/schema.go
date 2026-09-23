// Package modelsdev models the models.dev api.json format and reduces it to
// the providers the llms package supports. It performs no network IO and does
// not depend on the llms package, so the llms package can use it at runtime
// and cmd/genmodelinfo can use it at build time.
package modelsdev

// RawData mirrors the top level of models.dev's api.json: a map of provider
// id to the provider's entry.
type RawData map[string]RawProvider

// RawProvider mirrors a single provider entry. Only the fields the llms
// package consumes are modeled; unknown fields are ignored by encoding/json.
type RawProvider struct {
	ID     string              `json:"id"`
	Name   string              `json:"name"`
	Models map[string]RawModel `json:"models"`
}

// RawModel mirrors a single model entry under a provider's "models" map.
type RawModel struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Family      string `json:"family"`
	Attachment  bool   `json:"attachment"`
	Reasoning   bool   `json:"reasoning"`
	ToolCall    bool   `json:"tool_call"`
	Temperature bool   `json:"temperature"`
	// StructuredOutput is nil when models.dev does not state support.
	StructuredOutput *bool `json:"structured_output"`
	// ReasoningOptions lists the controls a reasoning model accepts.
	ReasoningOptions []RawReasoningOption `json:"reasoning_options"`
	Knowledge        string               `json:"knowledge"`
	ReleaseDate      string               `json:"release_date"`
	LastUpdated      string               `json:"last_updated"`
	// Status is "", "alpha", "beta" or "deprecated".
	Status     string        `json:"status"`
	Modalities RawModalities `json:"modalities"`
	Cost       RawCost       `json:"cost"`
	Limit      RawLimit      `json:"limit"`
}

// Reasoning option types used in RawReasoningOption.Type.
const (
	// ReasoningOptionEffort is an effort tier control; Values lists the tiers
	// in ascending order. A "none" value means reasoning can be turned off.
	ReasoningOptionEffort = "effort"
	// ReasoningOptionBudget is a thinking-token budget control with an
	// optional Min and Max.
	ReasoningOptionBudget = "budget_tokens"
	// ReasoningOptionToggle means reasoning can be switched on and off.
	ReasoningOptionToggle = "toggle"
)

// RawReasoningOption is one entry of a model's reasoning_options list.
type RawReasoningOption struct {
	Type   string   `json:"type"`
	Values []string `json:"values,omitempty"`
	Min    *int     `json:"min,omitempty"`
	Max    *int     `json:"max,omitempty"`
}

// RawModalities mirrors models.dev's modalities object.
type RawModalities struct {
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

// RawCost mirrors models.dev's cost object, in USD per 1M tokens. Fields are
// absent for dimensions a model does not support.
type RawCost struct {
	Input      float64       `json:"input"`
	Output     float64       `json:"output"`
	CacheRead  float64       `json:"cache_read"`
	CacheWrite float64       `json:"cache_write"`
	Reasoning  float64       `json:"reasoning"`
	Tiers      []RawCostTier `json:"tiers"`
}

// RawCostTier is a price that replaces the base cost when a request passes
// the tier threshold. Rates absent from the tier keep the base rate.
type RawCostTier struct {
	Input      float64     `json:"input"`
	Output     float64     `json:"output"`
	CacheRead  float64     `json:"cache_read"`
	CacheWrite float64     `json:"cache_write"`
	Reasoning  float64     `json:"reasoning"`
	Tier       RawTierSpec `json:"tier"`
}

// RawTierSpec describes when a cost tier applies. The only type models.dev
// uses is "context": the tier applies when the prompt is larger than Size
// tokens.
type RawTierSpec struct {
	Type string `json:"type"`
	Size int    `json:"size"`
}

// RawLimit mirrors models.dev's limit object.
type RawLimit struct {
	Context int `json:"context"`
	Input   int `json:"input"`
	Output  int `json:"output"`
}
