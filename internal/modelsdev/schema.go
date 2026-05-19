// Package modelsdev contains the build-time transform from the raw
// models.dev api.json shape into the compact ModelInfo artifact embedded by
// the llms package. It performs no IO so the transform is unit-testable.
package modelsdev

// RawData mirrors the top level of models.dev's api.json: a map of provider
// id to the provider's entry.
type RawData map[string]RawProvider

// RawProvider mirrors a single provider entry. Only the fields the transform
// consumes are modeled; unknown fields are ignored by encoding/json.
type RawProvider struct {
	ID     string              `json:"id"`
	Name   string              `json:"name"`
	Models map[string]RawModel `json:"models"`
}

// RawModel mirrors a single model entry under a provider's "models" map.
type RawModel struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Family      string        `json:"family"`
	Attachment  bool          `json:"attachment"`
	Reasoning   bool          `json:"reasoning"`
	ToolCall    bool          `json:"tool_call"`
	Temperature bool          `json:"temperature"`
	Knowledge   string        `json:"knowledge"`
	ReleaseDate string        `json:"release_date"`
	Modalities  RawModalities `json:"modalities"`
	Cost        RawCost       `json:"cost"`
	Limit       RawLimit      `json:"limit"`
}

// RawModalities mirrors models.dev's modalities object.
type RawModalities struct {
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

// RawCost mirrors models.dev's cost object, in USD per 1M tokens. Fields are
// absent for dimensions a model does not support.
type RawCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read"`
	CacheWrite float64 `json:"cache_write"`
}

// RawLimit mirrors models.dev's limit object.
type RawLimit struct {
	Context int `json:"context"`
	Output  int `json:"output"`
}
