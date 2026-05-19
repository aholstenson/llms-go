package llms

//go:generate go run ./cmd/genmodelinfo

// ModelInfo holds the embedded, build-time-generated metadata for a model.
// It is sourced from models.dev (see cmd/genmodelinfo) and is used both for
// pricing and for gating provider behavior (temperature, reasoning, tool
// calling, input modalities).
//
// The JSON keys are deliberately short because the data is embedded as a
// committed artifact (modelinfo_data.json); see cmd/genmodelinfo.
type ModelInfo struct {
	Cost       Cost         `json:"c"`
	Limits     Limits       `json:"l"`
	Caps       Capabilities `json:"f"`
	Modalities []string     `json:"m,omitempty"` // input modalities: text/image/audio/pdf/video
	Family     string       `json:"fam,omitempty"`
	Knowledge  string       `json:"k,omitempty"`  // training cutoff
	Released   string       `json:"rd,omitempty"` // release_date
}

// Cost is the per-model token pricing, in USD per 1 million tokens.
// A zero value means the dimension is unsupported or free.
type Cost struct {
	Input      float64 `json:"i"`
	Output     float64 `json:"o"`
	CacheRead  float64 `json:"r,omitempty"`
	CacheWrite float64 `json:"w,omitempty"`
}

// Limits describes the token limits for a model.
type Limits struct {
	Context int `json:"c,omitempty"`
	Output  int `json:"o,omitempty"`
}

// Capabilities describes which behaviors a model supports. These flags gate
// what providers send to the API.
type Capabilities struct {
	Temperature bool `json:"t"`
	Reasoning   bool `json:"r"`
	ToolCall    bool `json:"tc"`
	Attachment  bool `json:"a"`
}

// isUnknown reports whether this is the zero ModelInfo, i.e. the model was
// not found in the embedded data. Unknown models are treated permissively by
// the behavior gates so a model missing from models.dev never silently breaks.
func (mi ModelInfo) isUnknown() bool {
	return mi.Cost == (Cost{}) &&
		mi.Limits == (Limits{}) &&
		mi.Caps == (Capabilities{}) &&
		len(mi.Modalities) == 0 &&
		mi.Family == "" &&
		mi.Knowledge == "" &&
		mi.Released == ""
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

// clampMaxTokens clamps a requested max output token count to the model's
// declared output limit. It returns the (possibly reduced) value; a value of
// 0 (caller default) and unknown models are passed through unchanged.
func (mi ModelInfo) clampMaxTokens(requested int) (int, bool) {
	if mi.isUnknown() || requested == 0 || mi.Limits.Output == 0 {
		return requested, false
	}
	if requested > mi.Limits.Output {
		return mi.Limits.Output, true
	}
	return requested, false
}

// messagesContainImages reports whether any message part requires image input
// support (image URLs or inline binary data).
func messagesContainImages(messages []*Message) bool {
	for _, msg := range messages {
		for _, part := range msg.Parts {
			switch part.(type) {
			case *ImagePart, *BinaryPart:
				return true
			}
		}
	}
	return false
}

// LookupModelInfo returns the embedded ModelInfo for a fully qualified model
// name (e.g. "anthropic/claude-haiku-4-5" or
// "openrouter/google/gemini-2.5-flash-lite"). The boolean is false when the
// model is not present in the embedded data, in which case the zero ModelInfo
// is returned and callers should treat the model permissively.
func LookupModelInfo(qualifiedName string) (ModelInfo, bool) {
	info, ok := modelInfoData()[qualifiedName]
	return info, ok
}
