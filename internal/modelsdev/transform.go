package modelsdev

import (
	llms "github.com/aholstenson/llms-go"
)

// allowedProviders is the locked set of providers the llms package supports.
// Models from any other provider in api.json are dropped.
var allowedProviders = []string{"anthropic", "openai", "google", "openrouter"}

// Transform reduces the raw models.dev data down to the compact artifact the
// llms package embeds. Keys are "<provider>/<models.dev model id>", matching
// the stats/pricing key convention already used throughout the package
// (including nested ids such as "openrouter/google/gemini-2.5-flash-lite").
//
// The function performs no IO so it can be unit-tested.
func Transform(raw RawData) map[string]llms.ModelInfo {
	out := make(map[string]llms.ModelInfo)

	for _, provider := range allowedProviders {
		entry, ok := raw[provider]
		if !ok {
			continue
		}

		for id, model := range entry.Models {
			key := provider + "/" + id
			out[key] = llms.ModelInfo{
				Cost: llms.Cost{
					Input:      model.Cost.Input,
					Output:     model.Cost.Output,
					CacheRead:  model.Cost.CacheRead,
					CacheWrite: model.Cost.CacheWrite,
				},
				Limits: llms.Limits{
					Context: model.Limit.Context,
					Output:  model.Limit.Output,
				},
				Caps: llms.Capabilities{
					Temperature: model.Temperature,
					Reasoning:   model.Reasoning,
					ToolCall:    model.ToolCall,
					Attachment:  model.Attachment,
				},
				Modalities: model.Modalities.Input,
				Family:     model.Family,
				Knowledge:  model.Knowledge,
				Released:   model.ReleaseDate,
			}
		}
	}

	return out
}
