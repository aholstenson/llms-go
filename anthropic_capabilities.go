package llms

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

const (
	// anthropicCapabilityTimeout limits how long a request waits for the
	// Models API before it continues with the model data it already has.
	anthropicCapabilityTimeout = 5 * time.Second
	// anthropicCapabilityRetry is how long a failed lookup is remembered
	// before the Models API is asked again.
	anthropicCapabilityRetry = 5 * time.Minute
)

// anthropicCapabilities caches the Anthropic Models API answer for one model.
// The API is the authoritative source for which thinking configurations and
// effort tiers a model accepts, so it replaces those parts of the models.dev
// data and makes models that models.dev does not list yet usable.
type anthropicCapabilities struct {
	mu        sync.Mutex
	info      *anthropic.ModelInfo
	nextRetry time.Time
}

// modelInfo returns the model's ModelInfo with the Anthropic Models API
// capabilities applied when they are available. Entries from
// RegisterModelInfo are used as they are.
func (m *anthropicModel) modelInfo(ctx context.Context) ModelInfo {
	info := m.info.get()
	if m.capabilities == nil || m.info.registered() {
		return info
	}
	if api := m.capabilities.get(ctx, m); api != nil {
		info = applyAnthropicCapabilities(info, m.model, api)
	}
	return info
}

// get returns the cached Models API answer, asking the API on first use and
// again after anthropicCapabilityRetry when a lookup failed. Concurrent
// callers wait for one lookup instead of starting their own.
func (c *anthropicCapabilities) get(ctx context.Context, m *anthropicModel) *anthropic.ModelInfo {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.info != nil || time.Now().Before(c.nextRetry) {
		return c.info
	}

	// The answer is shared by every caller, so a cancelled request must not
	// cancel the lookup and delay the next attempt for the others.
	lookupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), anthropicCapabilityTimeout)
	defer cancel()
	info, err := m.client.Models.Get(lookupCtx, m.model, anthropic.ModelGetParams{}, option.WithMaxRetries(0))
	if err != nil {
		c.nextRetry = time.Now().Add(anthropicCapabilityRetry)
		m.logger.Debug("Anthropic Models API lookup failed, using known model data",
			slog.String("model", m.model), slog.Any("error", err))
		return nil
	}
	c.info = info
	return c.info
}

// applyAnthropicCapabilities replaces the thinking, effort, structured output,
// input and limit data in info with the Anthropic Models API answer. Pricing,
// family and dates stay as they are.
func applyAnthropicCapabilities(info ModelInfo, id string, api *anthropic.ModelInfo) ModelInfo {
	wasUnknown := info.isUnknown()
	caps := api.Capabilities

	info.Caps.Reasoning = caps.Thinking.Supported
	info.Caps.AdaptiveThinking = caps.Thinking.Types.Adaptive.Supported
	switch {
	case !caps.Thinking.Types.Enabled.Supported:
		info.Caps.ReasoningBudget = nil
	case info.Caps.ReasoningBudget == nil:
		info.Caps.ReasoningBudget = &TokenRange{Min: anthropicMinThinkingBudget}
	}
	info.Caps.ReasoningEfforts = anthropicEfforts(caps.Effort)
	info.Caps.ReasoningMandatory = info.Caps.Reasoning && isAnthropicAlwaysReasons(id)
	info.Caps.StructuredOutput = caps.StructuredOutputs.Supported

	modalities := []string{"text"}
	if caps.ImageInput.Supported {
		modalities = append(modalities, "image")
	}
	if caps.PDFInput.Supported {
		modalities = append(modalities, "pdf")
	}
	info.Modalities = modalities
	info.Caps.Attachment = len(modalities) > 1

	if api.MaxInputTokens > 0 {
		info.Limits.Context = int(api.MaxInputTokens)
	}
	if api.MaxTokens > 0 {
		info.Limits.Output = int(api.MaxTokens)
	}

	if wasUnknown {
		// The Models API does not report these. Every current Claude model
		// calls tools, and Anthropic removed sampling parameters together
		// with budget thinking (Opus 4.7 and later reject both).
		info.Caps.ToolCall = true
		info.Caps.Temperature = !caps.Thinking.Supported || caps.Thinking.Types.Enabled.Supported
	}
	return info
}

// anthropicEfforts returns the effort tiers an Anthropic effort capability
// lists as supported, lowest first. It reads the raw JSON so tiers the SDK
// has no field for (such as "xhigh") are included.
func anthropicEfforts(effort anthropic.EffortCapability) []Effort {
	if !effort.Supported {
		return nil
	}

	supported := map[Effort]bool{
		EffortLow:    effort.Low.Supported,
		EffortMedium: effort.Medium.Supported,
		EffortHigh:   effort.High.Supported,
		EffortMax:    effort.Max.Supported,
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(effort.RawJSON()), &raw); err == nil {
		for name, value := range raw {
			var tier struct {
				Supported bool `json:"supported"`
			}
			if effortRank(Effort(name)) > 0 && json.Unmarshal(value, &tier) == nil {
				supported[Effort(name)] = tier.Supported
			}
		}
	}

	var out []Effort
	for _, e := range []Effort{EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax} {
		if supported[e] {
			out = append(out, e)
		}
	}
	return out
}
