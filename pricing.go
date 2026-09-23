package llms

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"strconv"
	"strings"
)

// PricingManager resolves model pricing.
//
// Pricing comes from the model info (see LookupModelInfo). A user-supplied
// override file (LLM_PRICING_FILE) takes precedence, which is how manual
// pricing corrections are handled. The override file is a JSON object mapping
// the fully qualified model name to a Cost value, with the models.dev key
// names ("input", "output", "cache_read", "cache_write", "reasoning"). Tier
// breakdown counters are recorded from the model info, so tiers in the
// override file only change the price of tiers the model info also has.
type PricingManager struct {
	logger    *slog.Logger
	overrides map[string]Cost
}

// NewPricingManager creates a new PricingManager.
// It loads pricing overrides from the file specified in LLM_PRICING_FILE,
// or defaults to llms-pricing.json in the current directory. When the file is
// absent or fails to load, pricing falls back to the model info.
func NewPricingManager(logger *slog.Logger) *PricingManager {
	pm := &PricingManager{
		logger:    logger.With(slog.String("component", "pricing")),
		overrides: make(map[string]Cost),
	}

	pricingFile := os.Getenv("LLM_PRICING_FILE")
	if pricingFile == "" {
		pricingFile = "llms-pricing.json"
	}

	if err := pm.loadPricingFile(pricingFile); err != nil {
		if !os.IsNotExist(err) {
			logger.Warn("Failed to load pricing file, using model info",
				slog.String("file", pricingFile),
				slog.Any("error", err))
		} else {
			logger.Debug("Pricing file not found, using model info",
				slog.String("file", pricingFile))
		}
	} else {
		logger.Info("Loaded pricing overrides from file", slog.String("file", pricingFile))
	}

	return pm
}

// loadPricingFile loads pricing overrides from a JSON file.
func (pm *PricingManager) loadPricingFile(path string) error {
	data, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		return err
	}

	var pricing map[string]Cost
	if err := json.Unmarshal(data, &pricing); err != nil {
		return fmt.Errorf("failed to parse pricing file: %w", err)
	}

	// Loaded pricing takes precedence over the embedded data.
	for model, cost := range pricing {
		pm.overrides[model] = cost
	}

	return nil
}

// GetModelPricing returns the pricing for a specific model.
//
// Resolution order: (1) override file entry, (2) model info,
// (3) nil when nothing is known.
func (pm *PricingManager) GetModelPricing(serviceName string) *Cost {
	if cost, ok := pm.overrides[serviceName]; ok {
		return &cost
	}

	if info, ok := LookupModelInfo(serviceName); ok {
		cost := info.Cost
		return &cost
	}

	return nil
}

// tierCounterInfix joins a token counter name and the ContextOver of the cost
// tier its tokens were billed at, as in "input_tokens_over_200000".
const tierCounterInfix = "_over_"

// recordTierUsage records the tier breakdown for one request whose prompt is
// larger than one of the model's cost tiers: each token counter is copied to
// "<counter>_over_<ContextOver>". The base counters keep the full totals, and
// CalculateCosts adds the price difference for the copies.
func recordTierUsage(collector Collector, info ModelInfo, u TurnUsage) {
	prompt := u.InputTokens + u.CachedReadTokens + u.CachedWriteTokens
	tier := info.Cost.tierFor(int(prompt))
	if tier == nil {
		return
	}
	suffix := tierCounterInfix + strconv.Itoa(tier.ContextOver)
	for name, value := range map[string]int64{
		"input_tokens":        u.InputTokens,
		"output_tokens":       u.OutputTokens,
		"cached_read_tokens":  u.CachedReadTokens,
		"cached_write_tokens": u.CachedWriteTokens,
		"thinking_tokens":     u.ThinkingTokens,
	} {
		if value > 0 {
			collector.Counter(name + suffix).Add(int(value))
		}
	}
}

// tierFor returns the highest tier whose ContextOver is below the prompt
// size, or nil when the base prices apply.
func (c Cost) tierFor(promptTokens int) *CostTier {
	var found *CostTier
	for i := range c.Tiers {
		if promptTokens > c.Tiers[i].ContextOver {
			found = &c.Tiers[i]
		}
	}
	return found
}

// tier returns the tier with the given ContextOver, or nil.
func (c Cost) tier(contextOver int) *CostTier {
	for i := range c.Tiers {
		if c.Tiers[i].ContextOver == contextOver {
			return &c.Tiers[i]
		}
	}
	return nil
}

// rate returns the price per 1M tokens for a token counter name, and false
// for counters that are not billed.
func (c Cost) rate(counter string) (float64, bool) {
	switch counter {
	case "input_tokens":
		return c.Input, true
	case "output_tokens":
		return c.Output, true
	case "cached_read_tokens":
		return c.CacheRead, true
	case "cached_write_tokens":
		return c.CacheWrite, true
	case "thinking_tokens":
		// Thinking tokens that a provider reports apart from output tokens
		// (Gemini) are billed at the output rate unless a reasoning rate is
		// given.
		if c.Reasoning > 0 {
			return c.Reasoning, true
		}
		return c.Output, true
	default:
		return 0, false
	}
}

// rate returns the tier price for a token counter name. Rates the tier does
// not set keep the base rate.
func (t CostTier) rate(base Cost, counter string) float64 {
	var v float64
	switch counter {
	case "input_tokens":
		v = t.Input
	case "output_tokens":
		v = t.Output
	case "cached_read_tokens":
		v = t.CacheRead
	case "cached_write_tokens":
		v = t.CacheWrite
	case "thinking_tokens":
		v = t.Reasoning
		if v == 0 && base.Reasoning == 0 {
			v = t.Output
		}
	}
	if v == 0 {
		v, _ = base.rate(counter)
	}
	return v
}

// CalculateCosts calculates the total cost for each service based on token usage.
// Returns a map of service name to total cost in USD.
//
// Requests larger than a cost tier are billed at the tier price; see
// CostTier.
func (pm *PricingManager) CalculateCosts(stats CallStats) map[string]float64 {
	costs := make(map[string]float64)

	for service, counters := range stats.Success {
		pricing := pm.GetModelPricing(service)
		if pricing == nil {
			pm.logger.Warn("No pricing found for service", slog.String("service", service))
			continue
		}

		var cost float64
		for key, value := range counters {
			name, over, isTier := strings.Cut(key, tierCounterInfix)
			baseRate, billed := pricing.rate(name)
			if !billed {
				continue
			}

			// Costs in Cost are USD per 1M tokens.
			if !isTier {
				cost += float64(value) * baseRate / 1e6
				continue
			}

			// Tier counters repeat tokens already counted at the base rate,
			// so only the difference to the tier rate is added.
			contextOver, err := strconv.Atoi(over)
			if err != nil {
				continue
			}
			if tier := pricing.tier(contextOver); tier != nil {
				cost += float64(value) * (tier.rate(*pricing, name) - baseRate) / 1e6
			}
		}

		// Round the cost to 6 decimal places
		costs[service] = math.Round(cost*1000000) / 1000000
	}

	return costs
}
