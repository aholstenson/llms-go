package llms

import (
	"encoding/json"
	"log/slog"
	"math"
	"os"

	"github.com/cockroachdb/errors"
)

// PricingManager resolves model pricing.
//
// Pricing comes from the embedded models.dev-sourced ModelInfo data. A
// user-supplied override file (LLM_PRICING_FILE) takes precedence, which is
// how manual corrections or models missing from models.dev are handled. The
// override file is a JSON object mapping the fully qualified model name to a
// Cost value (the same models.dev-shaped Cost used by ModelInfo).
type PricingManager struct {
	logger    *slog.Logger
	overrides map[string]Cost
}

// NewPricingManager creates a new PricingManager.
// It loads pricing overrides from the file specified in LLM_PRICING_FILE,
// or defaults to llms-pricing.json in the current directory. When the file is
// absent or fails to load, pricing falls back to the embedded model info.
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
			logger.Warn("Failed to load pricing file, using embedded model info",
				slog.String("file", pricingFile),
				slog.Any("error", err))
		} else {
			logger.Debug("Pricing file not found, using embedded model info",
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
		return errors.Wrap(err, "failed to parse pricing file")
	}

	// Loaded pricing takes precedence over the embedded data.
	for model, cost := range pricing {
		pm.overrides[model] = cost
	}

	return nil
}

// GetModelPricing returns the pricing for a specific model.
//
// Resolution order: (1) override file entry, (2) embedded model info,
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

// CalculateCosts calculates the total cost for each service based on token usage.
// Returns a map of service name to total cost in USD.
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
			var rate float64
			switch key {
			case "input_tokens":
				rate = pricing.Input
			case "output_tokens":
				rate = pricing.Output
			case "cached_read_tokens":
				rate = pricing.CacheRead
			case "cached_write_tokens":
				rate = pricing.CacheWrite
			case "thinking_tokens":
				// models.dev has no dedicated thinking price. Gemini bills
				// thinking tokens at the regular output rate, so apply Output.
				rate = pricing.Output
			default:
				continue
			}

			// Costs in Cost are USD per 1M tokens.
			cost += float64(value) * rate / 1e6
		}

		// Round the cost to 6 decimal places
		costs[service] = math.Round(cost*1000000) / 1000000
	}

	return costs
}
