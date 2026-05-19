package llms_test

import (
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/aholstenson/llms-go"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("PricingManager", func() {
	var logger *slog.Logger

	BeforeEach(func() {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	})

	Describe("NewPricingManager", func() {
		It("should load pricing overrides from a JSON file when available", func() {
			// Create a temporary pricing file in the models.dev Cost shape.
			tmpDir := GinkgoT().TempDir()
			pricingFile := filepath.Join(tmpDir, "pricing.json")

			pricingJSON := `{
				"test-model": {
					"i": 1.00,
					"o": 2.00,
					"r": 0.10,
					"w": 0.20
				}
			}`

			err := os.WriteFile(pricingFile, []byte(pricingJSON), 0o644) //nolint:gosec
			Expect(err).NotTo(HaveOccurred())

			// Set environment variable
			GinkgoT().Setenv("LLM_PRICING_FILE", pricingFile)

			pm := llms.NewPricingManager(logger)

			// Verify loaded pricing
			pricing := pm.GetModelPricing("test-model")
			Expect(pricing).NotTo(BeNil())
			Expect(pricing.Input).To(Equal(1.00))
			Expect(pricing.Output).To(Equal(2.00))
			Expect(pricing.CacheRead).To(Equal(0.10))
			Expect(pricing.CacheWrite).To(Equal(0.20))

			// Embedded models should still be available
			defaultPricing := pm.GetModelPricing("openai/gpt-4.1")
			Expect(defaultPricing).NotTo(BeNil())
		})

		It("should let override file entries take precedence over embedded data", func() {
			tmpDir := GinkgoT().TempDir()
			pricingFile := filepath.Join(tmpDir, "pricing.json")

			// Override the embedded gpt-4.1 pricing.
			pricingJSON := `{"openai/gpt-4.1": {"i": 99.0, "o": 99.0}}`
			err := os.WriteFile(pricingFile, []byte(pricingJSON), 0o644) //nolint:gosec
			Expect(err).NotTo(HaveOccurred())

			GinkgoT().Setenv("LLM_PRICING_FILE", pricingFile)

			pm := llms.NewPricingManager(logger)

			pricing := pm.GetModelPricing("openai/gpt-4.1")
			Expect(pricing).NotTo(BeNil())
			Expect(pricing.Input).To(Equal(99.0))
			Expect(pricing.Output).To(Equal(99.0))
		})

		It("should handle missing pricing file gracefully", func() {
			GinkgoT().Setenv("LLM_PRICING_FILE", "/nonexistent/pricing.json")

			pm := llms.NewPricingManager(logger)

			// Should still have default pricing
			pricing := pm.GetModelPricing("openai/gpt-4.1")
			Expect(pricing).NotTo(BeNil())
		})

		It("should handle invalid JSON gracefully", func() {
			tmpDir := GinkgoT().TempDir()
			pricingFile := filepath.Join(tmpDir, "invalid.json")

			err := os.WriteFile(pricingFile, []byte("invalid json"), 0o644) //nolint:gosec
			Expect(err).NotTo(HaveOccurred())

			GinkgoT().Setenv("LLM_PRICING_FILE", pricingFile)

			pm := llms.NewPricingManager(logger)

			// Should still have default pricing
			pricing := pm.GetModelPricing("openai/gpt-4.1")
			Expect(pricing).NotTo(BeNil())
		})
	})

	Describe("GetModelPricing", func() {
		var pm *llms.PricingManager

		BeforeEach(func() {
			pm = llms.NewPricingManager(logger)
		})

		It("can get a model", func() {
			pricing := pm.GetModelPricing("openai/gpt-4.1")
			Expect(pricing).NotTo(BeNil())
			Expect(pricing.Input).To(Equal(2.00))
			Expect(pricing.Output).To(Equal(8.00))
			Expect(pricing.CacheRead).To(Equal(0.50))
			Expect(pricing.CacheWrite).To(Equal(0.00))
		})

		It("should return nil for unknown models", func() {
			pricing := pm.GetModelPricing("unknown-model")
			Expect(pricing).To(BeNil())
		})
	})

	Describe("CalculateCosts", func() {
		var pm *llms.PricingManager

		BeforeEach(func() {
			pm = llms.NewPricingManager(logger)
		})

		It("should calculate costs for input and output tokens", func() {
			stats := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {
						"input_tokens":  1_000_000, // 1M tokens
						"output_tokens": 500_000,   // 0.5M tokens
					},
				},
			}

			costs := pm.CalculateCosts(stats)

			// gpt-4.1: input $2/M, output $8/M
			// Expected: (1M * 2) + (0.5M * 8) = $2 + $4 = $6
			Expect(costs).To(HaveKey("openai/gpt-4.1"))
			Expect(costs["openai/gpt-4.1"]).To(BeNumerically("~", 6.0, 0.001))
		})

		It("should calculate costs with cached tokens", func() {
			stats := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {
						"input_tokens":       500_000, // 0.5M tokens
						"cached_read_tokens": 500_000, // 0.5M cached tokens
						"output_tokens":      100_000, // 0.1M tokens
					},
				},
			}

			costs := pm.CalculateCosts(stats)

			// gpt-4.1: input $2/M, cache read $0.50/M, output $8/M
			// Expected: (0.5M * 2) + (0.5M * 0.50) + (0.1M * 8) = $1 + $0.25 + $0.80 = $2.05
			Expect(costs).To(HaveKey("openai/gpt-4.1"))
			Expect(costs["openai/gpt-4.1"]).To(BeNumerically("~", 2.05, 0.001))
		})

		It("should calculate costs with cache write tokens for Anthropic", func() {
			stats := llms.CallStats{
				Success: map[string]map[string]int{
					"anthropic/claude-sonnet-4-5": {
						"input_tokens":        500_000, // 0.5M tokens
						"cached_write_tokens": 200_000, // 0.2M cache write tokens
						"cached_read_tokens":  100_000, // 0.1M cache read tokens
						"output_tokens":       100_000, // 0.1M tokens
					},
				},
			}

			costs := pm.CalculateCosts(stats)

			// claude-sonnet-4.5: input $3/M, cache write $3.75/M, cache read $0.30/M, output $15/M
			// Expected: (0.5M * 3) + (0.2M * 3.75) + (0.1M * 0.30) + (0.1M * 15)
			//         = $1.50 + $0.75 + $0.03 + $1.50 = $3.78
			Expect(costs).To(HaveKey("anthropic/claude-sonnet-4-5"))
			Expect(costs["anthropic/claude-sonnet-4-5"]).To(BeNumerically("~", 3.78, 0.001))
		})

		It("should calculate costs for multiple services", func() {
			stats := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {
						"input_tokens":  1_000_000,
						"output_tokens": 500_000,
					},
					"anthropic/claude-haiku-4-5": {
						"input_tokens":  2_000_000,
						"output_tokens": 1_000_000,
					},
				},
			}

			costs := pm.CalculateCosts(stats)

			// gpt-4.1: (1M * 2) + (0.5M * 8) = $6
			// claude-haiku-4.5: (2M * 1) + (1M * 5) = $7
			Expect(costs).To(HaveLen(2))
			Expect(costs["openai/gpt-4.1"]).To(BeNumerically("~", 6.0, 0.001))
			Expect(costs["anthropic/claude-haiku-4-5"]).To(BeNumerically("~", 7.0, 0.001))
		})

		It("should handle models without pricing", func() {
			stats := llms.CallStats{
				Success: map[string]map[string]int{
					"unknown-provider/unknown-model": {
						"input_tokens":  1_000_000,
						"output_tokens": 500_000,
					},
				},
			}

			costs := pm.CalculateCosts(stats)

			// Should not include unknown model
			Expect(costs).To(BeEmpty())
		})

		It("should handle zero token usage", func() {
			stats := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {
						"requests": 1, // Only requests, no tokens
					},
				},
			}

			costs := pm.CalculateCosts(stats)

			Expect(costs).To(HaveKey("openai/gpt-4.1"))
			Expect(costs["openai/gpt-4.1"]).To(Equal(0.0))
		})

		It("should calculate realistic costs for typical usage", func() {
			stats := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {
						"input_tokens":  10_000, // 10K tokens
						"output_tokens": 2_000,  // 2K tokens
						"requests":      1,
					},
				},
			}

			costs := pm.CalculateCosts(stats)

			// gpt-4.1: input $2/M, output $8/M
			// Expected: (10K * 2 / 1M) + (2K * 8 / 1M) = $0.02 + $0.016 = $0.036
			Expect(costs).To(HaveKey("openai/gpt-4.1"))
			Expect(costs["openai/gpt-4.1"]).To(BeNumerically("~", 0.036, 0.0001))
		})
	})
})

var _ = Describe("CallStats", func() {
	Describe("Add", func() {
		It("should aggregate costs when adding stats", func() {
			stats1 := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {"tokens": 100},
				},
				Costs: map[string]float64{
					"openai/gpt-4.1": 1.50,
				},
			}

			stats2 := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {"tokens": 50},
				},
				Costs: map[string]float64{
					"openai/gpt-4.1": 0.75,
				},
			}

			aggregated := stats1.Add(stats2)

			Expect(aggregated.Success["openai/gpt-4.1"]["tokens"]).To(Equal(150))
			Expect(aggregated.Costs["openai/gpt-4.1"]).To(BeNumerically("~", 2.25, 0.001))
		})

		It("should aggregate costs for multiple services", func() {
			stats1 := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {"tokens": 100},
				},
				Costs: map[string]float64{
					"openai/gpt-4.1": 1.00,
				},
			}

			stats2 := llms.CallStats{
				Success: map[string]map[string]int{
					"anthropic/claude-sonnet-4.5": {"tokens": 200},
				},
				Costs: map[string]float64{
					"anthropic/claude-sonnet-4.5": 2.00,
				},
			}

			aggregated := stats1.Add(stats2)

			Expect(aggregated.Costs).To(HaveLen(2))
			Expect(aggregated.Costs["openai/gpt-4.1"]).To(Equal(1.00))
			Expect(aggregated.Costs["anthropic/claude-sonnet-4.5"]).To(Equal(2.00))
		})

		It("should handle empty costs", func() {
			stats1 := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {"tokens": 100},
				},
			}

			stats2 := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {"tokens": 50},
				},
				Costs: map[string]float64{
					"openai/gpt-4.1": 0.50,
				},
			}

			aggregated := stats1.Add(stats2)

			Expect(aggregated.Costs["openai/gpt-4.1"]).To(Equal(0.50))
		})

		It("should aggregate costs with existing tokens and failure stats", func() {
			stats1 := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {"tokens": 100, "requests": 1},
				},
				Failure: map[string]map[string]int{
					"openai/gpt-4.1": {"errors": 1},
				},
				Costs: map[string]float64{
					"openai/gpt-4.1": 1.50,
				},
			}

			stats2 := llms.CallStats{
				Success: map[string]map[string]int{
					"openai/gpt-4.1": {"tokens": 50, "requests": 1},
				},
				Failure: map[string]map[string]int{
					"openai/gpt-4.1": {"errors": 2},
				},
				Costs: map[string]float64{
					"openai/gpt-4.1": 0.75,
				},
			}

			aggregated := stats1.Add(stats2)

			Expect(aggregated.Success["openai/gpt-4.1"]["tokens"]).To(Equal(150))
			Expect(aggregated.Success["openai/gpt-4.1"]["requests"]).To(Equal(2))
			Expect(aggregated.Failure["openai/gpt-4.1"]["errors"]).To(Equal(3))
			Expect(aggregated.Costs["openai/gpt-4.1"]).To(BeNumerically("~", 2.25, 0.001))
		})
	})
})
