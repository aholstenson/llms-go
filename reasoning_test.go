package llms

import (
	"bytes"
	"context"
	"io"
	"log/slog"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/openai/openai-go/v2/shared"
	"google.golang.org/genai"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func captureLogger() (*slog.Logger, *bytes.Buffer) {
	var buf bytes.Buffer
	h := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	return slog.New(h), &buf
}

// reasoningInfo builds a known, reasoning-capable ModelInfo with the given
// style/ceiling/mandatory flags so the gate helpers treat it as known.
func reasoningInfo(style, maxEffort string, mandatory bool) ModelInfo {
	return ModelInfo{
		Cost:   Cost{Input: 1, Output: 1},
		Limits: Limits{Output: 10000},
		Caps: Capabilities{
			Temperature:        true,
			Reasoning:          true,
			ToolCall:           true,
			ReasoningStyle:     style,
			MaxEffort:          maxEffort,
			ReasoningMandatory: mandatory,
		},
		Family: "test",
	}
}

func anthropicTurnFor(info ModelInfo, opts ...GenerateOption) *anthropicTurn {
	GinkgoHelper()
	m := &anthropicModel{logger: discardLogger(), model: "m", statsModel: "anthropic/m", info: info}
	base := []GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}
	s, err := m.newSession(context.Background(), append(base, opts...)...)
	Expect(err).NotTo(HaveOccurred())
	return s.turn.(*anthropicTurn)
}

func googleTurnFor(info ModelInfo, opts ...GenerateOption) *googleTurn {
	GinkgoHelper()
	m := &googleModel{logger: discardLogger(), model: "m", statsModel: "google/m", info: info}
	base := []GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}
	s, err := m.newSession(context.Background(), append(base, opts...)...)
	Expect(err).NotTo(HaveOccurred())
	return s.turn.(*googleTurn)
}

func openaiTurnFor(info ModelInfo, opts ...GenerateOption) *openaiTurn {
	GinkgoHelper()
	m := &openaiModel{logger: discardLogger(), model: "m", statsModel: "openai/m", info: info}
	base := []GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}
	s, err := m.newSession(context.Background(), append(base, opts...)...)
	Expect(err).NotTo(HaveOccurred())
	return s.turn.(*openaiTurn)
}

func openrouterTurnFor(info ModelInfo, opts ...GenerateOption) *openrouterTurn {
	GinkgoHelper()
	m := &openrouterModel{logger: discardLogger(), model: "m", statsModel: "openrouter/m", info: info}
	base := []GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}
	s, err := m.newSession(context.Background(), append(base, opts...)...)
	Expect(err).NotTo(HaveOccurred())
	return s.turn.(*openrouterTurn)
}

var _ = Describe("reasoning effort helpers", func() {
	Describe("effortRequested", func() {
		It("treats only explicit low/medium/high as requested", func() {
			Expect(effortRequested("")).To(BeFalse())
			Expect(effortRequested(EffortNone)).To(BeFalse())
			Expect(effortRequested(EffortLow)).To(BeTrue())
			Expect(effortRequested(EffortMedium)).To(BeTrue())
			Expect(effortRequested(EffortHigh)).To(BeTrue())
		})
	})

	Describe("clampEffort", func() {
		It("clamps over the ceiling and warns", func() {
			logger, buf := captureLogger()
			Expect(clampEffort(EffortHigh, "low", logger)).To(Equal(EffortLow))
			Expect(buf.String()).To(ContainSubstring("Clamping reasoning effort"))
		})

		It("leaves an effort under the ceiling unchanged and silent", func() {
			logger, buf := captureLogger()
			Expect(clampEffort(EffortLow, "high", logger)).To(Equal(EffortLow))
			Expect(buf.Len()).To(BeZero())
		})

		It("is a no-op for an empty ceiling", func() {
			Expect(clampEffort(EffortHigh, "", discardLogger())).To(Equal(EffortHigh))
		})

		It("does not clamp high under a forward-looking max ceiling", func() {
			Expect(clampEffort(EffortHigh, "max", discardLogger())).To(Equal(EffortHigh))
		})
	})

	Describe("effortToBudget", func() {
		info := ModelInfo{Limits: Limits{Output: 10000}}

		It("returns zero for none", func() {
			Expect(effortToBudget(EffortNone, info)).To(Equal(0))
		})

		It("uses a fraction of the output limit", func() {
			Expect(effortToBudget(EffortHigh, info)).To(Equal(8000))
			Expect(effortToBudget(EffortMedium, info)).To(Equal(5000))
		})

		It("applies a 1024 floor for small output limits", func() {
			Expect(effortToBudget(EffortLow, ModelInfo{Limits: Limits{Output: 1000}})).To(Equal(1024))
		})

		It("falls back to fixed budgets when the output limit is unknown", func() {
			Expect(effortToBudget(EffortHigh, ModelInfo{})).To(Equal(16384))
		})
	})
})

var _ = Describe("resolveReasoningRoute", func() {
	var (
		budgetModel    = reasoningInfo(reasoningStyleBudget, "", false)
		effortModel    = reasoningInfo(reasoningStyleEffort, "high", false)
		mandatoryModel = reasoningInfo(reasoningStyleAdaptive, "max", true)
		nonReasoning   = ModelInfo{Cost: Cost{Input: 1, Output: 1}, Family: "x"}
	)

	It("lets an explicit budget win where the style supports budgets", func() {
		r := resolveReasoningRoute(&generateContentOptions{MaxThinkingTokens: 2048, ReasoningEffort: EffortHigh}, budgetModel, true, discardLogger())
		Expect(r.Kind).To(Equal(reasoningKindBudget))
		Expect(r.Budget).To(Equal(2048))
	})

	It("ignores a budget for control on effort-only models and warns", func() {
		logger, buf := captureLogger()
		r := resolveReasoningRoute(&generateContentOptions{MaxThinkingTokens: 2048, ReasoningEffort: EffortLow}, effortModel, false, logger)
		Expect(r.Kind).To(Equal(reasoningKindEffort))
		Expect(r.Effort).To(Equal(EffortLow))
		Expect(buf.String()).To(ContainSubstring("WithMaxThinkingTokens"))
	})

	It("lets effort drive, clamped to the model ceiling", func() {
		r := resolveReasoningRoute(&generateContentOptions{ReasoningEffort: EffortHigh}, reasoningInfo(reasoningStyleEffort, "medium", false), false, discardLogger())
		Expect(r.Kind).To(Equal(reasoningKindEffort))
		Expect(r.Effort).To(Equal(EffortMedium))
	})

	It("disables reasoning by default on a disableable model", func() {
		r := resolveReasoningRoute(&generateContentOptions{}, effortModel, false, discardLogger())
		Expect(r.Kind).To(Equal(reasoningKindDisable))
	})

	It("omits params on a mandatory model with no option", func() {
		r := resolveReasoningRoute(&generateContentOptions{}, mandatoryModel, false, discardLogger())
		Expect(r.Kind).To(Equal(reasoningKindMandatory))
	})

	It("warns when EffortNone is set explicitly on a mandatory model", func() {
		logger, buf := captureLogger()
		r := resolveReasoningRoute(&generateContentOptions{ReasoningEffort: EffortNone}, mandatoryModel, false, logger)
		Expect(r.Kind).To(Equal(reasoningKindMandatory))
		Expect(buf.String()).To(ContainSubstring("always reasons"))
	})

	It("sends nothing for a non-reasoning model", func() {
		r := resolveReasoningRoute(&generateContentOptions{}, nonReasoning, false, discardLogger())
		Expect(r.Kind).To(Equal(reasoningKindSkip))
	})

	It("honors an explicit effort request on an unknown model", func() {
		var unknown ModelInfo
		r := resolveReasoningRoute(&generateContentOptions{ReasoningEffort: EffortHigh}, unknown, false, discardLogger())
		Expect(r.Kind).To(Equal(reasoningKindEffort))
		Expect(r.Effort).To(Equal(EffortHigh))
	})
})

var _ = Describe("effort enum mappers", func() {
	It("maps to the Anthropic output-config effort enum", func() {
		Expect(anthropicOutputEffort(EffortLow)).To(Equal(anthropic.BetaOutputConfigEffortLow))
		Expect(anthropicOutputEffort(EffortHigh)).To(Equal(anthropic.BetaOutputConfigEffortHigh))
		Expect(anthropicOutputEffort("max")).To(Equal(anthropic.BetaOutputConfigEffortMax))
	})

	It("maps to the Gemini thinking-level enum", func() {
		Expect(googleThinkingLevel(EffortMedium)).To(Equal(genai.ThinkingLevelMedium))
		Expect(googleThinkingLevel(EffortHigh)).To(Equal(genai.ThinkingLevelHigh))
	})

	It("maps to the OpenAI reasoning-effort enum", func() {
		Expect(openaiReasoningEffort(EffortLow)).To(Equal(shared.ReasoningEffortLow))
		Expect(openaiReasoningEffort(EffortHigh)).To(Equal(shared.ReasoningEffortHigh))
	})
})

var _ = Describe("Anthropic reasoning request", func() {
	It("sets output-config effort and adaptive thinking on adaptive models", func() {
		turn := anthropicTurnFor(reasoningInfo(reasoningStyleAdaptive, "max", false), WithReasoningEffort(EffortHigh))
		Expect(turn.params.OutputConfig.Effort).To(Equal(anthropic.BetaOutputConfigEffortHigh))
		Expect(turn.params.Thinking.OfAdaptive).NotTo(BeNil())
	})

	It("sets output-config effort only on effort-style models", func() {
		turn := anthropicTurnFor(reasoningInfo(reasoningStyleEffort, "", false), WithReasoningEffort(EffortMedium))
		Expect(turn.params.OutputConfig.Effort).To(Equal(anthropic.BetaOutputConfigEffortMedium))
		Expect(turn.params.Thinking.OfAdaptive).To(BeNil())
		Expect(turn.params.Thinking.OfEnabled).To(BeNil())
	})

	It("enables budget thinking with temperature 1.0 on budget-style models", func() {
		turn := anthropicTurnFor(reasoningInfo(reasoningStyleBudget, "", false), WithMaxThinkingTokens(3000))
		Expect(turn.params.Thinking.OfEnabled).NotTo(BeNil())
		Expect(turn.params.Thinking.OfEnabled.BudgetTokens).To(Equal(int64(3000)))
		Expect(turn.params.Temperature.Or(0)).To(Equal(1.0))
	})

	It("emits no reasoning params by default on a disableable model", func() {
		turn := anthropicTurnFor(reasoningInfo(reasoningStyleAdaptive, "", false))
		Expect(turn.params.Thinking.OfAdaptive).To(BeNil())
		Expect(turn.params.Thinking.OfEnabled).To(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(BeEmpty())
	})

	It("emits no reasoning params on a mandatory model", func() {
		turn := anthropicTurnFor(reasoningInfo(reasoningStyleAdaptive, "max", true))
		Expect(turn.params.Thinking.OfAdaptive).To(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(BeEmpty())
	})
})

var _ = Describe("Google reasoning request", func() {
	It("maps effort to a thinking level on level-style models", func() {
		turn := googleTurnFor(reasoningInfo(reasoningStyleLevel, "", false), WithReasoningEffort(EffortHigh))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingLevel).To(Equal(genai.ThinkingLevelHigh))
	})

	It("disables level-style reasoning with the minimal level", func() {
		turn := googleTurnFor(reasoningInfo(reasoningStyleLevel, "", false))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingLevel).To(Equal(genai.ThinkingLevelMinimal))
	})

	It("disables budget-style reasoning with budget 0", func() {
		turn := googleTurnFor(reasoningInfo(reasoningStyleBudget, "", false))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingBudget).NotTo(BeNil())
		Expect(*turn.config.ThinkingConfig.ThinkingBudget).To(Equal(int32(0)))
	})

	It("uses the explicit budget on budget-style models", func() {
		turn := googleTurnFor(reasoningInfo(reasoningStyleBudget, "", false), WithMaxThinkingTokens(4096))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingBudget).NotTo(BeNil())
		Expect(*turn.config.ThinkingConfig.ThinkingBudget).To(Equal(int32(4096)))
	})
})

var _ = Describe("OpenAI reasoning request", func() {
	It("lets effort drive", func() {
		turn := openaiTurnFor(reasoningInfo(reasoningStyleEffort, "high", false), WithReasoningEffort(EffortMedium))
		Expect(turn.params.Reasoning.Effort).To(Equal(shared.ReasoningEffortMedium))
	})

	It("disables a disableable model with effort none", func() {
		turn := openaiTurnFor(reasoningInfo(reasoningStyleEffort, "high", false))
		Expect(turn.params.Reasoning.Effort).To(Equal(shared.ReasoningEffort("none")))
	})

	It("emits no reasoning param on a mandatory model", func() {
		turn := openaiTurnFor(reasoningInfo(reasoningStyleEffort, "high", true))
		Expect(turn.params.Reasoning.Effort).To(BeEmpty())
	})

	It("emits no reasoning param on a non-reasoning model", func() {
		turn := openaiTurnFor(ModelInfo{Cost: Cost{Input: 1, Output: 1}, Family: "x"})
		Expect(turn.params.Reasoning.Effort).To(BeEmpty())
	})
})

var _ = Describe("OpenRouter reasoning request", func() {
	It("lets effort drive", func() {
		turn := openrouterTurnFor(reasoningInfo(reasoningStyleEffort, "high", false), WithReasoningEffort(EffortHigh))
		Expect(turn.params.Reasoning).NotTo(BeNil())
		Expect(turn.params.Reasoning.Effort).NotTo(BeNil())
		Expect(*turn.params.Reasoning.Effort).To(Equal("high"))
	})

	It("lets an explicit budget win over effort without setting both", func() {
		turn := openrouterTurnFor(reasoningInfo(reasoningStyleEffort, "high", false), WithReasoningEffort(EffortHigh), WithMaxThinkingTokens(5000))
		Expect(turn.params.Reasoning).NotTo(BeNil())
		Expect(turn.params.Reasoning.MaxTokens).NotTo(BeNil())
		Expect(*turn.params.Reasoning.MaxTokens).To(Equal(5000))
		Expect(turn.params.Reasoning.Effort).To(BeNil())
	})

	It("disables a disableable model with enabled false", func() {
		turn := openrouterTurnFor(reasoningInfo(reasoningStyleEffort, "high", false))
		Expect(turn.params.Reasoning).NotTo(BeNil())
		Expect(turn.params.Reasoning.Enabled).NotTo(BeNil())
		Expect(*turn.params.Reasoning.Enabled).To(BeFalse())
	})

	It("emits no reasoning param on a mandatory model", func() {
		turn := openrouterTurnFor(reasoningInfo(reasoningStyleEffort, "high", true))
		Expect(turn.params.Reasoning).To(BeNil())
	})

	It("emits no reasoning param on a non-reasoning model", func() {
		turn := openrouterTurnFor(ModelInfo{Cost: Cost{Input: 1, Output: 1}, Family: "x"})
		Expect(turn.params.Reasoning).To(BeNil())
	})
})
