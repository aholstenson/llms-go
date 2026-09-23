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
// reasoning controls so the gate helpers treat it as known.
func reasoningInfo(caps Capabilities) ModelInfo {
	caps.Temperature = true
	caps.Reasoning = true
	caps.ToolCall = true
	caps.StructuredOutput = true
	return ModelInfo{
		Cost:   Cost{Input: 1, Output: 1},
		Limits: Limits{Output: 10000},
		Caps:   caps,
		Family: "test",
	}
}

var (
	lowToHigh = []Effort{EffortLow, EffortMedium, EffortHigh}
	lowToMax  = []Effort{EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax}
	// Model shapes as derived from models.dev and the Anthropic Models API.
	adaptiveModel   = reasoningInfo(Capabilities{AdaptiveThinking: true, ReasoningEfforts: lowToMax})
	alwaysReasons   = reasoningInfo(Capabilities{AdaptiveThinking: true, ReasoningEfforts: lowToMax, ReasoningMandatory: true})
	budgetOnlyModel = reasoningInfo(Capabilities{ReasoningBudget: &TokenRange{Min: 1024}})
	budgetAndEffort = reasoningInfo(Capabilities{ReasoningBudget: &TokenRange{Min: 1024}, ReasoningEfforts: lowToHigh})
	effortModel     = reasoningInfo(Capabilities{ReasoningEfforts: lowToHigh})
	nonReasoning    = ModelInfo{Cost: Cost{Input: 1, Output: 1}, Caps: Capabilities{StructuredOutput: true}, Family: "x"}
)

func anthropicTurnFor(info ModelInfo, opts ...GenerateOption) *anthropicTurn {
	GinkgoHelper()
	m := &anthropicModel{logger: discardLogger(), model: "m", statsModel: "anthropic/m", info: fixedModelInfo(info)}
	base := []GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}
	s, err := m.newSession(context.Background(), append(base, opts...)...)
	Expect(err).NotTo(HaveOccurred())
	return s.turn.(*anthropicTurn)
}

func googleTurnFor(info ModelInfo, opts ...GenerateOption) *googleTurn {
	GinkgoHelper()
	m := &googleModel{logger: discardLogger(), model: "m", statsModel: "google/m", info: fixedModelInfo(info)}
	base := []GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}
	s, err := m.newSession(context.Background(), append(base, opts...)...)
	Expect(err).NotTo(HaveOccurred())
	return s.turn.(*googleTurn)
}

func openaiTurnFor(info ModelInfo, opts ...GenerateOption) *openaiTurn {
	GinkgoHelper()
	m := &openaiModel{logger: discardLogger(), model: "m", statsModel: "openai/m", info: fixedModelInfo(info)}
	base := []GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}
	s, err := m.newSession(context.Background(), append(base, opts...)...)
	Expect(err).NotTo(HaveOccurred())
	return s.turn.(*openaiTurn)
}

func openrouterTurnFor(info ModelInfo, opts ...GenerateOption) *openrouterTurn {
	GinkgoHelper()
	m := &openrouterModel{logger: discardLogger(), model: "m", statsModel: "openrouter/m", info: fixedModelInfo(info)}
	base := []GenerateOption{WithMessages(NewMessage(RoleUser, NewTextPart("hi")))}
	s, err := m.newSession(context.Background(), append(base, opts...)...)
	Expect(err).NotTo(HaveOccurred())
	return s.turn.(*openrouterTurn)
}

var _ = Describe("reasoning option helpers", func() {
	Describe("reasoningMode", func() {
		mode := func(opts ...GenerateOption) ReasoningMode {
			GinkgoHelper()
			o := &generateContentOptions{}
			for _, opt := range opts {
				Expect(opt(o)).To(Succeed())
			}
			return o.reasoningMode()
		}

		It("uses the provider default without options", func() {
			Expect(mode()).To(Equal(ReasoningDefault))
		})

		It("turns reasoning off for EffortNone", func() {
			Expect(mode(WithReasoningEffort(EffortNone))).To(Equal(ReasoningOff))
		})

		It("turns reasoning on for an effort or a budget", func() {
			Expect(mode(WithReasoningEffort(EffortLow))).To(Equal(ReasoningOn))
			Expect(mode(WithMaxThinkingTokens(2048))).To(Equal(ReasoningOn))
		})

		It("lets the last option win", func() {
			Expect(mode(WithReasoningEffort(EffortHigh), WithReasoning(ReasoningOff))).To(Equal(ReasoningOff))
			Expect(mode(WithReasoning(ReasoningOff), WithReasoningEffort(EffortHigh))).To(Equal(ReasoningOn))
			Expect(mode(WithReasoningEffort(EffortHigh), WithReasoning(ReasoningDefault))).To(Equal(ReasoningDefault))
		})

		It("keeps the effort when reasoning is turned on", func() {
			o := &generateContentOptions{}
			Expect(WithReasoningEffort(EffortHigh)(o)).To(Succeed())
			Expect(WithReasoning(ReasoningOn)(o)).To(Succeed())
			Expect(o.ReasoningEffort).To(Equal(EffortHigh))
		})

		It("lets an explicit off win over a budget", func() {
			Expect(mode(WithMaxThinkingTokens(2048), WithReasoning(ReasoningOff))).To(Equal(ReasoningOff))
		})
	})

	Describe("clampEffort", func() {
		It("moves an effort above the highest tier down and warns", func() {
			logger, buf := captureLogger()
			Expect(clampEffort(EffortMax, effortModel, logger)).To(Equal(EffortHigh))
			Expect(buf.String()).To(ContainSubstring("nearest supported effort"))
		})

		It("moves an effort below the lowest tier up", func() {
			Expect(clampEffort(EffortMinimal, effortModel, discardLogger())).To(Equal(EffortLow))
		})

		It("picks the nearest tier inside a gap, preferring the lower tier on a tie", func() {
			gappy := reasoningInfo(Capabilities{ReasoningEfforts: []Effort{EffortMinimal, EffortHigh}})
			Expect(clampEffort(EffortMedium, gappy, discardLogger())).To(Equal(EffortHigh))
			Expect(clampEffort(EffortLow, gappy, discardLogger())).To(Equal(EffortMinimal))
		})

		It("leaves an accepted effort unchanged and silent", func() {
			logger, buf := captureLogger()
			Expect(clampEffort(EffortLow, effortModel, logger)).To(Equal(EffortLow))
			Expect(buf.Len()).To(BeZero())
		})

		It("is a no-op when the tiers are unknown", func() {
			Expect(clampEffort(EffortMax, budgetOnlyModel, discardLogger())).To(Equal(EffortMax))
			Expect(clampEffort("", effortModel, discardLogger())).To(Equal(Effort("")))
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

		It("maps an empty effort like medium", func() {
			Expect(effortToBudget("", info)).To(Equal(5000))
		})

		It("stays below the output limit for max", func() {
			Expect(effortToBudget(EffortMax, info)).To(BeNumerically("<", 10000))
		})

		It("applies a 1024 floor for small output limits", func() {
			Expect(effortToBudget(EffortLow, ModelInfo{Limits: Limits{Output: 1000}})).To(Equal(1024))
		})

		It("keeps the budget inside the model's budget range", func() {
			ranged := ModelInfo{Limits: Limits{Output: 65536}, Caps: Capabilities{ReasoningBudget: &TokenRange{Min: 128, Max: 32768}}}
			Expect(effortToBudget(EffortHigh, ranged)).To(Equal(32768))
			small := ModelInfo{Limits: Limits{Output: 100}, Caps: Capabilities{ReasoningBudget: &TokenRange{Min: 128}}}
			Expect(effortToBudget(EffortLow, small)).To(Equal(128))
		})

		It("falls back to fixed budgets when the output limit is unknown", func() {
			Expect(effortToBudget(EffortHigh, ModelInfo{})).To(Equal(16384))
		})
	})
})

var _ = Describe("resolveReasoningRoute", func() {
	resolve := func(info ModelInfo, budgetSupported bool, opts ...GenerateOption) reasoningRoute {
		GinkgoHelper()
		o := &generateContentOptions{}
		for _, opt := range opts {
			Expect(opt(o)).To(Succeed())
		}
		return resolveReasoningRoute(o, info, budgetSupported, discardLogger())
	}

	It("sends nothing by default, even on a model that always reasons", func() {
		Expect(resolve(effortModel, false).Kind).To(Equal(reasoningKindDefault))
		Expect(resolve(alwaysReasons, false).Kind).To(Equal(reasoningKindDefault))
	})

	It("lets an explicit budget win where the model takes budgets", func() {
		r := resolve(budgetOnlyModel, true, WithMaxThinkingTokens(2048), WithReasoningEffort(EffortHigh))
		Expect(r.Kind).To(Equal(reasoningKindBudget))
		Expect(r.Budget).To(Equal(2048))
	})

	It("ignores a budget for control on effort-only models and warns", func() {
		logger, buf := captureLogger()
		o := &generateContentOptions{MaxThinkingTokens: 2048, ReasoningEffort: EffortLow}
		r := resolveReasoningRoute(o, effortModel, false, logger)
		Expect(r.Kind).To(Equal(reasoningKindEffort))
		Expect(r.Effort).To(Equal(EffortLow))
		Expect(buf.String()).To(ContainSubstring("WithMaxThinkingTokens"))
	})

	It("moves the effort to a tier the model accepts", func() {
		r := resolve(effortModel, false, WithReasoningEffort(EffortMax))
		Expect(r.Kind).To(Equal(reasoningKindEffort))
		Expect(r.Effort).To(Equal(EffortHigh))
	})

	It("turns reasoning on without an effort", func() {
		r := resolve(effortModel, false, WithReasoning(ReasoningOn))
		Expect(r.Kind).To(Equal(reasoningKindEffort))
		Expect(r.Effort).To(BeEmpty())
	})

	It("disables reasoning on a model that can turn it off", func() {
		Expect(resolve(effortModel, false, WithReasoning(ReasoningOff)).Kind).To(Equal(reasoningKindDisable))
	})

	It("uses the lowest effort when a model cannot turn reasoning off", func() {
		logger, buf := captureLogger()
		r := resolveReasoningRoute(&generateContentOptions{ReasoningEffort: EffortNone}, alwaysReasons, false, logger)
		Expect(r.Kind).To(Equal(reasoningKindEffort))
		Expect(r.Effort).To(Equal(EffortLow))
		Expect(buf.String()).To(ContainSubstring("cannot turn reasoning off"))
	})

	It("omits params when a model cannot turn reasoning off and has no effort tiers", func() {
		logger, buf := captureLogger()
		mandatoryBudget := reasoningInfo(Capabilities{ReasoningBudget: &TokenRange{Min: 128}, ReasoningMandatory: true})
		r := resolveReasoningRoute(&generateContentOptions{ReasoningMode: ReasoningOff}, mandatoryBudget, true, logger)
		Expect(r.Kind).To(Equal(reasoningKindMandatory))
		Expect(buf.String()).To(ContainSubstring("cannot turn reasoning off"))
	})

	It("sends nothing for a non-reasoning model in every mode", func() {
		Expect(resolve(nonReasoning, false).Kind).To(Equal(reasoningKindSkip))
		Expect(resolve(nonReasoning, false, WithReasoningEffort(EffortHigh)).Kind).To(Equal(reasoningKindSkip))
		Expect(resolve(nonReasoning, false, WithReasoning(ReasoningOff)).Kind).To(Equal(reasoningKindSkip))
	})

	It("honors an effort on an unknown model but does not send a disable form", func() {
		var unknown ModelInfo
		r := resolve(unknown, false, WithReasoningEffort(EffortHigh))
		Expect(r.Kind).To(Equal(reasoningKindEffort))
		Expect(r.Effort).To(Equal(EffortHigh))
		Expect(resolve(unknown, false, WithReasoning(ReasoningOff)).Kind).To(Equal(reasoningKindSkip))
		Expect(resolve(unknown, false).Kind).To(Equal(reasoningKindDefault))
	})
})

var _ = Describe("effort enum mappers", func() {
	It("maps to the Anthropic output-config effort enum", func() {
		Expect(anthropicOutputEffort(EffortLow)).To(Equal(anthropic.BetaOutputConfigEffortLow))
		Expect(anthropicOutputEffort(EffortHigh)).To(Equal(anthropic.BetaOutputConfigEffortHigh))
		Expect(anthropicOutputEffort(EffortXHigh)).To(Equal(anthropic.BetaOutputConfigEffort("xhigh")))
		Expect(anthropicOutputEffort(EffortMax)).To(Equal(anthropic.BetaOutputConfigEffortMax))
	})

	It("maps to the Gemini thinking-level enum", func() {
		Expect(googleThinkingLevel(EffortMinimal)).To(Equal(genai.ThinkingLevelMinimal))
		Expect(googleThinkingLevel(EffortMedium)).To(Equal(genai.ThinkingLevelMedium))
		Expect(googleThinkingLevel(EffortHigh)).To(Equal(genai.ThinkingLevelHigh))
	})

	It("maps to the OpenAI reasoning-effort enum", func() {
		Expect(openaiReasoningEffort(EffortMinimal)).To(Equal(shared.ReasoningEffortMinimal))
		Expect(openaiReasoningEffort(EffortLow)).To(Equal(shared.ReasoningEffortLow))
		Expect(openaiReasoningEffort(EffortHigh)).To(Equal(shared.ReasoningEffortHigh))
		Expect(openaiReasoningEffort(EffortXHigh)).To(Equal(shared.ReasoningEffort("xhigh")))
	})
})

var _ = Describe("Anthropic reasoning request", func() {
	It("sets adaptive thinking and the effort on adaptive models", func() {
		turn := anthropicTurnFor(adaptiveModel, WithReasoningEffort(EffortHigh))
		Expect(turn.params.Thinking.OfAdaptive).NotTo(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(Equal(anthropic.BetaOutputConfigEffortHigh))
	})

	It("sets adaptive thinking without an effort when reasoning is on", func() {
		turn := anthropicTurnFor(adaptiveModel, WithReasoning(ReasoningOn))
		Expect(turn.params.Thinking.OfAdaptive).NotTo(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(BeEmpty())
	})

	It("sends a budget and the effort on models that take both", func() {
		turn := anthropicTurnFor(budgetAndEffort, WithReasoningEffort(EffortMedium))
		Expect(turn.params.Thinking.OfEnabled).NotTo(BeNil())
		Expect(turn.params.Thinking.OfEnabled.BudgetTokens).To(Equal(int64(5000)))
		Expect(turn.params.OutputConfig.Effort).To(Equal(anthropic.BetaOutputConfigEffortMedium))
		Expect(turn.params.Temperature.Or(0)).To(Equal(1.0))
	})

	It("sends only a budget on budget-only models", func() {
		turn := anthropicTurnFor(budgetOnlyModel, WithReasoningEffort(EffortMedium))
		Expect(turn.params.Thinking.OfEnabled).NotTo(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(BeEmpty())
	})

	It("uses an explicit budget with temperature 1.0 on budget models", func() {
		turn := anthropicTurnFor(budgetOnlyModel, WithMaxThinkingTokens(3000))
		Expect(turn.params.Thinking.OfEnabled).NotTo(BeNil())
		Expect(turn.params.Thinking.OfEnabled.BudgetTokens).To(Equal(int64(3000)))
		Expect(turn.params.Temperature.Or(0)).To(Equal(1.0))
	})

	It("emits no reasoning params by default", func() {
		turn := anthropicTurnFor(adaptiveModel)
		Expect(turn.params.Thinking.OfAdaptive).To(BeNil())
		Expect(turn.params.Thinking.OfDisabled).To(BeNil())
		Expect(turn.params.Thinking.OfEnabled).To(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(BeEmpty())
	})

	It("sends the disable form on adaptive models when reasoning is off", func() {
		turn := anthropicTurnFor(adaptiveModel, WithReasoning(ReasoningOff))
		Expect(turn.params.Thinking.OfDisabled).NotTo(BeNil())
	})

	It("omits thinking on budget-only models when reasoning is off", func() {
		turn := anthropicTurnFor(budgetOnlyModel, WithReasoning(ReasoningOff))
		Expect(turn.params.Thinking.OfDisabled).To(BeNil())
		Expect(turn.params.Thinking.OfEnabled).To(BeNil())
	})

	It("never sends the disable form to a model that always reasons", func() {
		turn := anthropicTurnFor(alwaysReasons, WithReasoning(ReasoningOff))
		Expect(turn.params.Thinking.OfDisabled).To(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(Equal(anthropic.BetaOutputConfigEffortLow))
	})

	It("uses budget thinking on known models without listed controls", func() {
		turn := anthropicTurnFor(reasoningInfo(Capabilities{}), WithReasoningEffort(EffortLow))
		Expect(turn.params.Thinking.OfEnabled).NotTo(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(BeEmpty())
	})

	It("treats unknown models as adaptive", func() {
		turn := anthropicTurnFor(ModelInfo{}, WithReasoningEffort(EffortHigh))
		Expect(turn.params.Thinking.OfAdaptive).NotTo(BeNil())
		Expect(turn.params.OutputConfig.Effort).To(Equal(anthropic.BetaOutputConfigEffortHigh))
	})
})

var _ = Describe("Google reasoning request", func() {
	levelModel := reasoningInfo(Capabilities{ReasoningEfforts: []Effort{EffortMinimal, EffortLow, EffortMedium, EffortHigh}})
	levelAlwaysReasons := reasoningInfo(Capabilities{ReasoningEfforts: lowToHigh, ReasoningMandatory: true})

	It("maps effort to a thinking level on level models", func() {
		turn := googleTurnFor(levelModel, WithReasoningEffort(EffortHigh))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingLevel).To(Equal(genai.ThinkingLevelHigh))
	})

	It("turns reasoning on at the default depth without an effort", func() {
		turn := googleTurnFor(levelModel, WithReasoning(ReasoningOn))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.IncludeThoughts).To(BeTrue())
		Expect(turn.config.ThinkingConfig.ThinkingLevel).To(BeEmpty())
		Expect(turn.config.ThinkingConfig.ThinkingBudget).To(BeNil())
	})

	It("sends no thinking config by default", func() {
		Expect(googleTurnFor(levelModel).config.ThinkingConfig).To(BeNil())
		Expect(googleTurnFor(budgetOnlyModel).config.ThinkingConfig).To(BeNil())
	})

	It("turns level models off with the minimal level", func() {
		turn := googleTurnFor(levelModel, WithReasoning(ReasoningOff))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingLevel).To(Equal(genai.ThinkingLevelMinimal))
	})

	It("uses the lowest level when a level model cannot turn reasoning off", func() {
		turn := googleTurnFor(levelAlwaysReasons, WithReasoning(ReasoningOff))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingLevel).To(Equal(genai.ThinkingLevelLow))
	})

	It("turns budget models off with budget 0", func() {
		turn := googleTurnFor(budgetOnlyModel, WithReasoning(ReasoningOff))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingBudget).NotTo(BeNil())
		Expect(*turn.config.ThinkingConfig.ThinkingBudget).To(Equal(int32(0)))
	})

	It("uses the explicit budget on budget models", func() {
		turn := googleTurnFor(budgetOnlyModel, WithMaxThinkingTokens(4096))
		Expect(turn.config.ThinkingConfig).NotTo(BeNil())
		Expect(turn.config.ThinkingConfig.ThinkingBudget).NotTo(BeNil())
		Expect(*turn.config.ThinkingConfig.ThinkingBudget).To(Equal(int32(4096)))
	})
})

var _ = Describe("OpenAI reasoning request", func() {
	gpt5 := reasoningInfo(Capabilities{ReasoningEfforts: []Effort{EffortMinimal, EffortLow, EffortMedium, EffortHigh}, ReasoningMandatory: true})

	It("lets effort drive and asks for encrypted reasoning", func() {
		turn := openaiTurnFor(effortModel, WithReasoningEffort(EffortMedium))
		Expect(turn.params.Reasoning.Effort).To(Equal(shared.ReasoningEffortMedium))
		Expect(turn.params.Include).NotTo(BeEmpty())
	})

	It("uses medium when reasoning is on without an effort", func() {
		turn := openaiTurnFor(effortModel, WithReasoning(ReasoningOn))
		Expect(turn.params.Reasoning.Effort).To(Equal(shared.ReasoningEffortMedium))
	})

	It("sends effort none to turn a model off", func() {
		turn := openaiTurnFor(effortModel, WithReasoningEffort(EffortNone))
		Expect(turn.params.Reasoning.Effort).To(Equal(shared.ReasoningEffort("none")))
	})

	It("uses the lowest effort when a model cannot turn reasoning off", func() {
		turn := openaiTurnFor(gpt5, WithReasoning(ReasoningOff))
		Expect(turn.params.Reasoning.Effort).To(Equal(shared.ReasoningEffortMinimal))
	})

	It("sends no effort by default but keeps encrypted reasoning", func() {
		turn := openaiTurnFor(effortModel)
		Expect(turn.params.Reasoning.Effort).To(BeEmpty())
		Expect(turn.params.Include).NotTo(BeEmpty())
	})

	It("emits no reasoning param or include on a non-reasoning model", func() {
		turn := openaiTurnFor(nonReasoning, WithReasoningEffort(EffortHigh))
		Expect(turn.params.Reasoning.Effort).To(BeEmpty())
		Expect(turn.params.Include).To(BeEmpty())
	})
})

var _ = Describe("OpenRouter reasoning request", func() {
	It("lets effort drive", func() {
		turn := openrouterTurnFor(effortModel, WithReasoningEffort(EffortHigh))
		Expect(turn.params.Reasoning).NotTo(BeNil())
		Expect(turn.params.Reasoning.Effort).NotTo(BeNil())
		Expect(*turn.params.Reasoning.Effort).To(Equal("high"))
	})

	It("enables reasoning without an effort when reasoning is on", func() {
		turn := openrouterTurnFor(effortModel, WithReasoning(ReasoningOn))
		Expect(turn.params.Reasoning).NotTo(BeNil())
		Expect(turn.params.Reasoning.Enabled).NotTo(BeNil())
		Expect(*turn.params.Reasoning.Enabled).To(BeTrue())
		Expect(turn.params.Reasoning.Effort).To(BeNil())
	})

	It("lets an explicit budget win over effort without setting both", func() {
		turn := openrouterTurnFor(effortModel, WithReasoningEffort(EffortHigh), WithMaxThinkingTokens(5000))
		Expect(turn.params.Reasoning).NotTo(BeNil())
		Expect(turn.params.Reasoning.MaxTokens).NotTo(BeNil())
		Expect(*turn.params.Reasoning.MaxTokens).To(Equal(5000))
		Expect(turn.params.Reasoning.Effort).To(BeNil())
	})

	It("disables a model with enabled false", func() {
		turn := openrouterTurnFor(effortModel, WithReasoning(ReasoningOff))
		Expect(turn.params.Reasoning).NotTo(BeNil())
		Expect(turn.params.Reasoning.Enabled).NotTo(BeNil())
		Expect(*turn.params.Reasoning.Enabled).To(BeFalse())
	})

	It("emits no reasoning param by default", func() {
		Expect(openrouterTurnFor(effortModel).params.Reasoning).To(BeNil())
	})

	It("emits no reasoning param on a non-reasoning model", func() {
		Expect(openrouterTurnFor(nonReasoning, WithReasoningEffort(EffortHigh)).params.Reasoning).To(BeNil())
	})
})
