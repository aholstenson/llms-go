package llms

import "log/slog"

// Provider-default reasoning styles, used as the request-time fallback when a
// model carries no ReasoningStyle — i.e. a model absent from the embedded data.
// They mirror the gen-time defaults in internal/modelsdev so a known model and
// an unknown one from the same provider behave the same.
const (
	reasoningStyleBudget   = "budget"
	reasoningStyleEffort   = "effort"
	reasoningStyleAdaptive = "adaptive"
	reasoningStyleLevel    = "level"
)

// effortRank orders efforts so unsupported tiers can be clamped to the nearest
// supported one. Unknown strings rank -1. The range is wider than the public
// Effort enum so forward-looking tiers ("xhigh", "max") and provider-only
// values participate in clamping without code changes elsewhere.
func effortRank(e Effort) int {
	switch e {
	case EffortNone:
		return 0
	case "minimal":
		return 1
	case EffortLow:
		return 2
	case EffortMedium:
		return 3
	case EffortHigh:
		return 4
	case "xhigh":
		return 5
	case "max":
		return 6
	default:
		return -1
	}
}

// effortRequested reports whether the caller asked for reasoning via effort.
// The empty string ("no option given") and EffortNone are both "not requested".
func effortRequested(e Effort) bool {
	return e != "" && e != EffortNone
}

// clampEffort lowers an effort that exceeds the model's MaxEffort ceiling to
// that ceiling, warning when it does. An empty or unrecognized ceiling, or an
// unrecognized effort, is returned unchanged.
func clampEffort(e Effort, maxEffort string, logger *slog.Logger) Effort {
	if maxEffort == "" {
		return e
	}
	ceil := effortRank(Effort(maxEffort))
	if ceil < 0 {
		return e
	}
	if r := effortRank(e); r >= 0 && r > ceil {
		if logger != nil {
			logger.Warn("Clamping reasoning effort to model maximum",
				slog.String("requested", string(e)), slog.String("max", maxEffort))
		}
		return Effort(maxEffort)
	}
	return e
}

// effortForModel returns the effective effort to request and whether effort
// reasoning was requested at all. The returned effort is clamped to the model's
// MaxEffort ceiling.
func effortForModel(opts *generateContentOptions, info ModelInfo, logger *slog.Logger) (Effort, bool) {
	if !effortRequested(opts.ReasoningEffort) {
		return "", false
	}
	return clampEffort(opts.ReasoningEffort, info.Caps.MaxEffort, logger), true
}

// effortToBudget maps an effort to a thinking-token budget for budget-style
// models that have no native effort knob. It uses a fraction of the model's
// declared output limit with a 1024-token floor (the Anthropic minimum), or
// fixed budgets when the limit is unknown. Returns 0 for none/unrecognized.
func effortToBudget(e Effort, info ModelInfo) int {
	out := info.Limits.Output
	if out <= 0 {
		switch e {
		case EffortLow:
			return 4096
		case EffortMedium:
			return 8192
		case EffortHigh:
			return 16384
		default:
			return 0
		}
	}

	var frac float64
	switch e {
	case EffortLow:
		frac = 0.2
	case EffortMedium:
		frac = 0.5
	case EffortHigh:
		frac = 0.8
	default:
		return 0
	}

	budget := int(float64(out) * frac)
	if budget < 1024 {
		budget = 1024
	}
	return budget
}

// reasoningKind selects how a provider should configure reasoning for a
// request after precedence and the default-off policy have been resolved.
type reasoningKind int

const (
	// reasoningKindSkip: send no reasoning params (model cannot reason, or an
	// unknown model with no reasoning requested).
	reasoningKindSkip reasoningKind = iota
	// reasoningKindDisable: emit the provider's explicit disable form.
	reasoningKindDisable
	// reasoningKindMandatory: omit reasoning params; the model always reasons.
	reasoningKindMandatory
	// reasoningKindEffort: request reasoning at Effort. Budget-style providers
	// convert this to a budget via effortToBudget.
	reasoningKindEffort
	// reasoningKindBudget: request reasoning with an explicit token Budget.
	reasoningKindBudget
)

// reasoningRoute is the resolved, SDK-free reasoning decision for a request.
type reasoningRoute struct {
	Kind   reasoningKind
	Effort Effort // valid when Kind == reasoningKindEffort
	Budget int    // valid when Kind == reasoningKindBudget
}

// resolveReasoningRoute applies precedence and the default-off policy to
// produce a provider-independent reasoning decision.
//
// Precedence: an explicit WithMaxThinkingTokens budget wins where the model's
// style supports budgets (budgetSupported); otherwise WithReasoningEffort
// drives. On models whose style does not support budgets, a set budget is
// ignored for control (warn) — the caller may still use it to reserve output
// headroom.
//
// Default-off: with neither effort nor an applicable budget requested, a model
// that can disable reasoning gets reasoningKindDisable, a mandatory-reasoning
// model gets reasoningKindMandatory, and a non-reasoning or unknown model gets
// reasoningKindSkip.
func resolveReasoningRoute(opts *generateContentOptions, info ModelInfo, budgetSupported bool, logger *slog.Logger) reasoningRoute {
	wantBudget := opts.MaxThinkingTokens > 0

	if wantBudget && budgetSupported && info.allowsReasoning() {
		return reasoningRoute{Kind: reasoningKindBudget, Budget: opts.MaxThinkingTokens}
	}
	if wantBudget && !budgetSupported && logger != nil {
		logger.Warn("WithMaxThinkingTokens does not control reasoning for this model; use WithReasoningEffort. The value reserves output headroom only.")
	}

	if effort, ok := effortForModel(opts, info, logger); ok && info.allowsReasoning() {
		return reasoningRoute{Kind: reasoningKindEffort, Effort: effort}
	}

	// No reasoning requested: apply the default-off policy.
	if !info.Caps.Reasoning {
		// Non-reasoning or unknown model: send nothing.
		return reasoningRoute{Kind: reasoningKindSkip}
	}
	if info.Caps.ReasoningMandatory {
		if opts.ReasoningEffort == EffortNone && logger != nil {
			logger.Warn("Model always reasons; ignoring WithReasoningEffort(EffortNone)")
		}
		return reasoningRoute{Kind: reasoningKindMandatory}
	}
	return reasoningRoute{Kind: reasoningKindDisable}
}
