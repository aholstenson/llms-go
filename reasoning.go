package llms

import "log/slog"

// effortRank orders efforts so unsupported tiers can be moved to the nearest
// supported one. Unknown strings rank -1.
func effortRank(e Effort) int {
	switch e {
	case EffortNone:
		return 0
	case EffortMinimal:
		return 1
	case EffortLow:
		return 2
	case EffortMedium:
		return 3
	case EffortHigh:
		return 4
	case EffortXHigh:
		return 5
	case EffortMax:
		return 6
	default:
		return -1
	}
}

// clampEffort moves an effort the model does not accept to the nearest tier
// it does accept, warning when it does. When two tiers are equally near, the
// lower one wins. An empty effort, an unrecognized effort, and a model with
// no known tiers are returned unchanged.
func clampEffort(e Effort, info ModelInfo, logger *slog.Logger) Effort {
	accepted := info.Caps.ReasoningEfforts
	rank := effortRank(e)
	if len(accepted) == 0 || rank < 0 || info.acceptsEffort(e) {
		return e
	}

	best, bestDist := accepted[0], -1
	for _, a := range accepted {
		dist := effortRank(a) - rank
		if dist < 0 {
			dist = -dist
		}
		if bestDist < 0 || dist < bestDist {
			best, bestDist = a, dist
		}
	}
	if logger != nil {
		logger.Warn("Reasoning effort not supported by model, using nearest supported effort",
			slog.String("requested", string(e)), slog.String("effort", string(best)))
	}
	return best
}

// effortToBudget maps an effort to a thinking-token budget for budget-style
// models that have no native effort knob. It uses a fraction of the model's
// declared output limit, or fixed budgets when the limit is unknown, and keeps
// the result inside the model's budget range (at least 1024, the Anthropic
// minimum, when no range is known). An empty effort maps like EffortMedium.
// Returns 0 for none/unrecognized.
func effortToBudget(e Effort, info ModelInfo) int {
	if e == "" {
		e = EffortMedium
	}

	var frac float64
	var fixed int
	switch e {
	case EffortMinimal:
		frac, fixed = 0.1, 1024
	case EffortLow:
		frac, fixed = 0.2, 4096
	case EffortMedium:
		frac, fixed = 0.5, 8192
	case EffortHigh:
		frac, fixed = 0.8, 16384
	case EffortXHigh:
		frac, fixed = 0.85, 24576
	case EffortMax:
		// Stay below the output limit: Anthropic rejects a budget that is not
		// smaller than max_tokens.
		frac, fixed = 0.9, 32768
	default:
		return 0
	}

	budget := fixed
	if out := info.Limits.Output; out > 0 {
		budget = int(float64(out) * frac)
	}

	minBudget, maxBudget := anthropicMinThinkingBudget, 0
	if r := info.Caps.ReasoningBudget; r != nil {
		minBudget, maxBudget = max(r.Min, 1), r.Max
	}
	if budget < minBudget {
		budget = minBudget
	}
	if maxBudget > 0 && budget > maxBudget {
		budget = maxBudget
	}
	return budget
}

// reasoningMode returns the effective reasoning mode of a request. An effort
// wins over WithReasoning because WithReasoning clears the effort when it
// selects another mode. A thinking budget alone turns reasoning on.
func (o *generateContentOptions) reasoningMode() ReasoningMode {
	switch {
	case o.ReasoningEffort == EffortNone:
		return ReasoningOff
	case o.ReasoningEffort != "":
		return ReasoningOn
	case o.ReasoningMode == ReasoningDefault && o.MaxThinkingTokens > 0:
		return ReasoningOn
	default:
		return o.ReasoningMode
	}
}

// reasoningKind selects how a provider should configure reasoning for a
// request after the reasoning mode and precedence have been resolved.
type reasoningKind int

const (
	// reasoningKindSkip: send no reasoning params (model cannot reason, or an
	// unknown model that was asked to turn reasoning off).
	reasoningKindSkip reasoningKind = iota
	// reasoningKindDefault: send no reasoning params so the provider's
	// default applies.
	reasoningKindDefault
	// reasoningKindDisable: emit the provider's explicit disable form.
	reasoningKindDisable
	// reasoningKindMandatory: omit reasoning params; the model always reasons
	// and has no lower effort tier to fall back to.
	reasoningKindMandatory
	// reasoningKindEffort: request reasoning at Effort. An empty Effort means
	// reasoning on at the provider's default depth. Budget-style providers
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

// resolveReasoningRoute turns the request's reasoning options into a
// provider-independent reasoning decision.
//
// ReasoningDefault sends nothing. ReasoningOff disables reasoning where the
// model can turn it off; a model that always reasons gets its lowest effort
// tier instead (or no params when it has none), with a warning. ReasoningOn
// uses an explicit WithMaxThinkingTokens budget where the model takes budgets
// (budgetSupported), otherwise the effort, moved to the nearest tier the model
// accepts. On models that do not take budgets, a set budget is ignored for
// control (warn) — the caller may still use it to reserve output headroom.
func resolveReasoningRoute(opts *generateContentOptions, info ModelInfo, budgetSupported bool, logger *slog.Logger) reasoningRoute {
	if !info.allowsReasoning() {
		return reasoningRoute{Kind: reasoningKindSkip}
	}

	switch opts.reasoningMode() {
	case ReasoningOff:
		if !info.Caps.Reasoning {
			// Unknown model: a disable form could be rejected by a model
			// that does not reason, so send nothing.
			return reasoningRoute{Kind: reasoningKindSkip}
		}
		if !info.Caps.ReasoningMandatory {
			return reasoningRoute{Kind: reasoningKindDisable}
		}
		if lowest := info.minEffort(); lowest != "" {
			if logger != nil {
				logger.Warn("Model cannot turn reasoning off; using its lowest reasoning effort",
					slog.String("effort", string(lowest)))
			}
			return reasoningRoute{Kind: reasoningKindEffort, Effort: lowest}
		}
		if logger != nil {
			logger.Warn("Model cannot turn reasoning off; ignoring the request to turn it off")
		}
		return reasoningRoute{Kind: reasoningKindMandatory}

	case ReasoningOn:
		if opts.MaxThinkingTokens > 0 {
			if budgetSupported {
				return reasoningRoute{Kind: reasoningKindBudget, Budget: opts.MaxThinkingTokens}
			}
			if logger != nil {
				logger.Warn("WithMaxThinkingTokens does not control reasoning for this model; use WithReasoningEffort. The value reserves output headroom only.")
			}
		}
		return reasoningRoute{Kind: reasoningKindEffort, Effort: clampEffort(opts.ReasoningEffort, info, logger)}

	default:
		return reasoningRoute{Kind: reasoningKindDefault}
	}
}
