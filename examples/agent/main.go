// Command agent demonstrates the driveable Session API: the caller advances
// the agentic loop one model turn at a time, inspects what the model did,
// can Inject operator steering between turns, and uses the lightweight
// sub-agent pattern (a tool whose Execute calls another model with
// WithParentExecution so its token/step spend rolls up to the parent).
//
// Run with:
//
//	export ANTHROPIC_API_TOKEN=...      # or the token for your provider
//	go run ./examples/agent
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"

	llms "github.com/aholstenson/llms-go"
)

func modelName() string {
	if m := os.Getenv("LLM_EXAMPLE_MODEL"); m != "" {
		return m
	}
	return "anthropic/claude-sonnet-4-5"
}

// ResearchInput is the research tool's argument schema.
type ResearchInput struct {
	Topic string `json:"topic" jsonschema:"description=A focused question to research"`
}

// researchTool is a sub-agent: its Execute spins up a child generation that
// shares the parent's budget via WithParentExecution. Child token/step/tool
// spend accrues to the parent ExecutionContext.
type researchTool struct {
	child llms.Model
}

func (researchTool) Name() string { return "research" }

func (researchTool) Description() string {
	return "Delegate a focused question to a research sub-agent and get a short answer."
}

func (researchTool) Schema() *ResearchInput { return &ResearchInput{} }

func (t researchTool) Execute(ctx context.Context, in *ResearchInput) (string, error) {
	parent := llms.GetExecutionContext(ctx)

	maxSteps := 2
	if parent != nil {
		if r := parent.RemainingSteps(); r > 0 && r < maxSteps {
			maxSteps = r
		}
	}

	res, err := t.child.GenerateContent(ctx,
		llms.WithMessages(llms.NewMessage(llms.RoleUser,
			llms.NewTextPart("Answer in two sentences: "+in.Topic))),
		llms.WithMaxSteps(maxSteps),
		llms.WithMaxOutputTokens(256),
		llms.WithParentExecution(parent),
	)
	if err != nil {
		return "", err
	}
	if text, ok := res.(llms.TextResult); ok {
		return text.Text, nil
	}
	return "", nil
}

func (researchTool) ToString(out string) string { return out }

// DeployInput is the deploy tool's argument schema.
type DeployInput struct {
	Service string `json:"service" jsonschema:"description=The service to deploy"`
}

type deployTool struct{}

func (deployTool) Name() string               { return "deploy" }
func (deployTool) Description() string        { return "Deploy a service to production." }
func (deployTool) Schema() *DeployInput       { return &DeployInput{} }
func (deployTool) ToString(out string) string { return out }
func (deployTool) Execute(_ context.Context, in *DeployInput) (string, error) {
	fmt.Println("deploying", in.Service)
	return "deployed " + in.Service, nil
}

// needsApproval reports whether any tool call this turn is operator-gated.
func needsApproval(calls []llms.ToolCall) bool {
	for _, c := range calls {
		if c.Name == "deploy" {
			return true
		}
	}
	return false
}

func main() {
	ctx := context.Background()

	manager := llms.NewManager()

	model, err := manager.GetModel(modelName())
	if err != nil {
		log.Fatalf("could not load model: %v", err)
	}

	s, err := llms.NewSession(model,
		llms.WithSystemPrompt("You are an operations assistant. Use tools when helpful."),
		llms.WithMessages(llms.NewMessage(llms.RoleUser, llms.NewTextPart(
			"Research what a blue-green deployment is, then deploy the 'billing' service."))),
		llms.WithTools(
			llms.NewToolDef(researchTool{child: model}),
			llms.NewToolDef(deployTool{}),
		),
		llms.WithMaxSteps(8),
		llms.WithMaxOutputTokens(512),
	)
	if err != nil {
		log.Fatalf("could not create session: %v", err)
	}

	var lastExec llms.ExecutionContext
	for {
		// StepPlan runs the model turn but stops before any tool executes,
		// so the operator can decide what to do with the proposed calls.
		plan, done, err := s.StepPlan(ctx)
		if err != nil {
			log.Fatalf("plan failed: %v", err)
		}
		lastExec = plan.Exec

		for _, tc := range plan.ToolCalls {
			fmt.Printf("step %d: model wants %q\n", plan.Step, tc.Name)
		}

		if done {
			break
		}

		var outcomes []llms.ToolOutcome
		if needsApproval(plan.ToolCalls) {
			// Operator policy: block deploys before they run. Synthesize a
			// denial outcome for every call this turn and let the model react.
			fmt.Println("operator: deploy is policy-gated — denying turn")
			outcomes = make([]llms.ToolOutcome, len(plan.ToolCalls))
			for i, tc := range plan.ToolCalls {
				outcomes[i] = llms.ToolOutcome{
					ID:    tc.ID,
					Name:  tc.Name,
					Error: "operator policy: deployments are not permitted in this session; summarize instead",
				}
			}
		} else {
			// Approved: run the tools via the session's standard dispatch.
			outcomes, err = s.RunTools(ctx, plan.ToolCalls)
			if err != nil {
				log.Fatalf("tool run failed: %v", err)
			}
		}

		info, _, err := s.StepObserve(ctx, outcomes)
		if err != nil {
			log.Fatalf("observe failed: %v", err)
		}
		lastExec = info.Exec
	}

	res, err := s.Result()
	var mse *llms.MaxStepsError
	switch {
	case errors.As(err, &mse):
		fmt.Printf("\n[step-limited after %d steps] %s\n", mse.Result.Steps, mse.Result.FinalText)
	case err != nil:
		log.Fatalf("result failed: %v", err)
	default:
		if text, ok := res.(llms.TextResult); ok {
			fmt.Printf("\nfinal answer:\n%s\n", text.Text)
		}
	}

	if lastExec != nil {
		fmt.Printf("\nbudget (parent, incl. sub-agent rollup): steps=%d tools=%d in=%d out=%d\n",
			lastExec.CurrentStep(), lastExec.TotalToolCalls(),
			lastExec.InputTokens(), lastExec.OutputTokens())
	}
}
