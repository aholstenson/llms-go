# llms-go

A Go library for building LLM-powered applications with a single, unified API
across providers.

Write your code once against one `Model` interface and run it on Anthropic,
OpenAI, Google (Gemini), or OpenRouter. A `Manager` handles model registration,
aliases, and environment-variable overrides so you can swap models without
touching application code.

## Features

- **Multiple providers** — Anthropic, OpenAI, Google Gemini, and OpenRouter
  behind one interface.
- **Typed structured output** — get strongly-typed Go values back via
  generics; JSON schemas are derived automatically from your types.
- **Streaming** — stream text, thinking/reasoning tokens, and incrementally
  parsed structured output as it arrives.
- **Tool calling & agentic loops** — composable toolkits, multi-step
  execution, per-call timeouts, and built-in web search.
- **Cost & usage tracking** — automatic pricing from models.dev data plus
  OpenTelemetry GenAI metrics and local stats aggregation.
- **Capability-aware** — embedded model metadata gates temperature,
  reasoning, and modality behavior so missing features fail gracefully.

## Installation

```sh
go get github.com/aholstenson/llms-go
```

```go
import llms "github.com/aholstenson/llms-go"
```

## Quick start

Models are resolved through a `Manager` using fully-qualified
`provider/model` names (e.g. `anthropic/claude-sonnet-4-5`). API tokens are
read from environment variables: `ANTHROPIC_API_TOKEN`, `OPENAI_API_TOKEN`,
`OPENROUTER_API_TOKEN`, or `GOOGLE_API_TOKEN`.

```go
manager := llms.NewManager()

model, err := manager.GetModel("anthropic/claude-sonnet-4-5")
if err != nil {
    log.Fatal(err)
}

result, err := model.GenerateContent(ctx,
    llms.WithMessages(
        llms.NewMessage(llms.RoleUser, llms.NewTextPart("Say hello in one sentence.")),
    ),
)
if err != nil {
    log.Fatal(err)
}

fmt.Println(result.(llms.TextResult).Text)
```

### Aliases

Register friendly names and let environment variables override them at
deploy time without code changes:

```go
manager.RegisterAlias("fast", "anthropic/claude-haiku-4-5")
model, _ := manager.GetModel("fast") // or set LLM_MODEL_FAST=openai/gpt-4o
```

### Driving the loop with `Session`

`GenerateContent` runs the agentic loop to completion. `Session` exposes the
same loop one step at a time, so you can inspect what the model did, inject
operator steering, or gate tool calls between turns:

```go
s, err := llms.NewSession(model,
    llms.WithMessages(llms.NewMessage(llms.RoleUser, llms.NewTextPart("..."))),
    llms.WithTools(llms.NewToolDef(deployTool{})),
    llms.WithMaxSteps(8),
)
if err != nil {
    log.Fatal(err)
}

for {
    info, done, err := s.Step(ctx)
    if err != nil {
        log.Fatal(err)
    }

    for _, tc := range info.Output.ToolCalls {
        fmt.Printf("step %d: model called %q\n", info.Step, tc.Name)
    }

    if done {
        break
    }

    // Steer the next turn based on what just happened.
    s.Inject(llms.NewMessage(llms.RoleUser,
        llms.NewTextPart("Operator: skip deploy, summarize instead.")))
}

res, err := s.Result()
```

For finer control, `StepPlan` returns the model's tool calls without
executing them and `RunTools` runs them on demand, useful when tools
require approval or run out of process. See
[`examples/agent`](./examples/agent) for the full pattern.

## Examples

Runnable programs live in [`examples/`](./examples):

| Example | What it shows |
| --- | --- |
| [`examples/basic`](./examples/basic) | Minimal text generation |
| [`examples/streaming`](./examples/streaming) | Streaming text and thinking tokens |
| [`examples/structured`](./examples/structured) | Typed structured output via generics |
| [`examples/tools`](./examples/tools) | Defining a tool and running an agentic loop |
| [`examples/agent`](./examples/agent) | Multi-step tool calling with execution control |

Each example reads the model name from `LLM_EXAMPLE_MODEL` (defaulting to
`anthropic/claude-sonnet-4-5`) and the matching provider token from the
environment. Run one with:

```sh
go run ./examples/basic
```

## License

MIT
