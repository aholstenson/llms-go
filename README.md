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
- **Pluggable credentials** — environment variables by default, or supply
  your own source; credentials are resolved per request so they can rotate.

## Installation

```sh
go get github.com/aholstenson/llms-go
```

```go
import llms "github.com/aholstenson/llms-go"
```

## Quick start

Models are resolved through a `Manager` using fully-qualified
`provider/model` names (e.g. `anthropic/claude-sonnet-4-5`). By default API
keys are read from environment variables: `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `GEMINI_API_KEY` (also accepts
`GOOGLE_API_KEY`). See [Credentials](#credentials) to supply them yourself.

```go
manager := llms.NewManager()

model, err := manager.GetModel(ctx, "anthropic/claude-sonnet-4-5")
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
model, _ := manager.GetModel(ctx, "fast") // or set LLM_MODEL_FAST=openai/gpt-4o
```

### Credentials

A `Manager` gets its API keys from a `CredentialSource`. The default is
`EnvCredentials`, which reads the environment variables listed above and
memoizes what it finds. Applications that keep keys somewhere else — a secret
manager, a config file, a per-tenant lookup — supply their own:

```go
manager, err := llms.NewManager(
    llms.WithManagerCredentials(llms.CredentialFunc(
        func(ctx context.Context, provider string) (llms.Credential, error) {
            token, err := vault.Token(ctx, provider) // your own lookup
            if err != nil {
                return llms.Credential{}, err
            }
            return llms.Credential{APIKey: token}, nil
        },
    )),
)
```

The source is consulted twice: once when `GetModel` builds a model, so a
missing credential is reported there instead of surfacing as an opaque
transport error later, and then once per outbound HTTP request. The
per-request call is what makes rotation work — a short-lived token can expire
and be replaced without the model being rebuilt or the `Manager`'s model cache
being invalidated. It also puts the source in the hot path of every request,
so implementations should cache and refresh on expiry rather than doing real
work on every call.

`Credential.APIKey` is placed in whichever authentication header the provider
expects. For endpoints that need more than a key — a gateway token, a tenant
identifier — `Credential.Headers` is applied on top, and a credential carrying
only headers and no API key is valid.

`llms.StaticCredentials("...")` covers the single-provider case, and
`errors.Is(err, llms.ErrNoCredentials)` detects a missing credential
regardless of which source produced it.

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
