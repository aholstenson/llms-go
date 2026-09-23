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
- **Streaming** — stream text, thinking and reasoning tokens, and incrementally
  parsed structured output as it arrives.
- **Tool calling and agentic loops** — composable toolkits, multi-step
  execution, per-call timeouts, and built-in web search.
- **Cost and usage tracking** — automatic pricing from models.dev data,
  including long-context price tiers, plus OpenTelemetry GenAI metrics and
  local stats aggregation.
- **Capability-aware** — model metadata gates temperature, reasoning,
  structured output, and modality behavior so missing features fail
  gracefully. The metadata can be refreshed without a new release.
- **Retries you can watch** — rate limits and overloads are retried with
  backoff, and a callback reports each wait so you can see the progress.
- **Stalls become errors** — a call that goes silent fails, retries, and
  reports itself instead of hanging until an operator kills the job.
- **Pluggable credentials** — environment variables by default, or supply
  your own source; credentials are resolved per request so they can rotate.

## Installation

To install `llms-go`, run:

```sh
go get github.com/aholstenson/llms-go
```

Import the package in your code:

```go
import llms "github.com/aholstenson/llms-go"
```

## Quick start

Models are resolved through a `Manager` using fully-qualified
`provider/model` names (for example, `anthropic/claude-sonnet-4-5`). By default, API
keys are read from environment variables: `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `GEMINI_API_KEY` (also accepts
`GOOGLE_API_KEY`). See [Credentials](#credentials) to supply them yourself.

The following example initializes a manager and generates text:

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

## Aliases

Register friendly names and let environment variables override them at
deploy time without code changes:

```go
manager.RegisterAlias("fast", "anthropic/claude-haiku-4-5")
model, _ := manager.GetModel(ctx, "fast") // or set LLM_MODEL_FAST=openai/gpt-4o
```

## Reasoning

Reasoning (thinking) has three modes. Without an option, the provider's
default for the model applies:

```go
model.GenerateContent(ctx, llms.WithReasoning(llms.ReasoningOff), ...)  // turn it off
model.GenerateContent(ctx, llms.WithReasoning(llms.ReasoningOn), ...)   // on, default depth
model.GenerateContent(ctx, llms.WithReasoningEffort(llms.EffortHigh), ...) // on, at an effort
```

Efforts go from `EffortMinimal` to `EffortMax`. An effort the model does not
accept is moved to the nearest one it does accept. A model that cannot turn
reasoning off (for example Claude Opus 5.5) uses its lowest effort when asked
to turn it off. Both cases log a warning.

## Model data

Pricing, limits and capabilities come from [models.dev](https://models.dev),
embedded in the build. For Anthropic models, the Anthropic Models API is also
asked one time for each model, so new Claude models work before the embedded
data knows them.

You can use newer data without a new release:

```go
llms.RefreshModelInfo(ctx)             // get models.dev now and cache it for later processes
llms.LoadModelInfo(file)               // use a models.dev api.json that you downloaded
llms.RegisterModelInfo("openai/my-model", llms.ModelInfo{...}) // add or correct one model
```

Set `LLM_MODELS_FILE` to the path of a models.dev `api.json` to use that file
in place of the embedded data. Prices can be overridden with a JSON file in
`LLM_PRICING_FILE` (default `llms-pricing.json`) that maps model names to
`{"input": ..., "output": ..., "cache_read": ..., "cache_write": ...}` in USD
per million tokens.

To update the embedded data, run `go generate ./...`.

## Credentials

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

## Driving the loop with Session

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

## Retries

A request that fails with a rate limit (429), a service that is unavailable
(503), or an overload (529) is retried automatically. The library owns the
retry loop for all four providers, so the same options apply everywhere:

```go
res, err := model.GenerateContent(ctx,
    llms.WithMessages(llms.NewMessage(llms.RoleUser, llms.NewTextPart("..."))),
    // Up to 10 attempts in total.
    llms.WithMaxRetries(9),
    // Tell the user what is going on between attempts.
    llms.WithRetryNotify(func(_ context.Context, n llms.RetryNotice) {
        log.Printf("Failed attempt %d of %d, waiting %s", n.Attempt, n.MaxAttempts, n.Delay)
    }),
)
```

Defaults: 2 retries (3 attempts), exponential backoff from 500ms to 8s with
25% jitter, and a server `Retry-After` hint is used when there is one.
`WithRetryBackoff` replaces the delay policy and `WithRetryAfterCap` limits
how long a server hint can make the library wait.

When the retries run out the error is an `*UnavailableError`, which carries
the status code, the attempts made, and the last `Retry-After` hint:

```go
var ue *llms.UnavailableError
if errors.As(err, &ue) {
    log.Printf("gave up after %d attempts (status %d)", ue.Attempts, ue.StatusCode)
}
```

A stream is never retried after its first event reaches your streaming
callbacks. Such a failure is reported with the `ErrStreamingPartialOutput`
sentinel instead, so you never replay tokens you have already received. Up to
that point the whole model call — opening the request and reading the response
— is retried as one attempt.

## Stalls

A model call that goes silent is not a failure any provider reports: there is
no error and no token, only a gap. `WithStallTimeout` turns that gap into a
retryable error:

```go
res, err := model.GenerateContent(ctx,
    llms.WithMessages(llms.NewMessage(llms.RoleUser, llms.NewTextPart("..."))),
    // Fail an attempt that has been quiet for 90 seconds.
    llms.WithStallTimeout(90*time.Second),
)
```

The budget measures bytes from the provider, not the events the library hands
you. Keepalives and ping events are bytes, so a model that spends minutes
reasoning keeps resetting it; a connection that died does not. The budget
applies per attempt and only to the model call, so it never cuts short a slow
tool call or a long agentic loop.

A stall is an ordinary transient failure: it is reported through
`WithRetryNotify`, retried while nothing has reached your callbacks, and
surfaces as an `*UnavailableError` wrapping a `*StallError` when the attempts
run out. Detect it with `errors.Is(err, llms.ErrStall)`.

`WithRequestTimeout` bounds the other half — how long an attempt may wait for
the provider to start responding at all. Both default to off.

To tell whether the model is thinking or the connection is dead while a
turn produces no tokens, register `WithLivenessNotify`. It fires whenever data
arrives after a quiet second, and stops the moment the provider goes silent:

```go
llms.WithLivenessNotify(func(_ context.Context, n llms.LivenessNotice) {
    log.Printf("%s alive, quiet for %s", n.Provider, n.Idle)
})
```

## Recovering a failed turn

A model turn that fails transiently does not end a `Session`. The turn left no
trace — the conversation only grows when a turn succeeds — so the same session
can try again, instead of cancelling the run and losing the transcript
it has built up. This is what lets a long unattended job survive a stall:

```go
for {
    info, done, err := s.Step(ctx)
    if err != nil {
        if llms.IsRetryable(err) {
            log.Printf("step %d failed, retrying: %v", info.Step, err)
            continue // same session, same transcript, one more model call
        }
        return err
    }
    if done {
        break
    }
}
```

`Step` returns `done == false` alongside such an error, and `Session.Retryable`
reports the same thing. A retried turn does not consume step budget, and emits
a fresh `StreamingEventMessageStart`.

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
