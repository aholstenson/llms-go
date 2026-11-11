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

## Attachments

Send images, PDFs and text files as `BinaryPart`s. Give a part a name so the
model can refer to the file:

```go
msg := llms.NewMessage(llms.RoleUser,
	llms.NewTextPart("Compare the spec with the report."),
	llms.NewBinaryPart("text/markdown", spec).WithName("spec.md"),
	llms.NewBinaryPart("application/pdf", report).WithName("report.pdf"),
)
```

Check the input before you start a long run. `CheckMessages` runs the same
check that `GenerateContent` runs before it sends a request, and it makes no
network call:

```go
if err := llms.CheckMessages(model, []*llms.Message{msg}); err != nil {
	var ie *llms.InputError
	if errors.As(err, &ie) {
		// ie.Message and ie.Part tell which part failed.
	}
	if errors.Is(err, llms.ErrUnsupportedModality) { /* the model cannot read this kind of file */ }
	if errors.Is(err, llms.ErrAttachmentTooLarge) { /* send a list of files instead */ }
}
```

The errors are `ErrUnsupportedModality`, `ErrUnsupportedMediaType`,
`ErrAttachmentTooLarge` and `ErrInvalidAttachment` (for example a text file
that is not UTF-8). They are configuration errors, so `IsRetryable` is false.

`ModelInfoOf(model).Limits.Image` and `.Limits.PDF` give the provider limits:
the largest image in bytes and pixels, the image formats, the number of
images, and the largest PDF in bytes and pages. A zero value means the limit
is not known. The check does not count PDF pages, so compare `MaxPages` with
your own page count. Byte limits count the bytes in `BinaryPart.Data`, before
base64 encoding.

| Provider   | Images                         | PDFs                              | Limits                                   |
|------------|--------------------------------|-----------------------------------|------------------------------------------|
| Anthropic  | image block, name in a label   | document block, name as the title | 7.5 MB, 8000 px, 100 or 600 images and pages |
| OpenAI     | `input_image`, name in a label | `input_file`, name as the file name | 1500 images, 50 MB per file            |
| Google     | inline data, name in a label   | inline data, name in a label      | 3600 images, 50 MB and 1000 pages per PDF |
| OpenRouter | `image_url`, name in a label   | `file` part, name as the file name | not documented                          |

A label is a text part such as `File "chart.png":` just before the file.

Text files (`text/*`, JSON, XML and YAML) are sent as text. Anthropic gets a
plain-text document block with the name as its title. The other providers get
a text part with the content in a `<file>` element:

```text
<file name="spec.md" media_type="text/markdown">
...
</file>
```

Anthropic models with a context window under 1M tokens take 100 images and
100 PDF pages; other Claude models take 600. With more than 20 images, each
image must be 2000 px or smaller. Gemini documents its inline request limit
as 20 MB on one page and 100 MB on another, so no image byte limit is set for
Google. Correct a limit with `RegisterModelInfo`.

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

A request that fails with a temporary error is retried automatically. This
includes a rate limit (429), an overload (529), a timeout (408, 504), a
server or gateway error (500, 502, 503), and a Cloudflare error between
Cloudflare and the provider (520–524, for example 524 when the provider does
not answer in time). The library owns the
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
— is retried as one attempt. This includes a rate limit or an overload that the
provider sends inside a stream that already opened.

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

## Streams that end early

A proxy or a load balancer can close a response stream before the provider
sends its final event. The connection closes without an error, so the stream
looks complete but the response is missing. The library finds this for each
provider:

| Provider | Final event |
| --- | --- |
| OpenAI | `response.completed`, `response.failed` or `response.incomplete` |
| Anthropic | `message_stop` |
| Google | A chunk with a finish reason, or a blocked prompt |
| OpenRouter | A chunk with a finish reason |

A stream that ends early is a transient failure, like a stall. It is reported
through `WithRetryNotify`, retried while nothing has reached your callbacks,
and surfaces as an `*UnavailableError` when the attempts run out. Detect it
with `errors.Is(err, llms.ErrStreamEnded)`. The error message gives the number
of events received, the type of the last event and the response ID, if the
provider sent one:

```text
stream ended after 214 events without a final response (last event: response.reasoning_summary_text.delta, response ID: resp_123)
```

Dead HTTP/2 connections are found even without these options. After 15
seconds with no frame, the client sends a ping. If the ping gets no answer in
15 more seconds, the client closes the connection and removes it from the
pool. The model does not answer the ping, so a model that reasons for a long
time does not cause it to fail. The failed attempt is reported as a
`*StallError`, so it is retried like any other stall, and the next attempt
opens a new connection.

Pings are answered by the nearest HTTP/2 server. For most providers this is a
CDN or a load balancer, and in some networks it is a proxy. Pings thus find a
dead network path, such as a Wi-Fi change, a NAT timeout or a laptop that
slept, but not a stuck provider behind a healthy front end. Use
`WithStallTimeout` for that.

Each new model uses a copy of `http.DefaultTransport` with the pings turned
on. To use your own settings, set `http.DefaultTransport` before you create
the model. Changes to the fields of the same value after that are not seen.
When `http.DefaultTransport` is not an `*http.Transport`, for example a test
mock, the library uses it as it is, without pings.

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

The session also stays usable after a transient failure that comes after some
output already reached your streaming callback, for example a stream that was
cut after some reasoning deltas. `IsRetryable` and `Session.Retryable` are
false for it, because another attempt streams the turn again from the start.
The error also wraps `ErrStreamingPartialOutput`. Use `IsUnavailable` to find
all transient failures, with and without partial output:

```go
if llms.IsUnavailable(err) {
    if errors.Is(err, llms.ErrStreamingPartialOutput) {
        ui.DiscardCurrentTurn() // the next attempt streams the turn again
    }
    continue // same session, same transcript, one more model call
}
```

### Content policy blocks

A provider safety system can block a model call. The library reports this as
a `*ContentPolicyError` for every provider:

| Provider | Blocks |
| --- | --- |
| OpenAI | Error codes `cyber_policy`, `invalid_prompt`, `content_policy_violation`, `content_filter`, and a response that is incomplete because of `content_filter` |
| Anthropic | Stop reason `refusal` |
| Google | A blocked prompt (`promptFeedback.blockReason`), and the finish reasons `SAFETY`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT` |
| OpenRouter | Finish reason `content_filter`, finish reason `error` with a policy reason from the upstream provider, moderation errors (status 403), and policy errors from the upstream provider |

A refusal that the model writes itself, such as an OpenAI refusal content
part, is not a block: the turn completes, and `Session.Result` returns a
`*RefusalError`.

The library does not retry a `*ContentPolicyError` automatically: the
classifier is not deterministic, but it can also block the same transcript
each time. The session stays usable
(`done == false`), so you decide what to do:

```go
policyBlocks := 0
for {
    info, done, err := s.Step(ctx)
    if err != nil {
        if llms.IsRetryable(err) {
            continue
        }
        if llms.IsContentPolicy(err) && policyBlocks < 2 {
            policyBlocks++
            log.Printf("step %d blocked by content policy, trying again: %v", info.Step, err)
            continue
        }
        return err
    }
    if done {
        break
    }
}
```

`Session.Retryable` is false after a block, because the failure is not
transient. The session stays usable also when output already reached your
streaming callback. Then the error also wraps `ErrStreamingPartialOutput`, and
another attempt streams the turn again from the start.

A block on a long agentic run is often caused by tool output in the
transcript, not by the prompt. Log the latest tool results when you get this
error to find the cause.

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
