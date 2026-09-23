package llms

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

// ErrStreamingPartialOutput sentinels an error returned from a streaming
// generation where at least one content event had already been dispatched
// to the caller's StreamingFunc or StructuredStreamingFunc before the
// underlying stream errored. Callers should use errors.Is to detect this
// case and avoid blind retries that would replay tokens already shown to
// the user.
var ErrStreamingPartialOutput = errors.New("streaming output was partially delivered before error")

// ErrStructuredStreamParse sentinels an error returned when the jsonstream
// parser rejects mid-stream content during structured streaming. The parser
// state is unreliable after such an error, so the generation is aborted.
// Callers detect this via errors.Is; the underlying parser error is wrapped
// for diagnostics.
var ErrStructuredStreamParse = errors.New("structured stream parse failed")

// UnavailableError is the structured error returned by providers when a
// generation fails with a transient unavailability condition (rate limit,
// overload, server error). Callers use errors.As to recover it from a
// wrapped error chain.
//
// Fields:
//   - Provider: the GenAI system (e.g. "openai", "anthropic"), or "" if
//     not stamped.
//   - Model: the model ID requested, or "" if not stamped.
//   - StatusCode: HTTP status when known, 0 otherwise (e.g. OpenAI in-band
//     rate_limit_exceeded events).
//   - RetryAfter: the server's Retry-After hint (after RetryAfterCap is
//     applied), or 0 if no hint was provided. This is the raw hint, not
//     what the SDK actually slept (jitter/cap may differ); outer-loop
//     schedulers want the raw hint.
//   - HasRetryAfter: true when the server provided a hint.
//   - Attempts: number of attempts actually made (>=1). llms-go owns the
//     retry loop for every provider, so this count is exact.
//   - PartialOutput: true when at least one content event was dispatched
//     to StreamingFunc before the stream errored.
//   - PartialUsage: usage accumulated up to the point of a mid-stream
//     failure, if the provider reported any. Nil when no usage was seen
//     before the stream errored. Surface this to cost accounting so the
//     tokens already generated (and billed) are not silently dropped.
type UnavailableError struct {
	Provider      string
	Model         string
	StatusCode    int
	RetryAfter    time.Duration
	HasRetryAfter bool
	Attempts      int
	PartialOutput bool
	PartialUsage  *TurnUsage
	Cause         error
}

func (e *UnavailableError) Error() string {
	b := []byte("model unavailable")
	if e.Provider != "" || e.Model != "" {
		b = append(b, " ("...)
		if e.Provider != "" {
			b = append(b, e.Provider...)
			if e.Model != "" {
				b = append(b, '/')
			}
		}
		b = append(b, e.Model...)
		b = append(b, ')')
	}
	if e.StatusCode != 0 {
		b = append(b, fmt.Sprintf(" (status %d)", e.StatusCode)...)
	}
	if e.HasRetryAfter {
		b = append(b, fmt.Sprintf(" (retry after %s)", e.RetryAfter)...)
	}
	if e.Attempts > 0 {
		b = append(b, fmt.Sprintf(" after %d attempt(s)", e.Attempts)...)
	}
	if e.Cause != nil {
		b = append(b, ": "...)
		b = append(b, e.Cause.Error()...)
	}
	return string(b)
}

func (e *UnavailableError) Unwrap() error {
	return e.Cause
}

// IsRetryable reports whether a failed model call can be attempted again
// against the same conversation. It is true for a transient provider failure —
// a rate limit, an overload, or a stall (see WithStallTimeout) — that the
// library's own retries could not clear, and false once any of the turn's
// output has reached a streaming callback, because a fresh attempt would
// replay tokens the user has already seen.
//
// A retry is safe because a failed turn leaves the conversation untouched: the
// provider-native history only grows when a turn succeeds and its tool results
// are observed. So an unattended job can report the failure, wait, and call
// Session.StepPlan (or Step) again, instead of cancelling the run and throwing
// away the transcript it has built up:
//
//	info, done, err := session.Step(ctx)
//	if err != nil && llms.IsRetryable(err) {
//		// same session, same transcript, one more model call
//		continue
//	}
//
// A Session tracks this for its own last failure; see Session.Retryable.
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	// Output the caller has already seen cannot be replayed.
	if errors.Is(err, ErrStreamingPartialOutput) {
		return false
	}
	var ue *UnavailableError
	if errors.As(err, &ue) {
		return !ue.PartialOutput
	}
	return errors.Is(err, ErrStall)
}

// isUnavailableStatusCode returns true for HTTP status codes that indicate
// the provider is temporarily unavailable:
//   - 429: Too Many Requests / rate limited
//   - 503: Service Unavailable / overloaded
//   - 529: Overloaded (Anthropic-specific)
func isUnavailableStatusCode(code int) bool {
	return code == 429 || code == 503 || code == 529
}

// parseRetryAfterHeaders extracts a retry hint from an HTTP response header
// set. Headers are checked in preference order:
//  1. Retry-After-Ms: integer milliseconds (Anthropic-style).
//  2. Retry-After: integer seconds or RFC1123 HTTP-date.
//
// Returns (duration, true) on a successful parse, or (0, false) otherwise.
// Mirrors the Anthropic SDK's internal parser so behavior stays consistent
// across providers.
func parseRetryAfterHeaders(h http.Header) (time.Duration, bool) {
	if h == nil {
		return 0, false
	}

	if v := h.Get("Retry-After-Ms"); v != "" {
		if ms, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(ms * float64(time.Millisecond)), true
		}
	}

	if v := h.Get("Retry-After"); v != "" {
		if secs, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(secs * float64(time.Second)), true
		}
		if t, err := http.ParseTime(v); err == nil {
			return time.Until(t), true
		}
	}

	return 0, false
}

// extractRetryAfter pulls a Retry-After hint out of a provider error,
// preferring response headers attached to the SDK error and falling back to
// provider-specific shapes (e.g. Google google.rpc.RetryInfo in APIError
// Details). capturedHeaders is the latest response header set captured by
// a headerCapturingTransport, used for providers whose error type doesn't
// carry an *http.Response.
func extractRetryAfter(provider string, sdkErr error, capturedHeaders http.Header) (time.Duration, bool) {
	switch provider {
	case "anthropic":
		if resp := anthropicResponseFromError(sdkErr); resp != nil {
			if d, ok := parseRetryAfterHeaders(resp.Header); ok {
				return d, true
			}
		}
	case "openai":
		if resp := openaiResponseFromError(sdkErr); resp != nil {
			if d, ok := parseRetryAfterHeaders(resp.Header); ok {
				return d, true
			}
		}
	case "google":
		if d, ok := parseRetryAfterHeaders(capturedHeaders); ok {
			return d, true
		}
		if d, ok := googleRetryDelayFromError(sdkErr); ok {
			return d, true
		}
	case "openrouter":
		if d, ok := parseRetryAfterHeaders(capturedHeaders); ok {
			return d, true
		}
	}
	return 0, false
}
