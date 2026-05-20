package llms

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/cockroachdb/errors"
)

// ErrStreamingPartialOutput sentinels an error returned from a streaming
// generation where at least one content event had already been dispatched
// to the caller's StreamingFunc before the underlying stream errored.
// Callers should use errors.Is to detect this case and avoid blind retries
// that would replay tokens already shown to the user.
var ErrStreamingPartialOutput = errors.New("streaming output was partially delivered before error")

// UnavailableError is the structured error returned by providers when a
// generation fails with a transient unavailability condition (rate limit,
// overload, server error). Callers use errors.As to recover it from a
// wrapped error chain.
//
// Fields:
//   - StatusCode: HTTP status when known, 0 otherwise (e.g. OpenAI in-band
//     rate_limit_exceeded events).
//   - RetryAfter: the server's Retry-After hint (after RetryAfterCap is
//     applied), or 0 if no hint was provided. This is the raw hint, not
//     what the SDK actually slept (jitter/cap may differ); outer-loop
//     schedulers want the raw hint.
//   - HasRetryAfter: true when the server provided a hint.
//   - Attempts: number of attempts actually made (>=1). For Google/
//     OpenRouter (where llms-go owns the retry loop) this is exact. For
//     Anthropic/OpenAI (where the SDK runs its own retry loop and exposes
//     no counter) this is an upper bound: MaxRetries+1.
//   - PartialOutput: true when at least one content event was dispatched
//     to StreamingFunc before the stream errored.
type UnavailableError struct {
	StatusCode    int
	RetryAfter    time.Duration
	HasRetryAfter bool
	Attempts      int
	PartialOutput bool
	Cause         error
}

func (e *UnavailableError) Error() string {
	b := []byte("model unavailable")
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
