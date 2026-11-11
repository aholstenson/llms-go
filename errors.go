package llms

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ErrStreamingPartialOutput sentinels an error returned from a streaming
// generation where at least one content event had already been dispatched
// to the caller's StreamingFunc or StructuredStreamingFunc before the
// underlying stream errored. Callers use errors.Is to detect this case.
//
// The library does not retry such a call by itself, because a new attempt
// sends the output again from the start. The failed call does not change the
// conversation, so the caller can still try the turn again. When the error
// is also an UnavailableError (see IsUnavailable), a Session stays usable:
// the caller discards the output it received for the failed turn and calls
// Session.StepPlan (or Step) again.
var ErrStreamingPartialOutput = errors.New("streaming output was partially delivered before error")

// ErrStreamEnded is the sentinel for a response stream that ended before the
// provider sent its final event, for example an OpenAI stream without
// "response.completed" or an Anthropic stream without "message_stop". The
// connection closed without an error, so the cause is usually a proxy or a
// load balancer that cut the stream. Another attempt can succeed.
//
// The retry loop retries this failure while no output has reached the
// caller. When the attempts run out, or when output already reached the
// caller, the error is returned in an UnavailableError. Use errors.Is to
// find it:
//
//	if errors.Is(err, llms.ErrStreamEnded) { ... }
var ErrStreamEnded = errors.New("stream ended without a final response")

// streamEndedError is a stream that ended before its final event. It holds
// the diagnostics that help to find where the stream was cut.
type streamEndedError struct {
	// events is the number of events received before the stream ended.
	events int
	// lastEvent is the type of the last event received, or "" when no event
	// arrived.
	lastEvent string
	// responseID is the ID of the response, or "" when the provider did not
	// send one before the stream ended.
	responseID string
}

func (e *streamEndedError) Error() string {
	msg := fmt.Sprintf("stream ended after %d events without a final response", e.events)
	var details []string
	if e.lastEvent != "" {
		details = append(details, "last event: "+e.lastEvent)
	}
	if e.responseID != "" {
		details = append(details, "response ID: "+e.responseID)
	}
	if len(details) > 0 {
		msg += " (" + strings.Join(details, ", ") + ")"
	}
	return msg
}

func (e *streamEndedError) Unwrap() error { return ErrStreamEnded }

// isStreamEnded reports whether err is, or wraps, ErrStreamEnded.
func isStreamEnded(err error) bool {
	return errors.Is(err, ErrStreamEnded)
}

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
//   - StatusCode: HTTP status when known, 0 otherwise (e.g. an overload or
//     rate limit that Anthropic or OpenAI sent inside an open stream).
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

// ContentPolicyError is returned when a provider safety system blocks a
// model call, for example the OpenAI "cyber_policy" error. The provider
// rejects the call as a whole, so the model produces no turn. This is
// different from a RefusalError, where the model itself answers with a
// refusal and the turn completes.
//
// The classifiers that make these decisions are not deterministic. A call
// that is blocked can pass when it is sent again, but it can also fail the
// same way each time. Thus the library does not retry it automatically.
// The caller decides what to do. A Session stays usable after this error
// (Step returns done == false), so the caller can:
//   - call Step again a limited number of times,
//   - stop and keep the transcript for analysis,
//   - continue the work with a different model.
//
// Use errors.As or IsContentPolicy to find it in a wrapped error chain.
//
// Fields:
//   - Provider: the GenAI system (e.g. "openai").
//   - Model: the model ID requested.
//   - StatusCode: HTTP status when known, 0 when the provider sent the
//     error inside an open stream.
//   - Code: the provider's error code, e.g. "cyber_policy".
//   - Message: the provider's error message.
//   - PartialOutput: true when at least one content event was dispatched
//     to StreamingFunc before the call was blocked. Another attempt sends
//     that output again.
//   - PartialUsage: usage reported before the call was blocked, if any.
type ContentPolicyError struct {
	Provider      string
	Model         string
	StatusCode    int
	Code          string
	Message       string
	PartialOutput bool
	PartialUsage  *TurnUsage
	Cause         error
}

func (e *ContentPolicyError) Error() string {
	b := []byte("blocked by content policy")
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
	if e.Code != "" {
		b = append(b, ": "...)
		b = append(b, e.Code...)
	}
	if e.Message != "" {
		b = append(b, ": "...)
		b = append(b, e.Message...)
	}
	return string(b)
}

func (e *ContentPolicyError) Unwrap() error {
	return e.Cause
}

// contentPolicyFailure returns pe as the error of a turn. When output
// already reached the caller, it also wraps ErrStreamingPartialOutput, as
// the other turn errors do.
func contentPolicyFailure(pe *ContentPolicyError) error {
	if pe.PartialOutput {
		return errors.Join(pe, ErrStreamingPartialOutput)
	}
	return pe
}

// IsContentPolicy reports whether err contains a *ContentPolicyError: a
// provider safety system blocked the model call. See ContentPolicyError for
// the ways to recover.
func IsContentPolicy(err error) bool {
	var pe *ContentPolicyError
	return errors.As(err, &pe)
}

// IsRetryable reports whether a failed model call can be attempted again
// against the same conversation without replaying output. It is true for a
// transient provider failure — a rate limit, an overload, a stream event that
// could not be decoded, a stream that ended before its final event (see
// ErrStreamEnded), or a stall (see WithStallTimeout) — that the library's own
// retries could not clear. It is false once any of the turn's output has
// reached a streaming callback, because a fresh attempt replays tokens the
// user has already seen.
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
//
// A transient failure after partial output is not retryable in this sense,
// but the Session stays usable after it. The caller discards the partial
// output and decides whether to try again. See IsUnavailable.
//
// A *ContentPolicyError is not retryable in this sense: the failure is not
// transient. The Session stays usable after it, but the caller must decide
// whether to try again. See IsContentPolicy.
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
	return errors.Is(err, ErrStall) || errors.Is(err, ErrStreamEnded)
}

// IsUnavailable reports whether err contains an *UnavailableError: a model
// call failed with a transient condition, such as a rate limit, an overload,
// a stall or a stream that ended before its final event. It is true with and
// without partial output.
//
// A Session stays usable after such an error (Step returns done == false),
// because a failed turn does not change the conversation. Calling
// Session.StepPlan (or Step) again tries the turn again against the same
// transcript:
//   - Without partial output, IsRetryable is also true. The caller can try
//     again at once.
//   - With partial output (the error also wraps ErrStreamingPartialOutput),
//     IsRetryable is false. Some output of the failed turn already reached the
//     streaming callback. The caller must discard that output before it tries
//     again, because the next attempt streams the turn from the start.
func IsUnavailable(err error) bool {
	var ue *UnavailableError
	return errors.As(err, &ue)
}

// isUnavailableStatusCode returns true for HTTP status codes that indicate
// the provider is temporarily unavailable. The same request can succeed on
// a later attempt, so the retry loop tries it again:
//   - 408: Request Timeout
//   - 429: Too Many Requests / rate limited
//   - 500: Internal Server Error
//   - 502: Bad Gateway
//   - 503: Service Unavailable / overloaded
//   - 504: Gateway Timeout
//   - 520-524: Cloudflare errors between Cloudflare and the provider, such
//     as 524 when the provider does not answer in time
//   - 529: Overloaded (Anthropic-specific)
//
// Other 5xx codes, such as 501 Not Implemented or the Cloudflare TLS
// errors 525 and 526, fail the same way each time and are not retried.
func isUnavailableStatusCode(code int) bool {
	switch code {
	case 408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529:
		return true
	}
	return false
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
