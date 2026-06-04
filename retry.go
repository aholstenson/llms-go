package llms

import (
	"context"
	"math/rand"
	"net/http"
	"sync/atomic"
	"time"
)

// BackoffPolicy computes the delay before the next retry attempt.
//
// attempt is 0-indexed: 0 is the delay before the first retry (after the
// original try has just failed). When hasRetryAfter is true the server
// provided a hint (already clamped to RetryAfterCap by the retry loop) and
// policies are free to honor or override it; the default policy honors it.
type BackoffPolicy interface {
	Delay(attempt int, retryAfter time.Duration, hasRetryAfter bool) time.Duration
}

// BackoffFunc adapts a function value to the BackoffPolicy interface.
type BackoffFunc func(attempt int, retryAfter time.Duration, hasRetryAfter bool) time.Duration

// Delay implements BackoffPolicy.
func (f BackoffFunc) Delay(attempt int, retryAfter time.Duration, hasRetryAfter bool) time.Duration {
	return f(attempt, retryAfter, hasRetryAfter)
}

// ExponentialBackoff is the default BackoffPolicy and is also exported as a
// building block. Delay is Base * 2^attempt, capped at Max, with random
// jitter in [delay*(1-Jitter), delay). When the server provided a
// Retry-After hint, that hint is used verbatim instead.
//
// Defaults match the Anthropic and OpenAI SDK backoffs:
//
//	&ExponentialBackoff{Base: 500*time.Millisecond, Max: 8*time.Second, Jitter: 0.25}
type ExponentialBackoff struct {
	Base   time.Duration // initial delay; default 500ms
	Max    time.Duration // cap; default 8s
	Jitter float64       // 0..1; default 0.25. 0 disables jitter.
}

// Delay implements BackoffPolicy.
func (e *ExponentialBackoff) Delay(attempt int, retryAfter time.Duration, hasRetryAfter bool) time.Duration {
	if hasRetryAfter {
		return retryAfter
	}

	base := e.Base
	if base <= 0 {
		base = 500 * time.Millisecond
	}
	max := e.Max
	if max <= 0 {
		max = 8 * time.Second
	}

	d := base
	for i := 0; i < attempt && d < max; i++ {
		d *= 2
	}
	if d > max {
		d = max
	}

	jitter := e.Jitter
	if jitter < 0 {
		jitter = 0
	}
	if jitter > 1 {
		jitter = 1
	}
	if jitter > 0 {
		factor := 1 - jitter*rand.Float64() //nolint:gosec
		d = time.Duration(float64(d) * factor)
	}
	return d
}

func defaultBackoffPolicy() BackoffPolicy {
	return &ExponentialBackoff{
		Base:   500 * time.Millisecond,
		Max:    8 * time.Second,
		Jitter: 0.25,
	}
}

// headerCapturingTransport is an http.RoundTripper that records the latest
// response headers it observes. Used by providers whose SDK error types do
// not carry an *http.Response (Google, OpenRouter) so the retry loop can
// surface Retry-After hints on UnavailableError.
type headerCapturingTransport struct {
	inner http.RoundTripper
	last  atomic.Pointer[http.Header]
}

func newHeaderCapturingTransport(inner http.RoundTripper) *headerCapturingTransport {
	if inner == nil {
		inner = http.DefaultTransport
	}
	return &headerCapturingTransport{inner: inner}
}

// RoundTrip implements http.RoundTripper.
func (t *headerCapturingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.inner.RoundTrip(req)
	if resp != nil {
		h := resp.Header.Clone()
		t.last.Store(&h)
	}
	return resp, err
}

// LastHeaders returns the most recent response headers, or nil if no
// response has been observed yet.
func (t *headerCapturingTransport) LastHeaders() http.Header {
	p := t.last.Load()
	if p == nil {
		return nil
	}
	return *p
}

// retryClassifier inspects an error returned by a single attempt and
// reports whether it is retryable. statusCode, retryAfter, and
// hasRetryAfter are recorded so the retry loop can surface them on a final
// UnavailableError.
type retryClassifier func(err error) (retryable bool, statusCode int, retryAfter time.Duration, hasRetryAfter bool)

// retryLoop runs fn up to opts.MaxRetries+1 times, sleeping between
// attempts according to opts.RetryBackoff. classify decides whether an
// error is retryable. provider and model are stamped onto any
// *UnavailableError returned on final failure.
//
// Returns the function's value on success, or on final failure a
// *UnavailableError if the final attempt was retryable, or the raw error
// otherwise. ctx cancellation is honored mid-sleep.
func retryLoop[T any](
	ctx context.Context,
	opts *generateContentOptions,
	provider, model string,
	classify retryClassifier,
	fn func(ctx context.Context) (T, error),
) (T, error) {
	var zero T

	maxAttempts := opts.MaxRetries + 1
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	var lastErr error
	var lastStatus int
	var lastRetryAfter time.Duration
	var lastHasRetryAfter bool

	for attempt := 0; attempt < maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			if lastErr != nil {
				return zero, lastErr
			}
			return zero, err
		}

		v, err := fn(ctx)
		if err == nil {
			return v, nil
		}

		retryable, status, ra, hasRA := classify(err)
		lastErr = err
		lastStatus = status
		if hasRA {
			if opts.RetryAfterCap > 0 && ra > opts.RetryAfterCap {
				ra = opts.RetryAfterCap
			}
			lastRetryAfter = ra
			lastHasRetryAfter = true
		} else {
			lastRetryAfter = 0
			lastHasRetryAfter = false
		}

		if !retryable || attempt == maxAttempts-1 {
			if retryable {
				return zero, &UnavailableError{
					Provider:      provider,
					Model:         model,
					StatusCode:    lastStatus,
					RetryAfter:    lastRetryAfter,
					HasRetryAfter: lastHasRetryAfter,
					Attempts:      attempt + 1,
					Cause:         lastErr,
				}
			}
			return zero, lastErr
		}

		delay := opts.RetryBackoff.Delay(attempt, lastRetryAfter, lastHasRetryAfter)
		if delay < 0 {
			delay = 0
		}
		if delay > 0 {
			t := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				t.Stop()
				return zero, ctx.Err()
			case <-t.C:
			}
		}
	}

	return zero, lastErr
}
