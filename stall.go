package llms

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// ErrStall is the sentinel behind every StallError. Use errors.Is to detect a
// model call that went silent, without caring which provider or phase it was
// in:
//
//	if errors.Is(err, llms.ErrStall) { ... }
var ErrStall = errors.New("provider stopped sending data")

// StallPhase says where a model call went silent.
type StallPhase string

const (
	// StallPhaseResponse means the request was sent but the provider never
	// returned response headers. For a streaming provider this is the wait
	// for the stream to open; for a non-streaming one it covers the whole
	// generation, because nothing arrives until the answer is complete.
	StallPhaseResponse StallPhase = "response"
	// StallPhaseStream means the response had started and then went quiet.
	// Keepalives count as data, so a model that is merely thinking does not
	// reach this state — a dead connection or a wedged generation does.
	StallPhaseStream StallPhase = "stream"
)

// StallError reports a model call that stopped producing data for longer than
// the configured budget. It is what turns a silent hang into a failed attempt:
// the retry loop treats it as retryable, so it is reported through
// WithRetryNotify and, when the attempts run out, surfaces wrapped in an
// UnavailableError like any other transient provider failure.
//
// The budgets come from WithStallTimeout (quiet time inside a response) and
// WithRequestTimeout (time to the first byte of a response). Both are off
// unless set.
type StallError struct {
	// Provider is the GenAI system that went silent, e.g. "anthropic".
	Provider string
	// Model is the model ID that was requested, or "" when not known.
	Model string
	// Phase says whether the call was waiting for the response to start or
	// reading a response that had already started.
	Phase StallPhase
	// Timeout is the budget that was exceeded.
	Timeout time.Duration
}

func (e *StallError) Error() string {
	what := "no response"
	if e.Phase == StallPhaseStream {
		what = "no stream data"
	}
	target := e.Provider
	if e.Model != "" {
		if target != "" {
			target += "/"
		}
		target += e.Model
	}
	if target == "" {
		return fmt.Sprintf("model call stalled: %s for %s", what, e.Timeout)
	}
	return fmt.Sprintf("model call stalled: %s from %s for %s", what, target, e.Timeout)
}

func (e *StallError) Unwrap() error { return ErrStall }

// isStallError reports whether err is, or wraps, a *StallError.
func isStallError(err error) bool {
	var se *StallError
	return errors.As(err, &se)
}

// transientErrorType picks the metric error type for a failure the library
// treats as "try again later": a stall is reported as a timeout so it can be
// told apart from a provider that answered with a rate limit or an overload.
func transientErrorType(err error) GenAIErrorType {
	if isStallError(err) {
		return GenAIErrorTypeTimeout
	}
	return GenAIErrorTypeUnavailable
}

// LivenessNotice reports that data arrived from the provider after a quiet
// gap. It is given to the callback registered with WithLivenessNotify.
type LivenessNotice struct {
	// Provider is the GenAI system that sent data, e.g. "anthropic".
	Provider string
	// Model is the model ID that was requested.
	Model string
	// Idle is how long the call was quiet before this data arrived. Only
	// gaps of at least LivenessQuietThreshold are reported.
	Idle time.Duration
	// StallTimeout is the gap that would have failed the call, or 0 when
	// stall detection is off for this phase.
	StallTimeout time.Duration
	// Phase says whether the call was waiting for the response to start or
	// reading a response that had already started.
	Phase StallPhase
}

// LivenessNotifyFunc is called when data arrives after a quiet gap.
type LivenessNotifyFunc func(ctx context.Context, n LivenessNotice)

// LivenessQuietThreshold is the shortest gap that produces a LivenessNotice.
// Gaps below it are not reported: while tokens flow the caller already sees
// the stream, and a notice per token would be pure noise. A model that is
// thinking sends keepalives instead of tokens, and those land above the
// threshold — which is what makes "quiet but alive" visible.
const LivenessQuietThreshold = time.Second

// liveness carries the per-request stall budgets from the generate options
// down to the HTTP transport. The options are per call while the http.Client
// is per model, so the settings ride on the request context.
type liveness struct {
	model          string
	requestTimeout time.Duration
	stallTimeout   time.Duration
	notify         LivenessNotifyFunc
}

type livenessKey struct{}

// withLiveness attaches the request's stall budgets to ctx. It returns ctx
// unchanged when there is nothing to watch, so the transport stays on its
// zero-cost path.
func withLiveness(ctx context.Context, model string, opts *generateContentOptions) context.Context {
	if opts == nil {
		return ctx
	}
	if opts.RequestTimeout <= 0 && opts.StallTimeout <= 0 && opts.LivenessNotify == nil {
		return ctx
	}
	return context.WithValue(ctx, livenessKey{}, &liveness{
		model:          model,
		requestTimeout: opts.RequestTimeout,
		stallTimeout:   opts.StallTimeout,
		notify:         opts.LivenessNotify,
	})
}

func livenessFrom(ctx context.Context) *liveness {
	l, _ := ctx.Value(livenessKey{}).(*liveness)
	return l
}

// stallTransport wraps a RoundTripper so a request that goes silent fails with
// a *StallError instead of hanging forever.
//
// It watches bytes, not the events the library hands to the caller. That is
// the whole point: SSE keepalives and provider ping events are bytes, so a
// long thinking turn keeps the watchdog happy, while a dead connection does
// not. Watching at this level also means one implementation covers every
// provider, whatever its SDK surfaces.
type stallTransport struct {
	inner    http.RoundTripper
	provider string
}

func newStallTransport(inner http.RoundTripper, provider string) *stallTransport {
	if inner == nil {
		inner = http.DefaultTransport
	}
	return &stallTransport{inner: inner, provider: provider}
}

// RoundTrip implements http.RoundTripper.
func (t *stallTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	l := livenessFrom(req.Context())
	if l == nil {
		return t.inner.RoundTrip(req)
	}

	parent := req.Context()
	ctx, cancel := context.WithCancel(parent)
	w := newStallWatchdog(parent, cancel, t.provider, l)
	w.begin(StallPhaseResponse, l.requestTimeout)

	resp, err := t.inner.RoundTrip(req.WithContext(ctx))
	if err != nil {
		w.stop()
		cancel()
		if se := w.stallErr(); se != nil {
			return nil, se
		}
		return nil, err
	}

	w.data()
	w.begin(StallPhaseStream, l.stallTimeout)
	resp.Body = &stallBody{rc: resp.Body, w: w, cancel: cancel}
	return resp, nil
}

// stallBody is the response body wrapper that feeds the watchdog. Every read
// that returns data resets the budget; a read that fails after the watchdog
// tripped is reported as the stall it really was, not as the context
// cancellation the watchdog used to unblock it.
type stallBody struct {
	rc     io.ReadCloser
	w      *stallWatchdog
	cancel context.CancelFunc
}

func (b *stallBody) Read(p []byte) (int, error) {
	n, err := b.rc.Read(p)
	if n > 0 {
		b.w.data()
	}
	if err != nil && !errors.Is(err, io.EOF) {
		if se := b.w.stallErr(); se != nil {
			return n, se
		}
	}
	return n, err
}

func (b *stallBody) Close() error {
	b.w.stop()
	err := b.rc.Close()
	b.cancel()
	return err
}

// stallWatchdog cancels a request that has been quiet for too long. Cancelling
// the request context is what unblocks a read stuck on a dead connection; the
// watchdog then remembers that it was the one who cancelled, so the failure is
// reported as a stall rather than as a cancellation the caller did not ask
// for.
type stallWatchdog struct {
	ctx      context.Context
	cancel   context.CancelFunc
	provider string
	l        *liveness

	mu      sync.Mutex
	timer   *time.Timer
	phase   StallPhase
	timeout time.Duration
	last    time.Time
	fired   bool
	stopped bool
}

func newStallWatchdog(ctx context.Context, cancel context.CancelFunc, provider string, l *liveness) *stallWatchdog {
	return &stallWatchdog{
		ctx:      ctx,
		cancel:   cancel,
		provider: provider,
		l:        l,
		last:     time.Now(),
	}
}

// begin moves the watchdog to a new phase with its own budget. A budget of 0
// means the phase is not watched.
func (w *stallWatchdog) begin(phase StallPhase, timeout time.Duration) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stopped {
		return
	}
	w.phase = phase
	w.timeout = timeout
	w.last = time.Now()
	w.arm()
}

// arm (re)starts the timer for the current phase. Callers hold w.mu.
func (w *stallWatchdog) arm() {
	if w.timeout <= 0 {
		if w.timer != nil {
			w.timer.Stop()
		}
		return
	}
	if w.timer == nil {
		w.timer = time.AfterFunc(w.timeout, w.trip)
		return
	}
	w.timer.Stop()
	w.timer.Reset(w.timeout)
}

// data records that the provider is alive, restarting the budget and
// reporting a quiet gap to the liveness callback.
func (w *stallWatchdog) data() {
	w.mu.Lock()
	if w.stopped {
		w.mu.Unlock()
		return
	}
	now := time.Now()
	idle := now.Sub(w.last)
	w.last = now
	w.arm()
	notify := w.l.notify
	notice := LivenessNotice{
		Provider:     w.provider,
		Model:        w.l.model,
		Idle:         idle,
		StallTimeout: w.timeout,
		Phase:        w.phase,
	}
	w.mu.Unlock()

	// The callback runs outside the lock: it is caller code and may be slow.
	if notify != nil && idle >= LivenessQuietThreshold {
		notify(w.ctx, notice)
	}
}

// trip fires when a budget runs out. It cancels the request so the blocked
// read returns.
func (w *stallWatchdog) trip() {
	w.mu.Lock()
	if w.stopped || w.timeout <= 0 {
		w.mu.Unlock()
		return
	}
	// The timer goroutine can already be running when data arrives and resets
	// it, so a firing does not prove the budget ran out. The deadline that
	// counts is the last byte seen: if it is still ahead, re-arm for what is
	// left of it rather than failing a healthy stream.
	if elapsed := time.Since(w.last); elapsed < w.timeout {
		w.timer.Reset(w.timeout - elapsed)
		w.mu.Unlock()
		return
	}
	w.fired = true
	w.stopped = true
	w.mu.Unlock()
	w.cancel()
}

// stop retires the watchdog. It never fires again afterwards.
func (w *stallWatchdog) stop() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stopped {
		return
	}
	w.stopped = true
	if w.timer != nil {
		w.timer.Stop()
	}
}

// stallErr returns the error to report when the watchdog tripped, or nil when
// the failure came from somewhere else (the caller cancelled, the connection
// broke on its own).
func (w *stallWatchdog) stallErr() *StallError {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.fired {
		return nil
	}
	return &StallError{
		Provider: w.provider,
		Model:    w.l.model,
		Phase:    w.phase,
		Timeout:  w.timeout,
	}
}
