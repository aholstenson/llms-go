package llms

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// HTTP/2 health check timing for provider connections. When a connection
// receives no frame for healthCheckIdle, the transport sends a ping. When the
// ping gets no answer within healthCheckPingTimeout, the transport closes the
// connection. A dead connection is thus found within about 30 seconds and
// removed from the pool, so the next attempt opens a new one.
//
// The ping is answered by the provider's HTTP/2 stack, not by the model, so
// the check works the same while the model reasons, streams or waits for its
// first token.
const (
	healthCheckIdle        = 15 * time.Second
	healthCheckPingTimeout = 15 * time.Second
)

// sharedBase holds the base transport of every provider client. All providers
// use one value, so they share a connection pool, as they do with
// http.DefaultTransport.
var sharedBase struct {
	mu sync.Mutex
	// from is the http.DefaultTransport value that rt was copied from.
	from *http.Transport
	rt   *http.Transport
}

// sharedTransport returns the base transport for a new provider client: a copy
// of http.DefaultTransport with the HTTP/2 health check turned on.
//
// The copy is made again when http.DefaultTransport is set to a different
// value, so a program or a test that replaces it before it creates a model
// gets its own settings. Models that exist already keep the copy they got.
// Changes to the fields of the same value after the copy are not seen.
//
// When http.DefaultTransport is not an *http.Transport (for example an
// instrumentation wrapper or a test mock), it is used as it is, without
// health checks.
func sharedTransport() http.RoundTripper {
	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		return http.DefaultTransport
	}
	sharedBase.mu.Lock()
	defer sharedBase.mu.Unlock()
	if sharedBase.from != base {
		sharedBase.from = base
		sharedBase.rt = withHealthCheck(base, healthCheckIdle, healthCheckPingTimeout)
	}
	return sharedBase.rt
}

// withHealthCheck returns a copy of base that sends HTTP/2 pings on a quiet
// connection and closes the connection when a ping gets no answer. Values
// that base already sets are kept.
func withHealthCheck(base *http.Transport, idle, pingTimeout time.Duration) *http.Transport {
	t := base.Clone()
	if t.HTTP2 == nil {
		t.HTTP2 = &http.HTTP2Config{}
	}
	if t.HTTP2.SendPingTimeout == 0 {
		t.HTTP2.SendPingTimeout = idle
	}
	if t.HTTP2.PingTimeout == 0 {
		t.HTTP2.PingTimeout = pingTimeout
	}
	return t
}

// lostConnectionMessage is the text of the error that net/http gives to the
// requests on a connection it closed because a health check ping got no
// answer. net/http does not export this error, so it is found by its text.
const lostConnectionMessage = "http2: client connection lost"

func isLostConnection(err error) bool {
	return err != nil && strings.Contains(err.Error(), lostConnectionMessage)
}

// goDefaultPingTimeout is the ping timeout net/http uses when
// HTTP2Config.PingTimeout is zero.
const goDefaultPingTimeout = 15 * time.Second

// healthCheckWindow returns how long a connection can be silent before the
// HTTP/2 health check of rt closes it, or 0 when rt has no health check. It
// looks through the wrappers of this package to find the *http.Transport.
func healthCheckWindow(rt http.RoundTripper) time.Duration {
	for {
		switch t := rt.(type) {
		case *http.Transport:
			if t.HTTP2 == nil || t.HTTP2.SendPingTimeout <= 0 {
				return 0
			}
			ping := t.HTTP2.PingTimeout
			if ping <= 0 {
				ping = goDefaultPingTimeout
			}
			return t.HTTP2.SendPingTimeout + ping
		case *authTransport:
			rt = t.inner
		case *headerCapturingTransport:
			rt = t.inner
		default:
			return 0
		}
	}
}

// connLossTransport reports a connection that failed its HTTP/2 health check
// as a *StallError. Without it, the error is a plain error that the retry loop
// does not try again. A connection that does not answer pings has stopped
// sending data, which is what a stall is, so the retry loop and Session
// recovery then handle it like any other stall.
type connLossTransport struct {
	inner    http.RoundTripper
	provider string
	// window is the health check window of inner, given as the Timeout of
	// the StallError.
	window time.Duration
}

func newConnLossTransport(inner http.RoundTripper, provider string) *connLossTransport {
	if inner == nil {
		inner = sharedTransport()
	}
	return &connLossTransport{inner: inner, provider: provider, window: healthCheckWindow(inner)}
}

// RoundTrip implements http.RoundTripper.
func (t *connLossTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	model := ""
	if l := livenessFrom(req.Context()); l != nil {
		model = l.model
	}

	resp, err := t.inner.RoundTrip(req)
	if err != nil {
		if isLostConnection(err) {
			return nil, t.stall(model, StallPhaseResponse)
		}
		return nil, err
	}
	resp.Body = &connLossBody{rc: resp.Body, stall: func() error {
		return t.stall(model, StallPhaseStream)
	}}
	return resp, nil
}

func (t *connLossTransport) stall(model string, phase StallPhase) *StallError {
	return &StallError{
		Provider: t.provider,
		Model:    model,
		Phase:    phase,
		Timeout:  t.window,
	}
}

// connLossBody turns a read that failed on a lost connection into a
// *StallError.
type connLossBody struct {
	rc    io.ReadCloser
	stall func() error
}

func (b *connLossBody) Read(p []byte) (int, error) {
	n, err := b.rc.Read(p)
	if err != nil && !errors.Is(err, io.EOF) && isLostConnection(err) {
		return n, b.stall()
	}
	return n, err
}

func (b *connLossBody) Close() error {
	return b.rc.Close()
}
