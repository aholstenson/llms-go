package llms

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ExponentialBackoff", func() {
	It("returns server hint verbatim when hasRetryAfter is true", func() {
		eb := &ExponentialBackoff{Base: 500 * time.Millisecond, Max: 8 * time.Second, Jitter: 0}
		d := eb.Delay(0, 250*time.Millisecond, true)
		Expect(d).To(Equal(250 * time.Millisecond))
	})

	It("scales Base * 2^attempt without jitter", func() {
		eb := &ExponentialBackoff{Base: 100 * time.Millisecond, Max: 10 * time.Second, Jitter: 0}
		Expect(eb.Delay(0, 0, false)).To(Equal(100 * time.Millisecond))
		Expect(eb.Delay(1, 0, false)).To(Equal(200 * time.Millisecond))
		Expect(eb.Delay(2, 0, false)).To(Equal(400 * time.Millisecond))
	})

	It("caps at Max", func() {
		eb := &ExponentialBackoff{Base: 1 * time.Second, Max: 2 * time.Second, Jitter: 0}
		Expect(eb.Delay(10, 0, false)).To(Equal(2 * time.Second))
	})

	It("applies jitter in [delay*(1-Jitter), delay)", func() {
		eb := &ExponentialBackoff{Base: 1 * time.Second, Max: 10 * time.Second, Jitter: 0.5}
		for i := 0; i < 20; i++ {
			d := eb.Delay(0, 0, false)
			Expect(d).To(BeNumerically(">=", 500*time.Millisecond))
			Expect(d).To(BeNumerically("<=", 1*time.Second))
		}
	})
})

var _ = Describe("retryLoop", func() {
	defaultOpts := func() *generateContentOptions {
		return &generateContentOptions{
			MaxRetries:    2,
			RetryAfterCap: 60 * time.Second,
			RetryBackoff:  &ExponentialBackoff{Base: 1 * time.Millisecond, Max: 2 * time.Millisecond, Jitter: 0},
		}
	}

	It("returns the value on first success", func() {
		var calls int32
		v, err := retryLoop(context.Background(), defaultOpts(), "", "",
			func(error) (bool, int, time.Duration, bool) { return false, 0, 0, false },
			func(ctx context.Context) (int, error) {
				atomic.AddInt32(&calls, 1)
				return 42, nil
			})
		Expect(err).ToNot(HaveOccurred())
		Expect(v).To(Equal(42))
		Expect(calls).To(Equal(int32(1)))
	})

	It("retries up to MaxRetries+1 attempts and then returns UnavailableError", func() {
		var calls int32
		classify := func(error) (bool, int, time.Duration, bool) {
			return true, 429, 100 * time.Millisecond, true
		}
		_, err := retryLoop(context.Background(), defaultOpts(), "", "", classify,
			func(ctx context.Context) (int, error) {
				atomic.AddInt32(&calls, 1)
				return 0, errors.New("boom")
			})
		Expect(err).To(HaveOccurred())
		Expect(calls).To(Equal(int32(3)))

		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.StatusCode).To(Equal(429))
		Expect(ue.RetryAfter).To(Equal(100 * time.Millisecond))
		Expect(ue.HasRetryAfter).To(BeTrue())
		Expect(ue.Attempts).To(Equal(3))
	})

	It("MaxRetries=0 results in a single attempt", func() {
		var calls int32
		opts := defaultOpts()
		opts.MaxRetries = 0
		classify := func(error) (bool, int, time.Duration, bool) {
			return true, 503, 0, false
		}
		_, err := retryLoop(context.Background(), opts, "", "", classify,
			func(ctx context.Context) (int, error) {
				atomic.AddInt32(&calls, 1)
				return 0, errors.New("boom")
			})
		Expect(err).To(HaveOccurred())
		Expect(calls).To(Equal(int32(1)))
		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.Attempts).To(Equal(1))
	})

	It("does not retry on non-retryable errors", func() {
		var calls int32
		classify := func(error) (bool, int, time.Duration, bool) {
			return false, 400, 0, false
		}
		expected := errors.New("nope")
		_, err := retryLoop(context.Background(), defaultOpts(), "", "", classify,
			func(ctx context.Context) (int, error) {
				atomic.AddInt32(&calls, 1)
				return 0, expected
			})
		Expect(err).To(MatchError(expected))
		Expect(calls).To(Equal(int32(1)))
	})

	It("clamps the surfaced RetryAfter to RetryAfterCap", func() {
		opts := defaultOpts()
		opts.RetryAfterCap = 50 * time.Millisecond
		opts.MaxRetries = 0
		classify := func(error) (bool, int, time.Duration, bool) {
			return true, 429, 10 * time.Second, true
		}
		_, err := retryLoop(context.Background(), opts, "", "", classify,
			func(ctx context.Context) (int, error) { return 0, errors.New("boom") })
		var ue *UnavailableError
		Expect(errors.As(err, &ue)).To(BeTrue())
		Expect(ue.RetryAfter).To(Equal(50 * time.Millisecond))
	})

	It("succeeds when a later attempt returns no error", func() {
		var calls int32
		classify := func(error) (bool, int, time.Duration, bool) {
			return true, 429, 0, false
		}
		v, err := retryLoop(context.Background(), defaultOpts(), "", "", classify,
			func(ctx context.Context) (int, error) {
				n := atomic.AddInt32(&calls, 1)
				if n < 3 {
					return 0, errors.New("transient")
				}
				return 7, nil
			})
		Expect(err).ToNot(HaveOccurred())
		Expect(v).To(Equal(7))
		Expect(calls).To(Equal(int32(3)))
	})
})

var _ = Describe("headerCapturingTransport", func() {
	It("captures the latest response headers across requests", func() {
		i := int32(0)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			n := atomic.AddInt32(&i, 1)
			w.Header().Set("X-Test", "v")
			if n == 1 {
				w.Header().Set("Retry-After", "5")
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()

		transport := newHeaderCapturingTransport(http.DefaultTransport)
		client := &http.Client{Transport: transport}

		resp1, err := client.Get(srv.URL)
		Expect(err).ToNot(HaveOccurred())
		resp1.Body.Close()

		h := transport.LastHeaders()
		Expect(h).ToNot(BeNil())
		Expect(h.Get("Retry-After")).To(Equal("5"))

		resp2, err := client.Get(srv.URL)
		Expect(err).ToNot(HaveOccurred())
		resp2.Body.Close()

		h = transport.LastHeaders()
		Expect(h.Get("Retry-After")).To(Equal(""))
		Expect(h.Get("X-Test")).To(Equal("v"))
	})

	It("returns nil headers before any response is observed", func() {
		transport := newHeaderCapturingTransport(http.DefaultTransport)
		Expect(transport.LastHeaders()).To(BeNil())
	})
})
