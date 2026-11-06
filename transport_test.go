package llms

import (
	"bytes"
	"errors"
	"io"
	"net"
	"net/http"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
)

// silentH2Server accepts HTTP/2 connections without TLS. When answer is true it
// sends the headers and one chunk of body for each request; then it goes
// silent and does not answer pings or any other frame. This is how a dead
// connection looks to the client.
func silentH2Server(answer bool) (url string, stop func()) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	Expect(err).NotTo(HaveOccurred())
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go serveSilentH2(c, answer)
		}
	}()
	return "http://" + ln.Addr().String(), func() { _ = ln.Close() }
}

func serveSilentH2(c net.Conn, answer bool) {
	defer func() { _ = c.Close() }()
	preface := make([]byte, len(http2.ClientPreface))
	if _, err := io.ReadFull(c, preface); err != nil {
		return
	}
	fr := http2.NewFramer(c, c)
	if err := fr.WriteSettings(); err != nil {
		return
	}
	for {
		f, err := fr.ReadFrame()
		if err != nil {
			return
		}
		switch f := f.(type) {
		case *http2.SettingsFrame:
			if !f.IsAck() {
				_ = fr.WriteSettingsAck()
			}
		case *http2.HeadersFrame:
			if !answer {
				continue
			}
			var block bytes.Buffer
			_ = hpack.NewEncoder(&block).WriteField(hpack.HeaderField{Name: ":status", Value: "200"})
			_ = fr.WriteHeaders(http2.HeadersFrameParam{
				StreamID:      f.StreamID,
				BlockFragment: block.Bytes(),
				EndHeaders:    true,
			})
			_ = fr.WriteData(f.StreamID, false, []byte("data: x\n\n"))
		}
		// Pings and all other frames get no answer.
	}
}

// fastHealthCheckClient is a provider client whose base transport speaks
// HTTP/2 without TLS and finds a dead connection in about 200ms.
func fastHealthCheckClient() (*http.Client, *http.Transport) {
	t := withHealthCheck(http.DefaultTransport.(*http.Transport), 100*time.Millisecond, 100*time.Millisecond)
	t.Proxy = nil
	t.Protocols = new(http.Protocols)
	t.Protocols.SetUnencryptedHTTP2(true)
	return newAuthHTTPClient(t, StaticCredentials("k"), "openai", applyBearerCredential), t
}

var _ = Describe("HTTP/2 health check", Serial, func() {
	It("turns on pings for the shared transport", func() {
		t, ok := sharedTransport().(*http.Transport)
		Expect(ok).To(BeTrue())
		Expect(t.HTTP2).NotTo(BeNil())
		Expect(t.HTTP2.SendPingTimeout).To(Equal(healthCheckIdle))
		Expect(t.HTTP2.PingTimeout).To(Equal(healthCheckPingTimeout))
		Expect(t).NotTo(BeIdenticalTo(http.DefaultTransport))
	})

	It("makes a new copy when http.DefaultTransport is replaced", func() {
		original := http.DefaultTransport
		defer func() { http.DefaultTransport = original }()

		first := sharedTransport()
		Expect(sharedTransport()).To(BeIdenticalTo(first))

		http.DefaultTransport = &http.Transport{MaxIdleConns: 7}
		second, ok := sharedTransport().(*http.Transport)
		Expect(ok).To(BeTrue())
		Expect(second).NotTo(BeIdenticalTo(first))
		Expect(second).NotTo(BeIdenticalTo(http.DefaultTransport))
		Expect(second.MaxIdleConns).To(Equal(7))
		Expect(second.HTTP2.SendPingTimeout).To(Equal(healthCheckIdle))
	})

	It("uses an http.DefaultTransport that is not an *http.Transport as it is", func() {
		original := http.DefaultTransport
		defer func() { http.DefaultTransport = original }()

		mock := &mockTransport{}
		http.DefaultTransport = mock
		Expect(sharedTransport()).To(BeIdenticalTo(mock))
		Expect(healthCheckWindow(mock)).To(BeZero())
	})

	It("finds the health check window through the package wrappers", func() {
		base := withHealthCheck(&http.Transport{}, time.Second, 0)
		base.HTTP2.PingTimeout = 0
		wrapped := newAuthTransport(newHeaderCapturingTransport(base), StaticCredentials("k"), "google", applyGoogleCredential)
		Expect(healthCheckWindow(wrapped)).To(Equal(time.Second + goDefaultPingTimeout))
		Expect(healthCheckWindow(&http.Transport{})).To(BeZero())
	})

	It("keeps ping settings that the base transport already has", func() {
		base := &http.Transport{HTTP2: &http.HTTP2Config{SendPingTimeout: time.Minute}}
		t := withHealthCheck(base, time.Second, 2*time.Second)
		Expect(t.HTTP2.SendPingTimeout).To(Equal(time.Minute))
		Expect(t.HTTP2.PingTimeout).To(Equal(2 * time.Second))
		Expect(base.HTTP2.PingTimeout).To(BeZero())
	})

	It("reports a connection that dies mid-stream as a retryable stall", func() {
		url, stop := silentH2Server(true)
		defer stop()
		client, t := fastHealthCheckClient()
		defer t.CloseIdleConnections()

		resp, err := client.Get(url)
		Expect(err).NotTo(HaveOccurred())
		defer func() { _ = resp.Body.Close() }()
		Expect(resp.ProtoMajor).To(Equal(2))

		start := time.Now()
		_, err = io.ReadAll(resp.Body)
		Expect(time.Since(start)).To(BeNumerically("<", 5*time.Second))

		var se *StallError
		Expect(errors.As(err, &se)).To(BeTrue(), "got %v", err)
		Expect(se.Provider).To(Equal("openai"))
		Expect(se.Phase).To(Equal(StallPhaseStream))
		Expect(se.Timeout).To(Equal(200 * time.Millisecond))
		Expect(IsRetryable(err)).To(BeTrue())
	})

	It("reports a connection that dies before the response as a retryable stall", func() {
		url, stop := silentH2Server(false)
		defer stop()
		client, t := fastHealthCheckClient()
		defer t.CloseIdleConnections()

		start := time.Now()
		_, err := client.Get(url)
		Expect(time.Since(start)).To(BeNumerically("<", 5*time.Second))

		var se *StallError
		Expect(errors.As(err, &se)).To(BeTrue(), "got %v", err)
		Expect(se.Phase).To(Equal(StallPhaseResponse))
		Expect(IsRetryable(err)).To(BeTrue())
	})

	It("passes other transport errors through unchanged", func() {
		boom := errors.New("boom")
		t := newConnLossTransport(roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, boom
		}), "openai")
		req, _ := http.NewRequest(http.MethodGet, "http://example.invalid", nil)
		_, err := t.RoundTrip(req)
		Expect(err).To(BeIdenticalTo(boom))
	})
})

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// mockTransport stands in for a test mock installed as http.DefaultTransport.
type mockTransport struct{}

func (*mockTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("mock")
}
