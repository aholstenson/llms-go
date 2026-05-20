package llms

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("parseRetryAfterHeaders", func() {
	It("parses Retry-After-Ms in milliseconds", func() {
		h := http.Header{}
		h.Set("Retry-After-Ms", "1500")
		d, ok := parseRetryAfterHeaders(h)
		Expect(ok).To(BeTrue())
		Expect(d).To(Equal(1500 * time.Millisecond))
	})

	It("parses Retry-After in seconds", func() {
		h := http.Header{}
		h.Set("Retry-After", "2")
		d, ok := parseRetryAfterHeaders(h)
		Expect(ok).To(BeTrue())
		Expect(d).To(Equal(2 * time.Second))
	})

	It("parses Retry-After as an HTTP-date", func() {
		future := time.Now().Add(5 * time.Second).UTC().Format(http.TimeFormat)
		h := http.Header{}
		h.Set("Retry-After", future)
		d, ok := parseRetryAfterHeaders(h)
		Expect(ok).To(BeTrue())
		Expect(d).To(BeNumerically("~", 5*time.Second, 2*time.Second))
	})

	It("prefers Retry-After-Ms when both are present", func() {
		h := http.Header{}
		h.Set("Retry-After-Ms", "250")
		h.Set("Retry-After", "30")
		d, ok := parseRetryAfterHeaders(h)
		Expect(ok).To(BeTrue())
		Expect(d).To(Equal(250 * time.Millisecond))
	})

	It("returns (0,false) when neither header is set", func() {
		h := http.Header{}
		d, ok := parseRetryAfterHeaders(h)
		Expect(ok).To(BeFalse())
		Expect(d).To(BeZero())
	})

	It("returns (0,false) for a nil header set", func() {
		d, ok := parseRetryAfterHeaders(nil)
		Expect(ok).To(BeFalse())
		Expect(d).To(BeZero())
	})
})

var _ = Describe("UnavailableError", func() {
	It("recovers fields through errors.As", func() {
		original := &UnavailableError{
			StatusCode:    429,
			RetryAfter:    250 * time.Millisecond,
			HasRetryAfter: true,
			Attempts:      3,
			PartialOutput: true,
		}
		wrapped := fmt.Errorf("boom: %w", original)
		var recovered *UnavailableError
		Expect(errors.As(wrapped, &recovered)).To(BeTrue())
		Expect(recovered).To(Equal(original))
	})

	It("unwraps to the underlying cause", func() {
		cause := errors.New("rate limited by upstream")
		ue := &UnavailableError{StatusCode: 429, Cause: cause}
		Expect(errors.Is(ue, cause)).To(BeTrue())
	})
})
