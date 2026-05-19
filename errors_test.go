package llms

import (
	"fmt"

	"github.com/cockroachdb/errors"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("isUnavailableStatusCode", func() {
	DescribeTable("returns expected result for status code",
		func(code int, expected bool) {
			Expect(isUnavailableStatusCode(code)).To(Equal(expected))
		},
		Entry("429 (rate limited)", 429, true),
		Entry("503 (service unavailable)", 503, true),
		Entry("529 (overloaded)", 529, true),
		Entry("200 (ok)", 200, false),
		Entry("400 (bad request)", 400, false),
		Entry("401 (unauthorized)", 401, false),
		Entry("500 (internal server error)", 500, false),
		Entry("502 (bad gateway)", 502, false),
		Entry("504 (gateway timeout)", 504, false),
	)
})

var _ = Describe("ErrModelUnavailable", func() {
	It("is detected through Mark and Wrap", func() {
		providerErr := fmt.Errorf("anthropic API returned 503")
		wrapped := errors.Wrap(providerErr, "Anthropic model unavailable")
		marked := errors.Mark(wrapped, ErrModelUnavailable)

		Expect(errors.Is(marked, ErrModelUnavailable)).To(BeTrue())
	})

	It("is detected through additional wrapping by callers", func() {
		providerErr := fmt.Errorf("anthropic API returned 503")
		wrapped := errors.Wrap(providerErr, "Anthropic model unavailable")
		marked := errors.Mark(wrapped, ErrModelUnavailable)
		callerErr := errors.Wrap(marked, "failed to generate explore paths")

		Expect(errors.Is(callerErr, ErrModelUnavailable)).To(BeTrue())
	})

	It("is not detected on unrelated errors", func() {
		unrelated := errors.New("some other error")

		Expect(errors.Is(unrelated, ErrModelUnavailable)).To(BeFalse())
	})
})
