package llms

import (
	"errors"
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// These tests verify the contract callers rely on when distinguishing
// pre-stream failures (safe to retry) from mid-stream failures (where
// retrying would replay tokens already shown to the user).
//
// The provider streaming paths in anthropic.go, openai.go, google.go and
// openrouter.go set the PartialOutput flag on the returned UnavailableError
// and join the ErrStreamingPartialOutput sentinel onto the error before
// returning it. The asserts below mirror what a caller would do.
var _ = Describe("Streaming partial output sentinel", func() {
	It("a pre-stream UnavailableError is not flagged partial", func() {
		ue := &UnavailableError{
			StatusCode:    429,
			Attempts:      3,
			PartialOutput: false,
		}
		Expect(errors.Is(ue, ErrStreamingPartialOutput)).To(BeFalse())

		var recovered *UnavailableError
		Expect(errors.As(ue, &recovered)).To(BeTrue())
		Expect(recovered.PartialOutput).To(BeFalse())
	})

	It("a mid-stream UnavailableError is marked partial and joined with the sentinel", func() {
		ue := &UnavailableError{
			StatusCode:    529,
			Attempts:      1,
			PartialOutput: true,
		}
		joined := errors.Join(ue, ErrStreamingPartialOutput)

		Expect(errors.Is(joined, ErrStreamingPartialOutput)).To(BeTrue())

		var recovered *UnavailableError
		Expect(errors.As(joined, &recovered)).To(BeTrue())
		Expect(recovered.PartialOutput).To(BeTrue())
	})

	It("PartialOutput survives an additional wrap", func() {
		ue := &UnavailableError{PartialOutput: true}
		wrapped := fmt.Errorf("downstream: %w", errors.Join(ue, ErrStreamingPartialOutput))

		Expect(errors.Is(wrapped, ErrStreamingPartialOutput)).To(BeTrue())
		var recovered *UnavailableError
		Expect(errors.As(wrapped, &recovered)).To(BeTrue())
		Expect(recovered.PartialOutput).To(BeTrue())
	})
})
