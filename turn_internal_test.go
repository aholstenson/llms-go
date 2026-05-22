package llms

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("uncachedInputTokens", func() {
	It("subtracts cached tokens from the prompt total", func() {
		// Google/OpenAI/OpenRouter report a prompt total with cached nested
		// inside it; the disjoint, uncached input is total - cached.
		Expect(uncachedInputTokens(1_069_551, 674_094)).To(Equal(int64(395_457)))
	})

	It("keeps input + cached equal to the original prompt total", func() {
		const promptTotal int64 = 1_069_551
		const cached int64 = 674_094
		Expect(uncachedInputTokens(promptTotal, cached) + cached).To(Equal(promptTotal))
	})

	It("returns the full total when nothing was cached", func() {
		Expect(uncachedInputTokens(500, 0)).To(Equal(int64(500)))
	})

	It("clamps at zero when cached meets or exceeds the total", func() {
		Expect(uncachedInputTokens(500, 500)).To(Equal(int64(0)))
		Expect(uncachedInputTokens(500, 800)).To(Equal(int64(0)))
	})
})
