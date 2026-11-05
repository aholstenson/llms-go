package llms

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("isUnavailableStatusCode", func() {
	DescribeTable("returns expected result for status code",
		func(code int, expected bool) {
			Expect(isUnavailableStatusCode(code)).To(Equal(expected))
		},
		Entry("408 (request timeout)", 408, true),
		Entry("429 (rate limited)", 429, true),
		Entry("500 (internal server error)", 500, true),
		Entry("502 (bad gateway)", 502, true),
		Entry("503 (service unavailable)", 503, true),
		Entry("504 (gateway timeout)", 504, true),
		Entry("520 (Cloudflare unknown error)", 520, true),
		Entry("521 (Cloudflare web server down)", 521, true),
		Entry("522 (Cloudflare connection timed out)", 522, true),
		Entry("523 (Cloudflare origin unreachable)", 523, true),
		Entry("524 (Cloudflare timeout)", 524, true),
		Entry("529 (overloaded)", 529, true),
		Entry("200 (ok)", 200, false),
		Entry("400 (bad request)", 400, false),
		Entry("401 (unauthorized)", 401, false),
		Entry("403 (forbidden)", 403, false),
		Entry("404 (not found)", 404, false),
		Entry("409 (conflict)", 409, false),
		Entry("501 (not implemented)", 501, false),
		Entry("525 (Cloudflare SSL handshake failed)", 525, false),
		Entry("526 (Cloudflare invalid SSL certificate)", 526, false),
	)
})
