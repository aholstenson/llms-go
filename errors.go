package llms

import "github.com/cockroachdb/errors"

// ErrModelUnavailable indicates that the LLM provider is temporarily
// unavailable (overloaded, rate-limited, etc.). Callers can check for this
// with errors.Is to decide whether to degrade gracefully or surface a
// specific error code to the user.
var ErrModelUnavailable = errors.New("model is temporarily unavailable")

// isUnavailableStatusCode returns true for HTTP status codes that indicate
// the provider is temporarily unavailable:
//   - 429: Too Many Requests / rate limited
//   - 503: Service Unavailable / overloaded
//   - 529: Overloaded (Anthropic-specific)
func isUnavailableStatusCode(code int) bool {
	return code == 429 || code == 503 || code == 529
}
