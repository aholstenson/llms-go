package llms

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
)

// ErrNoCredentials sentinels a failure to obtain credentials for a provider.
// CredentialSource implementations should wrap it so callers can detect a
// missing-credential condition with errors.Is, regardless of where the
// credentials were supposed to come from.
var ErrNoCredentials = errors.New("no credentials available")

// Credential carries the authentication material for a single provider
// request.
//
// APIKey is placed in the provider's native authentication header
// (`X-Api-Key` for Anthropic, `Authorization: Bearer` for OpenAI and
// OpenRouter, `x-goog-api-key` for Google). Headers are applied on top and
// can carry anything else the endpoint needs, such as a gateway token or a
// tenant identifier. A Credential with an empty APIKey and non-empty Headers
// is valid: it means authentication is carried entirely by those headers.
type Credential struct {
	// APIKey is the provider API key. May be empty when Headers carry the
	// authentication instead.
	APIKey string

	// Headers are set on every request after APIKey is applied. Existing
	// values for the same header name are replaced.
	Headers http.Header
}

// CredentialSource supplies provider authentication to a Manager. provider is
// the lowercase provider segment of a resolved model name — "anthropic",
// "openai", "openrouter" or "google".
//
// A source is consulted twice per model: once when Manager.GetModel builds
// the model, so a missing credential fails fast rather than surfacing as an
// opaque transport error later, and then once per outbound HTTP request. The
// per-request call means rotating or short-lived credentials take effect
// without rebuilding the model, but it also means implementations sit in the
// hot path of every request and should keep resolution cheap — cache, and
// refresh on expiry rather than on every call.
//
// Implementations that cannot resolve a credential outside of a request (for
// example one that reads a tenant from the context) should return the zero
// Credential and a nil error for the construction-time call, and do the real
// work in the per-request call.
//
// Credential must be safe for concurrent use.
type CredentialSource interface {
	Credential(ctx context.Context, provider string) (Credential, error)
}

// CredentialFunc adapts a function value to the CredentialSource interface.
type CredentialFunc func(ctx context.Context, provider string) (Credential, error)

// Credential implements CredentialSource.
func (f CredentialFunc) Credential(ctx context.Context, provider string) (Credential, error) {
	return f(ctx, provider)
}

// StaticCredentials returns a CredentialSource that hands out the same API key
// for every provider. Useful when the calling application already holds the
// key and only ever talks to one provider.
func StaticCredentials(apiKey string) CredentialSource {
	return CredentialFunc(func(context.Context, string) (Credential, error) {
		if apiKey == "" {
			return Credential{}, fmt.Errorf("%w: static credential is empty", ErrNoCredentials)
		}
		return Credential{APIKey: apiKey}, nil
	})
}

// credentialEnvVars lists the environment variables EnvCredentials consults
// for each provider, in priority order.
var credentialEnvVars = map[string][]string{
	"anthropic":  {"ANTHROPIC_API_KEY"},
	"openai":     {"OPENAI_API_KEY"},
	"openrouter": {"OPENROUTER_API_KEY"},
	"google":     {"GEMINI_API_KEY", "GOOGLE_API_KEY"},
}

// EnvCredentials reads provider API keys from environment variables and is the
// default CredentialSource used by a Manager:
//
//	anthropic   ANTHROPIC_API_KEY
//	openai      OPENAI_API_KEY
//	openrouter  OPENROUTER_API_KEY
//	google      GEMINI_API_KEY, falling back to GOOGLE_API_KEY
//
// Because a source is consulted on every outbound request, resolved keys are
// memoized per provider — the process environment is read once per provider
// and every later request is served from the cache. Only successful lookups
// are cached, so an application that populates the environment after the first
// (failed) attempt still picks the key up.
//
// The flip side is that a key changed in the environment after it has been
// read is not observed. That is deliberate: an environment variable is a
// process-startup input, not a rotation channel. Applications with rotating
// credentials should supply their own CredentialSource, which is re-consulted
// per request.
//
// The zero value is ready for use and safe for concurrent use.
type EnvCredentials struct {
	cache sync.Map // provider -> string
}

// Credential implements CredentialSource.
func (e *EnvCredentials) Credential(_ context.Context, provider string) (Credential, error) {
	if cached, ok := e.cache.Load(provider); ok {
		return Credential{APIKey: cached.(string)}, nil
	}

	names, ok := credentialEnvVars[provider]
	if !ok {
		return Credential{}, fmt.Errorf("%w: no environment variable known for provider %q", ErrNoCredentials, provider)
	}

	for _, name := range names {
		if key := os.Getenv(name); key != "" {
			e.cache.Store(provider, key)
			return Credential{APIKey: key}, nil
		}
	}

	return Credential{}, fmt.Errorf("%w: %s is not set", ErrNoCredentials, strings.Join(names, " or "))
}

// credentialPlaceholder is handed to provider SDK constructors that reject an
// empty API key. It is never sent: authTransport overwrites the authentication
// header on every request with the credential resolved at that moment. Keeping
// the real key out of the SDK config also keeps it out of SDK error messages
// that dump their configuration.
const credentialPlaceholder = "resolved-per-request-by-credential-source"

// credentialApplier writes a Credential's API key into the authentication
// header a given provider expects.
type credentialApplier func(req *http.Request, cred Credential)

func applyAnthropicCredential(req *http.Request, cred Credential) {
	req.Header.Set("X-Api-Key", cred.APIKey)
}

func applyBearerCredential(req *http.Request, cred Credential) {
	req.Header.Set("Authorization", "Bearer "+cred.APIKey)
}

func applyGoogleCredential(req *http.Request, cred Credential) {
	req.Header.Set("x-goog-api-key", cred.APIKey)
}

// authTransport resolves a credential from its CredentialSource and applies it
// to every outbound request. Resolving per request (rather than baking a key
// into the SDK client at construction) is what lets a Manager keep caching
// models by name while the credentials behind them rotate.
type authTransport struct {
	inner    http.RoundTripper
	source   CredentialSource
	provider string
	apply    credentialApplier
}

func newAuthTransport(inner http.RoundTripper, source CredentialSource, provider string, apply credentialApplier) *authTransport {
	if inner == nil {
		inner = http.DefaultTransport
	}
	return &authTransport{inner: inner, source: source, provider: provider, apply: apply}
}

// RoundTrip implements http.RoundTripper.
func (t *authTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	cred, err := t.source.Credential(req.Context(), t.provider)
	if err != nil {
		return nil, fmt.Errorf("resolving %s credentials: %w", t.provider, err)
	}

	// RoundTrip must not mutate the request it is handed.
	r := req.Clone(req.Context())
	if cred.APIKey != "" {
		t.apply(r, cred)
	}
	for name, values := range cred.Headers {
		r.Header[http.CanonicalHeaderKey(name)] = append([]string(nil), values...)
	}
	return t.inner.RoundTrip(r)
}

// newAuthHTTPClient builds an http.Client that authenticates every request
// from source. No client-level timeout is set: the provider SDKs drive
// deadlines from the request context.
func newAuthHTTPClient(inner http.RoundTripper, source CredentialSource, provider string, apply credentialApplier) *http.Client {
	return &http.Client{Transport: newAuthTransport(inner, source, provider, apply)}
}
