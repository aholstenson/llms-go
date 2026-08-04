package llms

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/anthropics/anthropic-sdk-go/option"
)

var _ = Describe("Credentials", func() {
	Context("EnvCredentials", func() {
		It("reads the provider's environment variable", func() {
			GinkgoT().Setenv("ANTHROPIC_API_KEY", "ant-key")

			cred, err := (&EnvCredentials{}).Credential(context.Background(), "anthropic")
			Expect(err).NotTo(HaveOccurred())
			Expect(cred.APIKey).To(Equal("ant-key"))
		})

		It("falls back from GEMINI_API_KEY to GOOGLE_API_KEY for google", func() {
			GinkgoT().Setenv("GEMINI_API_KEY", "")
			GinkgoT().Setenv("GOOGLE_API_KEY", "google-key")

			cred, err := (&EnvCredentials{}).Credential(context.Background(), "google")
			Expect(err).NotTo(HaveOccurred())
			Expect(cred.APIKey).To(Equal("google-key"))
		})

		It("prefers GEMINI_API_KEY over GOOGLE_API_KEY", func() {
			GinkgoT().Setenv("GEMINI_API_KEY", "gemini-key")
			GinkgoT().Setenv("GOOGLE_API_KEY", "google-key")

			cred, err := (&EnvCredentials{}).Credential(context.Background(), "google")
			Expect(err).NotTo(HaveOccurred())
			Expect(cred.APIKey).To(Equal("gemini-key"))
		})

		It("caches a resolved key so later lookups do not re-read the environment", func() {
			GinkgoT().Setenv("OPENAI_API_KEY", "first")
			creds := &EnvCredentials{}

			cred, err := creds.Credential(context.Background(), "openai")
			Expect(err).NotTo(HaveOccurred())
			Expect(cred.APIKey).To(Equal("first"))

			// The source sits in the hot path of every request; once resolved
			// it must not go back to the environment.
			GinkgoT().Setenv("OPENAI_API_KEY", "second")
			cred, err = creds.Credential(context.Background(), "openai")
			Expect(err).NotTo(HaveOccurred())
			Expect(cred.APIKey).To(Equal("first"))
		})

		It("does not cache failures, so a late-populated environment still works", func() {
			GinkgoT().Setenv("OPENROUTER_API_KEY", "")
			creds := &EnvCredentials{}

			_, err := creds.Credential(context.Background(), "openrouter")
			Expect(errors.Is(err, ErrNoCredentials)).To(BeTrue())
			Expect(err.Error()).To(ContainSubstring("OPENROUTER_API_KEY"))

			GinkgoT().Setenv("OPENROUTER_API_KEY", "late-key")
			cred, err := creds.Credential(context.Background(), "openrouter")
			Expect(err).NotTo(HaveOccurred())
			Expect(cred.APIKey).To(Equal("late-key"))
		})

		It("caches each provider independently", func() {
			GinkgoT().Setenv("ANTHROPIC_API_KEY", "ant")
			GinkgoT().Setenv("OPENAI_API_KEY", "oai")
			creds := &EnvCredentials{}

			ant, err := creds.Credential(context.Background(), "anthropic")
			Expect(err).NotTo(HaveOccurred())
			oai, err := creds.Credential(context.Background(), "openai")
			Expect(err).NotTo(HaveOccurred())

			Expect(ant.APIKey).To(Equal("ant"))
			Expect(oai.APIKey).To(Equal("oai"))
		})

		It("reports an unknown provider as a missing credential", func() {
			_, err := (&EnvCredentials{}).Credential(context.Background(), "nope")
			Expect(errors.Is(err, ErrNoCredentials)).To(BeTrue())
		})
	})

	Context("authTransport", func() {
		// captureHeaders serves a request and records the headers it saw.
		newCapturingServer := func(seen *atomic.Pointer[http.Header]) *httptest.Server {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				h := r.Header.Clone()
				seen.Store(&h)
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{}`))
			}))
			DeferCleanup(srv.Close)
			return srv
		}

		It("applies each provider's native auth header", func() {
			cases := []struct {
				provider string
				apply    credentialApplier
				header   string
				want     string
			}{
				{"anthropic", applyAnthropicCredential, "X-Api-Key", "k"},
				{"openai", applyBearerCredential, "Authorization", "Bearer k"},
				{"openrouter", applyBearerCredential, "Authorization", "Bearer k"},
				{"google", applyGoogleCredential, "X-Goog-Api-Key", "k"},
			}

			for _, tc := range cases {
				var seen atomic.Pointer[http.Header]
				srv := newCapturingServer(&seen)

				client := newAuthHTTPClient(nil, StaticCredentials("k"), tc.provider, tc.apply)
				resp, err := client.Get(srv.URL)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.Body.Close()).To(Succeed())

				Expect(seen.Load().Get(tc.header)).To(Equal(tc.want), "provider %s", tc.provider)
			}
		})

		It("resolves the credential on every request so rotation takes effect", func() {
			var seen atomic.Pointer[http.Header]
			srv := newCapturingServer(&seen)

			var calls atomic.Int64
			source := CredentialFunc(func(context.Context, string) (Credential, error) {
				return Credential{APIKey: fmt.Sprintf("key-%d", calls.Add(1))}, nil
			})

			client := newAuthHTTPClient(nil, source, "anthropic", applyAnthropicCredential)
			for _, want := range []string{"key-1", "key-2", "key-3"} {
				resp, err := client.Get(srv.URL)
				Expect(err).NotTo(HaveOccurred())
				Expect(resp.Body.Close()).To(Succeed())
				Expect(seen.Load().Get("X-Api-Key")).To(Equal(want))
			}
		})

		It("applies extra headers on top of the API key", func() {
			var seen atomic.Pointer[http.Header]
			srv := newCapturingServer(&seen)

			source := CredentialFunc(func(context.Context, string) (Credential, error) {
				return Credential{
					APIKey:  "k",
					Headers: http.Header{"X-Tenant": []string{"acme"}},
				}, nil
			})

			client := newAuthHTTPClient(nil, source, "anthropic", applyAnthropicCredential)
			resp, err := client.Get(srv.URL)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Body.Close()).To(Succeed())

			Expect(seen.Load().Get("X-Api-Key")).To(Equal("k"))
			Expect(seen.Load().Get("X-Tenant")).To(Equal("acme"))
		})

		It("authenticates with headers alone when no API key is supplied", func() {
			var seen atomic.Pointer[http.Header]
			srv := newCapturingServer(&seen)

			source := CredentialFunc(func(context.Context, string) (Credential, error) {
				return Credential{Headers: http.Header{"Authorization": []string{"Bearer gateway"}}}, nil
			})

			client := newAuthHTTPClient(nil, source, "anthropic", applyAnthropicCredential)
			resp, err := client.Get(srv.URL)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Body.Close()).To(Succeed())

			Expect(seen.Load().Get("Authorization")).To(Equal("Bearer gateway"))
			Expect(seen.Load().Get("X-Api-Key")).To(BeEmpty())
		})

		It("surfaces a credential failure without issuing the request", func() {
			var seen atomic.Pointer[http.Header]
			srv := newCapturingServer(&seen)

			source := CredentialFunc(func(context.Context, string) (Credential, error) {
				return Credential{}, fmt.Errorf("%w: vault is down", ErrNoCredentials)
			})

			client := newAuthHTTPClient(nil, source, "anthropic", applyAnthropicCredential)
			_, err := client.Get(srv.URL)
			Expect(err).To(HaveOccurred())
			Expect(errors.Is(err, ErrNoCredentials)).To(BeTrue())
			Expect(seen.Load()).To(BeNil(), "the request must not reach the server")
		})

		It("passes the request context to the credential source", func() {
			var seen atomic.Pointer[http.Header]
			srv := newCapturingServer(&seen)

			type tenantKey struct{}
			source := CredentialFunc(func(ctx context.Context, _ string) (Credential, error) {
				tenant, _ := ctx.Value(tenantKey{}).(string)
				if tenant == "" {
					return Credential{}, ErrNoCredentials
				}
				return Credential{APIKey: "key-for-" + tenant}, nil
			})

			ctx := context.WithValue(context.Background(), tenantKey{}, "acme")
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL, nil)
			Expect(err).NotTo(HaveOccurred())

			client := newAuthHTTPClient(nil, source, "anthropic", applyAnthropicCredential)
			resp, err := client.Do(req)
			Expect(err).NotTo(HaveOccurred())
			Expect(resp.Body.Close()).To(Succeed())

			Expect(seen.Load().Get("X-Api-Key")).To(Equal("key-for-acme"))
		})
	})

	Context("wiring through a provider model", func() {
		It("sends the resolved key and never the SDK placeholder", func() {
			var seen atomic.Pointer[http.Header]
			// Anthropic generations always run in streaming mode, so the
			// stub speaks SSE.
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				h := r.Header.Clone()
				seen.Store(&h)
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				for _, evt := range []string{
					`event: message_start` + "\n" +
						`data: {"type":"message_start","message":{"id":"msg_1","type":"message",` +
						`"role":"assistant","model":"m","content":[],"stop_reason":null,` +
						`"usage":{"input_tokens":1,"output_tokens":0}}}`,
					`event: content_block_start` + "\n" +
						`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
					`event: content_block_delta` + "\n" +
						`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
					`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
					`event: message_delta` + "\n" +
						`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
					`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
				} {
					writeSSE(w, evt+"\n\n")
				}
			}))
			DeferCleanup(srv.Close)

			m := newAnthropicModel(discardLogger(), NewNoopMetrics(), StaticCredentials("real-key"), "m", nil,
				ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}},
				option.WithBaseURL(srv.URL))

			_, err := m.GenerateContent(context.Background(),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))))
			Expect(err).NotTo(HaveOccurred())

			Expect(seen.Load().Get("X-Api-Key")).To(Equal("real-key"))
			Expect(seen.Load().Get("X-Api-Key")).NotTo(Equal(credentialPlaceholder))
		})
	})

	Context("Manager", func() {
		It("uses a supplied credential source instead of the environment", func() {
			GinkgoT().Setenv("ANTHROPIC_API_KEY", "")

			manager, err := NewManager(
				WithManagerLogger(discardLogger()),
				WithManagerCredentials(StaticCredentials("from-source")),
			)
			Expect(err).NotTo(HaveOccurred())

			model, err := manager.GetModel(context.Background(), "anthropic/claude-sonnet-4-5")
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
		})

		It("fails fast when the source has no credential for the provider", func() {
			manager, err := NewManager(
				WithManagerLogger(discardLogger()),
				WithManagerCredentials(CredentialFunc(func(context.Context, string) (Credential, error) {
					return Credential{}, fmt.Errorf("%w: nothing configured", ErrNoCredentials)
				})),
			)
			Expect(err).NotTo(HaveOccurred())

			_, err = manager.GetModel(context.Background(), "openai/gpt-5")
			Expect(err).To(HaveOccurred())
			Expect(errors.Is(err, ErrNoCredentials)).To(BeTrue())
			Expect(err.Error()).To(ContainSubstring("openai/gpt-5"))
		})

		It("rejects a nil credential source", func() {
			_, err := NewManager(WithManagerCredentials(nil))
			Expect(err).To(HaveOccurred())
		})

		It("defaults to reading the environment", func() {
			GinkgoT().Setenv("ANTHROPIC_API_KEY", "env-key")

			manager, err := NewManager(WithManagerLogger(discardLogger()))
			Expect(err).NotTo(HaveOccurred())

			_, err = manager.GetModel(context.Background(), "anthropic/claude-sonnet-4-5")
			Expect(err).NotTo(HaveOccurred())
		})
	})
})
