package llms

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"google.golang.org/genai"
)

// flushingResponseWriter forces an explicit flush after each write so the
// streaming SDK sees an SSE chunk reach the wire before the next one is
// appended. Without this the entire body is buffered and our "first chunk
// succeeds, second chunk errors" timing collapses.
func writeSSE(w http.ResponseWriter, payload string) {
	if _, err := fmt.Fprint(w, payload); err != nil {
		return
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

var _ = Describe("Google", func() {
	Context("Google streaming mid-stream error", func() {
		It("attaches the partial usage already reported to the surfaced UnavailableError", func() {
			ctx := context.Background()

			// Server emits one valid chunk carrying text + usage, then a
			// well-formed APIError line. genai's stream decoder yields the
			// valid chunk first, then yields (nil, APIError) — exactly the
			// mid-stream-failure shape that drops usage if we don't carry
			// lastResponse out of handleStreaming.
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				writeSSE(w, "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hi\"}]}}],"+
					"\"usageMetadata\":{\"promptTokenCount\":11,\"candidatesTokenCount\":4,\"cachedContentTokenCount\":3,\"thoughtsTokenCount\":2}}\n\n")
				writeSSE(w, "{\"error\":{\"code\":503,\"message\":\"overloaded\",\"status\":\"UNAVAILABLE\"}}\n\n")
			}))
			DeferCleanup(srv.Close)

			client, err := genai.NewClient(ctx, &genai.ClientConfig{
				APIKey:      "test",
				Backend:     genai.BackendGeminiAPI,
				HTTPClient:  srv.Client(),
				HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
			})
			Expect(err).NotTo(HaveOccurred())

			m := &googleModel{
				logger:     discardLogger(),
				metrics:    NewNoopMetrics(),
				client:     client,
				statsModel: "google/m",
				model:      "m",
				info:       ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}},
			}

			var streamed string
			_, err = m.GenerateContent(ctx,
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
				WithStreamingFunc(func(_ context.Context, evt StreamingEvent) error {
					if chunk, ok := evt.(StreamingEventTextChunk); ok {
						streamed += chunk.Text
					}
					return nil
				}),
			)
			Expect(err).To(HaveOccurred())

			Expect(streamed).To(Equal("hi"))
			Expect(errors.Is(err, ErrStreamingPartialOutput)).To(BeTrue())

			var ue *UnavailableError
			Expect(errors.As(err, &ue)).To(BeTrue())
			Expect(ue.PartialOutput).To(BeTrue())
			Expect(ue.PartialUsage).NotTo(BeNil(), "PartialUsage must carry the usage reported before the mid-stream error")
			// PromptTokenCount(11) - CachedContentTokenCount(3) = 8 uncached input.
			Expect(ue.PartialUsage.InputTokens).To(Equal(int64(8)))
			Expect(ue.PartialUsage.OutputTokens).To(Equal(int64(4)))
			Expect(ue.PartialUsage.CachedReadTokens).To(Equal(int64(3)))
			Expect(ue.PartialUsage.ThinkingTokens).To(Equal(int64(2)))
		})

		It("returns a nil PartialUsage when the stream errors before any usage was reported", func() {
			ctx := context.Background()

			// Error chunk first — no usage observed before the failure.
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				writeSSE(w, "{\"error\":{\"code\":429,\"message\":\"rate limited\",\"status\":\"RESOURCE_EXHAUSTED\"}}\n\n")
			}))
			DeferCleanup(srv.Close)

			client, err := genai.NewClient(ctx, &genai.ClientConfig{
				APIKey:      "test",
				Backend:     genai.BackendGeminiAPI,
				HTTPClient:  srv.Client(),
				HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
			})
			Expect(err).NotTo(HaveOccurred())

			m := &googleModel{
				logger:     discardLogger(),
				metrics:    NewNoopMetrics(),
				client:     client,
				statsModel: "google/m",
				model:      "m",
				info:       ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}},
			}

			_, err = m.GenerateContent(ctx,
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))),
				WithStreamingFunc(func(_ context.Context, _ StreamingEvent) error { return nil }),
			)
			Expect(err).To(HaveOccurred())

			var ue *UnavailableError
			Expect(errors.As(err, &ue)).To(BeTrue())
			Expect(ue.PartialOutput).To(BeFalse())
			Expect(ue.PartialUsage).To(BeNil())
		})
	})
})
