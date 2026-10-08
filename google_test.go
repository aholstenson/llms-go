package llms

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
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
				info:       fixedModelInfo(ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}}),
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
				info:       fixedModelInfo(ModelInfo{Caps: Capabilities{Temperature: true, ToolCall: true}}),
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

	Context("temperature", func() {
		known := ModelInfo{Caps: Capabilities{Temperature: true}}

		DescribeTable("sends a custom temperature only to models that accept one",
			func(model string, info ModelInfo, sent bool) {
				turn := googleNamedTurnFor(model, info, WithTemperature(0.3))
				if sent {
					Expect(turn.config.Temperature).NotTo(BeNil())
					Expect(*turn.config.Temperature).To(Equal(float32(0.3)))
				} else {
					Expect(turn.config.Temperature).To(BeNil())
				}
			},
			Entry("Gemini 2.5", "gemini-2.5-pro", known, true),
			Entry("Gemini 3.5", "gemini-3.5-flash", known, true),
			Entry("Gemini 3 without a minor version", "gemini-3-pro-preview", known, true),
			Entry("Gemini 3.6", "gemini-3.6-flash", known, false),
			Entry("Gemini 3.10", "gemini-3.10-flash", known, false),
			Entry("Gemini 4", "gemini-4-flash", ModelInfo{}, false),
			Entry("a name with a models/ prefix", "models/gemini-3.6-flash", known, false),
			Entry("an alias", "gemini-flash-latest", known, false),
			Entry("an unknown model", "m", ModelInfo{}, false),
			Entry("a known model without a Gemini version", "gemma-4-31b-it", known, true),
			Entry("a model without temperature support", "gemini-2.5-pro", ModelInfo{Caps: Capabilities{ToolCall: true}}, false),
		)
	})

	Context("cost accounting", func() {
		It("splits Gemini usage into disjoint counters that price correctly", func() {
			ctx := context.Background()

			// Gemini reports promptTokenCount inclusive of
			// cachedContentTokenCount, and thoughtsTokenCount disjoint from
			// candidatesTokenCount (totalTokenCount is the sum of prompt,
			// candidates, tool-use prompt and thoughts).
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],`+
					`"usageMetadata":{"promptTokenCount":10000,"cachedContentTokenCount":4000,`+
					`"candidatesTokenCount":1000,"thoughtsTokenCount":500,"totalTokenCount":11500}}`)
			}))
			DeferCleanup(srv.Close)

			client, err := genai.NewClient(ctx, &genai.ClientConfig{
				APIKey:      "test",
				Backend:     genai.BackendGeminiAPI,
				HTTPClient:  srv.Client(),
				HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
			})
			Expect(err).NotTo(HaveOccurred())

			info, ok := LookupModelInfo("google/gemini-2.5-flash")
			Expect(ok).To(BeTrue())

			m := &googleModel{
				logger:     discardLogger(),
				metrics:    NewNoopMetrics(),
				client:     client,
				statsModel: "google/gemini-2.5-flash",
				model:      "gemini-2.5-flash",
				info:       fixedModelInfo(info),
			}

			recorder := NewMetricsRecorder()
			_, err = m.GenerateContent(WithMetrics(ctx, recorder),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))))
			Expect(err).NotTo(HaveOccurred())

			counters := recorder.GetStats().Success["google/gemini-2.5-flash"]
			// 10000 prompt - 4000 cached = 6000 fresh input; the cached tokens
			// are billed once, at the cache-read rate.
			Expect(counters["input_tokens"]).To(Equal(6000))
			Expect(counters["cached_read_tokens"]).To(Equal(4000))
			Expect(counters["output_tokens"]).To(Equal(1000))
			Expect(counters["thinking_tokens"]).To(Equal(500))

			pm := NewPricingManager(slog.New(slog.NewTextHandler(io.Discard, nil)))
			costs := pm.CalculateCosts(recorder.GetStats())

			// Fresh input at the input rate, cached prompt tokens at the
			// cache-read rate, and thinking tokens at the output rate on top
			// of the candidate tokens. Derived from the embedded rates so the
			// test survives a models.dev price refresh.
			want := (6000*info.Cost.Input + 4000*info.Cost.CacheRead + (1000+500)*info.Cost.Output) / 1e6
			Expect(costs["google/gemini-2.5-flash"]).To(BeNumerically("~", want, 1e-9))
		})
	})
})
