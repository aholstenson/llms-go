package llms

import (
	"context"
	"net/http"
	"net/http/httptest"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// valueRecorder is a MetricsRecorder with a type that cannot be compared.
type valueRecorder struct {
	calls map[string]int
}

func (v valueRecorder) RecordSuccess(serviceName string, _ Collector) { v.calls[serviceName]++ }
func (v valueRecorder) RecordFailure(serviceName string, _ Collector) { v.calls[serviceName]++ }

var _ = Describe("Manager recorder", func() {
	newCollector := func() Collector {
		collector := NewCollector()
		collector.Counter("calls").Add(1)
		return collector
	}

	Describe("recordSuccess and recordFailure", func() {
		It("records to the Manager recorder when the context has no recorder", func() {
			manager := NewMetricsRecorder()

			recordSuccess(context.Background(), manager, "svc", newCollector())
			recordFailure(context.Background(), manager, "svc", newCollector())

			Expect(manager.GetStats().Success["svc"]["calls"]).To(Equal(1))
			Expect(manager.GetStats().Failure["svc"]["calls"]).To(Equal(1))
		})

		It("records to the two recorders when the context has a recorder", func() {
			manager := NewMetricsRecorder()
			scoped := NewMetricsRecorder()
			ctx := WithMetrics(context.Background(), scoped)

			recordSuccess(ctx, manager, "svc", newCollector())
			recordFailure(ctx, manager, "svc", newCollector())

			for _, recorder := range []*DefaultMetricsRecorder{manager, scoped} {
				Expect(recorder.GetStats().Success["svc"]["calls"]).To(Equal(1))
				Expect(recorder.GetStats().Failure["svc"]["calls"]).To(Equal(1))
			}
		})

		It("records one time when the same recorder is in the two places", func() {
			recorder := NewMetricsRecorder()
			ctx := WithMetrics(context.Background(), recorder)

			recordSuccess(ctx, recorder, "svc", newCollector())
			recordFailure(ctx, recorder, "svc", newCollector())

			Expect(recorder.GetStats().Success["svc"]["calls"]).To(Equal(1))
			Expect(recorder.GetStats().Failure["svc"]["calls"]).To(Equal(1))
		})

		It("records to the context recorder when the Manager has no recorder", func() {
			scoped := NewMetricsRecorder()

			recordSuccess(WithMetrics(context.Background(), scoped), nil, "svc", newCollector())

			Expect(scoped.GetStats().Success["svc"]["calls"]).To(Equal(1))
		})

		It("accepts recorders that cannot be compared", func() {
			manager := valueRecorder{calls: map[string]int{}}
			scoped := valueRecorder{calls: map[string]int{}}

			recordSuccess(WithMetrics(context.Background(), scoped), manager, "svc", newCollector())

			Expect(manager.calls["svc"]).To(Equal(1))
			Expect(scoped.calls["svc"]).To(Equal(1))
		})
	})

	Describe("WithManagerMetricsRecorder", func() {
		It("rejects a nil recorder", func() {
			_, err := NewManager(WithManagerMetricsRecorder(nil))
			Expect(err).To(HaveOccurred())
		})

		It("records the calls of a model from the Manager", func() {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				for _, evt := range []string{
					`event: message_start` + "\n" +
						`data: {"type":"message_start","message":{"id":"msg_1","type":"message",` +
						`"role":"assistant","model":"m","content":[],"stop_reason":null,` +
						`"usage":{"input_tokens":7,"output_tokens":0}}}`,
					`event: content_block_start` + "\n" +
						`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
					`event: content_block_delta` + "\n" +
						`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
					`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}`,
					`event: message_delta` + "\n" +
						`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
					`event: message_stop` + "\n" + `data: {"type":"message_stop"}`,
				} {
					writeSSE(w, evt+"\n\n")
				}
			}))
			DeferCleanup(srv.Close)
			GinkgoT().Setenv("ANTHROPIC_BASE_URL", srv.URL)

			total := NewMetricsRecorder()
			manager, err := NewManager(
				WithManagerLogger(discardLogger()),
				WithManagerCredentials(StaticCredentials("key")),
				WithManagerMetricsRecorder(total),
			)
			Expect(err).NotTo(HaveOccurred())

			model, err := manager.GetModel(context.Background(), "anthropic/m")
			Expect(err).NotTo(HaveOccurred())

			// One call without a context recorder and one call with it.
			_, err = model.GenerateContent(context.Background(),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))))
			Expect(err).NotTo(HaveOccurred())

			scoped := NewMetricsRecorder()
			_, err = model.GenerateContent(WithMetrics(context.Background(), scoped),
				WithMessages(NewMessage(RoleUser, NewTextPart("hi"))))
			Expect(err).NotTo(HaveOccurred())

			Expect(total.GetStats().Success["anthropic/m"]).To(And(
				HaveKeyWithValue("calls", 2),
				HaveKeyWithValue("input_tokens", 14),
				HaveKeyWithValue("output_tokens", 6),
			))
			Expect(scoped.GetStats().Success["anthropic/m"]).To(And(
				HaveKeyWithValue("calls", 1),
				HaveKeyWithValue("input_tokens", 7),
				HaveKeyWithValue("output_tokens", 3),
			))
		})

		It("sets the recorder on the models of each provider", func() {
			total := NewMetricsRecorder()
			manager, err := NewManager(
				WithManagerLogger(discardLogger()),
				WithManagerCredentials(StaticCredentials("key")),
				WithManagerMetricsRecorder(total),
			)
			Expect(err).NotTo(HaveOccurred())

			for _, name := range []string{"anthropic/m", "openai/m", "openrouter/m", "google/m"} {
				model, err := manager.GetModel(context.Background(), name)
				Expect(err).NotTo(HaveOccurred())

				var recorder MetricsRecorder
				switch m := model.(type) {
				case *anthropicModel:
					recorder = m.recorder
				case *openaiModel:
					recorder = m.recorder
				case *openrouterModel:
					recorder = m.recorder
				case *googleModel:
					recorder = m.recorder
				}
				Expect(recorder).To(BeIdenticalTo(total), name)
			}
		})
	})
})
