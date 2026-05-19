package llms

import (
	"context"
	"time"

	"github.com/cockroachdb/errors"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
)

// GenAISystem represents a Generative AI system provider.
type GenAISystem string

// Common GenAI system identifiers following OpenTelemetry semantic conventions.
const (
	GenAISystemOpenAI    GenAISystem = "openai"
	GenAISystemAnthropic GenAISystem = "anthropic"
	GenAISystemGoogle    GenAISystem = "google"
)

// GenAIOperation represents the type of operation being performed.
type GenAIOperation string

// Common GenAI operations following OpenTelemetry semantic conventions.
const (
	GenAIOperationChat            GenAIOperation = "chat"
	GenAIOperationGenerateContent GenAIOperation = "generate_content"
)

// GenAIModel represents a specific model identifier.
type GenAIModel string

// GenAIErrorType represents the type of error that occurred.
type GenAIErrorType string

const (
	GenAIErrorTypeNoError          GenAIErrorType = ""
	GenAIErrorTypeTimeout          GenAIErrorType = "timeout"
	GenAIErrorTypeInternal         GenAIErrorType = "internal"
	GenAIErrorTypeStreamProcessing GenAIErrorType = "stream_processing"
	GenAIErrorTypeToolCall         GenAIErrorType = "tool_call"
	GenAIErrorTypeEmptyResponse    GenAIErrorType = "empty_response"
	GenAIErrorTypeUnavailable      GenAIErrorType = "unavailable"
)

// Metrics provides standardized GenAI metrics for the LLM implementations.
//
// Conventions and attribute names taken from the the specification at:
// https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
type Metrics struct {
	generateContent  metric.Int64Counter
	requests         metric.Int64Counter
	requestsDuration metric.Float64Histogram
	timeToFirstToken metric.Float64Histogram
	tokens           metric.Int64Counter
}

// NewMetrics creates a new Metrics instance with all the standardized GenAI metrics.
func NewMetrics(meter metric.Meter) (*Metrics, error) {
	generateRequests, err := meter.Int64Counter("gen_ai.generate_content",
		metric.WithDescription("Number of requests to generate content with LLMs"),
		metric.WithUnit("{requests}"),
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create requests counter")
	}

	requests, err := meter.Int64Counter("gen_ai.requests",
		metric.WithDescription("Total number of requests made to GenAI system APIs"),
		metric.WithUnit("{requests}"),
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create requests total counter")
	}

	requestsDuration, err := meter.Float64Histogram("gen_ai.requests.duration",
		metric.WithDescription("Duration of requests to GenAI systems"),
		metric.WithUnit("s"),
		metric.WithExplicitBucketBoundaries(0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create requests duration histogram")
	}

	timeToFirstToken, err := meter.Float64Histogram("gen_ai.time_to_first_token",
		metric.WithDescription("Time to generate first token for successful responses"),
		metric.WithUnit("s"),
		metric.WithExplicitBucketBoundaries(0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create time to first token histogram")
	}

	tokens, err := meter.Int64Counter("gen_ai.tokens",
		metric.WithDescription("Number of tokens used by GenAI systems"),
		metric.WithUnit("{tokens}"),
	)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create prompt tokens counter")
	}

	return &Metrics{
		generateContent:  generateRequests,
		requests:         requests,
		requestsDuration: requestsDuration,
		timeToFirstToken: timeToFirstToken,
		tokens:           tokens,
	}, nil
}

// RecordGenerateRequest records a new request to generate content with LLMs.
func (m *Metrics) RecordGenerateRequest(ctx context.Context, system GenAISystem, operation GenAIOperation, model GenAIModel) {
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.system", string(system)),
		attribute.String("gen_ai.operation.name", string(operation)),
		attribute.String("gen_ai.request.model", string(model)),
	}

	m.generateContent.Add(ctx, 1, metric.WithAttributes(attrs...))
}

// RecordCall records token usage for a single API call to a GenAI system.
func (m *Metrics) RecordCall(
	ctx context.Context,
	system GenAISystem,
	operation GenAIOperation,
	model GenAIModel,
	inputTokens int64,
	outputTokens int64,
	cachedReadTokens int64,
	cachedWriteTokens int64,
) {
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.system", string(system)),
		attribute.String("gen_ai.operation.name", string(operation)),
		attribute.String("gen_ai.request.model", string(model)),
	}

	m.requests.Add(ctx, 1, metric.WithAttributes(attrs...))

	if inputTokens > 0 {
		m.tokens.Add(ctx, inputTokens, metric.WithAttributes(attrs...), metric.WithAttributes(attribute.String("gen_ai.token.type", "input")))
	}

	if outputTokens > 0 {
		m.tokens.Add(ctx, outputTokens, metric.WithAttributes(attrs...), metric.WithAttributes(attribute.String("gen_ai.token.type", "output")))
	}

	if cachedReadTokens > 0 {
		m.tokens.Add(ctx, cachedReadTokens, metric.WithAttributes(attrs...), metric.WithAttributes(attribute.String("gen_ai.token.type", "cached_input")))
	}

	if cachedWriteTokens > 0 {
		m.tokens.Add(ctx, cachedWriteTokens, metric.WithAttributes(attrs...), metric.WithAttributes(attribute.String("gen_ai.token.type", "cached_output")))
	}
}

// RecordCallDuration records the duration of a request to GenAI systems.
func (m *Metrics) RecordCallDuration(ctx context.Context, system GenAISystem, operation GenAIOperation, model GenAIModel, duration time.Duration, errorType GenAIErrorType) {
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.system", string(system)),
		attribute.String("gen_ai.operation.name", string(operation)),
		attribute.String("gen_ai.request.model", string(model)),
	}

	if errorType != GenAIErrorTypeNoError {
		attrs = append(attrs, attribute.String("error.type", string(errorType)))
	}

	m.requestsDuration.Record(ctx, duration.Seconds(), metric.WithAttributes(attrs...))
}

// RecordTimeToFirstToken records the time to generate the first token for a GenAI response.
func (m *Metrics) RecordTimeToFirstToken(ctx context.Context, system GenAISystem, operation GenAIOperation, model GenAIModel, duration time.Duration) {
	attrs := []attribute.KeyValue{
		attribute.String("gen_ai.system", string(system)),
		attribute.String("gen_ai.operation.name", string(operation)),
		attribute.String("gen_ai.request.model", string(model)),
	}

	m.timeToFirstToken.Record(ctx, duration.Seconds(), metric.WithAttributes(attrs...))
}

// NewNoopMetrics creates a Metrics instance with no-op implementations.
func NewNoopMetrics() *Metrics {
	return &Metrics{
		generateContent:  noop.Int64Counter{},
		requests:         noop.Int64Counter{},
		requestsDuration: noop.Float64Histogram{},
		timeToFirstToken: noop.Float64Histogram{},
		tokens:           noop.Int64Counter{},
	}
}
