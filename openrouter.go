package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/aholstenson/llms-go/jsonstream"
	"github.com/invopop/jsonschema"
	openrouter "github.com/revrost/go-openrouter"
)

// GenAISystemOpenRouter identifies OpenRouter as the GenAI system for metrics.
const GenAISystemOpenRouter GenAISystem = "openrouter"

// OpenRouterOption configures an OpenRouter model at construction time.
type OpenRouterOption func(*openrouterModel)

// WithOpenRouterProvider attaches an OpenRouter provider routing configuration
// (provider ordering, fallbacks, data collection, quantization filters, etc.)
// to every request issued by this model.
func WithOpenRouterProvider(provider *openrouter.ChatProvider) OpenRouterOption {
	return func(m *openrouterModel) {
		m.provider = provider
	}
}

// WithOpenRouterReferer sets the HTTP-Referer header forwarded to OpenRouter,
// used for app attribution on openrouter.ai/rankings.
func WithOpenRouterReferer(referer string) OpenRouterOption {
	return func(m *openrouterModel) {
		m.clientOpts = append(m.clientOpts, openrouter.WithHTTPReferer(referer))
	}
}

// WithOpenRouterXTitle sets the X-Title header forwarded to OpenRouter, used
// alongside the referer for app attribution.
func WithOpenRouterXTitle(title string) OpenRouterOption {
	return func(m *openrouterModel) {
		m.clientOpts = append(m.clientOpts, openrouter.WithXTitle(title))
	}
}

type openrouterModel struct {
	logger            *slog.Logger
	metrics           *Metrics
	client            *openrouter.Client
	clientOpts        []openrouter.Option
	statsModel        string
	model             string
	info              ModelInfo
	subParserRegistry map[string]SubParserConfig
	provider          *openrouter.ChatProvider
	headerCapture     *headerCapturingTransport
}

// lastCapturedHeaders returns the most recent response headers seen by the
// transport, or nil if no response has been observed yet.
func (m *openrouterModel) lastCapturedHeaders() http.Header {
	if m.headerCapture == nil {
		return nil
	}
	return m.headerCapture.LastHeaders()
}

// NewOpenRouterModel creates a new model that routes requests through
// OpenRouter using the github.com/revrost/go-openrouter SDK. info carries
// embedded model metadata used to gate request parameters; the zero value is
// treated permissively.
func NewOpenRouterModel(logger *slog.Logger, metrics *Metrics, apiKey string, model string, registry map[string]SubParserConfig, info ModelInfo, opts ...OpenRouterOption) Model {
	m := &openrouterModel{
		logger:            logger.With(slog.String("provider", "openrouter")),
		metrics:           metrics,
		statsModel:        "openrouter/" + model,
		model:             model,
		info:              info,
		subParserRegistry: registry,
	}
	for _, opt := range opts {
		opt(m)
	}
	transport := newHeaderCapturingTransport(http.DefaultTransport)
	m.headerCapture = transport
	httpClient := &http.Client{Transport: transport}
	clientOpts := append([]openrouter.Option{withOpenRouterHTTPClient(httpClient)}, m.clientOpts...)
	m.client = openrouter.NewClient(apiKey, clientOpts...)
	return m
}

// withOpenRouterHTTPClient installs a custom http.Client on the OpenRouter
// ClientConfig so we can capture Retry-After headers on errors.
func withOpenRouterHTTPClient(c *http.Client) openrouter.Option {
	return func(cfg *openrouter.ClientConfig) {
		cfg.HTTPClient = c
	}
}

func (m *openrouterModel) GenerateContent(ctx context.Context, options ...GenerateOption) (Result, error) {
	s, err := m.newSession(options...)
	if err != nil {
		return nil, err
	}
	return runSession(ctx, s)
}

func (m *openrouterModel) newSession(options ...GenerateOption) (*Session, error) {
	opts := resolveGenerateContentOptions(m.subParserRegistry, options...)

	if len(opts.Tools) > 0 && !m.info.allowsToolCall() {
		return nil, fmt.Errorf("model %s does not support tool calling", m.statsModel)
	}
	if messagesContainImages(opts.Messages) && !m.info.allowsModality("image") {
		return nil, fmt.Errorf("model %s does not support image input", m.statsModel)
	}
	if clamped, didClamp := m.info.clampMaxTokens(opts.MaxTokens); didClamp {
		m.logger.Warn("Clamping max tokens to model output limit",
			slog.Int("requested", opts.MaxTokens), slog.Int("limit", clamped))
		opts.MaxTokens = clamped
	}

	messages, err := m.convertMessages(opts.SystemPrompt, opts.Messages)
	if err != nil {
		return nil, err
	}

	tools, toolMap, err := m.convertTools(opts.Tools)
	if err != nil {
		return nil, err
	}

	params := openrouter.ChatCompletionRequest{
		Model:    m.model,
		Messages: messages,
		Provider: m.provider,
	}

	if opts.Temperature != 0 && m.info.allowsTemperature() {
		params.Temperature = float32(opts.Temperature)
	}

	if v := m.info.resolveMaxTokens(opts.MaxTokens, 0); v > 0 {
		params.MaxTokens = v
	}

	if opts.MaxThinkingTokens > 0 && m.info.allowsReasoning() {
		budget := opts.MaxThinkingTokens
		params.Reasoning = &openrouter.ChatCompletionReasoning{
			MaxTokens: &budget,
		}
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	if opts.ResponseSchema != nil {
		schemaBytes, err := json.Marshal(opts.ResponseSchema.Schema.(*jsonschema.Schema))
		if err != nil {
			return nil, fmt.Errorf("failed to marshal response schema: %w", err)
		}
		params.ResponseFormat = &openrouter.ChatCompletionResponseFormat{
			Type: openrouter.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openrouter.ChatCompletionResponseFormatJSONSchema{
				Name:   opts.ResponseSchema.Name,
				Schema: rawJSONSchema(schemaBytes),
				Strict: true,
			},
		}
	}

	var jsParser *jsonstream.Parser
	if opts.StructuredStreamingFunc != nil && opts.StructuredStreamingSchema != nil {
		jsParser = jsonstream.New(opts.StructuredStreamingSchema)
	}

	turn := &openrouterTurn{
		m:        m,
		opts:     opts,
		params:   params,
		jsParser: jsParser,
	}

	return newSession(turn, newTracker(opts), toolMap, opts, m.logger), nil
}

// rawJSONSchema is a json.Marshaler wrapper around an already-encoded JSON
// schema body. OpenRouter's ResponseFormatJSONSchema.Schema is typed as
// json.Marshaler, so passing a map (as the OpenAI SDK accepts) does not
// compile here.
type rawJSONSchema []byte

func (r rawJSONSchema) MarshalJSON() ([]byte, error) {
	return []byte(r), nil
}

func (m *openrouterModel) convertMessages(systemPrompt string, messages []*Message) ([]openrouter.ChatCompletionMessage, error) {
	result := make([]openrouter.ChatCompletionMessage, 0, len(messages)+1)

	if systemPrompt != "" {
		result = append(result, openrouter.SystemMessage(systemPrompt))
	}

	for _, msg := range messages {
		switch msg.Role {
		case RoleUser:
			if len(msg.Parts) == 0 {
				return nil, errors.New("user message has no parts")
			}

			// Tool-result messages map to OpenRouter's separate tool role,
			// one message per result.
			if _, isToolResult := msg.Parts[0].(*ToolResultPart); isToolResult {
				for _, part := range msg.Parts {
					trp, ok := part.(*ToolResultPart)
					if !ok {
						return nil, fmt.Errorf("cannot mix tool-result and %T parts for OpenRouter", part)
					}
					text := trp.Text
					if trp.Error != "" {
						text = trp.Error
					}
					result = append(result, openrouter.ToolMessage(trp.ID, text))
				}
				continue
			}

			// Simple text-only message.
			if len(msg.Parts) == 1 {
				if textPart, ok := msg.Parts[0].(*TextPart); ok {
					result = append(result, openrouter.UserMessage(textPart.Text))
					continue
				}
			}

			parts := make([]openrouter.ChatMessagePart, 0, len(msg.Parts))
			for _, part := range msg.Parts {
				switch p := part.(type) {
				case *TextPart:
					parts = append(parts, openrouter.ChatMessagePart{
						Type: openrouter.ChatMessagePartTypeText,
						Text: p.Text,
					})
				case *ImagePart:
					parts = append(parts, openrouter.ChatMessagePart{
						Type: openrouter.ChatMessagePartTypeImageURL,
						ImageURL: &openrouter.ChatMessageImageURL{
							URL: p.URL,
						},
					})
				case *BinaryPart:
					if strings.HasPrefix(p.MediaType, "image/") {
						dataURL := fmt.Sprintf("data:%s;base64,%s", p.MediaType, base64.StdEncoding.EncodeToString(p.Data))
						parts = append(parts, openrouter.ChatMessagePart{
							Type: openrouter.ChatMessagePartTypeImageURL,
							ImageURL: &openrouter.ChatMessageImageURL{
								URL: dataURL,
							},
						})
					} else {
						return nil, fmt.Errorf("unsupported binary media type for OpenRouter: %s", p.MediaType)
					}
				default:
					return nil, fmt.Errorf("unsupported part type for OpenRouter: %T", part)
				}
			}
			result = append(result, openrouter.ChatCompletionMessage{
				Role:    openrouter.ChatMessageRoleUser,
				Content: openrouter.Content{Multi: parts},
			})

		case RoleAssistant:
			var content string
			var toolCalls []openrouter.ToolCall
			for _, part := range msg.Parts {
				switch p := part.(type) {
				case *ThinkingPart:
					// OpenRouter currently has no replay format for plain
					// thinking text; the model-side reasoning_details are
					// re-attached via the native stash in Observe.
					continue
				case *TextPart:
					content += p.Text
				case *ToolCallPart:
					toolCalls = append(toolCalls, openrouter.ToolCall{
						ID:   p.ID,
						Type: openrouter.ToolTypeFunction,
						Function: openrouter.FunctionCall{
							Name:      p.Name,
							Arguments: p.Arguments,
						},
					})
				default:
					return nil, fmt.Errorf("unsupported assistant part type for OpenRouter: %T", part)
				}
			}
			assistantMsg := openrouter.ChatCompletionMessage{
				Role: openrouter.ChatMessageRoleAssistant,
			}
			if content != "" {
				assistantMsg.Content = openrouter.Content{Text: content}
			}
			if len(toolCalls) > 0 {
				assistantMsg.ToolCalls = toolCalls
			}
			result = append(result, assistantMsg)

		default:
			return nil, fmt.Errorf("unsupported message role for OpenRouter: %s", msg.Role)
		}
	}

	return result, nil
}

func (m *openrouterModel) convertTools(tools []ToolDef) ([]openrouter.Tool, map[string]ToolDef, error) {
	if len(tools) == 0 {
		return nil, make(map[string]ToolDef), nil
	}

	result := make([]openrouter.Tool, 0, len(tools))
	toolMap := make(map[string]ToolDef, len(tools))

	for _, tool := range tools {
		if _, exists := toolMap[tool.Name()]; exists {
			continue
		}

		schema := jsonSchemaReflector.Reflect(tool.Schema())
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal tool schema: %w", err)
		}

		result = append(result, openrouter.Tool{
			Type: openrouter.ToolTypeFunction,
			Function: &openrouter.FunctionDefinition{
				Name:        tool.Name(),
				Description: tool.Description(),
				Parameters:  json.RawMessage(schemaBytes),
			},
		})
		toolMap[tool.Name()] = tool
	}

	return result, toolMap, nil
}

// openrouterTurn is the OpenRouter-specific Turn. It owns the native
// []ChatCompletionMessage history so the agentic loop never round-trips
// through neutral types.
type openrouterTurn struct {
	m    *openrouterModel
	opts *generateContentOptions

	params openrouter.ChatCompletionRequest

	jsParser                 *jsonstream.Parser
	structuredContentBuilder strings.Builder

	pending []*Message

	started   bool
	callCount int
	finalText string

	// assistantMsg is the most recent assistant message stashed by Next for
	// Observe to append natively, preserving tool_calls and reasoning_details.
	assistantMsg openrouter.ChatCompletionMessage
}

func (t *openrouterTurn) Inject(msgs ...*Message) {
	t.pending = append(t.pending, msgs...)
}

func (t *openrouterTurn) FinalText() string {
	return t.finalText
}

func (t *openrouterTurn) Observe(ctx context.Context, _ TurnOutput, outcomes []ToolOutcome) error {
	t.params.Messages = append(t.params.Messages, t.assistantMsg)
	return t.ObserveToolResults(ctx, nil, outcomes)
}

func (t *openrouterTurn) ObserveToolResults(_ context.Context, _ []ToolCall, outcomes []ToolOutcome) error {
	for _, o := range outcomes {
		result := o.Text
		if o.Error != "" {
			result = o.Error
		}
		t.params.Messages = append(t.params.Messages, openrouter.ToolMessage(o.ID, result))
	}
	return nil
}

func (t *openrouterTurn) Next(ctx context.Context) (TurnOutput, error) {
	m := t.m

	if !t.started {
		m.metrics.RecordGenerateRequest(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model))
		t.started = true
	}

	if len(t.pending) > 0 {
		conv, err := m.convertMessages("", t.pending)
		if err != nil {
			return TurnOutput{}, err
		}
		t.params.Messages = append(t.params.Messages, conv...)
		t.pending = nil
	}

	t.callCount++
	start := time.Now()

	collector := NewCollector()
	collector.Counter("calls").Add(1)
	if t.callCount == 1 {
		collector.Counter("requests").Add(1)
	}

	if t.opts.StreamingFunc != nil || t.opts.StructuredStreamingFunc != nil {
		return t.nextStreaming(ctx, start, collector)
	}
	return t.nextNonStreaming(ctx, start, collector)
}

func (t *openrouterTurn) nextNonStreaming(ctx context.Context, start time.Time, collector Collector) (TurnOutput, error) {
	m := t.m

	// Make sure streaming-only fields stay off for the non-streaming path.
	t.params.Stream = false
	t.params.StreamOptions = nil

	classify := func(err error) (bool, int, time.Duration, bool) {
		if err == nil {
			return false, 0, 0, false
		}
		status, ok := openrouter.HTTPStatusCode(err)
		if !ok || !isUnavailableStatusCode(status) {
			return false, status, 0, false
		}
		ra, hasRA := extractRetryAfter("openrouter", err, m.lastCapturedHeaders())
		return true, status, ra, hasRA
	}

	response, err := retryLoop(ctx, t.opts, classify, func(ctx context.Context) (openrouter.ChatCompletionResponse, error) {
		return m.client.CreateChatCompletion(ctx, t.params)
	})
	if err != nil {
		return t.reportError(ctx, err, start, collector, false, false)
	}

	if len(response.Choices) == 0 {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeEmptyResponse)
		return TurnOutput{}, errors.New("no completion choices returned from OpenRouter")
	}

	choice := response.Choices[0]
	message := choice.Message

	// Stash the full assistant message so Observe can replay tool_calls and
	// reasoning_details on the next turn.
	t.assistantMsg = message

	textContent := message.Content.Text

	// Emit reasoning to the streaming/thinking sink even in the non-streaming
	// path, so callers see thinking either way.
	if t.opts.StreamingFunc != nil {
		if message.Reasoning != nil && *message.Reasoning != "" {
			if err := t.opts.StreamingFunc(ctx, StreamingEventThinking{Text: *message.Reasoning}); err != nil {
				return TurnOutput{}, err
			}
		} else if message.ReasoningContent != nil && *message.ReasoningContent != "" {
			if err := t.opts.StreamingFunc(ctx, StreamingEventThinking{Text: *message.ReasoningContent}); err != nil {
				return TurnOutput{}, err
			}
		}
	}

	toolCalls := make([]ToolCall, 0, len(message.ToolCalls))
	for _, tc := range message.ToolCalls {
		toolCalls = append(toolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	thinking := extractOpenRouterThinking(&message)

	var usage TurnUsage
	if response.Usage != nil {
		usage = TurnUsage{
			InputTokens:      int64(response.Usage.PromptTokens),
			OutputTokens:     int64(response.Usage.CompletionTokens),
			CachedReadTokens: int64(response.Usage.PromptTokenDetails.CachedTokens),
		}
		collector.Counter("input_tokens").Add(response.Usage.PromptTokens)
		collector.Counter("output_tokens").Add(response.Usage.CompletionTokens)
		collector.Counter("cached_read_tokens").Add(response.Usage.PromptTokenDetails.CachedTokens)
		m.metrics.RecordCall(ctx,
			GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model),
			usage.InputTokens, usage.OutputTokens, usage.CachedReadTokens, 0,
		)
	}

	content := textContent
	if t.jsParser != nil {
		content = t.structuredContentBuilder.String()
	}
	t.finalText = content

	if metrics := GetMetrics(ctx); metrics != nil {
		metrics.RecordSuccess(m.statsModel, collector)
	}
	m.metrics.RecordCallDuration(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

	return TurnOutput{
		Text:       textContent,
		Thinking:   thinking,
		ToolCalls:  toolCalls,
		StopReason: openrouterStopReason(choice.FinishReason, len(toolCalls) > 0),
		Usage:      usage,
	}, nil
}

func (t *openrouterTurn) nextStreaming(ctx context.Context, start time.Time, collector Collector) (TurnOutput, error) {
	m := t.m

	params := t.params
	params.Stream = true
	params.StreamOptions = &openrouter.StreamOptions{IncludeUsage: true}

	stream, err := m.client.CreateChatCompletionStream(ctx, params)
	if err != nil {
		return t.reportError(ctx, err, start, collector, true, false)
	}
	defer stream.Close()

	hasRecordedFirstToken := false
	streamHasErrored := false
	streamingEmitted := false

	var textContent strings.Builder

	type accumulatedToolCall struct {
		ID        string
		Name      string
		Arguments strings.Builder
	}
	accToolCalls := make(map[int]*accumulatedToolCall)
	var toolCallOrder []int

	var reasoningText strings.Builder
	var reasoningDetails []openrouter.ChatCompletionReasoningDetails
	var finishReason openrouter.FinishReason
	var usage *openrouter.Usage

	for {
		chunk, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			err = recvErr
			break
		}

		if chunk.Usage != nil {
			usage = chunk.Usage
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		delta := choice.Delta

		if choice.FinishReason != "" {
			finishReason = choice.FinishReason
		}

		if delta.Content != "" && !streamHasErrored {
			if !hasRecordedFirstToken {
				m.metrics.RecordTimeToFirstToken(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model), time.Since(start))
				hasRecordedFirstToken = true
			}

			textContent.WriteString(delta.Content)

			if t.jsParser != nil && t.opts.StructuredStreamingFunc != nil {
				t.structuredContentBuilder.WriteString(delta.Content)
				events, perr := t.jsParser.Feed(delta.Content)
				if perr != nil {
					m.logger.Error("Error parsing structured JSON stream", slog.Any("error", perr))
				} else {
					for _, event := range events {
						if perr := t.opts.StructuredStreamingFunc(ctx, event); perr != nil {
							m.logger.Error("Error handling structured streaming event", slog.Any("error", perr))
							streamHasErrored = true
							break
						}
					}
				}
			}

			if t.opts.StreamingFunc != nil {
				streamingEmitted = true
				if perr := t.opts.StreamingFunc(ctx, StreamingEventTextChunk{Text: delta.Content}); perr != nil {
					m.logger.Error("Error handling OpenRouter response", slog.Any("error", perr))
					streamHasErrored = true
				}
			}
		}

		if delta.Reasoning != nil && *delta.Reasoning != "" {
			reasoningText.WriteString(*delta.Reasoning)
			if t.opts.StreamingFunc != nil && !streamHasErrored {
				streamingEmitted = true
				if perr := t.opts.StreamingFunc(ctx, StreamingEventThinking{Text: *delta.Reasoning}); perr != nil {
					m.logger.Error("Error handling OpenRouter reasoning", slog.Any("error", perr))
					streamHasErrored = true
				}
			}
		} else if delta.ReasoningContent != "" {
			reasoningText.WriteString(delta.ReasoningContent)
			if t.opts.StreamingFunc != nil && !streamHasErrored {
				streamingEmitted = true
				if perr := t.opts.StreamingFunc(ctx, StreamingEventThinking{Text: delta.ReasoningContent}); perr != nil {
					m.logger.Error("Error handling OpenRouter reasoning", slog.Any("error", perr))
					streamHasErrored = true
				}
			}
		}

		// reasoning_details arrive only on whole blocks (not deltas) — accumulate
		// the latest copy so we can preserve them on the next turn.
		if len(delta.ReasoningDetails) > 0 {
			reasoningDetails = append(reasoningDetails, delta.ReasoningDetails...)
		}

		for _, tc := range delta.ToolCalls {
			idx := 0
			if tc.Index != nil {
				idx = *tc.Index
			}
			acc, ok := accToolCalls[idx]
			if !ok {
				acc = &accumulatedToolCall{}
				accToolCalls[idx] = acc
				toolCallOrder = append(toolCallOrder, idx)
			}
			if tc.ID != "" {
				acc.ID = tc.ID
			}
			if tc.Function.Name != "" {
				acc.Name = tc.Function.Name
			}
			acc.Arguments.WriteString(tc.Function.Arguments)
		}
	}

	if t.jsParser != nil && t.opts.StructuredStreamingFunc != nil {
		events, ferr := t.jsParser.Flush()
		if ferr != nil {
			m.logger.Error("Error flushing structured JSON stream", slog.Any("error", ferr))
		} else {
			for _, event := range events {
				if ferr := t.opts.StructuredStreamingFunc(ctx, event); ferr != nil {
					m.logger.Error("Error handling structured streaming event", slog.Any("error", ferr))
					break
				}
			}
		}
	}

	if streamHasErrored {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeStreamProcessing)
		return TurnOutput{}, errors.New("stream handling failed")
	}
	if err != nil {
		return t.reportError(ctx, err, start, collector, true, streamingEmitted)
	}

	// Rebuild assistant message + neutral tool calls in deterministic order.
	finalText := textContent.String()
	toolCalls := make([]ToolCall, 0, len(toolCallOrder))
	nativeToolCalls := make([]openrouter.ToolCall, 0, len(toolCallOrder))
	for _, idx := range toolCallOrder {
		acc := accToolCalls[idx]
		args := acc.Arguments.String()
		toolCalls = append(toolCalls, ToolCall{
			ID:        acc.ID,
			Name:      acc.Name,
			Arguments: args,
		})
		nativeToolCalls = append(nativeToolCalls, openrouter.ToolCall{
			ID:   acc.ID,
			Type: openrouter.ToolTypeFunction,
			Function: openrouter.FunctionCall{
				Name:      acc.Name,
				Arguments: args,
			},
		})
	}

	assistant := openrouter.ChatCompletionMessage{
		Role: openrouter.ChatMessageRoleAssistant,
	}
	if finalText != "" {
		assistant.Content = openrouter.Content{Text: finalText}
	}
	if len(nativeToolCalls) > 0 {
		assistant.ToolCalls = nativeToolCalls
	}
	if reasoningText.Len() > 0 {
		r := reasoningText.String()
		assistant.Reasoning = &r
	}
	if len(reasoningDetails) > 0 {
		assistant.ReasoningDetails = reasoningDetails
	}
	t.assistantMsg = assistant

	var thinking []ThinkingBlock
	if reasoningText.Len() > 0 {
		thinking = append(thinking, ThinkingBlock{Text: reasoningText.String()})
	}

	var u TurnUsage
	if usage != nil {
		u = TurnUsage{
			InputTokens:      int64(usage.PromptTokens),
			OutputTokens:     int64(usage.CompletionTokens),
			CachedReadTokens: int64(usage.PromptTokenDetails.CachedTokens),
		}
		collector.Counter("input_tokens").Add(usage.PromptTokens)
		collector.Counter("output_tokens").Add(usage.CompletionTokens)
		collector.Counter("cached_read_tokens").Add(usage.PromptTokenDetails.CachedTokens)
		m.metrics.RecordCall(ctx,
			GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model),
			u.InputTokens, u.OutputTokens, u.CachedReadTokens, 0,
		)
	}

	content := finalText
	if t.jsParser != nil {
		content = t.structuredContentBuilder.String()
	}
	t.finalText = content

	if metrics := GetMetrics(ctx); metrics != nil {
		metrics.RecordSuccess(m.statsModel, collector)
	}
	m.metrics.RecordCallDuration(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

	return TurnOutput{
		Text:       finalText,
		Thinking:   thinking,
		ToolCalls:  toolCalls,
		StopReason: openrouterStopReason(finishReason, len(toolCalls) > 0),
		Usage:      u,
	}, nil
}

func (t *openrouterTurn) reportError(ctx context.Context, err error, start time.Time, collector Collector, streaming bool, partialEmitted bool) (TurnOutput, error) {
	m := t.m
	if metrics := GetMetrics(ctx); metrics != nil {
		metrics.RecordFailure(m.statsModel, collector)
	}

	// retryLoop already wrapped non-streaming retryable errors in
	// UnavailableError; just stamp PartialOutput and re-attach the streaming
	// sentinel where applicable.
	var ue *UnavailableError
	if errors.As(err, &ue) {
		ue.PartialOutput = partialEmitted
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
		if partialEmitted {
			return TurnOutput{}, errors.Join(ue, ErrStreamingPartialOutput)
		}
		return TurnOutput{}, ue
	}

	if status, ok := openrouter.HTTPStatusCode(err); ok && isUnavailableStatusCode(status) {
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
		ra, hasRA := extractRetryAfter("openrouter", err, m.lastCapturedHeaders())
		if hasRA && t.opts.RetryAfterCap > 0 && ra > t.opts.RetryAfterCap {
			ra = t.opts.RetryAfterCap
		}
		newUE := &UnavailableError{
			StatusCode:    status,
			RetryAfter:    ra,
			HasRetryAfter: hasRA,
			Attempts:      1,
			PartialOutput: partialEmitted,
			Cause:         err,
		}
		if partialEmitted {
			return TurnOutput{}, errors.Join(newUE, ErrStreamingPartialOutput)
		}
		return TurnOutput{}, newUE
	}
	m.metrics.RecordCallDuration(ctx, GenAISystemOpenRouter, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
	return TurnOutput{}, fmt.Errorf("error from OpenRouter: %w", err)
}

func extractOpenRouterThinking(msg *openrouter.ChatCompletionMessage) []ThinkingBlock {
	if msg == nil {
		return nil
	}
	var out []ThinkingBlock
	if msg.Reasoning != nil && *msg.Reasoning != "" {
		out = append(out, ThinkingBlock{Text: *msg.Reasoning})
	} else if msg.ReasoningContent != nil && *msg.ReasoningContent != "" {
		out = append(out, ThinkingBlock{Text: *msg.ReasoningContent})
	}
	return out
}

func openrouterStopReason(reason openrouter.FinishReason, hasToolCalls bool) StopReason {
	if hasToolCalls {
		return StopReasonToolUse
	}
	switch reason {
	case openrouter.FinishReasonToolCalls, openrouter.FinishReasonFunctionCall:
		return StopReasonToolUse
	case openrouter.FinishReasonLength:
		return StopReasonMaxTokens
	case openrouter.FinishReasonContentFilter:
		return StopReasonRefusal
	default:
		return StopReasonEndTurn
	}
}
