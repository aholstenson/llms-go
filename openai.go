package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"strings"
	"time"

	"github.com/aholstenson/llms-go/jsonstream"
	"github.com/cockroachdb/errors"
	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

type openaiModel struct {
	logger            *slog.Logger
	metrics           *Metrics
	client            openai.Client
	statsModel        string
	model             string
	info              ModelInfo
	subParserRegistry map[string]SubParserConfig
}

// NewOpenAIModel creates a new OpenAI model using the official OpenAI Go SDK.
// info carries embedded model metadata used to gate request parameters; the
// zero value is treated permissively.
func NewOpenAIModel(logger *slog.Logger, metrics *Metrics, apiKey string, model string, registry map[string]SubParserConfig, info ModelInfo) Model {
	client := openai.NewClient(option.WithAPIKey(apiKey))
	return &openaiModel{
		logger:            logger.With(slog.String("provider", "openai")),
		metrics:           metrics,
		client:            client,
		statsModel:        "openai/" + model,
		model:             model,
		info:              info,
		subParserRegistry: registry,
	}
}

func (m *openaiModel) GenerateContent(ctx context.Context, options ...GenerateOption) (Result, error) {
	s, err := m.newSession(options...)
	if err != nil {
		return nil, err
	}
	return runSession(ctx, s)
}

func (m *openaiModel) newSession(options ...GenerateOption) (*Session, error) {
	opts := resolveGenerateContentOptions(m.subParserRegistry, options...)

	// Gate request parameters against the model's known capabilities.
	if len(opts.Tools) > 0 && !m.info.allowsToolCall() {
		return nil, errors.Newf("model %s does not support tool calling", m.statsModel)
	}
	if messagesContainImages(opts.Messages) && !m.info.allowsModality("image") {
		return nil, errors.Newf("model %s does not support image input", m.statsModel)
	}
	if clamped, didClamp := m.info.clampMaxTokens(opts.MaxTokens); didClamp {
		m.logger.Warn("Clamping max tokens to model output limit",
			slog.Int("requested", opts.MaxTokens), slog.Int("limit", clamped))
		opts.MaxTokens = clamped
	}

	// Convert messages to OpenAI format
	messages, err := m.convertMessages(opts.SystemPrompt, opts.Messages)
	if err != nil {
		return nil, err
	}

	// Convert tools to OpenAI format if any
	tools, toolMap, err := m.convertTools(opts.Tools)
	if err != nil {
		return nil, err
	}

	// Build request parameters
	params := openai.ChatCompletionNewParams{
		Model:    m.model,
		Messages: messages,
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
	}

	if opts.Temperature != 0 && m.info.allowsTemperature() {
		params.Temperature = openai.Float(opts.Temperature)
	}

	if opts.MaxTokens != 0 {
		// Reasoning models (gpt-5.x) reject max_tokens and require
		// max_completion_tokens instead.
		switch m.model {
		case "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1":
			params.MaxCompletionTokens = openai.Int(int64(opts.MaxTokens))
		default:
			params.MaxTokens = openai.Int(int64(opts.MaxTokens))
		}
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	// Configure structured output if ResponseSchema is set
	if opts.ResponseSchema != nil {
		schemaBytes, err := json.Marshal(opts.ResponseSchema.Schema.(*jsonschema.Schema))
		if err != nil {
			return nil, errors.Wrap(err, "failed to marshal response schema")
		}

		var rawJSON map[string]any
		if err := json.Unmarshal(schemaBytes, &rawJSON); err != nil {
			return nil, errors.Wrap(err, "failed to unmarshal response schema")
		}

		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
				JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   opts.ResponseSchema.Name,
					Schema: rawJSON,
					Strict: openai.Bool(true),
				},
			},
		}
	}

	var jsParser *jsonstream.Parser
	if opts.StructuredStreamingFunc != nil && opts.StructuredStreamingSchema != nil {
		jsParser = jsonstream.New(opts.StructuredStreamingSchema)
	}

	turn := &openaiTurn{
		m:        m,
		opts:     opts,
		params:   params,
		jsParser: jsParser,
	}

	return newSession(turn, newTracker(opts), toolMap, opts, m.logger), nil
}

func (m *openaiModel) convertMessages(systemPrompt string, messages []*Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages)+1)

	// Add system prompt if provided
	if systemPrompt != "" {
		result = append(result, openai.SystemMessage(systemPrompt))
	}

	// Convert user messages
	for _, msg := range messages {
		switch msg.Role {
		case RoleUser:
			// Handle user message
			if len(msg.Parts) == 0 {
				return nil, errors.New("user message has no parts")
			}

			// Tool-result messages map to OpenAI's separate tool role, one
			// message per result, rather than parts of a user message.
			if _, isToolResult := msg.Parts[0].(*ToolResultPart); isToolResult {
				for _, part := range msg.Parts {
					trp, ok := part.(*ToolResultPart)
					if !ok {
						return nil, errors.Newf("cannot mix tool-result and %T parts for OpenAI", part)
					}
					text := trp.Text
					if trp.Error != "" {
						text = trp.Error
					}
					result = append(result, openai.ChatCompletionMessageParamUnion{
						OfTool: &openai.ChatCompletionToolMessageParam{
							Content: openai.ChatCompletionToolMessageParamContentUnion{
								OfString: openai.String(text),
							},
							ToolCallID: trp.ID,
						},
					})
				}
				continue
			}

			// Check if we have a single text part or multiple parts
			if len(msg.Parts) == 1 {
				if textPart, ok := msg.Parts[0].(*TextPart); ok {
					// Simple text message
					result = append(result, openai.UserMessage(textPart.Text))
					continue
				}
			}

			// Handle multi-part message (text and images)
			var contentParts []openai.ChatCompletionContentPartUnionParam

			for _, part := range msg.Parts {
				switch content := part.(type) {
				case *TextPart:
					contentParts = append(contentParts, openai.TextContentPart(content.Text))
				case *ImagePart:
					contentParts = append(contentParts, openai.ImageContentPart(
						openai.ChatCompletionContentPartImageImageURLParam{
							URL: content.URL,
						},
					))
				case *BinaryPart:
					data := base64.StdEncoding.EncodeToString(content.Data)
					contentParts = append(contentParts, openai.FileContentPart(
						openai.ChatCompletionContentPartFileFileParam{
							FileData: openai.String(data),
						},
					))
				default:
					return nil, errors.Newf("unsupported part type for OpenAI: %T", part)
				}
			}

			result = append(result, openai.UserMessage(contentParts))

		case RoleAssistant:
			content := ""
			var toolCalls []openai.ChatCompletionMessageToolCall
			for _, part := range msg.Parts {
				switch p := part.(type) {
				case *ThinkingPart:
					// OpenAI has no reasoning replay; skip silently.
					continue
				case *TextPart:
					content += p.Text
				case *ToolCallPart:
					toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCall{
						ID: p.ID,
						Function: openai.ChatCompletionMessageToolCallFunction{
							Name:      p.Name,
							Arguments: p.Arguments,
						},
					})
				default:
					return nil, errors.Newf("unsupported assistant part type for OpenAI: %T", part)
				}
			}

			if len(toolCalls) > 0 {
				// Mirror anthropicTurn/openaiTurn.Observe: build the native
				// assistant message via ChatCompletionMessage.ToParam so the
				// tool_calls round-trip exactly as the SDK expects.
				result = append(result, openai.ChatCompletionMessage{
					Content:   content,
					ToolCalls: toolCalls,
				}.ToParam())
			} else {
				result = append(result, openai.AssistantMessage(content))
			}

		default:
			return nil, errors.Newf("unsupported message role for OpenAI: %s", msg.Role)
		}
	}

	return result, nil
}

func (m *openaiModel) convertTools(tools []ToolDef) ([]openai.ChatCompletionToolParam, map[string]ToolDef, error) {
	if len(tools) == 0 {
		return nil, make(map[string]ToolDef), nil
	}

	result := make([]openai.ChatCompletionToolParam, 0, len(tools))
	toolMap := make(map[string]ToolDef, len(tools))

	for _, tool := range tools {
		if _, exists := toolMap[tool.Name()]; exists {
			continue
		}

		// Create JSON schema for tool parameters
		schema := jsonSchemaReflector.Reflect(tool.Schema())
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, nil, errors.Newf("failed to marshal tool schema: %w", err)
		}

		var rawJSON map[string]any
		if err := json.Unmarshal(schemaBytes, &rawJSON); err != nil {
			return nil, nil, errors.Newf("failed to unmarshal tool schema: %w", err)
		}

		// Create OpenAI function definition
		functionDef := openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Name(),
				Description: openai.String(tool.Description()),
				Parameters:  rawJSON,
			},
		}

		result = append(result, functionDef)
		toolMap[tool.Name()] = tool
	}

	return result, toolMap, nil
}

// openaiTurn is the OpenAI-specific Turn: it owns the native
// ChatCompletionNewParams history so the agentic loop never round-trips
// through neutral types.
type openaiTurn struct {
	m    *openaiModel
	opts *generateContentOptions

	params openai.ChatCompletionNewParams

	jsParser                 *jsonstream.Parser
	structuredContentBuilder strings.Builder

	pending []*Message

	started   bool
	callCount int
	finalText string

	// assistantMsg is the most recent assistant message (with empty-ID tool
	// calls filtered out), stashed by Next for Observe to append natively.
	assistantMsg openai.ChatCompletionMessage
}

func (t *openaiTurn) Inject(msgs ...*Message) {
	t.pending = append(t.pending, msgs...)
}

func (t *openaiTurn) FinalText() string {
	return t.finalText
}

func (t *openaiTurn) Observe(ctx context.Context, _ TurnOutput, outcomes []ToolOutcome) error {
	t.params.Messages = append(t.params.Messages, t.assistantMsg.ToParam())
	return t.ObserveToolResults(ctx, nil, outcomes)
}

// ObserveToolResults appends only the tool-result messages, used when the
// assistant message is already in native history (reconstructed turn).
func (t *openaiTurn) ObserveToolResults(_ context.Context, _ []ToolCall, outcomes []ToolOutcome) error {
	for _, o := range outcomes {
		result := o.Text
		if o.Error != "" {
			result = o.Error
		}
		t.params.Messages = append(t.params.Messages, openai.ChatCompletionMessageParamUnion{
			OfTool: &openai.ChatCompletionToolMessageParam{
				Content: openai.ChatCompletionToolMessageParamContentUnion{
					OfString: openai.String(result),
				},
				ToolCallID: o.ID,
			},
		})
	}
	return nil
}

func (t *openaiTurn) Next(ctx context.Context) (TurnOutput, error) {
	m := t.m

	if !t.started {
		m.metrics.RecordGenerateRequest(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model))
		t.started = true
	}

	// Apply any caller-injected messages before this call.
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
	hasRecordedFirstToken := false

	collector := NewCollector()
	collector.Counter("calls").Add(1)
	if t.callCount == 1 {
		collector.Counter("requests").Add(1)
	}

	streamHasErrored := false

	stream := m.client.Chat.Completions.NewStreaming(ctx, t.params)
	acc := openai.ChatCompletionAccumulator{}

	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)

		if _, ok := acc.JustFinishedContent(); ok { //nolint:staticcheck
			// TODO: Handle content stream finished
		}

		if _, ok := acc.JustFinishedRefusal(); ok { //nolint:staticcheck
			// TODO: Decide on how to handle refusals
		}

		if _, ok := acc.JustFinishedToolCall(); ok { //nolint:staticcheck
			// Tool calls are collected from the accumulator after the stream
			// completes; the Session drives execution.
		}

		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" && !streamHasErrored {
			if !hasRecordedFirstToken {
				m.metrics.RecordTimeToFirstToken(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start))
				hasRecordedFirstToken = true
			}

			content := chunk.Choices[0].Delta.Content

			if t.jsParser != nil && t.opts.StructuredStreamingFunc != nil {
				t.structuredContentBuilder.WriteString(content)
				events, err := t.jsParser.Feed(content)
				if err != nil {
					m.logger.Error("Error parsing structured JSON stream", slog.Any("error", err))
				} else {
					for _, event := range events {
						if err := t.opts.StructuredStreamingFunc(ctx, event); err != nil {
							m.logger.Error("Error handling structured streaming event", slog.Any("error", err))
							streamHasErrored = true
							break
						}
					}
				}
			}

			if t.opts.StreamingFunc != nil {
				if err := t.opts.StreamingFunc(ctx, StreamingEventTextChunk{Text: content}); err != nil {
					m.logger.Error("Error handling OpenAI response", slog.Any("error", err))
					streamHasErrored = true
					if cerr := stream.Close(); cerr != nil {
						m.logger.Warn("Error closing OpenAI stream", slog.Any("error", cerr))
					}
					break
				}
			}
		}
	}

	if err := stream.Close(); err != nil {
		m.logger.Warn("Error closing OpenAI stream", slog.Any("error", err))
	}

	promptTokens := acc.Usage.PromptTokens
	cachedTokens := acc.Usage.PromptTokensDetails.CachedTokens
	collector.Counter("input_tokens").Add(int(promptTokens))
	collector.Counter("output_tokens").Add(int(acc.Usage.CompletionTokens))
	collector.Counter("cached_read_tokens").Add(int(cachedTokens))

	m.metrics.RecordCall(
		ctx,
		GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model),
		promptTokens,
		acc.Usage.CompletionTokens,
		cachedTokens,
		0,
	)

	if t.jsParser != nil && t.opts.StructuredStreamingFunc != nil {
		events, err := t.jsParser.Flush()
		if err != nil {
			m.logger.Error("Error flushing structured JSON stream", slog.Any("error", err))
		} else {
			for _, event := range events {
				if err := t.opts.StructuredStreamingFunc(ctx, event); err != nil {
					m.logger.Error("Error handling structured streaming event", slog.Any("error", err))
					break
				}
			}
		}
	}

	if streamHasErrored {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeStreamProcessing)
		return TurnOutput{}, errors.New("stream handling failed")
	} else if stream.Err() != nil {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}

		openaiError := &openai.Error{}
		if errors.As(stream.Err(), &openaiError) && isUnavailableStatusCode(openaiError.StatusCode) {
			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
			return TurnOutput{}, errors.Mark(errors.Wrap(stream.Err(), "OpenAI model unavailable"), ErrModelUnavailable)
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, errors.Wrap(stream.Err(), "got error from OpenAI while streaming")
	}

	if len(acc.Choices) == 0 {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeEmptyResponse)
		return TurnOutput{}, errors.New("no completion choices returned")
	}

	choice := acc.Choices[0]

	// Tool calls sometimes return empty IDs; filter those out or the loop
	// would wait forever for a call that never happened.
	actualToolCalls := make([]openai.ChatCompletionMessageToolCall, 0, len(choice.Message.ToolCalls))
	for _, tc := range choice.Message.ToolCalls {
		if tc.ID == "" {
			continue
		}
		actualToolCalls = append(actualToolCalls, tc)
	}
	choice.Message.ToolCalls = actualToolCalls
	t.assistantMsg = choice.Message

	toolCalls := make([]ToolCall, 0, len(actualToolCalls))
	for _, tc := range actualToolCalls {
		toolCalls = append(toolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	content := choice.Message.Content
	if t.jsParser != nil {
		content = t.structuredContentBuilder.String()
	}
	t.finalText = content

	// This per-call accounting succeeded; record duration + success here so
	// metrics stay provider-local. Token rollup to the execution tracker is
	// done by the Session from TurnOutput.Usage.
	if metrics := GetMetrics(ctx); metrics != nil {
		metrics.RecordSuccess(m.statsModel, collector)
	}
	m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

	return TurnOutput{
		Text:       choice.Message.Content,
		ToolCalls:  toolCalls,
		StopReason: openaiStopReason(choice.FinishReason, len(toolCalls) > 0),
		Usage: TurnUsage{
			InputTokens:      promptTokens,
			OutputTokens:     acc.Usage.CompletionTokens,
			CachedReadTokens: cachedTokens,
		},
	}, nil
}

func openaiStopReason(finishReason string, hasToolCalls bool) StopReason {
	if hasToolCalls {
		return StopReasonToolUse
	}
	switch finishReason {
	case "tool_calls":
		return StopReasonToolUse
	case "length":
		return StopReasonMaxTokens
	case "content_filter":
		return StopReasonRefusal
	default:
		return StopReasonEndTurn
	}
}
