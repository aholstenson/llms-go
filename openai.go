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

// NewOpenRouterModel creates a new model that passes requests through OpenRouter.
// info carries embedded model metadata used to gate request parameters; the
// zero value is treated permissively.
func NewOpenRouterModel(logger *slog.Logger, metrics *Metrics, apiKey string, model string, registry map[string]SubParserConfig, info ModelInfo) Model {
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://openrouter.ai/api/v1"),
	)
	return &openaiModel{
		logger:            logger.With(slog.String("provider", "openrouter")),
		metrics:           metrics,
		client:            client,
		statsModel:        "openrouter/" + model,
		model:             model,
		info:              info,
		subParserRegistry: registry,
	}
}

func (m *openaiModel) GenerateContent(ctx context.Context, options ...GenerateOption) (Result, error) {
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

	maxSteps := opts.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10
	}

	// Create and inject execution tracker
	tracker := NewExecutionTracker(maxSteps)
	ctx = WithExecutionContext(ctx, tracker)

	// Set up custom streaming handler
	var streamHandler StreamingFunc
	if opts.StreamingFunc != nil {
		streamHandler = opts.StreamingFunc
	} else {
		// No-op handler when streaming is not requested
		streamHandler = func(_ context.Context, _ StreamingEvent) error {
			return nil
		}
	}

	// Set up structured streaming parser if configured
	var jsParser *jsonstream.Parser
	var structuredContentBuilder strings.Builder
	if opts.StructuredStreamingFunc != nil && opts.StructuredStreamingSchema != nil {
		jsParser = jsonstream.New(opts.StructuredStreamingSchema)
	}

	// Record a request to the LLM as we begin to generate content
	m.metrics.RecordGenerateRequest(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model))

	streamHasErrored := false
	steps := 0
	for {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		steps++
		if steps >= maxSteps {
			return nil, errors.New("max steps reached")
		}

		// Update execution tracker
		tracker.IncrementStep()

		start := time.Now()
		hasRecordedFirstToken := false

		// Each iteration which calls the LLM starts with fresh stats
		collector := NewCollector()
		collector.Counter("calls").Add(1)
		if steps == 1 {
			// If this is the first step, record a request
			collector.Counter("requests").Add(1)
		}

		if opts.StreamingFunc != nil {
			if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageStart{}); emitErr != nil {
				return nil, emitErr
			}
		}

		// Use the accumulator to collect streaming responses
		stream := m.client.Chat.Completions.NewStreaming(ctx, params)
		acc := openai.ChatCompletionAccumulator{}

		toolCalls := 0
		toolResultChan := make(chan toolResult)
		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			// When this fires, the current chunk value will not contain content data
			if _, ok := acc.JustFinishedContent(); ok { //nolint:staticcheck
				// TODO: Handle content stream finished
			}

			if _, ok := acc.JustFinishedRefusal(); ok { //nolint:staticcheck
				// TODO: Decide on how to handle refusals
			}

			if tool, ok := acc.JustFinishedToolCall(); ok {
				matchedTool, ok := toolMap[tool.Name]
				if !ok {
					toolResultChan <- toolResult{
						id:           tool.ID,
						functionName: tool.Name,
						errorString:  "Requested tool not found",
					}
					continue
				}

				toolCalls++
				go func() {
					result, err := doToolCall(ctx, m.logger, opts.StreamingFunc, opts.ToolCallTimeout, tool.ID, matchedTool, tool.Arguments)
					if err != nil {
						toolResultChan <- toolResult{
							id:           tool.ID,
							functionName: tool.Name,
							errorString:  err.Error(),
						}
					} else {
						toolResultChan <- toolResult{
							id:           tool.ID,
							functionName: tool.Name,
							results:      result,
						}
					}
				}()
			}

			// It's best to use chunks after handling JustFinished events.
			// Here we print the delta of the content, if it exists.
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" && !streamHasErrored {
				if !hasRecordedFirstToken {
					m.metrics.RecordTimeToFirstToken(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start))
					hasRecordedFirstToken = true
				}

				content := chunk.Choices[0].Delta.Content

				// Handle structured streaming if configured
				if jsParser != nil && opts.StructuredStreamingFunc != nil {
					structuredContentBuilder.WriteString(content)
					events, err := jsParser.Feed(content)
					if err != nil {
						m.logger.Error("Error parsing structured JSON stream", slog.Any("error", err))
					} else {
						for _, event := range events {
							if err := opts.StructuredStreamingFunc(ctx, event); err != nil {
								m.logger.Error("Error handling structured streaming event", slog.Any("error", err))
								streamHasErrored = true
								break
							}
						}
					}
				}

				err := streamHandler(ctx, StreamingEventTextChunk{Text: content})
				if err != nil {
					m.logger.Error("Error handling OpenAI response", slog.Any("error", err))
					streamHasErrored = true

					// Close the stream and break out of the loop
					err = stream.Close()
					if err != nil {
						m.logger.Warn("Error closing OpenAI stream", slog.Any("error", err))
					}
					break
				}
			}
		}

		// Make sure to close the stream after we're done with it
		err := stream.Close()
		if err != nil {
			m.logger.Warn("Error closing OpenAI stream", slog.Any("error", err))
		}

		// Accumulate stats for this call
		promptTokens := acc.Usage.PromptTokens
		cachedTokens := acc.Usage.PromptTokensDetails.CachedTokens
		collector.Counter("input_tokens").Add(int(promptTokens))
		collector.Counter("output_tokens").Add(int(acc.Usage.CompletionTokens))
		collector.Counter("cached_read_tokens").Add(int(cachedTokens))

		// Update execution tracker with token counts
		tracker.AddTokens(
			promptTokens,
			acc.Usage.CompletionTokens,
			cachedTokens,
		)

		m.metrics.RecordCall(
			ctx,
			GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model),
			promptTokens,
			acc.Usage.CompletionTokens,
			cachedTokens,
			0,
		)

		// Flush the jsonstream parser if we were using structured streaming
		if jsParser != nil && opts.StructuredStreamingFunc != nil {
			events, err := jsParser.Flush()
			if err != nil {
				m.logger.Error("Error flushing structured JSON stream", slog.Any("error", err))
			} else {
				for _, event := range events {
					if err := opts.StructuredStreamingFunc(ctx, event); err != nil {
						m.logger.Error("Error handling structured streaming event", slog.Any("error", err))
						break
					}
				}
			}
		}

		if streamHasErrored {
			// The stream handling has failed, record metrics and return
			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordFailure(m.statsModel, collector)
			}

			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeStreamProcessing)
			return nil, errors.New("stream handling failed")
		} else if stream.Err() != nil {
			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordFailure(m.statsModel, collector)
			}

			openaiError := &openai.Error{}
			if errors.As(stream.Err(), &openaiError) && isUnavailableStatusCode(openaiError.StatusCode) {
				m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
				return nil, errors.Mark(errors.Wrap(stream.Err(), "OpenAI model unavailable"), ErrModelUnavailable)
			}

			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
			return nil, errors.Wrap(stream.Err(), "got error from OpenAI while streaming")
		}

		// Get complete response from accumulator
		if len(acc.Choices) == 0 {
			// Record metrics for this call to the LLM as a failure
			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordFailure(m.statsModel, collector)
			}

			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeEmptyResponse)
			return nil, errors.New("no completion choices returned")
		}

		choice := acc.Choices[0]

		// Tool calls are executed early so to continue our loop with their
		// results collect them and add them to the conversation history.
		if len(choice.Message.ToolCalls) > 0 {
			// Emit an intermediate MessageEnd to signal that this is not the final reply
			if opts.StreamingFunc != nil {
				if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageEnd{Final: false}); emitErr != nil {
					return nil, emitErr
				}
			}
			// Tool calls sometimes return empty IDs, so we filter those out
			// or this will wait forever for a call that never happened
			var n int
			actualToolCalls := make([]openai.ChatCompletionMessageToolCall, 0, len(choice.Message.ToolCalls))
			for _, toolCall := range choice.Message.ToolCalls {
				if toolCall.ID == "" {
					continue
				}

				actualToolCalls = append(actualToolCalls, toolCall)
				n++
			}

			choice.Message.ToolCalls = actualToolCalls
			params.Messages = append(params.Messages, choice.Message.ToParam())

			// Wait for all tool calls to complete
			for i := 0; i < n; i++ {
				select {
				case toolResult := <-toolResultChan:
					result := ""
					if toolResult.errorString != "" {
						result = toolResult.errorString
					} else {
						result = toolResult.results
					}

					params.Messages = append(params.Messages, openai.ChatCompletionMessageParamUnion{
						OfTool: &openai.ChatCompletionToolMessageParam{
							Content: openai.ChatCompletionToolMessageParamContentUnion{
								OfString: openai.String(result),
							},
							ToolCallID: toolResult.id,
						},
					})
				case <-ctx.Done():
					// Record metrics for this call to the LLM as a failure
					if metrics := GetMetrics(ctx); metrics != nil {
						metrics.RecordFailure(m.statsModel, collector)
					}

					return nil, ctx.Err()
				}
			}

			// Record metrics for this call to the LLM as a success
			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordSuccess(m.statsModel, collector)
			}

			// Continue conversation with tool results
			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)
			continue
		}

		// Emit a final MessageEnd to signal that the agentic loop is done
		if opts.StreamingFunc != nil {
			if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageEnd{Final: true}); emitErr != nil {
				return nil, emitErr
			}
		}

		// LLM call loop is done, record metrics for this call to the LLM as a success
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordSuccess(m.statsModel, collector)
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

		// Return structured result if ResponseSchema was set
		if opts.ResponseSchema != nil {
			content := choice.Message.Content
			if jsParser != nil {
				// Use the accumulated content from structured streaming
				content = structuredContentBuilder.String()
			}
			result, err := opts.ResponseSchema.ParseInto([]byte(content))
			if err != nil {
				return nil, errors.WithDetail(
					errors.Wrap(err, "structured output parsing failed"),
					content,
				)
			}
			return result.(Result), nil
		}

		return TextResult{Text: choice.Message.Content}, nil
	}
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
			for _, part := range msg.Parts {
				if textPart, ok := part.(*TextPart); ok {
					content += textPart.Text
				}
			}

			result = append(result, openai.AssistantMessage(content))

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
