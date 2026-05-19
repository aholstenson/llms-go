package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"strings"
	"time"

	"github.com/aholstenson/llms-go/jsonstream"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/cockroachdb/errors"
	"github.com/invopop/jsonschema"
)

type anthropicModel struct {
	logger            *slog.Logger
	metrics           *Metrics
	client            anthropic.Client
	model             string
	statsModel        string
	info              ModelInfo
	subParserRegistry map[string]SubParserConfig
}

// NewAnthropicModel creates a new Anthropic model using the official Anthropic Go SDK.
// info carries embedded model metadata used to gate request parameters; the
// zero value is treated permissively. Optional SDK request options can be
// passed for customization (e.g., option.WithBaseURL for testing).
func NewAnthropicModel(logger *slog.Logger, metrics *Metrics, apiKey string, model string, registry map[string]SubParserConfig, info ModelInfo, opts ...option.RequestOption) Model {
	// Prepend the API key option so it can be overridden by caller options
	allOpts := append([]option.RequestOption{option.WithAPIKey(apiKey)}, opts...)
	client := anthropic.NewClient(allOpts...)
	return &anthropicModel{
		logger:            logger.With(slog.String("provider", "anthropic")),
		metrics:           metrics,
		client:            client,
		model:             model,
		statsModel:        "anthropic/" + model,
		info:              info,
		subParserRegistry: registry,
	}
}

// Beta message helper functions since SDK may not have beta message constructors
func newBetaUserMessage(content ...anthropic.BetaContentBlockParamUnion) anthropic.BetaMessageParam {
	return anthropic.BetaMessageParam{
		Role:    anthropic.BetaMessageParamRoleUser,
		Content: content,
	}
}

func newBetaAssistantMessage(content ...anthropic.BetaContentBlockParamUnion) anthropic.BetaMessageParam {
	return anthropic.BetaMessageParam{
		Role:    anthropic.BetaMessageParamRoleAssistant,
		Content: content,
	}
}

func betaMessageToParam(msg *anthropic.BetaMessage) anthropic.BetaMessageParam {
	content := make([]anthropic.BetaContentBlockParamUnion, len(msg.Content))
	for i, block := range msg.Content {
		switch b := block.AsAny().(type) {
		case anthropic.BetaTextBlock:
			content[i] = anthropic.BetaContentBlockParamUnion{
				OfText: &anthropic.BetaTextBlockParam{
					Text: b.Text,
				},
			}
		case anthropic.BetaToolUseBlock:
			content[i] = anthropic.BetaContentBlockParamUnion{
				OfToolUse: &anthropic.BetaToolUseBlockParam{
					ID:    b.ID,
					Name:  b.Name,
					Input: b.Input,
				},
			}
		case anthropic.BetaThinkingBlock:
			content[i] = anthropic.BetaContentBlockParamUnion{
				OfThinking: &anthropic.BetaThinkingBlockParam{
					Signature: b.Signature,
					Thinking:  b.Thinking,
				},
			}
		}
	}
	return anthropic.BetaMessageParam{
		Role:    anthropic.BetaMessageParamRole(msg.Role),
		Content: content,
	}
}

func (m *anthropicModel) GenerateContent(ctx context.Context, options ...GenerateOption) (Result, error) {
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

	// Convert messages to Anthropic format
	messages, systemPrompt, err := m.convertMessages(opts.SystemPrompt, opts.Messages)
	if err != nil {
		return nil, err
	}

	// Convert tools to Anthropic format if any
	tools, toolMap := m.convertTools(opts.Tools)

	// Build request parameters using beta types
	params := anthropic.BetaMessageNewParams{
		Model:    anthropic.Model(m.model),
		Messages: messages,
	}

	// Add system prompt if provided
	if systemPrompt != "" {
		params.System = []anthropic.BetaTextBlockParam{
			{
				Text: systemPrompt,
				// TODO: Ability to configure cache control of system prompt? Some prompts may include timestamps
				CacheControl: anthropic.NewBetaCacheControlEphemeralParam(),
			},
		}
	}

	if opts.Temperature != 0 && m.info.allowsTemperature() {
		params.Temperature = anthropic.Float(opts.Temperature)
	}

	if opts.MaxTokens != 0 {
		params.MaxTokens = int64(opts.MaxTokens)
	} else {
		params.MaxTokens = 1000
	}

	if opts.MaxThinkingTokens > 0 && m.info.allowsReasoning() {
		params.Thinking = anthropic.BetaThinkingConfigParamOfEnabled(int64(opts.MaxThinkingTokens))
		params.MaxTokens += int64(opts.MaxThinkingTokens)

		// Thinking mode requires a temperature of 1.0
		if m.info.allowsTemperature() {
			params.Temperature = anthropic.Float(1.0)
		}
	}

	if opts.WebSearch {
		// Enable the built-in web search tool
		tools = append(tools, anthropic.BetaToolUnionParam{OfWebSearchTool20250305: &anthropic.BetaWebSearchTool20250305Param{
			MaxUses: anthropic.Int(5),
		}})
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	// Add structured output format if ResponseSchema is set
	if opts.ResponseSchema != nil {
		schemaBytes, err := json.Marshal(opts.ResponseSchema.Schema.(*jsonschema.Schema))
		if err != nil {
			return nil, errors.Wrap(err, "failed to marshal response schema")
		}

		var rawJSON map[string]any
		if err := json.Unmarshal(schemaBytes, &rawJSON); err != nil {
			return nil, errors.Wrap(err, "failed to unmarshal response schema")
		}

		params.OutputFormat = anthropic.BetaJSONOutputFormatParam{Schema: rawJSON}
	}

	// Record a request to the LLM as we begin to generate content
	m.metrics.RecordGenerateRequest(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model))

	maxSteps := opts.MaxSteps
	if maxSteps == 0 {
		maxSteps = 10
	}

	// Create and inject execution tracker
	tracker := NewExecutionTracker(maxSteps)
	ctx = WithExecutionContext(ctx, tracker)

	// Set up structured streaming parser if configured
	var jsParser *jsonstream.Parser
	var structuredContentBuilder strings.Builder
	if opts.StructuredStreamingFunc != nil && opts.StructuredStreamingSchema != nil {
		jsParser = jsonstream.New(opts.StructuredStreamingSchema)
	}

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

		// Each iteration which calls the LLM starts with fresh stats
		collector := NewCollector()
		collector.Counter("calls").Add(1)
		if steps == 1 {
			// If this is the first step, record a request
			collector.Counter("requests").Add(1)
		}

		var response *anthropic.BetaMessage
		var err error

		if opts.StreamingFunc != nil {
			if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageStart{}); emitErr != nil {
				return nil, emitErr
			}
		}

		func() {
			ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
			defer cancel()

			if opts.StreamingFunc != nil || opts.StructuredStreamingFunc != nil {
				// Handle streaming
				response, err = m.handleStreaming(ctx, params, opts.StreamingFunc, opts.StructuredStreamingFunc, jsParser, &structuredContentBuilder)
			} else {
				// Handle non-streaming
				response, err = m.client.Beta.Messages.New(ctx, params,
					option.WithHeader("anthropic-beta", "structured-outputs-2025-11-13"),
				)
			}
		}()

		anthropicError := &anthropic.Error{}
		if errors.As(err, &anthropicError) {
			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordFailure(m.statsModel, collector)
			}

			if isUnavailableStatusCode(anthropicError.StatusCode) {
				m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
				return nil, errors.Mark(errors.Wrap(err, "Anthropic model unavailable"), ErrModelUnavailable)
			}

			m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
			return nil, errors.Wrap(err, "error from Anthropic")
		} else if err != nil {
			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordFailure(m.statsModel, collector)
			}
			m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
			return nil, err
		}

		// Check for tool calls
		var toolCalls []anthropic.BetaToolUseBlock
		textContent := ""

		for _, block := range response.Content {
			switch blockType := block.AsAny().(type) {
			case anthropic.BetaTextBlock:
				textContent += blockType.Text
			case anthropic.BetaToolUseBlock:
				toolCalls = append(toolCalls, blockType)
			}
		}

		collector.Counter("input_tokens").Add(int(response.Usage.InputTokens))
		collector.Counter("output_tokens").Add(int(response.Usage.OutputTokens))
		collector.Counter("cached_read_tokens").Add(int(response.Usage.CacheReadInputTokens))
		collector.Counter("cached_write_tokens").Add(int(response.Usage.CacheCreationInputTokens))

		// Update execution tracker with token counts
		tracker.AddTokens(
			response.Usage.InputTokens,
			response.Usage.OutputTokens,
			response.Usage.CacheReadInputTokens,
		)

		m.metrics.RecordCall(ctx,
			GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model),
			response.Usage.InputTokens,
			response.Usage.OutputTokens,
			response.Usage.CacheReadInputTokens,
			response.Usage.CacheCreationInputTokens,
		)

		// If no tool calls, return the text content
		if len(toolCalls) == 0 {
			// Emit a final MessageEnd to signal that the agentic loop is done
			if opts.StreamingFunc != nil {
				if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageEnd{Final: true}); emitErr != nil {
					return nil, emitErr
				}
			}

			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordSuccess(m.statsModel, collector)
			}
			m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

			// Return structured result if ResponseSchema was set
			if opts.ResponseSchema != nil {
				content := textContent
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

			return TextResult{Text: textContent}, nil
		}

		// Emit an intermediate MessageEnd to signal that this is not the final reply
		if opts.StreamingFunc != nil {
			if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageEnd{Final: false}); emitErr != nil {
				return nil, emitErr
			}
		}

		// Add assistant message to conversation
		params.Messages = append(params.Messages, betaMessageToParam(response))

		// Execute tool calls in parallel
		toolResultChan := make(chan toolResult, len(toolCalls))
		for _, toolCall := range toolCalls {
			matchedTool, ok := toolMap[toolCall.Name]
			if !ok {
				toolResultChan <- toolResult{
					id:           toolCall.ID,
					functionName: toolCall.Name,
					errorString:  "Requested tool not found",
				}
				continue
			}

			go func(call anthropic.BetaToolUseBlock) {
				// Convert input to JSON string for doToolCall
				inputBytes, _ := json.Marshal(call.Input)
				result, err := doToolCall(ctx, m.logger, opts.StreamingFunc, opts.ToolCallTimeout, call.ID, matchedTool, string(inputBytes))
				if err != nil {
					toolResultChan <- toolResult{
						id:           call.ID,
						functionName: call.Name,
						errorString:  "Tool call failed",
					}
				} else {
					toolResultChan <- toolResult{
						id:           call.ID,
						functionName: call.Name,
						results:      result,
					}
				}
			}(toolCall)
		}

		// Clear cache control values on old tool call messages
		for i := 0; i < len(params.Messages); i++ {
			content := params.Messages[i].Content
			for _, contentBlock := range content {
				if contentBlock.OfToolResult != nil {
					// Clear cache control by setting to zero value
					contentBlock.OfToolResult.CacheControl = anthropic.BetaCacheControlEphemeralParam{}
				}
			}
		}

		// Collect tool results and add them to the conversation
		var toolResultsContent []anthropic.BetaContentBlockParamUnion
		for i := 0; i < len(toolCalls); i++ {
			select {
			case toolResult := <-toolResultChan:
				result := ""
				if toolResult.errorString != "" {
					result = toolResult.errorString
				} else {
					result = toolResult.results
				}

				cacheControl := anthropic.BetaCacheControlEphemeralParam{}
				if i == len(toolCalls)-1 {
					cacheControl = anthropic.NewBetaCacheControlEphemeralParam()
				}

				block := anthropic.BetaContentBlockParamUnion{
					OfToolResult: &anthropic.BetaToolResultBlockParam{
						ToolUseID: toolResult.id,
						Content: []anthropic.BetaToolResultBlockParamContentUnion{
							{OfText: &anthropic.BetaTextBlockParam{Text: result}},
						},
						IsError:      anthropic.Bool(false),
						CacheControl: cacheControl,
					},
				}

				toolResultsContent = append(toolResultsContent, block)
			case <-ctx.Done():
				if metrics := GetMetrics(ctx); metrics != nil {
					metrics.RecordFailure(m.statsModel, collector)
				}

				m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
				return nil, ctx.Err()
			}
		}

		// Add tool results message to conversation
		params.Messages = append(params.Messages, newBetaUserMessage(toolResultsContent...))

		// Record metrics for this call to the LLM as a success
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordSuccess(m.statsModel, collector)
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

		// Continue the conversation loop
	}
}

func (m *anthropicModel) handleStreaming(
	ctx context.Context,
	params anthropic.BetaMessageNewParams,
	streamingFunc StreamingFunc,
	structuredStreamingFunc StructuredStreamingFunc,
	jsParser *jsonstream.Parser,
	structuredContentBuilder *strings.Builder,
) (*anthropic.BetaMessage, error) {
	stream := m.client.Beta.Messages.NewStreaming(ctx, params,
		option.WithHeader("anthropic-beta", "structured-outputs-2025-11-13"),
	)
	defer func() {
		err := stream.Close()
		if err != nil {
			m.logger.Warn("Failed to close stream", slog.Any("error", err))
		}
	}()

	message := anthropic.BetaMessage{}
	currentTool := ""
	partialJSON := ""
	citations := []StreamingEventCitation{}

	for stream.Next() {
		event := stream.Current()
		err := message.Accumulate(event)
		if err != nil {
			return nil, err
		}

		// Handle different event types for streaming
		switch eventVariant := event.AsAny().(type) {
		case anthropic.BetaRawContentBlockStartEvent:
			currentTool = ""
			partialJSON = ""
			citations = []StreamingEventCitation{}

			switch blockType := eventVariant.ContentBlock.AsAny().(type) {
			case anthropic.BetaTextBlock:
				if blockType.Text != "" {
					// Handle structured streaming
					if structuredStreamingFunc != nil && jsParser != nil {
						structuredContentBuilder.WriteString(blockType.Text)
						events, err := jsParser.Feed(blockType.Text)
						if err != nil {
							m.logger.Error("Error parsing structured JSON stream", slog.Any("error", err))
						} else {
							for _, evt := range events {
								if err := structuredStreamingFunc(ctx, evt); err != nil {
									m.logger.Error("Error handling structured streaming event", slog.Any("error", err))
									break
								}
							}
						}
					}

					if streamingFunc != nil {
						err := streamingFunc(ctx, StreamingEventTextChunk{Text: blockType.Text})
						if err != nil {
							return nil, err
						}
					}
				}
			case anthropic.BetaThinkingBlock:
				if streamingFunc != nil && blockType.Thinking != "" {
					if err := streamingFunc(ctx, StreamingEventThinking{Text: blockType.Thinking}); err != nil {
						return nil, err
					}
				}
			case anthropic.BetaServerToolUseBlock:
				currentTool = "web_search"
			case anthropic.BetaWebSearchToolResultBlock:
				if streamingFunc != nil {
					err := streamingFunc(ctx, StreamingEventToolResult{ToolID: currentTool, Result: &searchToolResult{
						count: -1,
						items: nil,
					}})
					if err != nil {
						return nil, err
					}
				}
			}
		case anthropic.BetaRawContentBlockDeltaEvent:
			switch deltaVariant := eventVariant.Delta.AsAny().(type) {
			case anthropic.BetaTextDelta:
				// Handle structured streaming
				if structuredStreamingFunc != nil && jsParser != nil {
					structuredContentBuilder.WriteString(deltaVariant.Text)
					events, err := jsParser.Feed(deltaVariant.Text)
					if err != nil {
						m.logger.Error("Error parsing structured JSON stream", slog.Any("error", err))
					} else {
						for _, evt := range events {
							if err := structuredStreamingFunc(ctx, evt); err != nil {
								m.logger.Error("Error handling structured streaming event", slog.Any("error", err))
								break
							}
						}
					}
				}

				if streamingFunc != nil {
					err := streamingFunc(ctx, StreamingEventTextChunk{Text: deltaVariant.Text})
					if err != nil {
						return nil, err
					}
				}
			case anthropic.BetaThinkingDelta:
				if streamingFunc != nil {
					if err := streamingFunc(ctx, StreamingEventThinking{Text: deltaVariant.Thinking}); err != nil {
						return nil, err
					}
				}
			case anthropic.BetaCitationsDelta:
				searchLocation := deltaVariant.Citation.AsAny()
				switch searchLocation := searchLocation.(type) {
				case anthropic.BetaCitationsWebSearchResultLocation:
					found := false
					for _, citation := range citations {
						if citation.URL == searchLocation.URL {
							found = true
							break
						}
					}

					if !found {
						citations = append(citations, StreamingEventCitation{
							Title: searchLocation.Title,
							URL:   searchLocation.URL,
						})
					}
				}
			case anthropic.BetaInputJSONDelta:
				if currentTool == "web_search" {
					partialJSON += deltaVariant.PartialJSON
					var searchToolArgs searchToolArgs
					err := json.Unmarshal([]byte(partialJSON), &searchToolArgs)
					var syntaxError *json.SyntaxError
					if errors.As(err, &syntaxError) {
						break
					} else if err != nil {
						return nil, err
					}

					if streamingFunc != nil {
						err = streamingFunc(ctx, StreamingEventToolUse{ToolID: currentTool, Arguments: &searchToolArgs})
						if err != nil {
							return nil, err
						}
					}
				}
			}
		case anthropic.BetaRawContentBlockStopEvent:
			for _, citation := range citations {
				if streamingFunc != nil {
					err := streamingFunc(ctx, citation)
					if err != nil {
						return nil, err
					}
				}
			}
		}
	}

	// Flush jsonstream parser
	if structuredStreamingFunc != nil && jsParser != nil {
		events, err := jsParser.Flush()
		if err != nil {
			m.logger.Error("Error flushing structured JSON stream", slog.Any("error", err))
		} else {
			for _, evt := range events {
				if err := structuredStreamingFunc(ctx, evt); err != nil {
					m.logger.Error("Error handling structured streaming event", slog.Any("error", err))
					break
				}
			}
		}
	}

	if stream.Err() != nil {
		return nil, errors.Wrap(stream.Err(), "streaming error from Anthropic")
	} else if message.Content == nil {
		return nil, errors.New("no content in message")
	}

	return &message, nil
}

func (m *anthropicModel) convertMessages(systemPrompt string, messages []*Message) ([]anthropic.BetaMessageParam, string, error) {
	result := make([]anthropic.BetaMessageParam, 0, len(messages))

	lastIndex := len(messages) - 1

	for i, msg := range messages {
		var msgParam anthropic.BetaMessageParam
		switch msg.Role {
		case RoleUser:
			// Handle user message
			if len(msg.Parts) == 0 {
				return nil, "", errors.New("user message has no parts")
			}

			var contentParts []anthropic.BetaContentBlockParamUnion

			// Track TextParts to only cache the last one if this is the last message
			lastTextPartIndex := -1
			for j, part := range msg.Parts {
				if _, ok := part.(*TextPart); ok {
					lastTextPartIndex = j
				}
			}

			for j, part := range msg.Parts {
				switch content := part.(type) {
				case *TextPart:
					var cacheControl anthropic.BetaCacheControlEphemeralParam
					if i == lastIndex && j == lastTextPartIndex {
						cacheControl = anthropic.NewBetaCacheControlEphemeralParam()
					}

					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfText: &anthropic.BetaTextBlockParam{
							Text:         content.Text,
							CacheControl: cacheControl,
						},
					})
				case *ImagePart:
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfImage: &anthropic.BetaImageBlockParam{
							Source: anthropic.BetaImageBlockParamSourceUnion{
								OfURL: &anthropic.BetaURLImageSourceParam{
									URL: content.URL,
								},
							},
						},
					})
				case *BinaryPart:
					data := base64.StdEncoding.EncodeToString(content.Data)
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfImage: &anthropic.BetaImageBlockParam{
							Source: anthropic.BetaImageBlockParamSourceUnion{
								OfBase64: &anthropic.BetaBase64ImageSourceParam{
									MediaType: anthropic.BetaBase64ImageSourceMediaType(content.MediaType),
									Data:      data,
								},
							},
						},
					})
				default:
					return nil, "", errors.Newf("unsupported part type for Anthropic: %T", part)
				}
			}

			msgParam = newBetaUserMessage(contentParts...)

		case RoleAssistant:
			content := ""
			for _, part := range msg.Parts {
				if textPart, ok := part.(*TextPart); ok {
					content += textPart.Text
				}
			}

			msgParam = newBetaAssistantMessage(anthropic.BetaContentBlockParamUnion{
				OfText: &anthropic.BetaTextBlockParam{
					Text: content,
				},
			})

		default:
			return nil, "", errors.Newf("unsupported message role for Anthropic: %s", msg.Role)
		}

		result = append(result, msgParam)
	}

	return result, systemPrompt, nil
}

func (m *anthropicModel) convertTools(tools []ToolDef) ([]anthropic.BetaToolUnionParam, map[string]ToolDef) {
	if len(tools) == 0 {
		return nil, make(map[string]ToolDef)
	}

	result := make([]anthropic.BetaToolUnionParam, 0, len(tools))
	toolMap := make(map[string]ToolDef, len(tools))

	for i, tool := range tools {
		if _, exists := toolMap[tool.Name()]; exists {
			continue
		}

		// Create JSON schema for tool parameters
		schema := jsonSchemaReflector.Reflect(tool.Schema())

		// Cache the available tools
		var cacheControl anthropic.BetaCacheControlEphemeralParam
		if i == len(tools)-1 {
			cacheControl = anthropic.NewBetaCacheControlEphemeralParam()
		}

		// Create Anthropic tool definition
		toolParam := anthropic.BetaToolParam{
			Name:        tool.Name(),
			Description: anthropic.String(tool.Description()),
			InputSchema: anthropic.BetaToolInputSchemaParam{
				Properties: schema.Properties,
				Type:       constant.Object(schema.Type),
			},
			CacheControl: cacheControl,
		}

		result = append(result, anthropic.BetaToolUnionParam{OfTool: &toolParam})
		toolMap[tool.Name()] = tool
	}

	return result, toolMap
}
