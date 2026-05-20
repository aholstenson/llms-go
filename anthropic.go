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
	s, err := m.newSession(options...)
	if err != nil {
		return nil, err
	}
	return runSession(ctx, s)
}

func (m *anthropicModel) newSession(options ...GenerateOption) (*Session, error) {
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

	var jsParser *jsonstream.Parser
	if opts.StructuredStreamingFunc != nil && opts.StructuredStreamingSchema != nil {
		jsParser = jsonstream.New(opts.StructuredStreamingSchema)
	}

	turn := &anthropicTurn{
		m:        m,
		opts:     opts,
		params:   params,
		jsParser: jsParser,
	}

	return newSession(turn, newTracker(opts), toolMap, opts, m.logger), nil
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
				case *ToolResultPart:
					// Replayed tool result. The live cache breakpoint on the
					// latest tool_result is owned by anthropicTurn.Observe; do
					// not stamp cache_control here.
					result := content.Text
					if content.Error != "" {
						result = content.Error
					}
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfToolResult: &anthropic.BetaToolResultBlockParam{
							ToolUseID: content.ID,
							Content: []anthropic.BetaToolResultBlockParamContentUnion{
								{OfText: &anthropic.BetaTextBlockParam{Text: result}},
							},
							IsError: anthropic.Bool(content.Error != ""),
						},
					})
				default:
					return nil, "", errors.Newf("unsupported part type for Anthropic: %T", part)
				}
			}

			msgParam = newBetaUserMessage(contentParts...)

		case RoleAssistant:
			var contentParts []anthropic.BetaContentBlockParamUnion
			// Anthropic requires thinking blocks first in the assistant turn;
			// assistantMessage already prepends them, but iterate them first
			// here too so any caller-built transcript stays valid.
			for _, part := range msg.Parts {
				tp, ok := part.(*ThinkingPart)
				if !ok {
					continue
				}
				if tp.Redacted {
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfRedactedThinking: &anthropic.BetaRedactedThinkingBlockParam{Data: tp.Data},
					})
				} else {
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfThinking: &anthropic.BetaThinkingBlockParam{
							Signature: tp.Signature,
							Thinking:  tp.Text,
						},
					})
				}
			}

			for _, part := range msg.Parts {
				switch content := part.(type) {
				case *ThinkingPart:
					// Handled above so it precedes text/tool_use.
					continue
				case *TextPart:
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfText: &anthropic.BetaTextBlockParam{Text: content.Text},
					})
				case *ToolCallPart:
					var input any
					if content.Arguments != "" {
						if err := json.Unmarshal([]byte(content.Arguments), &input); err != nil {
							return nil, "", errors.Wrapf(err, "invalid tool call arguments for %s", content.Name)
						}
					}
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfToolUse: &anthropic.BetaToolUseBlockParam{
							ID:    content.ID,
							Name:  content.Name,
							Input: input,
						},
					})
				default:
					return nil, "", errors.Newf("unsupported assistant part type for Anthropic: %T", part)
				}
			}

			msgParam = newBetaAssistantMessage(contentParts...)

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

// anthropicTurn is the Anthropic-specific Turn. The cache-control bookkeeping
// lives provider-local in Observe and operates on the native history, which is
// the request's only source of truth; the neutral transcript never feeds it.
type anthropicTurn struct {
	m    *anthropicModel
	opts *generateContentOptions

	params anthropic.BetaMessageNewParams

	jsParser                 *jsonstream.Parser
	structuredContentBuilder strings.Builder

	pending []*Message

	started   bool
	callCount int
	finalText string

	// response is stashed by Next for Observe to append natively.
	response *anthropic.BetaMessage
}

func (t *anthropicTurn) Inject(msgs ...*Message) {
	t.pending = append(t.pending, msgs...)
}

func (t *anthropicTurn) FinalText() string {
	return t.finalText
}

// Observe appends the assistant message and the tool results, then applies
// Anthropic's cache-control rule: only the last tool_result block carries
// CacheControl, so any earlier tool_result blocks are cleared first.
func (t *anthropicTurn) Observe(ctx context.Context, _ TurnOutput, outcomes []ToolOutcome) error {
	// Add assistant message to conversation
	t.params.Messages = append(t.params.Messages, betaMessageToParam(t.response))
	return t.ObserveToolResults(ctx, nil, outcomes)
}

// ObserveToolResults folds tool outcomes into native history without
// appending an assistant message. Used on a reconstructed turn where the
// assistant message is already present (rebuilt by convertMessages).
func (t *anthropicTurn) ObserveToolResults(_ context.Context, _ []ToolCall, outcomes []ToolOutcome) error {
	// Clear cache control values on old tool call messages
	for i := 0; i < len(t.params.Messages); i++ {
		content := t.params.Messages[i].Content
		for _, contentBlock := range content {
			if contentBlock.OfToolResult != nil {
				// Clear cache control by setting to zero value
				contentBlock.OfToolResult.CacheControl = anthropic.BetaCacheControlEphemeralParam{}
			}
		}
	}

	// Collect tool results and add them to the conversation
	var toolResultsContent []anthropic.BetaContentBlockParamUnion
	for i, o := range outcomes {
		result := o.Text
		if o.Error != "" {
			result = o.Error
		}

		cacheControl := anthropic.BetaCacheControlEphemeralParam{}
		if i == len(outcomes)-1 {
			cacheControl = anthropic.NewBetaCacheControlEphemeralParam()
		}

		toolResultsContent = append(toolResultsContent, anthropic.BetaContentBlockParamUnion{
			OfToolResult: &anthropic.BetaToolResultBlockParam{
				ToolUseID: o.ID,
				Content: []anthropic.BetaToolResultBlockParamContentUnion{
					{OfText: &anthropic.BetaTextBlockParam{Text: result}},
				},
				IsError:      anthropic.Bool(o.Error != ""),
				CacheControl: cacheControl,
			},
		})
	}

	t.params.Messages = append(t.params.Messages, newBetaUserMessage(toolResultsContent...))
	return nil
}

func (t *anthropicTurn) Next(ctx context.Context) (TurnOutput, error) {
	m := t.m

	if !t.started {
		m.metrics.RecordGenerateRequest(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model))
		t.started = true
	}

	if len(t.pending) > 0 {
		conv, _, err := m.convertMessages("", t.pending)
		if err != nil {
			return TurnOutput{}, err
		}
		// convertMessages stamps cache_control on the last text part of its
		// last message. Injected messages are not a cache prefix boundary:
		// leaving it would add a breakpoint per Inject and blow past
		// Anthropic's 4-block cache_control limit. The cache breakpoints
		// (system, tools, original user text, latest tool_result) are owned
		// by convertMessages/convertTools and Observe.
		clearAnthropicCacheControl(conv)
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

	var response *anthropic.BetaMessage
	var err error

	func() {
		ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
		defer cancel()

		if t.opts.StreamingFunc != nil || t.opts.StructuredStreamingFunc != nil {
			response, err = m.handleStreaming(ctx, t.params, t.opts.StreamingFunc, t.opts.StructuredStreamingFunc, t.jsParser, &t.structuredContentBuilder)
		} else {
			response, err = m.client.Beta.Messages.New(ctx, t.params,
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
			return TurnOutput{}, errors.Mark(errors.Wrap(err, "Anthropic model unavailable"), ErrModelUnavailable)
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, errors.Wrap(err, "error from Anthropic")
	} else if err != nil {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, err
	}

	var toolUseBlocks []anthropic.BetaToolUseBlock
	var thinking []ThinkingBlock
	textContent := ""
	// response is the accumulated BetaMessage for both the streaming and
	// non-streaming paths (message.Accumulate folds signature deltas into the
	// thinking block), so a single walk recovers signed thinking either way.
	for _, block := range response.Content {
		switch blockType := block.AsAny().(type) {
		case anthropic.BetaTextBlock:
			textContent += blockType.Text
		case anthropic.BetaToolUseBlock:
			toolUseBlocks = append(toolUseBlocks, blockType)
		case anthropic.BetaThinkingBlock:
			thinking = append(thinking, ThinkingBlock{
				Text:      blockType.Thinking,
				Signature: blockType.Signature,
			})
		case anthropic.BetaRedactedThinkingBlock:
			thinking = append(thinking, ThinkingBlock{
				Redacted: true,
				Data:     blockType.Data,
			})
		}
	}

	collector.Counter("input_tokens").Add(int(response.Usage.InputTokens))
	collector.Counter("output_tokens").Add(int(response.Usage.OutputTokens))
	collector.Counter("cached_read_tokens").Add(int(response.Usage.CacheReadInputTokens))
	collector.Counter("cached_write_tokens").Add(int(response.Usage.CacheCreationInputTokens))

	m.metrics.RecordCall(ctx,
		GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model),
		response.Usage.InputTokens,
		response.Usage.OutputTokens,
		response.Usage.CacheReadInputTokens,
		response.Usage.CacheCreationInputTokens,
	)

	t.response = response

	toolCalls := make([]ToolCall, 0, len(toolUseBlocks))
	for _, b := range toolUseBlocks {
		inputBytes, _ := json.Marshal(b.Input)
		toolCalls = append(toolCalls, ToolCall{
			ID:        b.ID,
			Name:      b.Name,
			Arguments: string(inputBytes),
		})
	}

	content := textContent
	if t.jsParser != nil {
		content = t.structuredContentBuilder.String()
	}
	t.finalText = content

	if metrics := GetMetrics(ctx); metrics != nil {
		metrics.RecordSuccess(m.statsModel, collector)
	}
	m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

	return TurnOutput{
		Text:       textContent,
		Thinking:   thinking,
		ToolCalls:  toolCalls,
		StopReason: anthropicStopReason(string(response.StopReason), len(toolCalls) > 0),
		Usage: TurnUsage{
			InputTokens:       response.Usage.InputTokens,
			OutputTokens:      response.Usage.OutputTokens,
			CachedReadTokens:  response.Usage.CacheReadInputTokens,
			CachedWriteTokens: response.Usage.CacheCreationInputTokens,
		},
	}, nil
}

// clearAnthropicCacheControl zeroes any cache_control set by convertMessages
// on the given messages. Used for injected messages so they do not consume a
// cache_control breakpoint (Anthropic permits at most 4).
func clearAnthropicCacheControl(msgs []anthropic.BetaMessageParam) {
	for _, msg := range msgs {
		for _, block := range msg.Content {
			if block.OfText != nil {
				block.OfText.CacheControl = anthropic.BetaCacheControlEphemeralParam{}
			}
			if block.OfToolResult != nil {
				block.OfToolResult.CacheControl = anthropic.BetaCacheControlEphemeralParam{}
			}
		}
	}
}

func anthropicStopReason(reason string, hasToolCalls bool) StopReason {
	if hasToolCalls {
		return StopReasonToolUse
	}
	switch reason {
	case "tool_use":
		return StopReasonToolUse
	case "max_tokens":
		return StopReasonMaxTokens
	case "refusal":
		return StopReasonRefusal
	default:
		return StopReasonEndTurn
	}
}
