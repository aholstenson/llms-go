package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/aholstenson/llms-go/jsonstream"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/invopop/jsonschema"
)

// anthropicResponseFromError pulls the *http.Response out of an Anthropic
// SDK error so callers can read Retry-After-style headers.
func anthropicResponseFromError(err error) *http.Response {
	if err == nil {
		return nil
	}
	var ae *anthropic.Error
	if errors.As(err, &ae) && ae != nil {
		return ae.Response
	}
	return nil
}

type anthropicModel struct {
	logger            *slog.Logger
	metrics           *Metrics
	client            anthropic.Client
	model             string
	statsModel        string
	info              ModelInfo
	subParserRegistry map[string]SubParserConfig
}

// newAnthropicModel creates a new Anthropic model using the official Anthropic Go SDK.
// creds is consulted on every request so rotating credentials take effect
// without rebuilding the model. info carries embedded model metadata used to
// gate request parameters; the zero value is treated permissively. Optional
// SDK request options can be passed for customization (e.g.,
// option.WithBaseURL for testing).
func newAnthropicModel(logger *slog.Logger, metrics *Metrics, creds CredentialSource, model string, registry map[string]SubParserConfig, info ModelInfo, opts ...option.RequestOption) Model {
	// Prepend the auth options so they can be overridden by caller options.
	// The placeholder key satisfies the SDK; the transport replaces the
	// X-Api-Key header per request.
	httpClient := newAuthHTTPClient(nil, creds, "anthropic", applyAnthropicCredential)
	allOpts := append([]option.RequestOption{
		option.WithAPIKey(credentialPlaceholder),
		option.WithHTTPClient(httpClient),
	}, opts...)
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
	s, err := m.newSession(ctx, options...)
	if err != nil {
		return nil, err
	}
	return runSession(ctx, s)
}

func (m *anthropicModel) newSession(ctx context.Context, options ...GenerateOption) (*Session, error) {
	opts, err := resolveGenerateContentOptions(m.subParserRegistry, options...)
	if err != nil {
		return nil, err
	}

	opts.Tools, err = filterAvailableTools(ctx, opts.Tools)
	if err != nil {
		return nil, err
	}

	// Gate request parameters against the model's known capabilities.
	if len(opts.Tools) > 0 && !m.info.allowsToolCall() {
		return nil, fmt.Errorf("model %s does not support tool calling", m.statsModel)
	}
	if modality := firstUnsupportedModality(opts.Messages, m.info); modality != "" {
		return nil, fmt.Errorf("model %s does not support %s input", m.statsModel, modality)
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

	// Anthropic requires max_tokens. Default to the model's declared output
	// limit; for unknown models fall back to a conservative ceiling that fits
	// every current Claude model. Thinking tokens count against this cap, so
	// they're added before clamping against the model's output limit.
	maxOutput := m.info.resolveMaxOutputTokens(opts.MaxOutputTokens, 4096)

	// Resolve reasoning. The effective style picks the SDK shape: legacy budget
	// (pre-4.5) uses thinking budget_tokens; effort (4.5) uses output_config
	// effort; adaptive (4.6/4.7) adds the adaptive thinking config. Only the
	// budget style accepts an explicit WithMaxThinkingTokens budget.
	style := m.info.Caps.ReasoningStyle
	if style == "" {
		style = reasoningStyleBudget
	}
	switch route := resolveReasoningRoute(opts, m.info, style == reasoningStyleBudget, m.logger); route.Kind {
	case reasoningKindBudget:
		params.Thinking = anthropic.BetaThinkingConfigParamOfEnabled(int64(route.Budget))
		maxOutput += route.Budget
		// Budget-style thinking requires temperature 1.0.
		if m.info.allowsTemperature() {
			params.Temperature = anthropic.Float(1.0)
		}
	case reasoningKindEffort:
		if style == reasoningStyleBudget {
			budget := effortToBudget(route.Effort, m.info)
			params.Thinking = anthropic.BetaThinkingConfigParamOfEnabled(int64(budget))
			maxOutput += budget
			if m.info.allowsTemperature() {
				params.Temperature = anthropic.Float(1.0)
			}
		} else {
			params.OutputConfig.Effort = anthropicOutputEffort(route.Effort)
			if style == reasoningStyleAdaptive {
				params.Thinking = anthropic.BetaThinkingConfigParamUnion{
					OfAdaptive: &anthropic.BetaThinkingConfigAdaptiveParam{},
				}
			}
		}
	case reasoningKindDisable:
		// Budget- and effort-style models think only when asked, so omitting the
		// param is enough. Adaptive-style models may reason by default (Opus 5
		// does), so they need the explicit disable form.
		if style == reasoningStyleAdaptive {
			params.Thinking = anthropic.BetaThinkingConfigParamUnion{
				OfDisabled: &anthropic.BetaThinkingConfigDisabledParam{},
			}
		}
	case reasoningKindMandatory, reasoningKindSkip:
		// Mandatory-reasoning models (Opus 4.7, Fable 5) reject the disable form
		// and keep their built-in adaptive reasoning; unknown or non-reasoning
		// models get no reasoning params at all.
	}

	if clamped, didClamp := m.info.clampMaxOutputTokens(maxOutput); didClamp {
		m.logger.Warn("Clamping max tokens to model output limit",
			slog.Int("requested", maxOutput), slog.Int("limit", clamped))
		maxOutput = clamped
	}
	params.MaxTokens = int64(maxOutput)

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
			return nil, fmt.Errorf("failed to marshal response schema: %w", err)
		}

		var rawJSON map[string]any
		if err := json.Unmarshal(schemaBytes, &rawJSON); err != nil {
			return nil, fmt.Errorf("failed to unmarshal response schema: %w", err)
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

// openStream sends the request and returns the stream once the server has
// accepted it. A failure to open the stream is retried, so a rate limit or
// an overload before the first event only reaches the caller when the
// retries run out. It also returns how many attempts were made, which the
// caller stamps onto UnavailableError.
func (m *anthropicModel) openStream(
	ctx context.Context,
	params anthropic.BetaMessageNewParams,
	opts *generateContentOptions,
) (*ssestream.Stream[anthropic.BetaRawMessageStreamEventUnion], int, error) {
	reqOpts := []option.RequestOption{
		option.WithHeader("anthropic-beta", "structured-outputs-2025-11-13"),
		// llms-go owns the retry loop so WithRetryBackoff and
		// WithRetryNotify apply to Anthropic like to every other provider.
		option.WithMaxRetries(0),
	}

	classify := func(err error) (bool, int, time.Duration, bool) {
		ae := &anthropic.Error{}
		if !errors.As(err, &ae) {
			return false, 0, 0, false
		}
		if !isUnavailableStatusCode(ae.StatusCode) {
			return false, ae.StatusCode, 0, false
		}
		ra, hasRA := extractRetryAfter("anthropic", err, nil)
		return true, ae.StatusCode, ra, hasRA
	}

	attempts := 0
	stream, err := retryLoop(ctx, opts, string(GenAISystemAnthropic), m.model, classify,
		func(ctx context.Context) (*ssestream.Stream[anthropic.BetaRawMessageStreamEventUnion], error) {
			attempts++
			stream := m.client.Beta.Messages.NewStreaming(ctx, params, reqOpts...)
			// The SDK completes the request before it returns the stream, so
			// an error here means the response never started.
			if err := stream.Err(); err != nil {
				if cerr := stream.Close(); cerr != nil {
					m.logger.Warn("Failed to close stream", slog.Any("error", cerr))
				}
				return nil, err
			}
			return stream, nil
		})
	return stream, attempts, err
}

// handleStreaming reads an open stream to the end. The stream is never
// reopened here: once events flow, a failure is reported to the caller with
// whatever content and usage arrived first.
func (m *anthropicModel) handleStreaming(
	ctx context.Context,
	stream *ssestream.Stream[anthropic.BetaRawMessageStreamEventUnion],
	streamingFunc StreamingFunc,
	structuredStreamingFunc StructuredStreamingFunc,
	jsParser *jsonstream.Parser,
	structuredContentBuilder *strings.Builder,
	streamingEmitted *bool,
) (*anthropic.BetaMessage, error) {
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
	var structuredStreamErr error

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
					if structuredStreamErr == nil && structuredStreamingFunc != nil && jsParser != nil {
						structuredContentBuilder.WriteString(blockType.Text)
						events, err := jsParser.Feed(blockType.Text)
						if err != nil {
							structuredStreamErr = fmt.Errorf("%w: %w", ErrStructuredStreamParse, err)
						} else {
							for _, evt := range events {
								if streamingEmitted != nil {
									*streamingEmitted = true
								}
								if err := structuredStreamingFunc(ctx, evt); err != nil {
									structuredStreamErr = err
									break
								}
							}
						}
					}

					if streamingFunc != nil {
						if streamingEmitted != nil {
							*streamingEmitted = true
						}
						err := streamingFunc(ctx, StreamingEventTextChunk{Text: blockType.Text})
						if err != nil {
							return nil, err
						}
					}
				}
			case anthropic.BetaThinkingBlock:
				if streamingFunc != nil && blockType.Thinking != "" {
					if streamingEmitted != nil {
						*streamingEmitted = true
					}
					if err := streamingFunc(ctx, StreamingEventThinking{Text: blockType.Thinking}); err != nil {
						return nil, err
					}
				}
			case anthropic.BetaServerToolUseBlock:
				currentTool = "web_search"
			case anthropic.BetaWebSearchToolResultBlock:
				if streamingFunc != nil {
					if streamingEmitted != nil {
						*streamingEmitted = true
					}
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
				if structuredStreamErr == nil && structuredStreamingFunc != nil && jsParser != nil {
					structuredContentBuilder.WriteString(deltaVariant.Text)
					events, err := jsParser.Feed(deltaVariant.Text)
					if err != nil {
						structuredStreamErr = fmt.Errorf("%w: %w", ErrStructuredStreamParse, err)
					} else {
						for _, evt := range events {
							if streamingEmitted != nil {
								*streamingEmitted = true
							}
							if err := structuredStreamingFunc(ctx, evt); err != nil {
								structuredStreamErr = err
								break
							}
						}
					}
				}

				if streamingFunc != nil {
					if streamingEmitted != nil {
						*streamingEmitted = true
					}
					err := streamingFunc(ctx, StreamingEventTextChunk{Text: deltaVariant.Text})
					if err != nil {
						return nil, err
					}
				}
			case anthropic.BetaThinkingDelta:
				if streamingFunc != nil {
					if streamingEmitted != nil {
						*streamingEmitted = true
					}
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
						if streamingEmitted != nil {
							*streamingEmitted = true
						}
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
					if streamingEmitted != nil {
						*streamingEmitted = true
					}
					err := streamingFunc(ctx, citation)
					if err != nil {
						return nil, err
					}
				}
			}
		}
	}

	// Flush jsonstream parser
	if structuredStreamErr == nil && structuredStreamingFunc != nil && jsParser != nil {
		events, err := jsParser.Flush()
		if err != nil {
			structuredStreamErr = fmt.Errorf("%w: %w", ErrStructuredStreamParse, err)
		} else {
			for _, evt := range events {
				if streamingEmitted != nil {
					*streamingEmitted = true
				}
				if err := structuredStreamingFunc(ctx, evt); err != nil {
					structuredStreamErr = err
					break
				}
			}
		}
	}

	if stream.Err() != nil {
		// Return the accumulated message so the caller can recover any
		// usage emitted before the stream errored (Anthropic reports input
		// usage on message_start, very early in the stream).
		return &message, fmt.Errorf("streaming error from Anthropic: %w", stream.Err())
	}
	if structuredStreamErr != nil {
		if streamingEmitted != nil && *streamingEmitted {
			return nil, errors.Join(ErrStreamingPartialOutput, structuredStreamErr)
		}
		return nil, structuredStreamErr
	}
	if message.Content == nil {
		return nil, errors.New("no content in message")
	}

	return &message, nil
}

// extractUsage converts the accumulated BetaMessage usage into TurnUsage.
// Returns nil when no tokens have been reported yet — handleStreaming can
// surface an empty zero-value message if it errors before message_start.
func (m *anthropicModel) extractUsage(msg *anthropic.BetaMessage) *TurnUsage {
	if msg == nil {
		return nil
	}

	if msg.Usage.InputTokens == 0 && msg.Usage.OutputTokens == 0 &&
		msg.Usage.CacheReadInputTokens == 0 && msg.Usage.CacheCreationInputTokens == 0 {
		return nil
	}

	return &TurnUsage{
		InputTokens:       msg.Usage.InputTokens,
		OutputTokens:      msg.Usage.OutputTokens,
		CachedReadTokens:  msg.Usage.CacheReadInputTokens,
		CachedWriteTokens: msg.Usage.CacheCreationInputTokens,
	}
}

// betaImageBlock converts a neutral *ImagePart / *BinaryPart into an
// Anthropic image block param. It is shared by user-message conversion and
// tool-result attachment passthrough.
func betaImageBlock(part MessagePart) (*anthropic.BetaImageBlockParam, error) {
	switch content := part.(type) {
	case *ImagePart:
		return &anthropic.BetaImageBlockParam{
			Source: anthropic.BetaImageBlockParamSourceUnion{
				OfURL: &anthropic.BetaURLImageSourceParam{URL: content.URL},
			},
		}, nil
	case *BinaryPart:
		data := base64.StdEncoding.EncodeToString(content.Data)
		return &anthropic.BetaImageBlockParam{
			Source: anthropic.BetaImageBlockParamSourceUnion{
				OfBase64: &anthropic.BetaBase64ImageSourceParam{
					MediaType: anthropic.BetaBase64ImageSourceMediaType(content.MediaType),
					Data:      data,
				},
			},
		}, nil
	default:
		return nil, fmt.Errorf("unsupported attachment type for Anthropic: %T", part)
	}
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
				case *ImagePart, *BinaryPart:
					img, err := betaImageBlock(part)
					if err != nil {
						return nil, "", err
					}
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{OfImage: img})
				case *ToolResultPart:
					// Replayed tool result. The live cache breakpoint on the
					// latest tool_result is owned by anthropicTurn.Observe; do
					// not stamp cache_control here.
					result := content.Text
					if content.Error != "" {
						result = content.Error
					}
					resultContent := []anthropic.BetaToolResultBlockParamContentUnion{
						{OfText: &anthropic.BetaTextBlockParam{Text: result}},
					}
					// Attachments only accompany a successful result.
					if content.Error == "" {
						for _, att := range content.Attachments {
							img, err := betaImageBlock(att)
							if err != nil {
								return nil, "", err
							}
							resultContent = append(resultContent, anthropic.BetaToolResultBlockParamContentUnion{OfImage: img})
						}
					}
					contentParts = append(contentParts, anthropic.BetaContentBlockParamUnion{
						OfToolResult: &anthropic.BetaToolResultBlockParam{
							ToolUseID: content.ID,
							Content:   resultContent,
							IsError:   anthropic.Bool(content.Error != ""),
						},
					})
				default:
					return nil, "", fmt.Errorf("unsupported part type for Anthropic: %T", part)
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
							return nil, "", fmt.Errorf("invalid tool call arguments for %s: %w", content.Name, err)
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
					return nil, "", fmt.Errorf("unsupported assistant part type for Anthropic: %T", part)
				}
			}

			msgParam = newBetaAssistantMessage(contentParts...)

		default:
			return nil, "", fmt.Errorf("unsupported message role for Anthropic: %s", msg.Role)
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

	for _, tool := range tools {
		if _, exists := toolMap[tool.Name()]; exists {
			continue
		}

		// Create JSON schema for tool parameters
		schema := jsonSchemaReflector.Reflect(tool.Schema())

		// Create Anthropic tool definition
		toolParam := anthropic.BetaToolParam{
			Name:        tool.Name(),
			Description: anthropic.String(tool.Description()),
			InputSchema: anthropic.BetaToolInputSchemaParam{
				Properties: schema.Properties,
				Type:       constant.Object(schema.Type),
			},
		}

		result = append(result, anthropic.BetaToolUnionParam{OfTool: &toolParam})
		toolMap[tool.Name()] = tool
	}

	if len(result) > 0 {
		result[len(result)-1].OfTool.CacheControl = anthropic.NewBetaCacheControlEphemeralParam()
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
	if t.response != nil {
		t.params.Messages = append(t.params.Messages, betaMessageToParam(t.response))
	}
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
		if o.Error != nil {
			result = o.ModelError()
		}

		cacheControl := anthropic.BetaCacheControlEphemeralParam{}
		if i == len(outcomes)-1 {
			cacheControl = anthropic.NewBetaCacheControlEphemeralParam()
		}

		resultContent := []anthropic.BetaToolResultBlockParamContentUnion{
			{OfText: &anthropic.BetaTextBlockParam{Text: result}},
		}
		// Attachments only accompany a successful result.
		if o.Error == nil {
			for _, att := range o.Attachments {
				img, err := betaImageBlock(att)
				if err != nil {
					return err
				}
				resultContent = append(resultContent, anthropic.BetaToolResultBlockParamContentUnion{OfImage: img})
			}
		}

		toolResultsContent = append(toolResultsContent, anthropic.BetaContentBlockParamUnion{
			OfToolResult: &anthropic.BetaToolResultBlockParam{
				ToolUseID:    o.ID,
				Content:      resultContent,
				IsError:      anthropic.Bool(o.Error != nil),
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
	var streamingEmitted bool

	// Anthropic's API rejects non-streaming requests for any generation that
	// may exceed 10 minutes so we run in streaming mode for all requests.
	stream, attempts, err := m.openStream(ctx, t.params, t.opts)
	if err != nil {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}

		// The stream never started, so there is no partial output or usage
		// to preserve. A retryable failure is already an UnavailableError
		// with the exact attempt count on it.
		var ue *UnavailableError
		if errors.As(err, &ue) {
			m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
			return TurnOutput{}, ue
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, fmt.Errorf("error from Anthropic (model %s): %w", m.model, err)
	}

	response, err = m.handleStreaming(ctx, stream, t.opts.StreamingFunc, t.opts.StructuredStreamingFunc, t.jsParser, &t.structuredContentBuilder, &streamingEmitted)

	anthropicError := &anthropic.Error{}
	if errors.As(err, &anthropicError) {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}

		// On a mid-stream failure the SDK may already have surfaced
		// message_start usage; preserve it so cost accounting is not lost.
		partialUsage := m.extractUsage(response)
		if partialUsage != nil {
			collector.Counter("input_tokens").Add(int(partialUsage.InputTokens))
			collector.Counter("output_tokens").Add(int(partialUsage.OutputTokens))
			collector.Counter("cached_read_tokens").Add(int(partialUsage.CachedReadTokens))
			collector.Counter("cached_write_tokens").Add(int(partialUsage.CachedWriteTokens))
			m.metrics.RecordCall(ctx,
				GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model),
				partialUsage.InputTokens,
				partialUsage.OutputTokens,
				partialUsage.CachedReadTokens,
				partialUsage.CachedWriteTokens,
			)
		}

		if isUnavailableStatusCode(anthropicError.StatusCode) {
			m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
			ra, hasRA := extractRetryAfter("anthropic", err, nil)
			if hasRA && t.opts.RetryAfterCap > 0 && ra > t.opts.RetryAfterCap {
				ra = t.opts.RetryAfterCap
			}
			ue := &UnavailableError{
				Provider:      string(GenAISystemAnthropic),
				Model:         m.model,
				StatusCode:    anthropicError.StatusCode,
				RetryAfter:    ra,
				HasRetryAfter: hasRA,
				Attempts:      attempts,
				PartialOutput: streamingEmitted,
				PartialUsage:  partialUsage,
				Cause:         err,
			}
			if streamingEmitted {
				return TurnOutput{}, errors.Join(ue, ErrStreamingPartialOutput)
			}
			return TurnOutput{}, ue
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, fmt.Errorf("error from Anthropic (model %s): %w", m.model, err)
	} else if err != nil {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}

		// Non-SDK errors (e.g. structured stream parse failure) can still
		// land after message_start surfaced usage; preserve it.
		if partialUsage := m.extractUsage(response); partialUsage != nil {
			collector.Counter("input_tokens").Add(int(partialUsage.InputTokens))
			collector.Counter("output_tokens").Add(int(partialUsage.OutputTokens))
			collector.Counter("cached_read_tokens").Add(int(partialUsage.CachedReadTokens))
			collector.Counter("cached_write_tokens").Add(int(partialUsage.CachedWriteTokens))
			m.metrics.RecordCall(ctx,
				GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model),
				partialUsage.InputTokens,
				partialUsage.OutputTokens,
				partialUsage.CachedReadTokens,
				partialUsage.CachedWriteTokens,
			)
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemAnthropic, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, fmt.Errorf("error from Anthropic (model %s): %w", m.model, err)
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
		arguments := b.JSON.Input.Raw()
		if arguments == "" {
			inputBytes, err := json.Marshal(b.Input)
			if err != nil {
				return TurnOutput{}, fmt.Errorf("marshal tool use input for %s: %w", b.Name, err)
			}
			arguments = string(inputBytes)
		}
		toolCalls = append(toolCalls, ToolCall{
			ID:        b.ID,
			Name:      b.Name,
			Arguments: arguments,
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

// anthropicOutputEffort maps a portable Effort to the Anthropic output-config
// effort enum. EffortNone never reaches here (it resolves to the off path);
// unrecognized values fall back to medium.
func anthropicOutputEffort(e Effort) anthropic.BetaOutputConfigEffort {
	switch e {
	case EffortLow:
		return anthropic.BetaOutputConfigEffortLow
	case EffortMedium:
		return anthropic.BetaOutputConfigEffortMedium
	case EffortHigh:
		return anthropic.BetaOutputConfigEffortHigh
	case "max":
		return anthropic.BetaOutputConfigEffortMax
	default:
		return anthropic.BetaOutputConfigEffortMedium
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
