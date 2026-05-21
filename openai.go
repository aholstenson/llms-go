package llms

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/aholstenson/llms-go/jsonstream"
	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// openaiResponseFromError pulls the *http.Response out of an OpenAI SDK
// error so callers can read Retry-After-style headers.
func openaiResponseFromError(err error) *http.Response {
	if err == nil {
		return nil
	}
	var oe *openai.Error
	if errors.As(err, &oe) && oe != nil {
		return oe.Response
	}
	return nil
}

type openaiModel struct {
	logger            *slog.Logger
	metrics           *Metrics
	client            openai.Client
	statsModel        string
	model             string
	info              ModelInfo
	subParserRegistry map[string]SubParserConfig
}

// newOpenAIModel creates a new OpenAI model using the official OpenAI Go SDK
// against the Responses API. info carries embedded model metadata used to
// gate request parameters; the zero value is treated permissively. Optional
// SDK request options can be passed for customization (e.g.
// option.WithBaseURL for testing).
func newOpenAIModel(logger *slog.Logger, metrics *Metrics, apiKey string, model string, registry map[string]SubParserConfig, info ModelInfo, opts ...option.RequestOption) Model {
	allOpts := append([]option.RequestOption{option.WithAPIKey(apiKey)}, opts...)
	client := openai.NewClient(allOpts...)
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
	s, err := m.newSession(ctx, options...)
	if err != nil {
		return nil, err
	}
	return runSession(ctx, s)
}

func (m *openaiModel) newSession(ctx context.Context, options ...GenerateOption) (*Session, error) {
	opts, err := resolveGenerateContentOptions(m.subParserRegistry, options...)
	if err != nil {
		return nil, err
	}

	opts.Tools, err = filterAvailableTools(ctx, opts.Tools)
	if err != nil {
		return nil, err
	}

	if len(opts.Tools) > 0 && !m.info.allowsToolCall() {
		return nil, fmt.Errorf("model %s does not support tool calling", m.statsModel)
	}
	if modality := firstUnsupportedModality(opts.Messages, m.info); modality != "" {
		return nil, fmt.Errorf("model %s does not support %s input", m.statsModel, modality)
	}

	inputItems, err := m.convertMessages(opts.Messages)
	if err != nil {
		return nil, err
	}

	tools, toolMap, err := m.convertTools(opts.Tools)
	if err != nil {
		return nil, err
	}

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(m.model),
		// Drive the loop statelessly: we re-send the full input every turn
		// and disable server-side storage. Request encrypted reasoning so
		// o-series reasoning items survive replay between turns.
		Store: openai.Bool(false),
		Include: []responses.ResponseIncludable{
			responses.ResponseIncludableReasoningEncryptedContent,
		},
	}

	if opts.SystemPrompt != "" {
		params.Instructions = openai.String(opts.SystemPrompt)
	}

	if opts.Temperature != 0 && m.info.allowsTemperature() {
		params.Temperature = openai.Float(opts.Temperature)
	}

	if v := m.info.resolveMaxOutputTokens(opts.MaxOutputTokens, 0); v > 0 {
		if clamped, didClamp := m.info.clampMaxOutputTokens(v); didClamp {
			m.logger.Warn("Clamping max tokens to model output limit",
				slog.Int("requested", v), slog.Int("limit", clamped))
			v = clamped
		}
		params.MaxOutputTokens = openai.Int(int64(v))
	}

	if len(tools) > 0 {
		params.Tools = tools
	}

	if opts.ResponseSchema != nil {
		schemaBytes, err := json.Marshal(opts.ResponseSchema.Schema.(*jsonschema.Schema))
		if err != nil {
			return nil, fmt.Errorf("failed to marshal response schema: %w", err)
		}
		var rawJSON map[string]any
		if err := json.Unmarshal(schemaBytes, &rawJSON); err != nil {
			return nil, fmt.Errorf("failed to unmarshal response schema: %w", err)
		}
		params.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
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
		m:          m,
		opts:       opts,
		params:     params,
		inputItems: inputItems,
		jsParser:   jsParser,
	}

	return newSession(turn, newTracker(opts), toolMap, opts, m.logger), nil
}

func (m *openaiModel) convertMessages(messages []*Message) (responses.ResponseInputParam, error) {
	result := make(responses.ResponseInputParam, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case RoleUser:
			if len(msg.Parts) == 0 {
				return nil, errors.New("user message has no parts")
			}

			// Tool-result messages map to standalone function_call_output
			// items in the Responses input, one per result.
			if _, isToolResult := msg.Parts[0].(*ToolResultPart); isToolResult {
				for _, part := range msg.Parts {
					trp, ok := part.(*ToolResultPart)
					if !ok {
						return nil, fmt.Errorf("cannot mix tool-result and %T parts for OpenAI", part)
					}
					text := trp.Text
					if trp.Error != "" {
						text = trp.Error
					}
					result = append(result, responses.ResponseInputItemUnionParam{
						OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
							CallID: trp.ID,
							Output: text,
						},
					})
				}
				continue
			}

			content, err := convertUserParts(msg.Parts)
			if err != nil {
				return nil, err
			}
			result = append(result, responses.ResponseInputItemUnionParam{
				OfInputMessage: &responses.ResponseInputItemMessageParam{
					Role:    "user",
					Content: content,
				},
			})

		case RoleAssistant:
			var textBuf strings.Builder
			var toolCalls []responses.ResponseFunctionToolCallParam
			for _, part := range msg.Parts {
				switch p := part.(type) {
				case *ThinkingPart:
					// Responses API needs reasoning items with id +
					// encrypted_content to replay; the neutral ThinkingPart
					// carries only text, so we can't reconstruct a valid
					// reasoning item from user-supplied history.
					continue
				case *TextPart:
					textBuf.WriteString(p.Text)
				case *ToolCallPart:
					toolCalls = append(toolCalls, responses.ResponseFunctionToolCallParam{
						CallID:    p.ID,
						Name:      p.Name,
						Arguments: p.Arguments,
					})
				default:
					return nil, fmt.Errorf("unsupported assistant part type for OpenAI: %T", part)
				}
			}
			if textBuf.Len() > 0 {
				result = append(result, responses.ResponseInputItemParamOfMessage(textBuf.String(), responses.EasyInputMessageRoleAssistant))
			}
			for i := range toolCalls {
				result = append(result, responses.ResponseInputItemUnionParam{
					OfFunctionCall: &toolCalls[i],
				})
			}

		default:
			return nil, fmt.Errorf("unsupported message role for OpenAI: %s", msg.Role)
		}
	}

	return result, nil
}

// filenameForMediaType returns a synthetic filename for an input_file part.
// The Responses API requires a filename alongside file_data, but BinaryPart
// carries only a media type, so we pick an extension from the mime registry
// and fall back to ".bin" for unknown types.
func filenameForMediaType(mediaType string) string {
	exts, _ := mime.ExtensionsByType(mediaType)
	if len(exts) > 0 {
		return "file" + exts[0]
	}
	return "file.bin"
}

func convertUserParts(parts []MessagePart) (responses.ResponseInputMessageContentListParam, error) {
	out := make(responses.ResponseInputMessageContentListParam, 0, len(parts))
	for _, part := range parts {
		switch content := part.(type) {
		case *TextPart:
			out = append(out, responses.ResponseInputContentUnionParam{
				OfInputText: &responses.ResponseInputTextParam{Text: content.Text},
			})
		case *ImagePart:
			out = append(out, responses.ResponseInputContentUnionParam{
				OfInputImage: &responses.ResponseInputImageParam{
					ImageURL: openai.String(content.URL),
					Detail:   responses.ResponseInputImageDetailAuto,
				},
			})
		case *BinaryPart:
			b64 := base64.StdEncoding.EncodeToString(content.Data)
			if strings.HasPrefix(content.MediaType, "image/") {
				dataURL := fmt.Sprintf("data:%s;base64,%s", content.MediaType, b64)
				out = append(out, responses.ResponseInputContentUnionParam{
					OfInputImage: &responses.ResponseInputImageParam{
						ImageURL: openai.String(dataURL),
						Detail:   responses.ResponseInputImageDetailAuto,
					},
				})
			} else {
				dataURL := fmt.Sprintf("data:%s;base64,%s", content.MediaType, b64)
				out = append(out, responses.ResponseInputContentUnionParam{
					OfInputFile: &responses.ResponseInputFileParam{
						FileData: openai.String(dataURL),
						Filename: openai.String(filenameForMediaType(content.MediaType)),
					},
				})
			}
		default:
			return nil, fmt.Errorf("unsupported part type for OpenAI: %T", part)
		}
	}
	return out, nil
}

func (m *openaiModel) convertTools(tools []ToolDef) ([]responses.ToolUnionParam, map[string]ToolDef, error) {
	if len(tools) == 0 {
		return nil, make(map[string]ToolDef), nil
	}

	result := make([]responses.ToolUnionParam, 0, len(tools))
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

		var rawJSON map[string]any
		if err := json.Unmarshal(schemaBytes, &rawJSON); err != nil {
			return nil, nil, fmt.Errorf("failed to unmarshal tool schema: %w", err)
		}

		result = append(result, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.Name(),
				Description: openai.String(tool.Description()),
				Parameters:  rawJSON,
				Strict:      openai.Bool(false),
			},
		})
		toolMap[tool.Name()] = tool
	}

	return result, toolMap, nil
}

// openaiTurn is the OpenAI Responses-API-specific Turn: it owns the native
// ResponseInputParam history so the agentic loop never round-trips through
// neutral types.
type openaiTurn struct {
	m    *openaiModel
	opts *generateContentOptions

	params     responses.ResponseNewParams
	inputItems responses.ResponseInputParam

	jsParser                 *jsonstream.Parser
	structuredContentBuilder strings.Builder

	pending []*Message

	started   bool
	callCount int
	finalText string

	// outputItems is the most recent assistant turn's native output items
	// (messages, function calls, reasoning), stashed by Next so Observe can
	// append them back as input items before the next call.
	outputItems []responses.ResponseInputItemUnionParam
}

func (t *openaiTurn) Inject(msgs ...*Message) {
	t.pending = append(t.pending, msgs...)
}

func (t *openaiTurn) FinalText() string {
	return t.finalText
}

func (t *openaiTurn) Observe(ctx context.Context, _ TurnOutput, outcomes []ToolOutcome) error {
	t.inputItems = append(t.inputItems, t.outputItems...)
	return t.ObserveToolResults(ctx, nil, outcomes)
}

// ObserveToolResults appends only the tool-result items, used when the
// assistant turn is already in native history (reconstructed turn).
func (t *openaiTurn) ObserveToolResults(_ context.Context, _ []ToolCall, outcomes []ToolOutcome) error {
	for _, o := range outcomes {
		text := o.Text
		if o.Error != nil {
			text = o.ModelError()
		}
		t.inputItems = append(t.inputItems, responses.ResponseInputItemUnionParam{
			OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
				CallID: o.ID,
				Output: text,
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

	if len(t.pending) > 0 {
		conv, err := m.convertMessages(t.pending)
		if err != nil {
			return TurnOutput{}, err
		}
		t.inputItems = append(t.inputItems, conv...)
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

	t.params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: t.inputItems,
	}

	streamHasErrored := false
	streamingEmitted := false
	var structuredStreamErr error

	stream := m.client.Responses.NewStreaming(ctx, t.params, option.WithMaxRetries(t.opts.MaxRetries))

	var finalResponse *responses.Response

	for stream.Next() {
		event := stream.Current()

		switch ev := event.AsAny().(type) {
		case responses.ResponseTextDeltaEvent:
			if streamHasErrored {
				continue
			}
			if !hasRecordedFirstToken {
				m.metrics.RecordTimeToFirstToken(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start))
				hasRecordedFirstToken = true
			}

			delta := ev.Delta
			if structuredStreamErr == nil && t.jsParser != nil && t.opts.StructuredStreamingFunc != nil {
				t.structuredContentBuilder.WriteString(delta)
				parsed, err := t.jsParser.Feed(delta)
				if err != nil {
					structuredStreamErr = fmt.Errorf("%w: %w", ErrStructuredStreamParse, err)
				} else {
					for _, e := range parsed {
						streamingEmitted = true
						if err := t.opts.StructuredStreamingFunc(ctx, e); err != nil {
							structuredStreamErr = err
							break
						}
					}
				}
			}

			if t.opts.StreamingFunc != nil && !streamHasErrored {
				streamingEmitted = true
				if err := t.opts.StreamingFunc(ctx, StreamingEventTextChunk{Text: delta}); err != nil {
					m.logger.Error("Error handling OpenAI response", slog.Any("error", err))
					streamHasErrored = true
				}
			}

		case responses.ResponseReasoningSummaryTextDeltaEvent:
			if streamHasErrored || t.opts.StreamingFunc == nil {
				continue
			}
			streamingEmitted = true
			if err := t.opts.StreamingFunc(ctx, StreamingEventThinking{Text: ev.Delta}); err != nil {
				m.logger.Error("Error handling OpenAI thinking", slog.Any("error", err))
				streamHasErrored = true
			}

		case responses.ResponseCompletedEvent:
			r := ev.Response
			finalResponse = &r

		case responses.ResponseFailedEvent:
			r := ev.Response
			finalResponse = &r

		case responses.ResponseIncompleteEvent:
			r := ev.Response
			finalResponse = &r

		case responses.ResponseErrorEvent:
			m.logger.Error("OpenAI stream error",
				slog.String("code", ev.Code), slog.String("message", ev.Message))
			streamHasErrored = true
		}
	}

	if err := stream.Close(); err != nil {
		m.logger.Warn("Error closing OpenAI stream", slog.Any("error", err))
	}

	if structuredStreamErr == nil && t.jsParser != nil && t.opts.StructuredStreamingFunc != nil {
		parsed, err := t.jsParser.Flush()
		if err != nil {
			structuredStreamErr = fmt.Errorf("%w: %w", ErrStructuredStreamParse, err)
		} else {
			for _, e := range parsed {
				streamingEmitted = true
				if err := t.opts.StructuredStreamingFunc(ctx, e); err != nil {
					structuredStreamErr = err
					break
				}
			}
		}
	}

	if structuredStreamErr != nil {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeStreamProcessing)
		if streamingEmitted {
			return TurnOutput{}, errors.Join(ErrStreamingPartialOutput, structuredStreamErr)
		}
		return TurnOutput{}, structuredStreamErr
	}

	if streamHasErrored {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeStreamProcessing)
		return TurnOutput{}, errors.New("stream handling failed")
	}
	if stream.Err() != nil {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		openaiError := &openai.Error{}
		if errors.As(stream.Err(), &openaiError) && isUnavailableStatusCode(openaiError.StatusCode) {
			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
			ra, hasRA := extractRetryAfter("openai", stream.Err(), nil)
			if hasRA && t.opts.RetryAfterCap > 0 && ra > t.opts.RetryAfterCap {
				ra = t.opts.RetryAfterCap
			}
			ue := &UnavailableError{
				StatusCode:    openaiError.StatusCode,
				RetryAfter:    ra,
				HasRetryAfter: hasRA,
				Attempts:      t.opts.MaxRetries + 1,
				PartialOutput: streamingEmitted,
				Cause:         stream.Err(),
			}
			if streamingEmitted {
				return TurnOutput{}, errors.Join(ue, ErrStreamingPartialOutput)
			}
			return TurnOutput{}, ue
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, fmt.Errorf("got error from OpenAI while streaming: %w", stream.Err())
	}

	if finalResponse == nil {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeEmptyResponse)
		return TurnOutput{}, errors.New("no completed response from OpenAI")
	}

	if finalResponse.Status == responses.ResponseStatusFailed {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}
		respErr := finalResponse.Error
		err := fmt.Errorf("OpenAI response failed: %s: %s", respErr.Code, respErr.Message)
		if respErr.Code == "rate_limit_exceeded" {
			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
			ue := &UnavailableError{
				StatusCode:    0,
				Attempts:      t.opts.MaxRetries + 1,
				PartialOutput: streamingEmitted,
				Cause:         err,
			}
			if streamingEmitted {
				return TurnOutput{}, errors.Join(ue, ErrStreamingPartialOutput)
			}
			return TurnOutput{}, ue
		}
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, err
	}

	// Walk the response output, capturing neutral results and stashing the
	// native items so Observe can replay them on the next turn.
	var textBuilder strings.Builder
	var thinking []ThinkingBlock
	var toolCalls []ToolCall
	hasRefusal := false
	outputItems := make([]responses.ResponseInputItemUnionParam, 0, len(finalResponse.Output))

	for _, item := range finalResponse.Output {
		switch v := item.AsAny().(type) {
		case responses.ResponseOutputMessage:
			for _, c := range v.Content {
				switch c.Type {
				case "output_text":
					textBuilder.WriteString(c.Text)
				case "refusal":
					hasRefusal = true
				}
			}
			msgParam := v.ToParam()
			outputItems = append(outputItems, responses.ResponseInputItemUnionParam{
				OfOutputMessage: &msgParam,
			})

		case responses.ResponseFunctionToolCall:
			toolCalls = append(toolCalls, ToolCall{
				ID:        v.CallID,
				Name:      v.Name,
				Arguments: v.Arguments,
			})
			fcParam := v.ToParam()
			outputItems = append(outputItems, responses.ResponseInputItemUnionParam{
				OfFunctionCall: &fcParam,
			})

		case responses.ResponseReasoningItem:
			var summaryText strings.Builder
			for _, s := range v.Summary {
				summaryText.WriteString(s.Text)
			}
			if summaryText.Len() > 0 {
				thinking = append(thinking, ThinkingBlock{Text: summaryText.String()})
			}
			reasoningParam := v.ToParam()
			outputItems = append(outputItems, responses.ResponseInputItemUnionParam{
				OfReasoning: &reasoningParam,
			})

		default:
			// Skip built-in tool outputs (web search, file search, computer
			// use, etc.) — not surfaced through the neutral interface.
		}
	}

	t.outputItems = outputItems

	usage := finalResponse.Usage
	promptTokens := usage.InputTokens
	cachedTokens := usage.InputTokensDetails.CachedTokens
	outputTokens := usage.OutputTokens
	reasoningTokens := usage.OutputTokensDetails.ReasoningTokens

	collector.Counter("input_tokens").Add(int(promptTokens))
	collector.Counter("output_tokens").Add(int(outputTokens))
	collector.Counter("cached_read_tokens").Add(int(cachedTokens))

	m.metrics.RecordCall(
		ctx,
		GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model),
		promptTokens, outputTokens, cachedTokens, 0,
	)

	textContent := textBuilder.String()
	content := textContent
	if t.jsParser != nil {
		content = t.structuredContentBuilder.String()
	}
	t.finalText = content

	if metrics := GetMetrics(ctx); metrics != nil {
		metrics.RecordSuccess(m.statsModel, collector)
	}
	m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

	return TurnOutput{
		Text:       textContent,
		Thinking:   thinking,
		ToolCalls:  toolCalls,
		StopReason: openaiStopReason(finalResponse, len(toolCalls) > 0, hasRefusal),
		Usage: TurnUsage{
			InputTokens:      promptTokens,
			OutputTokens:     outputTokens,
			CachedReadTokens: cachedTokens,
			ThinkingTokens:   reasoningTokens,
		},
	}, nil
}

func openaiStopReason(r *responses.Response, hasToolCalls, hasRefusal bool) StopReason {
	if hasToolCalls {
		return StopReasonToolUse
	}
	if hasRefusal {
		return StopReasonRefusal
	}
	switch r.IncompleteDetails.Reason {
	case "max_output_tokens":
		return StopReasonMaxTokens
	case "content_filter":
		return StopReasonRefusal
	}
	return StopReasonEndTurn
}
