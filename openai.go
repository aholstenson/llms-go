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
	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
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

// openaiStreamError is an error that OpenAI sent inside a stream that had
// already opened with status 200. It comes from an "error" event, or from
// the SDK when an event has an "error" field.
type openaiStreamError struct {
	Type    string
	Code    string
	Message string
	// err is the SDK error, if the SDK made one.
	err error
}

func (e *openaiStreamError) Error() string {
	if e.err != nil {
		return e.err.Error()
	}
	return fmt.Sprintf("received error while streaming: %s: %s", e.Code, e.Message)
}

func (e *openaiStreamError) Unwrap() error {
	return e.err
}

// unavailable reports whether the error is a rate limit or an overload.
func (e *openaiStreamError) unavailable() bool {
	return openaiUnavailableCode(e.Code) || e.Type == "service_unavailable_error"
}

// openaiUnavailableCode reports whether an OpenAI error code means that
// the request can succeed if it is sent again later.
func openaiUnavailableCode(code string) bool {
	return code == "rate_limit_exceeded" || code == "server_is_overloaded"
}

// openaiPolicyCode reports whether an OpenAI error code means that a safety
// system blocked the request or the response.
func openaiPolicyCode(code string) bool {
	switch code {
	case "cyber_policy", "invalid_prompt", "content_policy_violation", "content_filter":
		return true
	}
	return false
}

// openaiContentPolicy reports whether err is a content policy block from
// OpenAI. It finds the block in an HTTP error and in an error that OpenAI
// sent inside an open stream. status is 0 for an error inside a stream.
func openaiContentPolicy(err error) (status int, code, message string, ok bool) {
	var oe *openai.Error
	if errors.As(err, &oe) {
		return oe.StatusCode, oe.Code, oe.Message, openaiPolicyCode(oe.Code)
	}
	var se *openaiStreamError
	if errors.As(err, &se) {
		return 0, se.Code, se.Message, openaiPolicyCode(se.Code)
	}
	return 0, "", "", false
}

// openaiStreamDecodeError is a stream event that the SDK could not decode
// because its data is not complete JSON, for example when a connection
// cuts an event. The fault is in the transport and not in the request, so
// another attempt can succeed. A frame with no "data:" line, such as a
// keep-alive comment, does not cause this error: the SDK skips it.
type openaiStreamDecodeError struct {
	err error
}

func (e *openaiStreamDecodeError) Error() string {
	return "could not decode stream event: " + e.err.Error()
}

func (e *openaiStreamDecodeError) Unwrap() error {
	return e.err
}

// asOpenAIStreamError changes an SDK stream error into an
// *openaiStreamError when the SDK made it from an "error" field in the
// stream. The SDK keeps the event that had the field, so the code and the
// message are read from it. It changes a JSON syntax error into an
// *openaiStreamDecodeError. Other errors are returned unchanged.
func asOpenAIStreamError(err error) error {
	if err == nil {
		return nil
	}
	// Only a syntax error is a damaged event. A type error means that the
	// event does not match the SDK types, and it would occur again.
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		return &openaiStreamDecodeError{err: err}
	}
	var streamErr *ssestream.StreamError
	if !errors.As(err, &streamErr) {
		return err
	}
	var event struct {
		Error struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(streamErr.Event.Data, &event) != nil {
		return err
	}
	f := event.Error
	return &openaiStreamError{Type: f.Type, Code: f.Code, Message: f.Message, err: err}
}

// openaiUnavailable reports whether err means that OpenAI is rate limited or
// overloaded, or that it sent a stream event that could not be decoded.
// status is the HTTP status, or 0 when the error came inside an open stream.
func openaiUnavailable(err error) (status int, ok bool) {
	var oe *openai.Error
	if errors.As(err, &oe) {
		return oe.StatusCode, isUnavailableStatusCode(oe.StatusCode)
	}
	var se *openaiStreamError
	if errors.As(err, &se) {
		return 0, se.unavailable()
	}
	var de *openaiStreamDecodeError
	if errors.As(err, &de) {
		return 0, true
	}
	return 0, false
}

type openaiModel struct {
	logger            *slog.Logger
	metrics           *Metrics
	recorder          MetricsRecorder
	client            openai.Client
	statsModel        string
	model             string
	info              modelInfoRef
	subParserRegistry map[string]SubParserConfig
}

// newOpenAIModel creates a new OpenAI model using the official OpenAI Go SDK
// against the Responses API. creds is consulted on every request so rotating
// credentials take effect without rebuilding the model. info supplies model
// metadata used to gate request parameters; an unknown model is treated
// permissively. Optional SDK request options can be passed for customization
// (e.g. option.WithBaseURL for testing).
func newOpenAIModel(logger *slog.Logger, metrics *Metrics, creds CredentialSource, model string, registry map[string]SubParserConfig, info modelInfoRef, opts ...option.RequestOption) Model {
	// The placeholder key satisfies the SDK; the transport replaces the
	// Authorization header per request.
	httpClient := newAuthHTTPClient(nil, creds, "openai", applyBearerCredential)
	allOpts := append([]option.RequestOption{
		option.WithAPIKey(credentialPlaceholder),
		option.WithHTTPClient(httpClient),
	}, opts...)
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

// gateInfo returns the qualified model name and the ModelInfo that gates
// requests. It makes no network call.
func (m *openaiModel) gateInfo() (string, ModelInfo) {
	return m.statsModel, m.info.get()
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

	info := m.info.get()
	if err := checkRequestCapabilities(m.statsModel, info, opts); err != nil {
		return nil, err
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
		// and disable server-side storage. Encrypted reasoning is requested
		// below only when the model actually reasons — non-reasoning models
		// reject the include with a 400.
		Store: openai.Bool(false),
	}

	if opts.SystemPrompt != "" {
		params.Instructions = openai.String(opts.SystemPrompt)
	}

	if opts.Temperature != 0 && info.allowsTemperature() {
		params.Temperature = openai.Float(opts.Temperature)
	}

	maxOutput := info.resolveMaxOutputTokens(opts.MaxOutputTokens, 0)

	// Resolve reasoning. OpenAI has no token budget — effort drives. A
	// WithMaxThinkingTokens value cannot control reasoning (warned in
	// resolveReasoningRoute) but still reserves output headroom, since OpenAI
	// counts reasoning tokens against max_output_tokens. Known reasoning
	// models always ask for encrypted reasoning so reasoning items replay.
	encryptedReasoning := []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent}
	switch route := resolveReasoningRoute(opts, info, false, m.logger); route.Kind {
	case reasoningKindEffort:
		effort := route.Effort
		if effort == "" {
			// Some models default to no reasoning, so "on" needs a tier.
			effort = clampEffort(EffortMedium, info, m.logger)
		}
		params.Reasoning = shared.ReasoningParam{Effort: openaiReasoningEffort(effort)}
		params.Include = encryptedReasoning
	case reasoningKindDisable:
		// Models that list a "none" effort accept it to turn reasoning off.
		params.Reasoning = shared.ReasoningParam{Effort: shared.ReasoningEffort(EffortNone)}
	case reasoningKindDefault, reasoningKindMandatory:
		// No reasoning param, so the model's default applies.
		if info.Caps.Reasoning {
			params.Include = encryptedReasoning
		}
	case reasoningKindSkip, reasoningKindBudget:
		// Non-reasoning models get no reasoning param and no encrypted-content
		// include (would 400). Budget is not applicable.
	}

	if opts.MaxThinkingTokens > 0 && maxOutput > 0 {
		maxOutput += opts.MaxThinkingTokens
	}

	if maxOutput > 0 {
		if clamped, didClamp := info.clampMaxOutputTokens(maxOutput); didClamp {
			m.logger.Warn("Clamping max tokens to model output limit",
				slog.Int("requested", maxOutput), slog.Int("limit", clamped))
			maxOutput = clamped
		}
		params.MaxOutputTokens = openai.Int(int64(maxOutput))
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
		info:       info,
		opts:       opts,
		params:     params,
		inputItems: inputItems,
		jsParser:   jsParser,
	}

	return newSession(turn, newTracker(opts), toolMap, opts, m.logger), nil
}

// extractUsage converts an OpenAI Response's usage block into TurnUsage.
// Returns nil when no tokens have been reported (e.g. the stream errored
// before a Completed/Failed/Incomplete event surfaced a final response).
func (m *openaiModel) extractUsage(resp *responses.Response) *TurnUsage {
	if resp == nil {
		return nil
	}

	u := resp.Usage
	if u.InputTokens == 0 && u.OutputTokens == 0 &&
		u.InputTokensDetails.CachedTokens == 0 && u.OutputTokensDetails.ReasoningTokens == 0 {
		return nil
	}

	cachedTokens := u.InputTokensDetails.CachedTokens
	return &TurnUsage{
		InputTokens:      uncachedInputTokens(u.InputTokens, cachedTokens),
		OutputTokens:     u.OutputTokens,
		CachedReadTokens: cachedTokens,
		ThinkingTokens:   u.OutputTokensDetails.ReasoningTokens,
	}
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
					var item responses.ResponseInputItemUnionParam
					var err error
					if trp.Error != "" {
						item, err = openaiFunctionCallOutput(trp.ID, trp.Error, nil)
					} else {
						item, err = openaiFunctionCallOutput(trp.ID, trp.Text, trp.Attachments)
					}
					if err != nil {
						return nil, err
					}
					result = append(result, item)
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

// convertUserParts converts the parts of a user message. Text files become
// input_text parts and other files become input_file parts with Name as the
// file name (see expandFileParts).
func convertUserParts(parts []MessagePart) (responses.ResponseInputMessageContentListParam, error) {
	parts = expandFileParts(parts, true)
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
			if partModality(content) == "image" {
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
						Filename: openai.String(partFilename(content)),
					},
				})
			}
		default:
			return nil, fmt.Errorf("unsupported part type for OpenAI: %T", part)
		}
	}
	return out, nil
}

// openaiFunctionCallOutput builds the function_call_output item for one tool
// result. A result without attachments is sent as a plain string. A result
// with attachments is sent as a list of content items: the text first, then
// each attachment. The attachments stay in the tool result, so the model
// connects them to the call that made them.
func openaiFunctionCallOutput(callID, text string, attachments []MessagePart) (responses.ResponseInputItemUnionParam, error) {
	var output responses.ResponseInputItemFunctionCallOutputOutputUnionParam
	if len(attachments) == 0 {
		output.OfString = openai.String(text)
	} else {
		attachments = expandFileParts(attachments, true)
		items := make(responses.ResponseFunctionCallOutputItemListParam, 0, len(attachments)+1)
		if text != "" {
			items = append(items, responses.ResponseFunctionCallOutputItemUnionParam{
				OfInputText: &responses.ResponseInputTextContentParam{Text: text},
			})
		}
		for _, part := range attachments {
			item, err := openaiFunctionCallOutputItem(part)
			if err != nil {
				return responses.ResponseInputItemUnionParam{}, err
			}
			items = append(items, item)
		}
		output.OfResponseFunctionCallOutputItemArray = items
	}
	return responses.ResponseInputItemUnionParam{
		OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
			CallID: openai.String(callID),
			Output: output,
		},
	}, nil
}

// openaiFunctionCallOutputItem converts one tool-result attachment to a
// content item of a function_call_output. Images become input_image items.
// Other binary data becomes an input_file item.
func openaiFunctionCallOutputItem(part MessagePart) (responses.ResponseFunctionCallOutputItemUnionParam, error) {
	switch content := part.(type) {
	case *TextPart:
		return responses.ResponseFunctionCallOutputItemUnionParam{
			OfInputText: &responses.ResponseInputTextContentParam{Text: content.Text},
		}, nil
	case *ImagePart:
		return responses.ResponseFunctionCallOutputItemUnionParam{
			OfInputImage: &responses.ResponseInputImageContentParam{
				ImageURL: openai.String(content.URL),
				Detail:   responses.ResponseInputImageContentDetailAuto,
			},
		}, nil
	case *BinaryPart:
		dataURL := fmt.Sprintf("data:%s;base64,%s", content.MediaType, base64.StdEncoding.EncodeToString(content.Data))
		if partModality(content) == "image" {
			return responses.ResponseFunctionCallOutputItemUnionParam{
				OfInputImage: &responses.ResponseInputImageContentParam{
					ImageURL: openai.String(dataURL),
					Detail:   responses.ResponseInputImageContentDetailAuto,
				},
			}, nil
		}
		return responses.ResponseFunctionCallOutputItemUnionParam{
			OfInputFile: &responses.ResponseInputFileContentParam{
				FileData: openai.String(dataURL),
				Filename: openai.String(partFilename(content)),
			},
		}, nil
	default:
		return responses.ResponseFunctionCallOutputItemUnionParam{}, fmt.Errorf("unsupported tool-result attachment type for OpenAI: %T", part)
	}
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
	info ModelInfo
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
		var item responses.ResponseInputItemUnionParam
		var err error
		if o.Error != nil {
			item, err = openaiFunctionCallOutput(o.ID, o.ModelError(), nil)
		} else {
			item, err = openaiFunctionCallOutput(o.ID, o.Text, o.Attachments)
		}
		if err != nil {
			return err
		}
		t.inputItems = append(t.inputItems, item)
	}
	return nil
}

// openaiStreamResult is what one drained stream attempt produced. The two
// error flags are terminal for the turn — unlike a transport failure they say
// nothing about whether another attempt would fare better — so they travel
// beside the transport error rather than as one.
type openaiStreamResult struct {
	// response is the final response event, or nil when none arrived.
	response *responses.Response
	// handling is true when a streaming callback or an in-band error event
	// failed, so the rest of the stream was ignored.
	handling bool
	// parseErr is a structured-output parse or dispatch failure.
	parseErr error
}

// runStream sends the request and reads the response to the end, retrying the
// whole attempt while nothing has reached the caller. Opening and draining sit
// inside one retry loop because the failure this guards against — a rate
// limit, an overload, or a stalled connection — is just as likely after the
// server accepted the request as before it, and a stream that dies before its
// first event is as safe to replay as one that never opened. This includes a
// rate limit or an overload that OpenAI sends inside a stream that opened with
// status 200, and a stream event that the SDK could not decode.
//
// The moment an event reaches a streaming callback the attempt becomes final:
// retrying would replay tokens the user has already seen. It returns how many
// attempts were made and, on failure, the last attempt's partial result so
// usage reported before the failure is not lost.
func (t *openaiTurn) runStream(
	ctx context.Context,
	start time.Time,
	hasRecordedFirstToken *bool,
	streamingEmitted *bool,
) (openaiStreamResult, int, error) {
	m := t.m

	classify := func(err error) (bool, int, time.Duration, bool) {
		status, unavailable := openaiUnavailable(err)

		if *streamingEmitted {
			// Events already reached the caller, so this attempt stands.
			return false, status, 0, false
		}

		// A stall or a stream that ended early is a transient failure like
		// any other, except the provider never got as far as telling us so.
		if isStallError(err) || isStreamEnded(err) {
			return true, 0, 0, false
		}

		if !unavailable {
			return false, status, 0, false
		}
		ra, hasRA := extractRetryAfter("openai", err, nil)
		return true, status, ra, hasRA
	}

	attempts := 0
	var partial openaiStreamResult
	result, err := retryLoop(ctx, t.opts, string(GenAISystemOpenAI), m.model, classify,
		func(ctx context.Context) (openaiStreamResult, error) {
			attempts++
			// Each attempt receives the whole response from the start, so
			// structured output state from an earlier attempt must not
			// linger. This includes an attempt from an earlier call to Next
			// that failed after output: the Session lets the caller try a
			// turn again after a content policy block.
			if t.jsParser != nil {
				t.jsParser.Reset()
			}
			t.structuredContentBuilder.Reset()
			partial = openaiStreamResult{}

			// llms-go owns the retry loop so WithRetryBackoff and
			// WithRetryNotify apply to OpenAI like to every other provider.
			stream := m.client.Responses.NewStreaming(ctx, t.params, option.WithMaxRetries(0))
			// The SDK completes the request before it returns the stream, so
			// an error here means the response never started.
			if err := stream.Err(); err != nil {
				if cerr := stream.Close(); cerr != nil {
					m.logger.Warn("Error closing OpenAI stream", slog.Any("error", cerr))
				}
				return openaiStreamResult{}, err
			}

			res, streamErr := t.drainStream(ctx, stream, start, hasRecordedFirstToken, streamingEmitted)
			partial = res
			// A parse or handling failure is the turn's answer, whatever the
			// transport went on to do, and retrying it would fail the same
			// way. Report it instead of the transport error.
			if streamErr != nil && res.parseErr == nil && !res.handling {
				return openaiStreamResult{}, streamErr
			}
			return res, nil
		})
	if err != nil {
		return partial, attempts, err
	}
	return result, attempts, nil
}

// drainStream reads an open stream to the end, dispatching events to the
// caller's callbacks. The returned error is the transport error, if any; the
// result carries the failures that belong to this turn rather than to the
// connection.
func (t *openaiTurn) drainStream(
	ctx context.Context,
	stream *ssestream.Stream[responses.ResponseStreamEventUnion],
	start time.Time,
	hasRecordedFirstToken *bool,
	streamingEmitted *bool,
) (openaiStreamResult, error) {
	m := t.m

	var out openaiStreamResult
	var structuredStreamErr error
	// inBandErr is a rate limit, an overload or a content policy block that
	// OpenAI sent inside the stream. Unlike other error events, it is
	// reported with its code: another attempt can succeed after a rate limit
	// or an overload, and the caller decides what to do after a block.
	var inBandErr error
	// ended describes the stream for the error when it ends before its
	// final event.
	var ended streamEndedError

events:
	for stream.Next() {
		event := stream.Current()
		ended.events++
		ended.lastEvent = event.Type

		switch ev := event.AsAny().(type) {
		case responses.ResponseCreatedEvent:
			ended.responseID = ev.Response.ID

		case responses.ResponseInProgressEvent:
			if ended.responseID == "" {
				ended.responseID = ev.Response.ID
			}

		case responses.ResponseTextDeltaEvent:
			if out.handling {
				continue
			}
			if !*hasRecordedFirstToken {
				m.metrics.RecordTimeToFirstToken(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start))
				*hasRecordedFirstToken = true
			}

			delta := ev.Delta
			if structuredStreamErr == nil && t.jsParser != nil && t.opts.StructuredStreamingFunc != nil {
				t.structuredContentBuilder.WriteString(delta)
				parsed, err := t.jsParser.Feed(delta)
				if err != nil {
					structuredStreamErr = fmt.Errorf("%w: %w", ErrStructuredStreamParse, err)
				} else {
					for _, e := range parsed {
						*streamingEmitted = true
						if err := t.opts.StructuredStreamingFunc(ctx, e); err != nil {
							structuredStreamErr = err
							break
						}
					}
				}
			}

			if t.opts.StreamingFunc != nil && !out.handling {
				*streamingEmitted = true
				if err := t.opts.StreamingFunc(ctx, StreamingEventTextChunk{Text: delta}); err != nil {
					m.logger.Error("Error handling OpenAI response", slog.Any("error", err))
					out.handling = true
				}
			}

		case responses.ResponseReasoningSummaryTextDeltaEvent:
			if out.handling || t.opts.StreamingFunc == nil {
				continue
			}
			*streamingEmitted = true
			if err := t.opts.StreamingFunc(ctx, StreamingEventThinking{Text: ev.Delta}); err != nil {
				m.logger.Error("Error handling OpenAI thinking", slog.Any("error", err))
				out.handling = true
			}

		case responses.ResponseCompletedEvent:
			r := ev.Response
			out.response = &r

		case responses.ResponseFailedEvent:
			r := ev.Response
			out.response = &r

		case responses.ResponseIncompleteEvent:
			r := ev.Response
			out.response = &r

		case responses.ResponseErrorEvent:
			if se := (&openaiStreamError{Code: ev.Code, Message: ev.Message}); se.unavailable() || openaiPolicyCode(ev.Code) {
				inBandErr = se
				break events
			}
			m.logger.Error("OpenAI stream error",
				slog.String("code", ev.Code), slog.String("message", ev.Message))
			out.handling = true
		}
	}

	if err := stream.Close(); err != nil {
		m.logger.Warn("Error closing OpenAI stream", slog.Any("error", err))
	}

	// The stream ended without an error but before its final event, so it
	// was cut. A failed streaming callback is the turn's answer, so it is
	// reported instead.
	cutOff := stream.Err() == nil && inBandErr == nil && out.response == nil && !out.handling
	if cutOff {
		out.parseErr = structuredStreamErr
		if err := ctx.Err(); err != nil {
			return out, err
		}
		// The structured output is not complete, so it is not flushed: the
		// parser error would hide the transient failure.
		return out, &ended
	}

	if structuredStreamErr == nil && t.jsParser != nil && t.opts.StructuredStreamingFunc != nil {
		parsed, err := t.jsParser.Flush()
		if err != nil {
			structuredStreamErr = fmt.Errorf("%w: %w", ErrStructuredStreamParse, err)
		} else {
			for _, e := range parsed {
				*streamingEmitted = true
				if err := t.opts.StructuredStreamingFunc(ctx, e); err != nil {
					structuredStreamErr = err
					break
				}
			}
		}
	}

	out.parseErr = structuredStreamErr
	if err := asOpenAIStreamError(stream.Err()); err != nil {
		return out, err
	}
	if inBandErr != nil {
		return out, inBandErr
	}
	// A response that failed with a rate limit or an overload can succeed on
	// another attempt, so it is a transport error and not the turn's answer.
	// A response that a content policy blocked goes the same way, so that it
	// is reported with its code. This includes a response that a content
	// filter stopped before it was complete.
	if r := out.response; r != nil {
		if r.Status == responses.ResponseStatusFailed {
			if code := string(r.Error.Code); openaiUnavailableCode(code) || openaiPolicyCode(code) {
				return out, &openaiStreamError{Code: code, Message: r.Error.Message}
			}
		}
		if r.Status == responses.ResponseStatusIncomplete && r.IncompleteDetails.Reason == "content_filter" {
			return out, &openaiStreamError{Code: "content_filter", Message: "the response was stopped by a content filter"}
		}
	}
	return out, nil
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

	streamingEmitted := false

	// The stall budgets ride on the context so the transport can fail a
	// silent request instead of hanging on it.
	result, attempts, err := t.runStream(withLiveness(ctx, m.model, t.opts), start, &hasRecordedFirstToken, &streamingEmitted)
	if err != nil {
		recordFailure(ctx, m.recorder, m.statsModel, collector)

		// If a ResponseIncomplete/Failed event landed before the transport
		// errored, the partial result already carries usage — preserve it so
		// cost accounting is not lost.
		partialUsage := m.extractUsage(result.response)
		if partialUsage != nil {
			collector.Counter("input_tokens").Add(int(partialUsage.InputTokens))
			collector.Counter("output_tokens").Add(int(partialUsage.OutputTokens))
			collector.Counter("cached_read_tokens").Add(int(partialUsage.CachedReadTokens))
			recordTierUsage(collector, t.info, *partialUsage)
			m.metrics.RecordCall(ctx,
				GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model),
				partialUsage.InputTokens,
				partialUsage.OutputTokens,
				partialUsage.CachedReadTokens,
				0,
			)
		}

		// A failure the retry loop kept retrying is already an
		// UnavailableError with the exact attempt count on it.
		var ue *UnavailableError
		if errors.As(err, &ue) {
			ue.PartialOutput = streamingEmitted
			ue.PartialUsage = partialUsage
			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), transientErrorType(err))
			if streamingEmitted {
				return TurnOutput{}, errors.Join(ue, ErrStreamingPartialOutput)
			}
			return TurnOutput{}, ue
		}

		// A transient failure the retry loop declined to retry, because
		// output had already reached the caller. Report it in the same shape
		// so callers see one error type for "try again later".
		status, unavailable := openaiUnavailable(err)
		if isStallError(err) || isStreamEnded(err) || unavailable {
			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), transientErrorType(err))
			var ra time.Duration
			var hasRA bool
			if unavailable {
				ra, hasRA = extractRetryAfter("openai", err, nil)
				if hasRA && t.opts.RetryAfterCap > 0 && ra > t.opts.RetryAfterCap {
					ra = t.opts.RetryAfterCap
				}
			} else {
				status = 0
			}
			ue := &UnavailableError{
				Provider:      string(GenAISystemOpenAI),
				Model:         m.model,
				StatusCode:    status,
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

		if status, code, message, ok := openaiContentPolicy(err); ok {
			m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeContentPolicy)
			return TurnOutput{}, contentPolicyFailure(&ContentPolicyError{
				Provider:      string(GenAISystemOpenAI),
				Model:         m.model,
				StatusCode:    status,
				Code:          code,
				Message:       message,
				PartialOutput: streamingEmitted,
				PartialUsage:  partialUsage,
				Cause:         err,
			})
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, fmt.Errorf("got error from OpenAI (model %s) while streaming: %w", m.model, err)
	}

	finalResponse := result.response

	if result.parseErr != nil {
		recordFailure(ctx, m.recorder, m.statsModel, collector)
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeStreamProcessing)
		if streamingEmitted {
			return TurnOutput{}, errors.Join(ErrStreamingPartialOutput, result.parseErr)
		}
		return TurnOutput{}, result.parseErr
	}

	if result.handling {
		recordFailure(ctx, m.recorder, m.statsModel, collector)
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeStreamProcessing)
		return TurnOutput{}, fmt.Errorf("stream handling failed for OpenAI (model %s)", m.model)
	}

	if finalResponse == nil {
		// drainStream reports a stream without a final event as a transport
		// error, so this is a defensive check only.
		recordFailure(ctx, m.recorder, m.statsModel, collector)
		m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeEmptyResponse)
		return TurnOutput{}, fmt.Errorf("no completed response from OpenAI (model %s)", m.model)
	}

	if finalResponse.Status == responses.ResponseStatusFailed {
		recordFailure(ctx, m.recorder, m.statsModel, collector)

		// A ResponseStatusFailed event carries final usage (the model
		// generated tokens before the server-side failure), so preserve it.
		partialUsage := m.extractUsage(finalResponse)
		if partialUsage != nil {
			collector.Counter("input_tokens").Add(int(partialUsage.InputTokens))
			collector.Counter("output_tokens").Add(int(partialUsage.OutputTokens))
			collector.Counter("cached_read_tokens").Add(int(partialUsage.CachedReadTokens))
			recordTierUsage(collector, t.info, *partialUsage)
			m.metrics.RecordCall(ctx,
				GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model),
				partialUsage.InputTokens,
				partialUsage.OutputTokens,
				partialUsage.CachedReadTokens,
				0,
			)
		}

		respErr := finalResponse.Error
		// A failure with a rate limit, an overload or a content policy block
		// does not come here: drainStream reports it as a transport error,
		// so it is retried or reported as a ContentPolicyError.
		err := fmt.Errorf("OpenAI response failed (model %s): %s: %s", m.model, respErr.Code, respErr.Message)
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
	// OpenAI's InputTokens is the total prompt size and *includes* the cached
	// tokens reported in InputTokensDetails, so subtract to get the disjoint,
	// uncached input the rest of the stack expects (see uncachedInputTokens).
	cachedTokens := usage.InputTokensDetails.CachedTokens
	inputTokens := uncachedInputTokens(usage.InputTokens, cachedTokens)
	outputTokens := usage.OutputTokens
	reasoningTokens := usage.OutputTokensDetails.ReasoningTokens

	collector.Counter("input_tokens").Add(int(inputTokens))
	collector.Counter("output_tokens").Add(int(outputTokens))
	collector.Counter("cached_read_tokens").Add(int(cachedTokens))
	recordTierUsage(collector, t.info, TurnUsage{
		InputTokens:      inputTokens,
		OutputTokens:     outputTokens,
		CachedReadTokens: cachedTokens,
	})

	m.metrics.RecordCall(
		ctx,
		GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model),
		inputTokens, outputTokens, cachedTokens, 0,
	)

	textContent := textBuilder.String()
	content := textContent
	if t.jsParser != nil {
		content = t.structuredContentBuilder.String()
	}
	t.finalText = content

	recordSuccess(ctx, m.recorder, m.statsModel, collector)
	m.metrics.RecordCallDuration(ctx, GenAISystemOpenAI, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

	return TurnOutput{
		Text:       textContent,
		Thinking:   thinking,
		ToolCalls:  toolCalls,
		StopReason: openaiStopReason(finalResponse, len(toolCalls) > 0, hasRefusal),
		Usage: TurnUsage{
			InputTokens:      inputTokens,
			OutputTokens:     outputTokens,
			CachedReadTokens: cachedTokens,
			ThinkingTokens:   reasoningTokens,
		},
	}, nil
}

// openaiReasoningEffort maps a portable Effort to the OpenAI reasoning-effort
// enum. EffortNone never reaches here (it resolves to the off path, which emits
// the raw "none" string); unrecognized values fall back to medium.
func openaiReasoningEffort(e Effort) shared.ReasoningEffort {
	switch e {
	case EffortMinimal:
		return shared.ReasoningEffortMinimal
	case EffortXHigh, EffortMax:
		// The SDK has no constants for these tiers yet; the API accepts them
		// on models that list them.
		return shared.ReasoningEffort(e)
	case EffortLow:
		return shared.ReasoningEffortLow
	case EffortMedium:
		return shared.ReasoningEffortMedium
	case EffortHigh:
		return shared.ReasoningEffortHigh
	default:
		return shared.ReasoningEffortMedium
	}
}

func openaiStopReason(r *responses.Response, hasToolCalls, hasRefusal bool) StopReason {
	if hasToolCalls {
		return StopReasonToolUse
	}
	if hasRefusal {
		return StopReasonRefusal
	}
	// An incomplete response with the reason "content_filter" does not come
	// here: drainStream reports it as a ContentPolicyError.
	if r.IncompleteDetails.Reason == "max_output_tokens" {
		return StopReasonMaxTokens
	}
	return StopReasonEndTurn
}

func (m *openaiModel) setRecorder(recorder MetricsRecorder) {
	m.recorder = recorder
}
