package llms

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/aholstenson/llms-go/jsonstream"
	"github.com/invopop/jsonschema"
	"google.golang.org/genai"
)

// lastCapturedHeaders returns the most recent response headers seen by the
// transport, or nil if no response has been observed yet.
func (m *googleModel) lastCapturedHeaders() http.Header {
	if m.headerCapture == nil {
		return nil
	}
	return m.headerCapture.LastHeaders()
}

// googleErrorDetails inspects an error from the genai SDK and reports
// (retryAfter, retryable, statusCode). retryAfter is the duration parsed
// from the SDK error (Details), if any. retryable is true when the status
// is one of the unavailable codes (429/503/529). statusCode is 0 when no
// APIError is found in the chain.
func googleErrorDetails(err error) (time.Duration, bool, int) {
	if err == nil {
		return 0, false, 0
	}
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return 0, false, 0
	}
	d, _ := googleRetryDelayFromError(err)
	return d, isUnavailableStatusCode(apiErr.Code), apiErr.Code
}

// googleRetryDelayFromError walks genai.APIError.Details looking for a
// google.rpc.RetryInfo entry and decodes its retryDelay field.
func googleRetryDelayFromError(err error) (time.Duration, bool) {
	if err == nil {
		return 0, false
	}
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return 0, false
	}
	for _, d := range apiErr.Details {
		t, _ := d["@type"].(string)
		if !strings.HasSuffix(t, "google.rpc.RetryInfo") {
			continue
		}
		delay, ok := d["retryDelay"].(string)
		if !ok {
			continue
		}
		if dur, perr := time.ParseDuration(delay); perr == nil {
			return dur, true
		}
	}
	return 0, false
}

type googleModel struct {
	logger            *slog.Logger
	metrics           *Metrics
	client            *genai.Client
	statsModel        string
	model             string
	info              ModelInfo
	subParserRegistry map[string]SubParserConfig
	headerCapture     *headerCapturingTransport
}

// NewGoogleModel creates a new Google Gemini model. info carries embedded
// model metadata used to gate request parameters; the zero value is treated
// permissively.
func NewGoogleModel(logger *slog.Logger, metrics *Metrics, apiKey string, model string, registry map[string]SubParserConfig, info ModelInfo) (Model, error) {
	ctx := context.Background()
	transport := newHeaderCapturingTransport(http.DefaultTransport)
	httpClient := &http.Client{Transport: transport}
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:     apiKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: httpClient,
	})
	if err != nil {
		return nil, err
	}
	return &googleModel{
		logger:            logger.With(slog.String("provider", "google")),
		metrics:           metrics,
		client:            client,
		statsModel:        "google/" + model,
		model:             model,
		info:              info,
		subParserRegistry: registry,
		headerCapture:     transport,
	}, nil
}

func (m *googleModel) GenerateContent(ctx context.Context, options ...GenerateOption) (Result, error) {
	s, err := m.newSession(options...)
	if err != nil {
		return nil, err
	}
	return runSession(ctx, s)
}

func (m *googleModel) newSession(options ...GenerateOption) (*Session, error) {
	opts := resolveGenerateContentOptions(m.subParserRegistry, options...)

	// Gate request parameters against the model's known capabilities.
	if len(opts.Tools) > 0 && !m.info.allowsToolCall() {
		return nil, fmt.Errorf("model %s does not support tool calling", m.statsModel)
	}
	if modality := firstUnsupportedModality(opts.Messages, m.info); modality != "" {
		return nil, fmt.Errorf("model %s does not support %s input", m.statsModel, modality)
	}
	if clamped, didClamp := m.info.clampMaxTokens(opts.MaxTokens); didClamp {
		m.logger.Warn("Clamping max tokens to model output limit",
			slog.Int("requested", opts.MaxTokens), slog.Int("limit", clamped))
		opts.MaxTokens = clamped
	}

	messages, err := m.convertMessages(opts.Messages)
	if err != nil {
		return nil, err
	}

	config := &genai.GenerateContentConfig{}

	if opts.SystemPrompt != "" {
		config.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{
				{
					Text: opts.SystemPrompt,
				},
			},
		}
	}

	if opts.Temperature != 0 && m.info.allowsTemperature() {
		t := float32(opts.Temperature)
		config.Temperature = &t
	}

	if v := m.info.resolveMaxTokens(opts.MaxTokens, 0); v > 0 {
		config.MaxOutputTokens = int32(v) //nolint:gosec
	}

	if opts.MaxThinkingTokens != 0 && m.info.allowsReasoning() {
		thinkingBudget := int32(opts.MaxThinkingTokens) //nolint:gosec
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget:  &thinkingBudget,
			IncludeThoughts: true,
		}

		// Gemini seems to include thinking tokens in the max output tokens
		config.MaxOutputTokens += int32(opts.MaxThinkingTokens) //nolint:gosec
	} else {
		// Some Gemini models enable thinking by default, which silently
		// includes thinking tokens in the max output tokens.
		zero := int32(0)
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget: &zero,
		}
	}

	var toolMap map[string]ToolDef
	if len(opts.Tools) > 0 {
		var tools []*genai.Tool
		tools, toolMap = m.convertTools(opts.Tools)
		config.Tools = tools
	}

	// Configure structured output if ResponseSchema is set
	if opts.ResponseSchema != nil {
		config.ResponseMIMEType = "application/json"
		config.ResponseSchema = ConvertToGenaiSchema(opts.ResponseSchema.Schema.(*jsonschema.Schema))
	}

	var jsParser *jsonstream.Parser
	if opts.StructuredStreamingFunc != nil && opts.StructuredStreamingSchema != nil {
		jsParser = jsonstream.New(opts.StructuredStreamingSchema)
	}

	turn := &googleTurn{
		m:        m,
		opts:     opts,
		config:   config,
		messages: messages,
		jsParser: jsParser,
	}

	return newSession(turn, newTracker(opts), toolMap, opts, m.logger), nil
}

func (m *googleModel) handleStreaming(
	ctx context.Context,
	config *genai.GenerateContentConfig,
	messages []*genai.Content,
	streamingFunc StreamingFunc,
	structuredStreamingFunc StructuredStreamingFunc,
	jsParser *jsonstream.Parser,
	structuredContentBuilder *strings.Builder,
	start time.Time,
	hasRecordedFirstToken *bool,
	streamingEmitted *bool,
) (*genai.GenerateContentResponse, error) {
	var lastResponse *genai.GenerateContentResponse

	for resp, err := range m.client.Models.GenerateContentStream(ctx, m.model, messages, config) {
		if err != nil {
			return nil, err
		}

		if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
			for _, p := range resp.Candidates[0].Content.Parts {
				if p.Text == "" {
					continue
				}

				if p.Thought {
					if streamingFunc != nil {
						if streamingEmitted != nil {
							*streamingEmitted = true
						}
						if err := streamingFunc(ctx, StreamingEventThinking{Text: p.Text}); err != nil {
							return nil, err
						}
					}
				} else {
					if !*hasRecordedFirstToken {
						m.metrics.RecordTimeToFirstToken(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start))
						*hasRecordedFirstToken = true
					}

					// Handle structured streaming
					if structuredStreamingFunc != nil && jsParser != nil {
						structuredContentBuilder.WriteString(p.Text)
						events, err := jsParser.Feed(p.Text)
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
						if streamingEmitted != nil {
							*streamingEmitted = true
						}
						if err := streamingFunc(ctx, StreamingEventTextChunk{Text: p.Text}); err != nil {
							return nil, err
						}
					}
				}
			}
		}

		// Update lastResponse to keep track of the full context (usage, function calls)
		// Note: GenerateContentStream responses are separate chunks.
		// However, the GenAI SDK's GenerateContentResponse objects from a stream
		// are designed to be used as chunks. To get the final text or function calls,
		// we may need to merge them.
		if lastResponse == nil {
			lastResponse = resp
			// Clean the first candidate's parts of any empty parts
			if len(lastResponse.Candidates) > 0 && lastResponse.Candidates[0].Content != nil {
				var cleanParts []*genai.Part
				for _, p := range lastResponse.Candidates[0].Content.Parts {
					if !m.isPartEmpty(p) {
						cleanParts = append(cleanParts, p)
					}
				}
				lastResponse.Candidates[0].Content.Parts = cleanParts
			}
		} else {
			// Merge candidates from chunks
			if len(resp.Candidates) > 0 && len(lastResponse.Candidates) > 0 {
				if resp.Candidates[0].Content != nil && lastResponse.Candidates[0].Content != nil {
					for _, p := range resp.Candidates[0].Content.Parts {
						if m.isPartEmpty(p) {
							continue
						}
						lastResponse.Candidates[0].Content.Parts = append(lastResponse.Candidates[0].Content.Parts, p)
					}
				}

				finishReason := resp.Candidates[0].FinishReason
				if finishReason != "" && finishReason != genai.FinishReasonUnspecified {
					lastResponse.Candidates[0].FinishReason = finishReason
				}
			}

			// Merge usage metadata
			if resp.UsageMetadata != nil {
				lastResponse.UsageMetadata = resp.UsageMetadata
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

	if lastResponse == nil {
		return nil, errors.New("no response from Google")
	}

	return lastResponse, nil
}

func (m *googleModel) extractText(resp *genai.GenerateContentResponse) string {
	if resp == nil || len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return ""
	}
	var text string
	for _, p := range resp.Candidates[0].Content.Parts {
		if p.Text != "" && !p.Thought {
			text += p.Text
		}
	}
	return text
}

func (m *googleModel) isPartEmpty(p *genai.Part) bool {
	if p == nil {
		return true
	}
	return p.Text == "" &&
		p.InlineData == nil &&
		p.FileData == nil &&
		p.FunctionCall == nil &&
		p.FunctionResponse == nil &&
		p.ExecutableCode == nil &&
		p.CodeExecutionResult == nil &&
		!p.Thought
}

func (m *googleModel) convertMessages(messages []*Message) ([]*genai.Content, error) {
	result := make([]*genai.Content, 0, len(messages)+1)
	for _, message := range messages {
		parts := make([]*genai.Part, 0, len(message.Parts))
		for _, part := range message.Parts {
			var p *genai.Part
			switch part := part.(type) {
			case *ThinkingPart:
				p = &genai.Part{
					Text:    part.Text,
					Thought: true,
				}
			case *TextPart:
				p = &genai.Part{
					Text: part.Text,
				}
			case *BinaryPart:
				p = &genai.Part{
					InlineData: &genai.Blob{
						Data:     part.Data,
						MIMEType: part.MediaType,
					},
				}
			case *ImagePart:
				// TODO: This probably doesn't work very well
				p = &genai.Part{
					FileData: &genai.FileData{
						FileURI: part.URL,
					},
				}
			case *ToolCallPart:
				var args map[string]any
				if part.Arguments != "" {
					if err := json.Unmarshal([]byte(part.Arguments), &args); err != nil {
						return nil, fmt.Errorf("invalid tool call arguments for %s: %w", part.Name, err)
					}
				}
				p = &genai.Part{
					FunctionCall: &genai.FunctionCall{
						ID:   part.ID,
						Name: part.Name,
						Args: args,
					},
				}
			case *ToolResultPart:
				responseMap := make(map[string]any)
				if part.Error != "" {
					responseMap["error"] = part.Error
				} else {
					responseMap["output"] = part.Text
				}
				p = &genai.Part{
					FunctionResponse: &genai.FunctionResponse{
						ID:       part.ID,
						Name:     part.Name,
						Response: responseMap,
					},
				}
			}

			if p != nil && !m.isPartEmpty(p) {
				parts = append(parts, p)
			}
		}

		if len(parts) > 0 {
			switch message.Role {
			case RoleUser:
				result = append(result, &genai.Content{
					Role:  genai.RoleUser,
					Parts: parts,
				})
			case RoleAssistant:
				result = append(result, &genai.Content{
					Role:  genai.RoleModel,
					Parts: parts,
				})
			default:
				return nil, fmt.Errorf("unsupported message role for Google: %s", message.Role)
			}
		}
	}
	return result, nil
}

func (m *googleModel) convertTools(tools []ToolDef) ([]*genai.Tool, map[string]ToolDef) {
	if len(tools) == 0 {
		return nil, make(map[string]ToolDef)
	}

	result := make([]*genai.Tool, 0, len(tools))
	toolMap := make(map[string]ToolDef, len(tools))

	for _, tool := range tools {
		if _, exists := toolMap[tool.Name()]; exists {
			continue
		}

		toolMap[tool.Name()] = tool

		schema := jsonSchemaReflector.Reflect(tool.Schema())

		tool := &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:                 tool.Name(),
					Description:          tool.Description(),
					ParametersJsonSchema: schema,
				},
			},
		}
		result = append(result, tool)
	}
	return result, toolMap
}

// googleTurn is the Google-specific Turn. It owns the native []*genai.Content
// history; stream-merge and empty-part handling stay in
// googleModel.handleStreaming and are never exposed through neutral types.
type googleTurn struct {
	m    *googleModel
	opts *generateContentOptions

	config   *genai.GenerateContentConfig
	messages []*genai.Content

	jsParser                 *jsonstream.Parser
	structuredContentBuilder strings.Builder

	pending []*Message

	started   bool
	callCount int
	finalText string

	// lastResponse is stashed by Next for Observe to derive the native
	// assistant message parts.
	lastResponse *genai.GenerateContentResponse
}

func (t *googleTurn) Inject(msgs ...*Message) {
	t.pending = append(t.pending, msgs...)
}

func (t *googleTurn) FinalText() string {
	return t.finalText
}

func (t *googleTurn) Observe(ctx context.Context, _ TurnOutput, outcomes []ToolOutcome) error {
	resp := t.lastResponse

	assistantMsg := &genai.Content{Role: genai.RoleModel}
	if resp != nil && len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, p := range resp.Candidates[0].Content.Parts {
			if !t.m.isPartEmpty(p) {
				assistantMsg.Parts = append(assistantMsg.Parts, p)
			}
		}
	}
	if len(assistantMsg.Parts) > 0 {
		t.messages = append(t.messages, assistantMsg)
	}

	return t.ObserveToolResults(ctx, nil, outcomes)
}

// ObserveToolResults appends only the function-response message, used when
// the assistant message is already in native history (reconstructed turn).
func (t *googleTurn) ObserveToolResults(_ context.Context, _ []ToolCall, outcomes []ToolOutcome) error {
	toolResultsMsg := &genai.Content{Role: genai.RoleUser}
	for _, o := range outcomes {
		responseMap := make(map[string]any)
		if o.Error != "" {
			responseMap["error"] = o.Error
		} else {
			responseMap["output"] = o.Text
		}
		toolResultsMsg.Parts = append(toolResultsMsg.Parts, &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				Name:     o.Name,
				ID:       o.ID,
				Response: responseMap,
			},
		})
	}
	t.messages = append(t.messages, toolResultsMsg)
	return nil
}

func (t *googleTurn) Next(ctx context.Context) (TurnOutput, error) {
	m := t.m

	if !t.started {
		m.metrics.RecordGenerateRequest(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model))
		t.started = true
	}

	if len(t.pending) > 0 {
		conv, err := m.convertMessages(t.pending)
		if err != nil {
			return TurnOutput{}, err
		}
		t.messages = append(t.messages, conv...)
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

	var response *genai.GenerateContentResponse
	var err error
	streamingEmitted := false
	streamingPath := t.opts.StreamingFunc != nil || t.opts.StructuredStreamingFunc != nil

	if streamingPath {
		response, err = m.handleStreaming(ctx, t.config, t.messages, t.opts.StreamingFunc, t.opts.StructuredStreamingFunc, t.jsParser, &t.structuredContentBuilder, start, &hasRecordedFirstToken, &streamingEmitted)
	} else {
		classify := func(err error) (bool, int, time.Duration, bool) {
			if err == nil {
				return false, 0, 0, false
			}
			_, retryable, status := googleErrorDetails(err)
			if !retryable {
				return false, status, 0, false
			}
			ra, hasRA := extractRetryAfter("google", err, m.lastCapturedHeaders())
			return true, status, ra, hasRA
		}
		response, err = retryLoop(ctx, t.opts, classify, func(ctx context.Context) (*genai.GenerateContentResponse, error) {
			return m.client.Models.GenerateContent(ctx, m.model, t.messages, t.config)
		})
	}

	if err != nil {
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordFailure(m.statsModel, collector)
		}

		var ue *UnavailableError
		if errors.As(err, &ue) {
			ue.PartialOutput = streamingEmitted
			m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
			if streamingEmitted {
				return TurnOutput{}, errors.Join(ue, ErrStreamingPartialOutput)
			}
			return TurnOutput{}, ue
		}

		// Streaming path: build UnavailableError ourselves since retryLoop
		// is bypassed.
		if streamingPath {
			if _, retryable, status := googleErrorDetails(err); retryable {
				ra, hasRA := extractRetryAfter("google", err, m.lastCapturedHeaders())
				if hasRA && t.opts.RetryAfterCap > 0 && ra > t.opts.RetryAfterCap {
					ra = t.opts.RetryAfterCap
				}
				se := &UnavailableError{
					StatusCode:    status,
					RetryAfter:    ra,
					HasRetryAfter: hasRA,
					Attempts:      1,
					PartialOutput: streamingEmitted,
					Cause:         err,
				}
				m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
				if streamingEmitted {
					return TurnOutput{}, errors.Join(se, ErrStreamingPartialOutput)
				}
				return TurnOutput{}, se
			}
		}

		var apiErr genai.APIError
		if errors.As(err, &apiErr) {
			m.logger.Error("Google API error", slog.Any("error", err), slog.Any("messages", t.messages))
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
		return TurnOutput{}, err
	}

	var usage TurnUsage
	if response.UsageMetadata != nil {
		collector.Counter("input_tokens").Add(int(response.UsageMetadata.PromptTokenCount))
		collector.Counter("output_tokens").Add(int(response.UsageMetadata.CandidatesTokenCount))
		collector.Counter("cached_read_tokens").Add(int(response.UsageMetadata.CachedContentTokenCount))
		if response.UsageMetadata.ThoughtsTokenCount > 0 {
			collector.Counter("thinking_tokens").Add(int(response.UsageMetadata.ThoughtsTokenCount))
		}

		usage = TurnUsage{
			InputTokens:      int64(response.UsageMetadata.PromptTokenCount),
			OutputTokens:     int64(response.UsageMetadata.CandidatesTokenCount),
			CachedReadTokens: int64(response.UsageMetadata.CachedContentTokenCount),
			ThinkingTokens:   int64(response.UsageMetadata.ThoughtsTokenCount),
		}

		m.metrics.RecordCall(
			ctx,
			GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model),
			int64(response.UsageMetadata.PromptTokenCount),
			int64(response.UsageMetadata.CandidatesTokenCount),
			int64(response.UsageMetadata.CachedContentTokenCount),
			0,
		)
	}

	t.lastResponse = response
	functionCalls := response.FunctionCalls()

	toolCalls := make([]ToolCall, 0, len(functionCalls))
	for _, fc := range functionCalls {
		argsBytes, _ := json.Marshal(fc.Args)
		toolCalls = append(toolCalls, ToolCall{
			ID:        fc.ID,
			Name:      fc.Name,
			Arguments: string(argsBytes),
		})
	}

	var thinking []ThinkingBlock
	if len(response.Candidates) > 0 && response.Candidates[0].Content != nil {
		for _, p := range response.Candidates[0].Content.Parts {
			if p.Thought && p.Text != "" {
				// The vendored genai SDK exposes no thought-signature field,
				// so Google thinking round-trips text only.
				thinking = append(thinking, ThinkingBlock{Text: p.Text})
			}
		}
	}

	rawText := m.extractText(response)
	t.finalText = rawText
	if t.jsParser != nil {
		t.finalText = t.structuredContentBuilder.String()
	}

	var finishReason genai.FinishReason
	if len(response.Candidates) > 0 {
		finishReason = response.Candidates[0].FinishReason
	}
	if len(functionCalls) == 0 && finishReason != genai.FinishReasonStop {
		m.logger.Warn("Google API returned a non-stop finish reason", slog.String("finish_reason", string(finishReason)))
	}

	if metrics := GetMetrics(ctx); metrics != nil {
		metrics.RecordSuccess(m.statsModel, collector)
	}
	m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

	return TurnOutput{
		Text:       rawText,
		Thinking:   thinking,
		ToolCalls:  toolCalls,
		StopReason: googleStopReason(finishReason, len(functionCalls) > 0),
		Usage:      usage,
	}, nil
}

func googleStopReason(reason genai.FinishReason, hasToolCalls bool) StopReason {
	if hasToolCalls {
		return StopReasonToolUse
	}
	switch reason {
	case genai.FinishReasonMaxTokens:
		return StopReasonMaxTokens
	case genai.FinishReasonSafety, genai.FinishReasonProhibitedContent, genai.FinishReasonBlocklist:
		return StopReasonRefusal
	default:
		return StopReasonEndTurn
	}
}
