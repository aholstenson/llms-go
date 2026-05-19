package llms

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"time"

	"github.com/aholstenson/llms-go/jsonstream"
	"github.com/cockroachdb/errors"
	"github.com/invopop/jsonschema"
	"google.golang.org/genai"
)

type googleModel struct {
	logger            *slog.Logger
	metrics           *Metrics
	client            *genai.Client
	statsModel        string
	model             string
	info              ModelInfo
	subParserRegistry map[string]SubParserConfig
}

// NewGoogleModel creates a new Google Gemini model. info carries embedded
// model metadata used to gate request parameters; the zero value is treated
// permissively.
func NewGoogleModel(logger *slog.Logger, metrics *Metrics, apiKey string, model string, registry map[string]SubParserConfig, info ModelInfo) (Model, error) {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
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
	}, nil
}

func (m *googleModel) GenerateContent(ctx context.Context, options ...GenerateOption) (Result, error) {
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

	if opts.MaxTokens != 0 {
		config.MaxOutputTokens = int32(opts.MaxTokens) //nolint:gosec
	} else {
		config.MaxOutputTokens = 1000
	}

	if opts.MaxThinkingTokens != 0 && m.info.allowsReasoning() {
		thinkingBudget := int32(opts.MaxThinkingTokens) //nolint:gosec
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget: &thinkingBudget,
		}

		// Gemini seems to include thinking tokens in the max output tokens (more testing needed)
		config.MaxOutputTokens += int32(opts.MaxThinkingTokens) //nolint:gosec
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

	// Record a request to the LLM as we begin to generate content
	m.metrics.RecordGenerateRequest(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model))

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

		var response *genai.GenerateContentResponse
		var err error

		if opts.StreamingFunc != nil {
			if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageStart{}); emitErr != nil {
				return nil, emitErr
			}
		}

		if opts.StreamingFunc != nil || opts.StructuredStreamingFunc != nil {
			response, err = m.handleStreaming(ctx, config, messages, opts.StreamingFunc, opts.StructuredStreamingFunc, jsParser, &structuredContentBuilder, start, &hasRecordedFirstToken)
		} else {
			response, err = m.client.Models.GenerateContent(ctx, m.model, messages, config)
		}

		if err != nil {
			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordFailure(m.statsModel, collector)
			}

			var apiErr *genai.APIError
			if errors.As(err, &apiErr) {
				if isUnavailableStatusCode(apiErr.Code) {
					m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeUnavailable)
					return nil, errors.Mark(errors.Wrap(err, "Google model unavailable"), ErrModelUnavailable)
				}

				m.logger.Error("Google API error", slog.Any("error", err), slog.Any("messages", messages))
			}

			m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
			return nil, err
		}

		// Record metrics
		if response.UsageMetadata != nil {
			collector.Counter("input_tokens").Add(int(response.UsageMetadata.PromptTokenCount))
			collector.Counter("output_tokens").Add(int(response.UsageMetadata.CandidatesTokenCount))
			collector.Counter("cached_read_tokens").Add(int(response.UsageMetadata.CachedContentTokenCount))
			if response.UsageMetadata.ThoughtsTokenCount > 0 {
				collector.Counter("thinking_tokens").Add(int(response.UsageMetadata.ThoughtsTokenCount))
			}

			// Update execution tracker with token counts
			tracker.AddTokens(
				int64(response.UsageMetadata.PromptTokenCount),
				int64(response.UsageMetadata.CandidatesTokenCount),
				int64(response.UsageMetadata.CachedContentTokenCount),
			)

			m.metrics.RecordCall(
				ctx,
				GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model),
				int64(response.UsageMetadata.PromptTokenCount),
				int64(response.UsageMetadata.CandidatesTokenCount),
				int64(response.UsageMetadata.CachedContentTokenCount),
				0,
			)
		}

		// Check for function calls
		functionCalls := response.FunctionCalls()
		if len(functionCalls) == 0 {
			// Emit a final MessageEnd to signal that the agentic loop is done
			if opts.StreamingFunc != nil {
				if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageEnd{Final: true}); emitErr != nil {
					return nil, emitErr
				}
			}

			if metrics := GetMetrics(ctx); metrics != nil {
				metrics.RecordSuccess(m.statsModel, collector)
			}
			m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)
			finishReason := response.Candidates[0].FinishReason
			if finishReason != genai.FinishReasonStop {
				m.logger.Warn("Google API returned a non-stop finish reason", slog.String("finish_reason", string(finishReason)))
			}

			// Return structured result if ResponseSchema was set
			if opts.ResponseSchema != nil {
				content := m.extractText(response)
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

			return TextResult{Text: m.extractText(response)}, nil
		}

		// Emit an intermediate MessageEnd to signal that this is not the final reply
		if opts.StreamingFunc != nil {
			if emitErr := opts.StreamingFunc(ctx, StreamingEventMessageEnd{Final: false}); emitErr != nil {
				return nil, emitErr
			}
		}

		// Add assistant response to messages
		assistantMsg := &genai.Content{
			Role: genai.RoleModel,
		}
		if len(response.Candidates) > 0 {
			for _, p := range response.Candidates[0].Content.Parts {
				if !m.isPartEmpty(p) {
					assistantMsg.Parts = append(assistantMsg.Parts, p)
				}
			}
		}

		if len(assistantMsg.Parts) > 0 {
			messages = append(messages, assistantMsg)
		}

		// Execute function calls in parallel
		toolResultChan := make(chan toolResult, len(functionCalls))
		for _, fc := range functionCalls {
			tool, ok := toolMap[fc.Name]
			if !ok {
				toolResultChan <- toolResult{
					id:           fc.ID,
					functionName: fc.Name,
					errorString:  "Requested tool not found",
				}
				continue
			}

			go func(fc *genai.FunctionCall, tool ToolDef) {
				argsBytes, _ := json.Marshal(fc.Args)
				result, err := doToolCall(ctx, m.logger, opts.StreamingFunc, opts.ToolCallTimeout, fc.ID, tool, string(argsBytes))
				if err != nil {
					toolResultChan <- toolResult{
						id:           fc.ID,
						functionName: fc.Name,
						errorString:  "Tool call failed",
					}
				} else {
					toolResultChan <- toolResult{
						id:           fc.ID,
						functionName: fc.Name,
						results:      result,
					}
				}
			}(fc, tool)
		}

		// Collect tool results and add them to the conversation
		toolResultsMsg := &genai.Content{
			Role: genai.RoleUser,
		}
		for range len(functionCalls) {
			select {
			case tr := <-toolResultChan:
				responseMap := make(map[string]any)
				if tr.errorString != "" {
					responseMap["error"] = tr.errorString
				} else {
					responseMap["output"] = tr.results
				}

				toolResultsMsg.Parts = append(toolResultsMsg.Parts, &genai.Part{
					FunctionResponse: &genai.FunctionResponse{
						Name:     tr.functionName,
						ID:       tr.id,
						Response: responseMap,
					},
				})
			case <-ctx.Done():
				if metrics := GetMetrics(ctx); metrics != nil {
					metrics.RecordFailure(m.statsModel, collector)
				}
				m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeInternal)
				return nil, ctx.Err()
			}
		}
		messages = append(messages, toolResultsMsg)

		// Record metrics for this call to the LLM as a success
		if metrics := GetMetrics(ctx); metrics != nil {
			metrics.RecordSuccess(m.statsModel, collector)
		}

		m.metrics.RecordCallDuration(ctx, GenAISystemGoogle, GenAIOperationChat, GenAIModel(m.model), time.Since(start), GenAIErrorTypeNoError)

		// Continue the conversation loop
	}
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
			if len(lastResponse.Candidates) > 0 {
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
				for _, p := range resp.Candidates[0].Content.Parts {
					if m.isPartEmpty(p) {
						continue
					}
					lastResponse.Candidates[0].Content.Parts = append(lastResponse.Candidates[0].Content.Parts, p)
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
				return nil, errors.Newf("unsupported message role for Google: %s", message.Role)
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
