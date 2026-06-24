package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"reflect"
	"time"

	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
)

// ToolResult is what a tool renders from a successful Execute. Text is what
// the model reads as the result; Attachments optionally carry rich content
// (*ImagePart / *BinaryPart) that is delivered to the model either natively
// (Anthropic tool-result blocks) or as a trailing synthetic user message
// (OpenAI / Google / OpenRouter).
//
// Render only concerns the success path: errors continue to flow through
// ToolOutcome.Error / ModelError / VisibleToolError.
type ToolResult struct {
	// Text is what the model reads as the tool result.
	Text string
	// Attachments are optional rich-content parts (*ImagePart / *BinaryPart)
	// delivered to the model alongside the textual result.
	Attachments []MessagePart
}

// TextToolResult is a convenience constructor for the common text-only tool
// result. (Named to avoid colliding with the TextResult Result type.)
func TextToolResult(text string) ToolResult { return ToolResult{Text: text} }

// WithImage returns a copy of the result with an image attachment appended.
// It chains off TextToolResult (or a bare ToolResult{}) for the rich-content
// case, e.g. TextToolResult("done").WithImage("https://...").
func (r ToolResult) WithImage(url string) ToolResult {
	r.Attachments = append(r.Attachments, NewImagePart(url))
	return r
}

// WithBinary returns a copy of the result with a binary attachment appended,
// e.g. TextToolResult("rendered chart").WithBinary("image/png", png).
func (r ToolResult) WithBinary(mediaType string, data []byte) ToolResult {
	r.Attachments = append(r.Attachments, NewBinaryPart(mediaType, data))
	return r
}

// appendToolAttachment appends a labeled attachment group to parts: a text
// part naming the tool call, followed by the attachments themselves. It is a
// no-op when the call produced no attachments.
//
// Providers without native in-result attachment support (OpenAI, Google,
// OpenRouter) deliver tool-result attachments in a trailing user message. The
// label gives the model an explicit back-reference so each attachment is
// attributed to its tool call rather than read as fresh user input — which
// matters when several tool calls in one batch each return attachments.
func appendToolAttachment(parts []MessagePart, name, id string, attachments []MessagePart) []MessagePart {
	if len(attachments) == 0 {
		return parts
	}
	parts = append(parts, NewTextPart(fmt.Sprintf("Attachment(s) from tool %q (call %s):", name, id)))
	return append(parts, attachments...)
}

// Tool is used to describe a tool that can be used by the LLM.
type Tool[I any, O any] interface {
	// Name is the unique name of the tool.
	Name() string

	// Description is a short description of the tool. Used by the LLM to
	// determine if the tool is relevant to the current task.
	Description() string

	// Schema is the schema of the tool's input. Should be a pointer to a struct.
	Schema() I

	// Execute is the function that will be called when the tool is invoked.
	Execute(ctx context.Context, input I) (O, error)

	// Render turns a successful Execute output into the result delivered to
	// the model. Use TextResult for the common text-only case, or set
	// Attachments to deliver images/binary content.
	Render(O) ToolResult
}

type ToolDef interface {
	Name() string
	Description() string
	Schema() any
	Execute(ctx context.Context, in any) (any, error)
	Render(any) ToolResult
}

// ConditionalTool is an optional interface that ToolDef values may implement
// to signal runtime availability. When IsAvailable returns (false, nil) at
// session creation, the tool is silently excluded from the LLM's tool list
// and from the session's toolMap. A non-nil error aborts session creation.
type ConditionalTool interface {
	IsAvailable(ctx context.Context) (bool, error)
}

type toolWrapper[I any, O any] struct {
	tool Tool[I, O]
}

func (tw *toolWrapper[I, O]) Name() string {
	return tw.tool.Name()
}

func (tw *toolWrapper[I, O]) Description() string {
	return tw.tool.Description()
}

func (tw *toolWrapper[I, O]) Schema() any {
	return tw.tool.Schema()
}

func (tw *toolWrapper[I, O]) Execute(ctx context.Context, in any) (any, error) {
	return tw.tool.Execute(ctx, in.(I))
}

func (tw *toolWrapper[I, O]) Render(out any) ToolResult {
	return tw.tool.Render(out.(O))
}

// IsAvailable delegates to the wrapped Tool when it implements
// ConditionalTool; otherwise the tool is always available.
func (tw *toolWrapper[I, O]) IsAvailable(ctx context.Context) (bool, error) {
	if c, ok := any(tw.tool).(ConditionalTool); ok {
		return c.IsAvailable(ctx)
	}
	return true, nil
}

func NewToolDef[I any, O any](tool Tool[I, O]) ToolDef {
	return &toolWrapper[I, O]{tool: tool}
}

// filterAvailableTools returns the subset of tools whose IsAvailable returns
// true. Tools that don't implement ConditionalTool are kept. Order is
// preserved. The input slice is returned unchanged (no allocation) when no
// tool is filtered out.
func filterAvailableTools(ctx context.Context, tools []ToolDef) ([]ToolDef, error) {
	if len(tools) == 0 {
		return tools, nil
	}
	var out []ToolDef
	for i, t := range tools {
		c, ok := t.(ConditionalTool)
		if !ok {
			if out != nil {
				out = append(out, t)
			}
			continue
		}
		avail, err := c.IsAvailable(ctx)
		if err != nil {
			return nil, fmt.Errorf("tool %q availability check: %w", t.Name(), err)
		}
		if avail {
			if out != nil {
				out = append(out, t)
			}
			continue
		}
		if out == nil {
			out = make([]ToolDef, 0, len(tools)-1)
			out = append(out, tools[:i]...)
		}
	}
	if out == nil {
		return tools, nil
	}
	return out, nil
}

type toolResult struct {
	id           string
	functionName string
	results      string
	errorString  string
}

var jsonSchemaReflector = jsonschema.Reflector{
	Anonymous:      true,
	ExpandedStruct: true,
}

func newPublicID() string {
	return uuid.New().String()
}

// doToolCall is a helper function to make a tool call using the given JSON-encoded
// arguments.
func doToolCall(ctx context.Context, logger *slog.Logger, streamingFunc StreamingFunc, timeout time.Duration, id string, tool ToolDef, arguments string) (result ToolResult, err error) {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	eventID := newPublicID()
	started := false
	var out any

	// Defer delivery of results so we always emit a result/error event.
	defer func() {
		if streamingFunc == nil || !started {
			return
		}
		var event StreamingEvent
		if err != nil {
			event = StreamingEventToolError{ID: eventID, ToolID: tool.Name(), Error: err}
		} else {
			event = StreamingEventToolResult{ID: eventID, ToolID: tool.Name(), Result: out}
		}
		if nerr := streamingFunc(context.WithoutCancel(ctx), event); nerr != nil {
			logger.Error("Tool notify failed", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.Any("error", nerr))
		}
	}()

	defer func() {
		if r := recover(); r != nil {
			logger.Error("Tool call panicked", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.Int("inputLength", len(arguments)), slog.Any("panic", r))
			err = fmt.Errorf("%w: %v", NewVisibleToolError("tool execution failed due to panic"), r)
		}
	}()

	// Record tool call in execution tracker
	if tracker, ok := ctx.Value(executionContextKey{}).(*executionTracker); ok {
		tracker.RecordToolCall(tool.Name())
	}

	logger.Debug("Executing tool", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("input", arguments))

	// Parse into typed struct from the tool
	schemaType := reflect.TypeOf(tool.Schema()).Elem()
	args := reflect.New(schemaType).Interface()
	if err := json.Unmarshal([]byte(arguments), args); err != nil {
		logger.Error("Tool call failed, could not parse arguments", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.Int("inputLength", len(arguments)), slog.Any("error", err))
		return ToolResult{}, fmt.Errorf("%w: %w", NewVisibleToolError("invalid arguments"), err)
	}

	// From here on the ToolUse event has been sent (or attempted), so the
	// deferred emitter owns sending the matching terminal event.
	if streamingFunc != nil {
		started = true
		if nerr := streamingFunc(ctx, StreamingEventToolUse{ID: eventID, ToolID: tool.Name(), Arguments: args}); nerr != nil {
			logger.Error("Tool notify failed", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.Int("inputLength", len(arguments)), slog.Any("error", nerr))
		}
	}

	// Execute the tool
	out, err = tool.Execute(ctx, args)
	if err != nil {
		logger.Error("Tool call failed", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.Int("inputLength", len(arguments)), slog.Any("error", err))
		return ToolResult{}, fmt.Errorf("tool execution failed: %w", err)
	}

	result = tool.Render(out)

	logger.Debug("Tool call succeeded", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("result", result.Text))

	return result, nil
}
