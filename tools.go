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

	ToString(O) string
}

type ToolDef interface {
	Name() string
	Description() string
	Schema() any
	Execute(ctx context.Context, in any) (any, error)
	ToString(any) string
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

func (tw *toolWrapper[I, O]) ToString(out any) string {
	return tw.tool.ToString(out.(O))
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
func doToolCall(ctx context.Context, logger *slog.Logger, streamingFunc StreamingFunc, timeout time.Duration, id string, tool ToolDef, arguments string) (result string, err error) {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

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

	eventID := newPublicID()
	logger.Debug("Executing tool", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("input", arguments))

	// Parse into typed struct from the tool
	schemaType := reflect.TypeOf(tool.Schema()).Elem()
	args := reflect.New(schemaType).Interface()
	if err := json.Unmarshal([]byte(arguments), args); err != nil {
		logger.Error("Tool call failed, could not parse arguments", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.Int("inputLength", len(arguments)), slog.Any("error", err))
		return "", fmt.Errorf("%w: %w", NewVisibleToolError("invalid arguments"), err)
	}

	if streamingFunc != nil {
		err := streamingFunc(ctx, StreamingEventToolUse{ID: eventID, ToolID: tool.Name(), Arguments: args})
		if err != nil {
			logger.Error("Tool notify failed", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.Int("inputLength", len(arguments)), slog.Any("error", err))
		}
	}

	// Execute the tool
	var out any
	out, err = tool.Execute(ctx, args)
	if err != nil {
		logger.Error("Tool call failed", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.Int("inputLength", len(arguments)), slog.Any("error", err))
		return "", fmt.Errorf("tool execution failed: %w", err)
	}

	result = tool.ToString(out)

	logger.Debug("Tool call succeeded", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("result", result))
	if streamingFunc != nil {
		err := streamingFunc(ctx, StreamingEventToolResult{ID: eventID, ToolID: tool.Name(), Result: out})
		if err != nil {
			logger.Error("Tool notify failed", slog.String("tool", tool.Name()), slog.Int("inputLength", len(arguments)), slog.Any("error", err))
		}
	}

	return result, nil
}
