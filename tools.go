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

func NewToolDef[I any, O any](tool Tool[I, O]) ToolDef {
	return &toolWrapper[I, O]{tool: tool}
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
			logger.Error("Tool call panicked", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("input", arguments), slog.Any("panic", r))
			err = fmt.Errorf("tool execution failed due to panic: %v", r)
		}
	}()

	// Record tool call in execution tracker
	if tracker, ok := ctx.Value(executionContextKey{}).(*ExecutionTracker); ok {
		tracker.RecordToolCall(tool.Name())
	}

	eventID := newPublicID()
	logger.Debug("Executing tool", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("input", arguments))

	// Parse into typed struct from the tool
	schemaType := reflect.TypeOf(tool.Schema()).Elem()
	args := reflect.New(schemaType).Interface()
	if err := json.Unmarshal([]byte(arguments), args); err != nil {
		logger.Error("Tool call failed, could not parse arguments", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("input", arguments), slog.Any("error", err))
		return "", fmt.Errorf("invalid arguments: %w", err)
	}

	if streamingFunc != nil {
		err := streamingFunc(ctx, StreamingEventToolUse{ID: eventID, ToolID: tool.Name(), Arguments: args})
		if err != nil {
			logger.Error("Tool notify failed", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("input", arguments), slog.Any("error", err))
		}
	}

	// Execute the tool
	var out any
	out, err = tool.Execute(ctx, args)
	if err != nil {
		logger.Error("Tool call failed", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("input", arguments), slog.Any("error", err))
		return "", fmt.Errorf("tool execution failed: %w", err)
	}

	result = tool.ToString(out)

	logger.Debug("Tool call succeeded", slog.String("tool", tool.Name()), slog.String("toolCallID", id), slog.String("result", result))
	if streamingFunc != nil {
		err := streamingFunc(ctx, StreamingEventToolResult{ID: eventID, ToolID: tool.Name(), Result: out})
		if err != nil {
			logger.Error("Tool notify failed", slog.String("tool", tool.Name()), slog.String("input", arguments), slog.Any("error", err))
		}
	}

	return result, nil
}
