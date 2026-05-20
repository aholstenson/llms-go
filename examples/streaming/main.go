// Command streaming demonstrates streaming a response token-by-token,
// including reasoning/thinking tokens when the model supports them.
//
// Run with:
//
//	export ANTHROPIC_API_TOKEN=...      # or the token for your provider
//	go run ./examples/streaming
package main

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"

	llms "github.com/aholstenson/llms-go"
)

func modelName() string {
	if m := os.Getenv("LLM_EXAMPLE_MODEL"); m != "" {
		return m
	}
	return "anthropic/claude-sonnet-4-5"
}

func main() {
	ctx := context.Background()

	manager := llms.NewManager(slog.Default(), llms.NewNoopMetrics())

	model, err := manager.GetModel(modelName())
	if err != nil {
		log.Fatalf("could not load model: %v", err)
	}

	stream := func(ctx context.Context, event llms.StreamingEvent) error {
		switch e := event.(type) {
		case llms.StreamingEventThinking:
			fmt.Print("\033[2m" + e.Text + "\033[0m") // dim reasoning tokens
		case llms.StreamingEventTextChunk:
			fmt.Print(e.Text)
		case llms.StreamingEventMessageEnd:
			if e.Final {
				fmt.Println()
			}
		}
		return nil
	}

	_, err = model.GenerateContent(ctx,
		llms.WithMessages(
			llms.NewMessage(llms.RoleUser,
				llms.NewTextPart("Write a short haiku about concurrency in Go.")),
		),
		llms.WithStreamingFunc(stream),
		llms.WithMaxOutputTokens(256),
	)
	if err != nil {
		log.Fatalf("generation failed: %v", err)
	}
}
