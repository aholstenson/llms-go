// Command streaming demonstrates streaming a response token-by-token,
// including reasoning/thinking tokens when the model supports them.
//
// Run with:
//
//	export ANTHROPIC_API_KEY=...        # or the key for your provider
//	go run ./examples/streaming
package main

import (
	"context"
	"fmt"
	"log"
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

	manager, err := llms.NewManager()
	if err != nil {
		log.Fatalf("could not create manager: %v", err)
	}

	model, err := manager.GetModel(ctx, modelName())
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
