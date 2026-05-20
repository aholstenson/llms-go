// Command basic demonstrates the simplest possible text generation with
// llms-go: resolve a model through a Manager and ask it a question.
//
// Run with:
//
//	export ANTHROPIC_API_TOKEN=...      # or the token for your provider
//	go run ./examples/basic
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

	manager := llms.NewManager()

	model, err := manager.GetModel(modelName())
	if err != nil {
		log.Fatalf("could not load model: %v", err)
	}

	result, err := model.GenerateContent(ctx,
		llms.WithSystemPrompt("You are a concise assistant."),
		llms.WithMessages(
			llms.NewMessage(llms.RoleUser,
				llms.NewTextPart("Explain what a Go interface is in two sentences.")),
		),
		llms.WithMaxOutputTokens(256),
	)
	if err != nil {
		log.Fatalf("generation failed: %v", err)
	}

	// Plain prompts return a TextResult.
	if text, ok := result.(llms.TextResult); ok {
		fmt.Println(text.Text)
	}
}
