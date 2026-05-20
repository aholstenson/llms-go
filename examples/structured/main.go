// Command structured demonstrates typed structured output. A JSON schema is
// derived automatically from the Go type, and the response is unmarshalled
// back into that type.
//
// Run with:
//
//	export ANTHROPIC_API_KEY=...        # or the key for your provider
//	go run ./examples/structured
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

// Recipe is the shape we want the model to fill in. Struct tags drive the
// generated JSON schema and the field descriptions sent to the model.
type Recipe struct {
	Name        string   `json:"name" jsonschema:"description=The dish name"`
	PrepMinutes int      `json:"prep_minutes" jsonschema:"description=Preparation time in minutes"`
	Ingredients []string `json:"ingredients" jsonschema:"description=List of ingredients"`
	Steps       []string `json:"steps" jsonschema:"description=Ordered preparation steps"`
}

func main() {
	ctx := context.Background()

	manager := llms.NewManager()

	model, err := manager.GetModel(modelName())
	if err != nil {
		log.Fatalf("could not load model: %v", err)
	}

	result, err := model.GenerateContent(ctx,
		llms.WithMessages(
			llms.NewMessage(llms.RoleUser,
				llms.NewTextPart("Give me a simple recipe for pancakes.")),
		),
		llms.WithResponseSchema[Recipe](),
		llms.WithMaxOutputTokens(1024),
	)
	if err != nil {
		log.Fatalf("generation failed: %v", err)
	}

	structured, ok := result.(llms.StructuredResult[Recipe])
	if !ok {
		log.Fatalf("unexpected result type %T", result)
	}

	recipe := structured.Data
	fmt.Printf("%s (%d min)\n\n", recipe.Name, recipe.PrepMinutes)
	fmt.Println("Ingredients:")
	for _, ing := range recipe.Ingredients {
		fmt.Printf("  - %s\n", ing)
	}
	fmt.Println("\nSteps:")
	for i, step := range recipe.Steps {
		fmt.Printf("  %d. %s\n", i+1, step)
	}
}
