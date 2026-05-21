// Command tools demonstrates defining a tool and letting the model call it
// in an agentic loop. The model decides when to invoke the tool, the result
// is fed back, and generation continues until a final answer is produced.
//
// Run with:
//
//	export ANTHROPIC_API_KEY=...        # or the key for your provider
//	go run ./examples/tools
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

// WeatherInput is the tool's argument schema. The pointer to this type is
// returned from Schema() and the JSON schema is derived from the struct tags.
type WeatherInput struct {
	City string `json:"city" jsonschema:"description=The city to look up weather for"`
}

// WeatherOutput is what Execute returns.
type WeatherOutput struct {
	City      string `json:"city"`
	Temp      int    `json:"temp_celsius"`
	Condition string `json:"condition"`
}

// weatherTool implements llms.Tool[*WeatherInput, WeatherOutput].
type weatherTool struct{}

func (weatherTool) Name() string { return "get_weather" }

func (weatherTool) Description() string {
	return "Get the current weather for a city."
}

func (weatherTool) Schema() *WeatherInput { return &WeatherInput{} }

func (weatherTool) Execute(ctx context.Context, in *WeatherInput) (WeatherOutput, error) {
	// A real tool would call an API here; we return canned data.
	return WeatherOutput{City: in.City, Temp: 18, Condition: "partly cloudy"}, nil
}

func (weatherTool) ToString(out WeatherOutput) string {
	return fmt.Sprintf("%s: %d°C, %s", out.City, out.Temp, out.Condition)
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

	result, err := model.GenerateContent(ctx,
		llms.WithMessages(
			llms.NewMessage(llms.RoleUser,
				llms.NewTextPart("What's the weather like in Gothenburg? Reply in one sentence.")),
		),
		llms.WithTools(llms.NewToolDef(weatherTool{})),
		llms.WithMaxSteps(5),
		llms.WithMaxOutputTokens(512),
	)
	if err != nil {
		log.Fatalf("generation failed: %v", err)
	}

	if text, ok := result.(llms.TextResult); ok {
		fmt.Println(text.Text)
	}
}
