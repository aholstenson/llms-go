package modelsdev_test

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/aholstenson/llms-go/internal/modelsdev"
)

func readSample(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/modelsdev_sample.json")
	if err != nil {
		t.Fatalf("reading fixture: %v", err)
	}
	return data
}

func TestFilterKeepsSupportedProviders(t *testing.T) {
	filtered, err := modelsdev.Filter(readSample(t))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	raw, err := modelsdev.Parse(filtered)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	for _, want := range []string{"anthropic", "openai", "openrouter"} {
		if _, ok := raw[want]; !ok {
			t.Errorf("expected provider %q in filtered data", want)
		}
	}
	if _, ok := raw["deepseek"]; ok {
		t.Error("unsupported provider deepseek leaked into filtered data")
	}
	if _, ok := raw["openrouter"].Models["google/gemini-2.5-flash-lite"]; !ok {
		t.Error("expected nested OpenRouter model id to survive filtering")
	}
}

func TestFilterDropsDescriptionsAndKeepsOtherFields(t *testing.T) {
	in := []byte(`{"openai":{"id":"openai","models":{"gpt-x":{
		"id":"gpt-x","description":"a long text","reasoning":true,
		"reasoning_options":[{"type":"effort","values":["none","low","high"]}],
		"future_field":{"kept":true}}}}}`)

	filtered, err := modelsdev.Filter(in)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if bytes.Contains(filtered, []byte("description")) {
		t.Error("expected description to be removed")
	}
	for _, kept := range []string{"reasoning_options", "future_field"} {
		if !bytes.Contains(filtered, []byte(kept)) {
			t.Errorf("expected %s to be kept", kept)
		}
	}
}

func TestFilterIsDeterministic(t *testing.T) {
	a, err := modelsdev.Filter(readSample(t))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	b, err := modelsdev.Filter(readSample(t))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	ca, _ := modelsdev.Compress(a)
	cb, _ := modelsdev.Compress(b)
	if !bytes.Equal(ca, cb) {
		t.Error("expected equal input to give equal compressed output")
	}
}

func TestFilterRejectsDataWithoutSupportedProviders(t *testing.T) {
	_, err := modelsdev.Filter([]byte(`{"deepseek":{"id":"deepseek","models":{}}}`))
	if err == nil || !strings.Contains(err.Error(), "supported providers") {
		t.Fatalf("expected an error about supported providers, got %v", err)
	}
}

func TestParseReadsGzip(t *testing.T) {
	filtered, err := modelsdev.Filter(readSample(t))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	compressed, err := modelsdev.Compress(filtered)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}
	raw, err := modelsdev.Parse(compressed)
	if err != nil {
		t.Fatalf("Parse gzip: %v", err)
	}
	if _, ok := raw["anthropic"].Models["claude-haiku-4-5"]; !ok {
		t.Error("expected claude-haiku-4-5 after gzip round trip")
	}
}

func TestVersionIsNewestLastUpdated(t *testing.T) {
	raw := modelsdev.RawData{
		"openai": {Models: map[string]modelsdev.RawModel{
			"a": {LastUpdated: "2026-01-01"},
			"b": {LastUpdated: "2026-03-05"},
		}},
		"deepseek": {Models: map[string]modelsdev.RawModel{
			"c": {LastUpdated: "2027-01-01"},
		}},
	}
	if got := modelsdev.Version(raw); got != "2026-03-05" {
		t.Errorf("Version = %q, want 2026-03-05 (unsupported providers ignored)", got)
	}
}
