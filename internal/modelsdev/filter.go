package modelsdev

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
)

// APIURL is the models.dev endpoint that serves the full api.json.
const APIURL = "https://models.dev/api.json"

// Providers is the set of models.dev providers the llms package supports.
// Models from other providers are dropped by Filter.
var Providers = []string{"anthropic", "openai", "google", "openrouter"}

// droppedModelFields are models.dev model fields the llms package never
// reads and that make up a large part of the data.
var droppedModelFields = []string{"description"}

// Filter reduces a models.dev api.json document (plain or gzip) to the
// supported providers, removes unused large fields, and returns it as
// minified JSON in the models.dev format. The output is deterministic
// because encoding/json sorts map keys.
func Filter(data []byte) ([]byte, error) {
	data, err := decompress(data)
	if err != nil {
		return nil, err
	}

	var all map[string]map[string]json.RawMessage
	if err := json.Unmarshal(data, &all); err != nil {
		return nil, fmt.Errorf("parsing models.dev data: %w", err)
	}

	out := make(map[string]map[string]json.RawMessage, len(Providers))
	for _, provider := range Providers {
		entry, ok := all[provider]
		if !ok {
			continue
		}

		var models map[string]map[string]json.RawMessage
		if err := json.Unmarshal(entry["models"], &models); err != nil {
			return nil, fmt.Errorf("parsing %s models: %w", provider, err)
		}
		for _, model := range models {
			for _, field := range droppedModelFields {
				delete(model, field)
			}
		}

		encoded, err := json.Marshal(models)
		if err != nil {
			return nil, fmt.Errorf("encoding %s models: %w", provider, err)
		}
		entry["models"] = encoded
		out[provider] = entry
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("models.dev data has none of the supported providers %v", Providers)
	}

	return json.Marshal(out)
}

// Parse decodes a models.dev api.json document, plain or gzip.
func Parse(data []byte) (RawData, error) {
	data, err := decompress(data)
	if err != nil {
		return nil, err
	}

	var raw RawData
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parsing models.dev data: %w", err)
	}
	return raw, nil
}

// Version returns the newest last_updated date of the supported providers'
// models. It identifies how current a copy of the data is, so a newer copy
// can win over an older one.
func Version(raw RawData) string {
	var newest string
	for _, provider := range Providers {
		for _, model := range raw[provider].Models {
			if model.LastUpdated > newest {
				newest = model.LastUpdated
			}
		}
	}
	return newest
}

// Compress gzips data with a fixed header, so equal input gives equal output.
func Compress(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	w, err := gzip.NewWriterLevel(&buf, gzip.BestCompression)
	if err != nil {
		return nil, err
	}
	if _, err := w.Write(data); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// decompress returns data unchanged unless it starts with the gzip magic
// bytes, in which case it returns the decompressed content.
func decompress(data []byte) ([]byte, error) {
	if len(data) < 2 || data[0] != 0x1f || data[1] != 0x8b {
		return data, nil
	}
	r, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("reading gzip data: %w", err)
	}
	defer r.Close() //nolint:errcheck
	out, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("reading gzip data: %w", err)
	}
	return out, nil
}
