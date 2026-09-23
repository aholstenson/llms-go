// Command genmodelinfo regenerates modelinfo_data.json.gz from models.dev.
//
// Usage:
//
//	go run ./cmd/genmodelinfo            # fetch https://models.dev/api.json
//	go run ./cmd/genmodelinfo -in api.json -o modelinfo_data.json.gz
//
// The output is models.dev's api.json reduced to the providers the llms
// package supports, without model descriptions, minified and gzipped. It
// stays in the models.dev format, so the same data can also be given to
// llms.LoadModelInfo or LLM_MODELS_FILE. The output is deterministic, so
// equal input gives an equal file. A summary of added and removed models is
// written to stderr for use in pull request descriptions.
package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"time"

	"github.com/aholstenson/llms-go/internal/modelsdev"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "genmodelinfo:", err)
		os.Exit(1)
	}
}

func run() error {
	in := flag.String("in", "", "path to a local models.dev api.json (plain or gzip); if empty, fetched over HTTP")
	out := flag.String("o", "modelinfo_data.json.gz", "output path for the generated file")
	flag.Parse()

	data, err := loadRaw(*in)
	if err != nil {
		return err
	}

	filtered, err := modelsdev.Filter(data)
	if err != nil {
		return err
	}
	raw, err := modelsdev.Parse(filtered)
	if err != nil {
		return err
	}
	compressed, err := modelsdev.Compress(filtered)
	if err != nil {
		return err
	}

	previous := readPrevious(*out)

	if err := os.WriteFile(*out, compressed, 0o644); err != nil { //nolint:gosec
		return fmt.Errorf("writing %s: %w", *out, err)
	}

	current := modelKeys(raw)
	fmt.Fprintf(os.Stderr, "genmodelinfo: wrote %d models (data version %s, %d bytes) to %s\n",
		len(current), modelsdev.Version(raw), len(compressed), *out)
	if previous != nil {
		printDiff(previous, current)
	}
	return nil
}

func loadRaw(path string) ([]byte, error) {
	if path != "" {
		data, err := os.ReadFile(path) //nolint:gosec
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", path, err)
		}
		return data, nil
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Get(modelsdev.APIURL)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", modelsdev.APIURL, err)
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching %s: unexpected status %s", modelsdev.APIURL, resp.Status)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}
	return data, nil
}

// readPrevious returns the model keys of an earlier generated file, or nil
// when there is none.
func readPrevious(path string) []string {
	data, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		return nil
	}
	raw, err := modelsdev.Parse(data)
	if err != nil {
		return nil
	}
	return modelKeys(raw)
}

func modelKeys(raw modelsdev.RawData) []string {
	var keys []string
	for _, provider := range modelsdev.Providers {
		for id := range raw[provider].Models {
			keys = append(keys, provider+"/"+id)
		}
	}
	slices.Sort(keys)
	return keys
}

func printDiff(previous, current []string) {
	for _, k := range current {
		if _, found := slices.BinarySearch(previous, k); !found {
			fmt.Fprintln(os.Stderr, "  added:  ", k)
		}
	}
	for _, k := range previous {
		if _, found := slices.BinarySearch(current, k); !found {
			fmt.Fprintln(os.Stderr, "  removed:", k)
		}
	}
}
