// Command genmodelinfo regenerates modelinfo_data.json from models.dev.
//
// Usage:
//
//	go run ./cmd/genmodelinfo            # fetch https://models.dev/api.json
//	go run ./cmd/genmodelinfo -in api.json -o modelinfo_data.json
//
// CI runs the generator and then `git diff --exit-code modelinfo_data.json`
// to detect drift. Output is deterministic (encoding/json sorts map keys) so
// diffs are clean. The -in flag lets CI pin a committed snapshot for
// reproducibility.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/aholstenson/llms-go/internal/modelsdev"
)

const apiURL = "https://models.dev/api.json"

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "genmodelinfo:", err)
		os.Exit(1)
	}
}

func run() error {
	in := flag.String("in", "", "path to a local models.dev api.json; if empty, fetched over HTTP")
	out := flag.String("o", "modelinfo_data.json", "output path for the generated artifact")
	flag.Parse()

	data, err := loadRaw(*in)
	if err != nil {
		return err
	}

	var raw modelsdev.RawData
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("parsing api.json: %w", err)
	}

	info := modelsdev.Transform(raw)

	// encoding/json marshals map keys in sorted order, giving deterministic
	// output for clean diffs.
	encoded, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return fmt.Errorf("encoding artifact: %w", err)
	}
	encoded = append(encoded, '\n')

	if err := os.WriteFile(*out, encoded, 0o644); err != nil { //nolint:gosec
		return fmt.Errorf("writing %s: %w", *out, err)
	}

	fmt.Fprintf(os.Stderr, "genmodelinfo: wrote %d models to %s\n", len(info), *out)
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
	resp, err := client.Get(apiURL)
	if err != nil {
		return nil, fmt.Errorf("fetching %s: %w", apiURL, err)
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetching %s: unexpected status %s", apiURL, resp.Status)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}
	return data, nil
}
