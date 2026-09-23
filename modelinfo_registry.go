package llms

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/aholstenson/llms-go/internal/modelsdev"
)

// embeddedModelsDev is models.dev data for the supported providers, filtered
// and gzipped by cmd/genmodelinfo.
//
//go:embed modelinfo_data.json.gz
var embeddedModelsDev []byte

// ModelsFileEnv names the environment variable that points to a models.dev
// api.json file (plain or gzip). When it is set, that file is used in place
// of the embedded data and the refresh cache.
const ModelsFileEnv = "LLM_MODELS_FILE"

// modelsDevURL is where RefreshModelInfo gets models.dev data.
var modelsDevURL = modelsdev.APIURL

// modelsDevCachePath returns the file RefreshModelInfo writes and later
// processes read. It is a variable so tests can move it.
var modelsDevCachePath = func() (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "llms-go", "models.dev.json.gz"), nil
}

// modelInfoRegistry holds the model info for the process: the models.dev
// data in use and the entries added with RegisterModelInfo.
type modelInfoRegistry struct {
	loadOnce sync.Once

	mu         sync.RWMutex
	base       map[string]ModelInfo
	registered map[string]ModelInfo
}

var modelInfos = &modelInfoRegistry{registered: make(map[string]ModelInfo)}

// LookupModelInfo returns the ModelInfo for a fully qualified model name
// (e.g. "anthropic/claude-haiku-4-5" or
// "openrouter/google/gemini-2.5-flash-lite"). Entries from RegisterModelInfo
// win over models.dev data. The boolean is false when the model is not known,
// in which case the zero ModelInfo is returned and callers should treat the
// model permissively.
func LookupModelInfo(qualifiedName string) (ModelInfo, bool) {
	r := modelInfos
	r.ensureLoaded()

	r.mu.RLock()
	defer r.mu.RUnlock()
	if info, ok := r.registered[qualifiedName]; ok {
		return info, true
	}
	info, ok := r.base[qualifiedName]
	return info, ok
}

// RegisterModelInfo adds or replaces the ModelInfo for a fully qualified
// model name. Use it for models that models.dev does not list or to correct
// its data. Registered entries win over all models.dev data and over the
// Anthropic Models API answer, and apply to models that already exist.
//
// The capability flags gate requests: a model registered with
// Caps.ToolCall=false rejects tools, so set every capability the model has.
func RegisterModelInfo(qualifiedName string, info ModelInfo) {
	r := modelInfos
	r.mu.Lock()
	defer r.mu.Unlock()
	r.registered[qualifiedName] = info
}

// LoadModelInfo replaces the models.dev data for this process with a
// models.dev api.json document (plain or gzip) read from rd. Only the
// supported providers are used.
func LoadModelInfo(rd io.Reader) error {
	data, err := io.ReadAll(rd)
	if err != nil {
		return fmt.Errorf("reading model info: %w", err)
	}
	raw, err := modelsdev.Parse(data)
	if err != nil {
		return err
	}
	return modelInfos.setBase(raw)
}

// RefreshModelInfo gets the current data from models.dev, uses it for this
// process, and saves it in the user cache directory. Later processes use the
// saved copy when it is newer than the data embedded in the build. When
// LLM_MODELS_FILE is set, later processes use that file instead.
func RefreshModelInfo(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsDevURL, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("fetching %s: %w", modelsDevURL, err)
	}
	defer resp.Body.Close() //nolint:errcheck
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fetching %s: unexpected status %s", modelsDevURL, resp.Status)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("reading %s: %w", modelsDevURL, err)
	}

	filtered, err := modelsdev.Filter(body)
	if err != nil {
		return err
	}
	raw, err := modelsdev.Parse(filtered)
	if err != nil {
		return err
	}
	if err := modelInfos.setBase(raw); err != nil {
		return err
	}

	// The refreshed data is already in use; a failure to save it only
	// affects later processes.
	if err := writeModelsDevCache(filtered); err != nil {
		slog.Warn("llms: failed to save refreshed model info", slog.Any("error", err))
	}
	return nil
}

// ensureLoaded loads the models.dev data on first use: LLM_MODELS_FILE when
// set, otherwise the newer of the embedded data and the refresh cache.
func (r *modelInfoRegistry) ensureLoaded() {
	r.loadOnce.Do(func() {
		raw := r.initialData()
		infos := modelInfoFromModelsDev(raw)
		r.mu.Lock()
		// A LoadModelInfo or RefreshModelInfo that ran first wins.
		if r.base == nil {
			r.base = infos
		}
		r.mu.Unlock()
	})
}

func (r *modelInfoRegistry) initialData() modelsdev.RawData {
	if path := os.Getenv(ModelsFileEnv); path != "" {
		raw, err := readModelsDevFile(path)
		if err == nil {
			return raw
		}
		slog.Warn("llms: failed to load model info file, using embedded data",
			slog.String("file", path), slog.Any("error", err))
	}

	raw, err := modelsdev.Parse(embeddedModelsDev)
	if err != nil {
		// The embedded data is generated and committed; a parse failure is
		// a build-time mistake, not a runtime condition.
		panic("llms: failed to parse embedded model info: " + err.Error())
	}

	if path, err := modelsDevCachePath(); err == nil {
		if cached, err := readModelsDevFile(path); err == nil && modelsdev.Version(cached) > modelsdev.Version(raw) {
			return cached
		}
	}
	return raw
}

func (r *modelInfoRegistry) setBase(raw modelsdev.RawData) error {
	infos := modelInfoFromModelsDev(raw)
	if len(infos) == 0 {
		return errors.New("model info has no models for the supported providers")
	}
	// Mark the initial load as done so it cannot replace this data.
	r.loadOnce.Do(func() {})
	r.mu.Lock()
	defer r.mu.Unlock()
	r.base = infos
	return nil
}

func readModelsDevFile(path string) (modelsdev.RawData, error) {
	data, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		return nil, err
	}
	return modelsdev.Parse(data)
}

// writeModelsDevCache saves filtered models.dev data to the cache file. It
// writes a temporary file and renames it, so readers never see a partial file.
func writeModelsDevCache(filtered []byte) error {
	path, err := modelsDevCachePath()
	if err != nil {
		return err
	}
	compressed, err := modelsdev.Compress(filtered)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil { //nolint:gosec
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".models.dev-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) //nolint:errcheck
	if _, err := tmp.Write(compressed); err != nil {
		tmp.Close() //nolint:errcheck,gosec
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// modelInfoRef gives a model its current ModelInfo. A ref with a key reads
// the registry on every call, so RegisterModelInfo, LoadModelInfo and
// RefreshModelInfo apply to models that already exist. A ref without a key
// returns its fixed ModelInfo.
type modelInfoRef struct {
	key   string
	fixed ModelInfo
}

// registeredModelInfo returns a ref that follows the registry entry for key.
func registeredModelInfo(key string) modelInfoRef {
	return modelInfoRef{key: key}
}

// fixedModelInfo returns a ref that always returns info.
func fixedModelInfo(info ModelInfo) modelInfoRef {
	return modelInfoRef{fixed: info}
}

// registered reports whether the ref's model has a RegisterModelInfo entry.
func (r modelInfoRef) registered() bool {
	if r.key == "" {
		return false
	}
	modelInfos.mu.RLock()
	defer modelInfos.mu.RUnlock()
	_, ok := modelInfos.registered[r.key]
	return ok
}

func (r modelInfoRef) get() ModelInfo {
	if r.key == "" {
		return r.fixed
	}
	info, _ := LookupModelInfo(r.key)
	return info
}
