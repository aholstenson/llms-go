package llms

import (
	"bytes"
	"context"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aholstenson/llms-go/internal/modelsdev"
)

// isolateModelInfos gives a test its own cache file and restores the model
// info registry, the models.dev URL and the cache path when the test ends.
// It returns the cache file path.
func isolateModelInfos(t *testing.T) string {
	t.Helper()
	modelInfos.ensureLoaded()

	modelInfos.mu.Lock()
	base, registered := modelInfos.base, maps.Clone(modelInfos.registered)
	modelInfos.mu.Unlock()
	url, cachePath := modelsDevURL, modelsDevCachePath

	path := filepath.Join(t.TempDir(), "models.dev.json.gz")
	modelsDevCachePath = func() (string, error) { return path, nil }

	t.Cleanup(func() {
		modelInfos.mu.Lock()
		modelInfos.base, modelInfos.registered = base, registered
		modelInfos.mu.Unlock()
		modelsDevURL, modelsDevCachePath = url, cachePath
	})
	return path
}

// modelsDevDoc returns a small models.dev document with one OpenAI model.
func modelsDevDoc(id, lastUpdated string) []byte {
	return []byte(`{"openai":{"id":"openai","models":{"` + id + `":{"id":"` + id +
		`","description":"dropped","last_updated":"` + lastUpdated +
		`","tool_call":true,"cost":{"input":1,"output":2}}}}}`)
}

func TestRegisterModelInfoWinsAndAppliesToExistingRefs(t *testing.T) {
	isolateModelInfos(t)

	ref := registeredModelInfo("openai/gpt-4.1")
	if ref.get().Cost.Input == 42 {
		t.Fatal("unexpected starting price")
	}

	RegisterModelInfo("openai/gpt-4.1", ModelInfo{Cost: Cost{Input: 42}})
	if got := ref.get().Cost.Input; got != 42 {
		t.Errorf("existing ref should see the registered entry, got input price %v", got)
	}

	RegisterModelInfo("custom/model", ModelInfo{Family: "custom"})
	if info, ok := LookupModelInfo("custom/model"); !ok || info.Family != "custom" {
		t.Errorf("expected registered custom model, got %+v ok=%v", info, ok)
	}
}

func TestLoadModelInfoReplacesModelsDevData(t *testing.T) {
	isolateModelInfos(t)
	RegisterModelInfo("openai/registered", ModelInfo{Family: "kept"})

	if err := LoadModelInfo(bytes.NewReader(modelsDevDoc("gpt-loaded", "2030-01-01"))); err != nil {
		t.Fatalf("LoadModelInfo: %v", err)
	}
	if _, ok := LookupModelInfo("openai/gpt-loaded"); !ok {
		t.Error("expected loaded model")
	}
	if _, ok := LookupModelInfo("openai/gpt-4.1"); ok {
		t.Error("expected embedded models to be replaced")
	}
	if _, ok := LookupModelInfo("openai/registered"); !ok {
		t.Error("expected registered models to survive a load")
	}
}

func TestLoadModelInfoRejectsEmptyData(t *testing.T) {
	isolateModelInfos(t)

	err := LoadModelInfo(strings.NewReader(`{"deepseek":{"models":{"x":{}}}}`))
	if err == nil {
		t.Fatal("expected an error for data without supported providers")
	}
	if _, ok := LookupModelInfo("openai/gpt-4.1"); !ok {
		t.Error("a failed load must keep the current data")
	}
}

func TestRefreshModelInfoFetchesAndCaches(t *testing.T) {
	cachePath := isolateModelInfos(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(modelsDevDoc("gpt-refreshed", "2030-01-01"))
	}))
	defer srv.Close()
	modelsDevURL = srv.URL

	if err := RefreshModelInfo(context.Background()); err != nil {
		t.Fatalf("RefreshModelInfo: %v", err)
	}
	if _, ok := LookupModelInfo("openai/gpt-refreshed"); !ok {
		t.Error("expected refreshed model to be in use")
	}

	cached, err := readModelsDevFile(cachePath)
	if err != nil {
		t.Fatalf("reading cache: %v", err)
	}
	if _, ok := cached["openai"].Models["gpt-refreshed"]; !ok {
		t.Error("expected refreshed model in the cache file")
	}
	data, _ := os.ReadFile(cachePath) //nolint:gosec
	if bytes.Contains(data, []byte("dropped")) {
		t.Error("expected the cache to hold filtered data")
	}
}

func TestRefreshModelInfoKeepsDataOnHTTPError(t *testing.T) {
	isolateModelInfos(t)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()
	modelsDevURL = srv.URL

	if err := RefreshModelInfo(context.Background()); err == nil {
		t.Fatal("expected an error for a failed fetch")
	}
	if _, ok := LookupModelInfo("openai/gpt-4.1"); !ok {
		t.Error("a failed refresh must keep the current data")
	}
}

func TestInitialDataUsesModelsFile(t *testing.T) {
	isolateModelInfos(t)

	path := filepath.Join(t.TempDir(), "api.json")
	if err := os.WriteFile(path, modelsDevDoc("gpt-from-file", "2000-01-01"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(ModelsFileEnv, path)

	raw := (&modelInfoRegistry{}).initialData()
	if _, ok := raw["openai"].Models["gpt-from-file"]; !ok {
		t.Error("expected LLM_MODELS_FILE data, even when older than the embedded data")
	}
}

func TestInitialDataFallsBackWhenModelsFileIsBad(t *testing.T) {
	isolateModelInfos(t)
	t.Setenv(ModelsFileEnv, filepath.Join(t.TempDir(), "missing.json"))

	raw := (&modelInfoRegistry{}).initialData()
	if _, ok := raw["openai"].Models["gpt-4.1"]; !ok {
		t.Error("expected the embedded data when LLM_MODELS_FILE cannot be read")
	}
}

func TestInitialDataUsesNewerCacheOnly(t *testing.T) {
	cachePath := isolateModelInfos(t)
	t.Setenv(ModelsFileEnv, "")

	write := func(doc []byte) {
		t.Helper()
		compressed, err := modelsdev.Compress(doc)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(cachePath, compressed, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	write(modelsDevDoc("gpt-old-cache", "2000-01-01"))
	if raw := (&modelInfoRegistry{}).initialData(); raw["openai"].Models["gpt-old-cache"].ID != "" {
		t.Error("an older cache must not replace the embedded data")
	}

	write(modelsDevDoc("gpt-new-cache", "2999-01-01"))
	if raw := (&modelInfoRegistry{}).initialData(); raw["openai"].Models["gpt-new-cache"].ID == "" {
		t.Error("a newer cache should replace the embedded data")
	}
}
