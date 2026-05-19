package llms

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"sync"

	"github.com/cockroachdb/errors"
)

// Manager is used to get access to different LLM models, with support for
// dynamic selection based on environment variables.
type Manager struct {
	mu         sync.Mutex
	logger     *slog.Logger
	metrics    *Metrics
	models     map[string]Model
	aliases    map[string]string
	overrides  map[string]string
	subParsers map[string]SubParserConfig
}

func NewManager(logger *slog.Logger, metrics *Metrics) *Manager {
	return &Manager{
		logger:     logger,
		metrics:    metrics,
		models:     make(map[string]Model),
		aliases:    map[string]string{},
		overrides:  make(map[string]string),
		subParsers: make(map[string]SubParserConfig),
	}
}

func (m *Manager) RegisterAlias(alias, target string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.aliases[alias] = target
}

// RegisterSubParser registers a sub-parser config by name. Struct fields
// tagged with `jsonstream:"<name>"` will use this config when building
// streaming schemas for models created by this Manager.
func (m *Manager) RegisterSubParser(name string, config SubParserConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.subParsers[name] = config
}

// subParserRegistrySnapshot returns a copy of the sub-parser registry.
// Must be called with m.mu held.
func (m *Manager) subParserRegistrySnapshot() map[string]SubParserConfig {
	result := make(map[string]SubParserConfig, len(m.subParsers))
	for k, v := range m.subParsers {
		result[k] = v
	}
	return result
}

// RegisterOverride sets a model override that takes priority over both
// environment variables and registered aliases. Use this for explicit
// user-provided flags (e.g. CLI --interactive-model).
func (m *Manager) RegisterOverride(alias, target string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.overrides[alias] = target
}

// ResolveModelName resolves an alias to the actual provider/model name.
func (m *Manager) ResolveModelName(name string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.resolveModelNameLocked(name)
}

func (m *Manager) resolveModelNameLocked(name string) (string, error) {
	current := name
	visited := make(map[string]bool)

	for {
		if strings.Contains(current, "/") {
			return current, nil
		}

		if visited[current] {
			return "", errors.Newf("alias loop detected: %s", name)
		}
		visited[current] = true

		// Check explicit overrides first (CLI flags take highest priority)
		if target, ok := m.overrides[current]; ok {
			current = target
			continue
		}

		// Check environment variables (allows overriding registered aliases)
		envSuffix := strings.ToUpper(strings.ReplaceAll(current, "-", "_"))
		if target := os.Getenv("LLM_MODEL_" + envSuffix); target != "" {
			current = target
			continue
		}

		// Check registered aliases
		if target, ok := m.aliases[current]; ok {
			current = target
			continue
		}

		return "", errors.Newf("could not resolve model name: %s", current)
	}
}

func (m *Manager) GetModel(name string) (Model, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	name, err := m.resolveModelNameLocked(name)
	if err != nil {
		return nil, err
	}

	model, ok := m.models[name]
	if ok {
		// Model has been loaded
		return model, nil
	}

	slashIdx := strings.Index(name, "/")
	if slashIdx == -1 {
		return nil, errors.Newf("invalid model name: %s", name)
	}

	registry := m.subParserRegistrySnapshot()

	modelName := name[slashIdx+1:]
	modelProvider := name[:slashIdx]

	// Embedded model metadata (pricing/capabilities). A miss yields the zero
	// ModelInfo, which the behavior gates treat permissively.
	info, _ := LookupModelInfo(name)

	switch modelProvider {
	case "anthropic":
		apiToken := os.Getenv("ANTHROPIC_API_TOKEN")
		if apiToken == "" {
			return nil, errors.New("ANTHROPIC_API_TOKEN is not set")
		}

		model = NewAnthropicModel(m.logger, m.metrics, apiToken, modelName, registry, info)
	case "openai":
		apiToken := os.Getenv("OPENAI_API_TOKEN")
		if apiToken == "" {
			return nil, errors.New("OPENAI_API_TOKEN is not set")
		}

		model = NewOpenAIModel(m.logger, m.metrics, apiToken, modelName, registry, info)
	case "openrouter":
		apiToken := os.Getenv("OPENROUTER_API_TOKEN")
		if apiToken == "" {
			return nil, errors.New("OPENROUTER_API_TOKEN is not set")
		}

		model = NewOpenRouterModel(m.logger, m.metrics, apiToken, modelName, registry, info)
	case "google":
		apiToken := os.Getenv("GOOGLE_API_TOKEN")
		if apiToken == "" {
			return nil, errors.New("GOOGLE_API_TOKEN is not set")
		}

		var err error
		model, err = NewGoogleModel(m.logger, m.metrics, apiToken, modelName, registry, info)
		if err != nil {
			return nil, err
		}
	case "test":
		model = &testModel{name: modelName}
	default:
		return nil, errors.Newf("unknown model provider: %s", modelProvider)
	}

	m.logger.Info("Loaded model", slog.String("name", name), slog.String("model", modelName))
	m.models[name] = model
	return model, nil
}

type testModel struct {
	name string
}

func (m *testModel) GenerateContent(ctx context.Context, opts ...GenerateOption) (Result, error) {
	return nil, errors.New("test model not implemented")
}
