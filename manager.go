package llms

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
)

// Manager is used to get access to different LLM models, with support for
// dynamic selection based on environment variables.
type Manager struct {
	mu          sync.Mutex
	logger      *slog.Logger
	metrics     *Metrics
	credentials CredentialSource
	models      map[string]Model
	aliases     map[string]string
	overrides   map[string]string
	subParsers  map[string]SubParserConfig
}

// ManagerOption configures a Manager during construction.
type ManagerOption func(*Manager) error

// WithManagerLogger sets the logger used by the Manager and the models it
// creates. Defaults to slog.Default() if not provided.
func WithManagerLogger(logger *slog.Logger) ManagerOption {
	return func(m *Manager) error {
		m.logger = logger
		return nil
	}
}

// WithManagerMetrics sets the metrics used by the Manager and the models it
// creates. Defaults to NewNoopMetrics() if not provided.
func WithManagerMetrics(metrics *Metrics) ManagerOption {
	return func(m *Manager) error {
		m.metrics = metrics
		return nil
	}
}

// WithManagerCredentials sets the CredentialSource used to authenticate
// requests for the models this Manager creates. Defaults to EnvCredentials,
// which reads the conventional per-provider environment variables.
//
// The source is consulted once per model at construction and then once per
// outbound HTTP request, so credentials can rotate without the models being
// rebuilt. See CredentialSource for the full contract.
func WithManagerCredentials(credentials CredentialSource) ManagerOption {
	return func(m *Manager) error {
		if credentials == nil {
			return errors.New("credential source must not be nil")
		}
		m.credentials = credentials
		return nil
	}
}

func NewManager(opts ...ManagerOption) (*Manager, error) {
	m := &Manager{
		logger:      slog.Default(),
		metrics:     NewNoopMetrics(),
		credentials: &EnvCredentials{},
		models:      make(map[string]Model),
		aliases:     map[string]string{},
		overrides:   make(map[string]string),
		subParsers:  make(map[string]SubParserConfig),
	}
	for _, opt := range opts {
		if err := opt(m); err != nil {
			return nil, err
		}
	}
	return m, nil
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
			return "", fmt.Errorf("alias loop detected: %s", name)
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

		return "", fmt.Errorf("could not resolve model name: %s", current)
	}
}

func (m *Manager) GetModel(ctx context.Context, name string) (Model, error) {
	m.mu.Lock()
	resolved, err := m.resolveModelNameLocked(name)
	if err != nil {
		m.mu.Unlock()
		return nil, err
	}
	if model, ok := m.models[resolved]; ok {
		m.mu.Unlock()
		return model, nil
	}
	registry := m.subParserRegistrySnapshot()
	logger := m.logger
	metrics := m.metrics
	credentials := m.credentials
	m.mu.Unlock()

	slashIdx := strings.Index(resolved, "/")
	if slashIdx == -1 {
		return nil, fmt.Errorf("invalid model name: %s", resolved)
	}

	modelName := resolved[slashIdx+1:]
	modelProvider := resolved[:slashIdx]

	// Embedded model metadata (pricing/capabilities). A miss yields the zero
	// ModelInfo, which the behavior gates treat permissively.
	info, _ := LookupModelInfo(resolved)

	// Probe the credential source so a missing credential fails here rather
	// than as an opaque transport error on the first generation. The models
	// themselves resolve credentials per request, not from this result.
	if modelProvider != "test" {
		if _, err := credentials.Credential(ctx, modelProvider); err != nil {
			return nil, fmt.Errorf("credentials for %s: %w", resolved, err)
		}
	}

	var model Model
	switch modelProvider {
	case "anthropic":
		model = newAnthropicModel(logger, metrics, credentials, modelName, registry, info)
	case "openai":
		model = newOpenAIModel(logger, metrics, credentials, modelName, registry, info)
	case "openrouter":
		model, err = newOpenRouterModel(logger, metrics, credentials, modelName, registry, info)
		if err != nil {
			return nil, err
		}
	case "google":
		model, err = newGoogleModel(ctx, logger, metrics, credentials, modelName, registry, info)
		if err != nil {
			return nil, err
		}
	case "test":
		model = &testModel{name: modelName}
	default:
		return nil, fmt.Errorf("unknown model provider: %s", modelProvider)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if existing, ok := m.models[resolved]; ok {
		// Another goroutine built the same model while we were constructing
		// ours; discard ours and return the winner.
		return existing, nil
	}
	logger.Info("Loaded model", slog.String("name", resolved), slog.String("model", modelName))
	m.models[resolved] = model
	return model, nil
}

type testModel struct {
	name string
}

func (m *testModel) GenerateContent(ctx context.Context, opts ...GenerateOption) (Result, error) {
	return nil, errors.New("test model not implemented")
}
