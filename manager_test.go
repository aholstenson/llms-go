package llms_test

import (
	"io"
	"log/slog"

	"github.com/aholstenson/llms-go"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Manager", func() {
	var manager *llms.Manager
	var logger *slog.Logger
	var metrics *llms.Metrics

	BeforeEach(func() {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
		metrics = llms.NewNoopMetrics()
		manager = llms.NewManager(logger, metrics)

		// Clear relevant env vars using GinkgoT() for automatic cleanup
		GinkgoT().Setenv("LLM_MODEL_TESTING", "")
		GinkgoT().Setenv("LLM_MODEL_RECURSIVE", "")
		GinkgoT().Setenv("LLM_MODEL_LEGACY", "")
	})

	Describe("RegisterAlias", func() {
		It("should allow registering and resolving an alias", func() {
			manager.RegisterAlias("testing", "test/model-name")

			model, err := manager.GetModel("testing")
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
		})
	})

	Describe("RegisterOverride", func() {
		It("should take priority over environment variables", func() {
			GinkgoT().Setenv("LLM_MODEL_TESTING", "test/env-model")
			manager.RegisterOverride("testing", "test/override-model")

			resolved, err := manager.ResolveModelName("testing")
			Expect(err).NotTo(HaveOccurred())
			Expect(resolved).To(Equal("test/override-model"))
		})

		It("should take priority over registered aliases", func() {
			manager.RegisterAlias("testing", "test/alias-model")
			manager.RegisterOverride("testing", "test/override-model")

			resolved, err := manager.ResolveModelName("testing")
			Expect(err).NotTo(HaveOccurred())
			Expect(resolved).To(Equal("test/override-model"))
		})
	})

	Describe("Dynamic Lookup", func() {
		It("should resolve aliases recursively", func() {
			manager.RegisterAlias("alias1", "alias2")
			manager.RegisterAlias("alias2", "test/model-name")

			model, err := manager.GetModel("alias1")
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
		})

		It("should detect alias loops", func() {
			manager.RegisterAlias("loop1", "loop2")
			manager.RegisterAlias("loop2", "loop1")

			_, err := manager.GetModel("loop1")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("alias loop detected"))
		})

		It("should resolve from environment variables", func() {
			GinkgoT().Setenv("LLM_MODEL_TESTING", "test/model-name")

			model, err := manager.GetModel("testing")
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
		})

		It("should allow environment variables to override registered aliases", func() {
			manager.RegisterAlias("testing", "openai/gpt-4")         // Register as openai
			GinkgoT().Setenv("LLM_MODEL_TESTING", "test/model-name") // Override with test

			model, err := manager.GetModel("testing")
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
		})

		It("should resolve recursively through environment variables and aliases", func() {
			manager.RegisterAlias("start", "env_alias")
			GinkgoT().Setenv("LLM_MODEL_ENV_ALIAS", "test/model-name")

			model, err := manager.GetModel("start")
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
		})
	})
})
