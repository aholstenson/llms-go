package llms_test

import (
	"log/slog"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestLlms(t *testing.T) {
	RegisterFailHandler(Fail)
	slog.SetDefault(slog.New(slog.NewTextHandler(GinkgoWriter, nil)))
	RunSpecs(t, "Llms Suite")
}
