package jsonstream_test

import (
	"log/slog"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestJsonstream(t *testing.T) {
	RegisterFailHandler(Fail)
	slog.SetDefault(slog.New(slog.NewTextHandler(GinkgoWriter, nil)))
	RunSpecs(t, "Jsonstream Suite")
}
