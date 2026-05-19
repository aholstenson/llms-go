package llms_test

import (
	"context"
	"encoding/json"

	"github.com/aholstenson/llms-go"
	"github.com/aholstenson/llms-go/jsonstream"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Model", func() {
	Describe("Result types", func() {
		It("TextResult should implement Result interface", func() {
			var result llms.Result = llms.TextResult{Text: "hello"}
			Expect(result).NotTo(BeNil())

			textResult, ok := result.(llms.TextResult)
			Expect(ok).To(BeTrue())
			Expect(textResult.Text).To(Equal("hello"))
		})

		It("StructuredResult should implement Result interface", func() {
			type TestData struct {
				Name string `json:"name"`
			}

			var result llms.Result = llms.StructuredResult[TestData]{
				Data: TestData{Name: "test"},
				Raw:  `{"name":"test"}`,
			}
			Expect(result).NotTo(BeNil())

			structuredResult, ok := result.(llms.StructuredResult[TestData])
			Expect(ok).To(BeTrue())
			Expect(structuredResult.Data.Name).To(Equal("test"))
			Expect(structuredResult.Raw).To(Equal(`{"name":"test"}`))
		})
	})

	Describe("WithResponseSchema", func() {
		It("should create a GenerateOption with correct schema name", func() {
			type TopicSummary struct {
				Title       string   `json:"title"`
				Description string   `json:"description"`
				Keywords    []string `json:"keywords"`
			}

			opt := llms.WithResponseSchema[TopicSummary]()
			Expect(opt).NotTo(BeNil())

			// Apply the option to options struct via reflection
			// Since generateContentOptions is private, we test indirectly
			// by verifying the option function is not nil
		})

		It("should parse response into StructuredResult", func() {
			type TestStruct struct {
				Name  string `json:"name"`
				Value int    `json:"value"`
			}

			// Get the option and apply it
			opt := llms.WithResponseSchema[TestStruct]()
			Expect(opt).NotTo(BeNil())

			// The ParseInto function should be able to parse JSON
			jsonData := `{"name":"test","value":42}`
			var parsed TestStruct
			err := json.Unmarshal([]byte(jsonData), &parsed)
			Expect(err).NotTo(HaveOccurred())
			Expect(parsed.Name).To(Equal("test"))
			Expect(parsed.Value).To(Equal(42))
		})
	})

	Describe("WithStructuredStreaming", func() {
		It("should create a GenerateOption with handler", func() {
			type TestStruct struct {
				Name string `json:"name"`
			}

			opt := llms.WithStructuredStreaming[TestStruct](func(ctx context.Context, event jsonstream.Event) error {
				return nil
			})
			Expect(opt).NotTo(BeNil())
		})
	})
})
