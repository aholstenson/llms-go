package llms_test

import (
	"github.com/aholstenson/llms-go"
	jsonstream "github.com/aholstenson/llms-go/jsonstream"
	"github.com/invopop/jsonschema"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"google.golang.org/genai"
)

// mockSubParserConfig implements SubParserConfig for testing.
type mockSubParserConfig struct{}

func (m *mockSubParserConfig) CreateSubParser() jsonstream.SubParser {
	return &mockSubParser{}
}

type mockSubParser struct{}

func (m *mockSubParser) Feed(chunk string) ([]any, error) { return nil, nil }
func (m *mockSubParser) Flush() ([]any, error)            { return nil, nil }
func (m *mockSubParser) Reset()                            {}

// testRegistry returns a sub-parser registry with a "markdown" entry for testing.
func testRegistry() map[string]llms.SubParserConfig {
	return map[string]llms.SubParserConfig{
		"markdown": &mockSubParserConfig{},
	}
}

var _ = Describe("Schema", func() {
	Describe("ConvertToJsonstreamSchemaFromType", func() {
		It("should enable streaming on all string fields", func() {
			type TestStruct struct {
				Name        string `json:"name"`
				Description string `json:"description"`
			}

			jsSchema := llms.ConvertToJsonstreamSchemaFromType[TestStruct](nil)

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Type).To(Equal(jsonstream.TypeObject))
			Expect(jsSchema.Root.Children).To(HaveKey("name"))
			Expect(jsSchema.Root.Children).To(HaveKey("description"))
			Expect(jsSchema.Root.Children["name"].Streaming).To(BeTrue())
			Expect(jsSchema.Root.Children["description"].Streaming).To(BeTrue())
		})

		It("should resolve sub-parser for fields with jsonstream tag via registry", func() {
			type TestStruct struct {
				Topic string `json:"topic"`
				Text  string `json:"text" jsonstream:"markdown"`
			}

			jsSchema := llms.ConvertToJsonstreamSchemaFromType[TestStruct](testRegistry())

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Children["topic"].SubParser).To(BeNil())
			Expect(jsSchema.Root.Children["text"].SubParser).NotTo(BeNil())
		})

		It("should ignore unregistered jsonstream tags", func() {
			type TestStruct struct {
				Text string `json:"text" jsonstream:"unknown"`
			}

			jsSchema := llms.ConvertToJsonstreamSchemaFromType[TestStruct](testRegistry())

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Children["text"].SubParser).To(BeNil())
			Expect(jsSchema.Root.Children["text"].Streaming).To(BeTrue())
		})

		It("should allow ConfigureSubParser to override struct tags", func() {
			type TestStruct struct {
				Text string `json:"text" jsonstream:"markdown"`
			}

			customConfig := &mockSubParserConfig{}
			jsSchema := llms.ConvertToJsonstreamSchemaFromType[TestStruct](
				testRegistry(),
				llms.ConfigureSubParser("text", customConfig),
			)

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Children["text"].SubParser).NotTo(BeNil())
		})

		It("should handle nested objects with correct field paths", func() {
			type Inner struct {
				Content string `json:"content" jsonstream:"markdown"`
			}
			type Outer struct {
				Title    string `json:"title"`
				Response Inner  `json:"response"`
			}

			jsSchema := llms.ConvertToJsonstreamSchemaFromType[Outer](testRegistry())

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Children["title"].Streaming).To(BeTrue())
			Expect(jsSchema.Root.Children["title"].SubParser).To(BeNil())

			responseConfig := jsSchema.Root.Children["response"]
			Expect(responseConfig.Type).To(Equal(jsonstream.TypeObject))
			Expect(responseConfig.Children["content"].Streaming).To(BeTrue())
			Expect(responseConfig.Children["content"].SubParser).NotTo(BeNil())
		})

		It("should allow ConfigureSubParser with dot notation for nested fields", func() {
			type Inner struct {
				Content string `json:"content"`
			}
			type Outer struct {
				Response Inner `json:"response"`
			}

			jsSchema := llms.ConvertToJsonstreamSchemaFromType[Outer](
				nil,
				llms.ConfigureSubParser("response.content", &mockSubParserConfig{}),
			)

			Expect(jsSchema).NotTo(BeNil())
			responseConfig := jsSchema.Root.Children["response"]
			Expect(responseConfig.Children["content"].SubParser).NotTo(BeNil())
		})

		It("should handle arrays of objects", func() {
			type Item struct {
				Name        string `json:"name"`
				Description string `json:"description" jsonstream:"markdown"`
			}
			type Container struct {
				Items []Item `json:"items"`
			}

			jsSchema := llms.ConvertToJsonstreamSchemaFromType[Container](testRegistry())

			Expect(jsSchema).NotTo(BeNil())
			itemsConfig := jsSchema.Root.Children["items"]
			Expect(itemsConfig.Type).To(Equal(jsonstream.TypeArray))
			Expect(itemsConfig.ItemConfig).NotTo(BeNil())
			Expect(itemsConfig.ItemConfig.Type).To(Equal(jsonstream.TypeObject))
			Expect(itemsConfig.ItemConfig.Children["name"].Streaming).To(BeTrue())
			Expect(itemsConfig.ItemConfig.Children["description"].Streaming).To(BeTrue())
			Expect(itemsConfig.ItemConfig.Children["description"].SubParser).NotTo(BeNil())
		})

		It("should not enable streaming on non-string fields", func() {
			type TestStruct struct {
				Count   int     `json:"count"`
				Price   float64 `json:"price"`
				Enabled bool    `json:"enabled"`
			}

			jsSchema := llms.ConvertToJsonstreamSchemaFromType[TestStruct](nil)

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Children["count"].Streaming).To(BeFalse())
			Expect(jsSchema.Root.Children["price"].Streaming).To(BeFalse())
			Expect(jsSchema.Root.Children["enabled"].Streaming).To(BeFalse())
		})

		It("should handle arrays of strings", func() {
			type TestStruct struct {
				Tags []string `json:"tags"`
			}

			jsSchema := llms.ConvertToJsonstreamSchemaFromType[TestStruct](nil)

			Expect(jsSchema).NotTo(BeNil())
			tagsConfig := jsSchema.Root.Children["tags"]
			Expect(tagsConfig.Type).To(Equal(jsonstream.TypeArray))
			Expect(tagsConfig.ItemConfig).NotTo(BeNil())
			Expect(tagsConfig.ItemConfig.Type).To(Equal(jsonstream.TypeString))
			Expect(tagsConfig.ItemConfig.Streaming).To(BeTrue())
		})
	})

	Describe("ConvertToJsonstreamSchema", func() {
		It("should convert a simple object schema", func() {
			// Create a jsonschema for a simple struct
			type TestStruct struct {
				Name string `json:"name"`
				Age  int    `json:"age"`
			}

			reflector := jsonschema.Reflector{}
			schema := reflector.Reflect(&TestStruct{})

			jsSchema := llms.ConvertToJsonstreamSchema(schema)

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Type).To(Equal(jsonstream.TypeObject))
			Expect(jsSchema.Root.Children).To(HaveKey("name"))
			Expect(jsSchema.Root.Children).To(HaveKey("age"))
			Expect(jsSchema.Root.Children["name"].Type).To(Equal(jsonstream.TypeString))
			Expect(jsSchema.Root.Children["age"].Type).To(Equal(jsonstream.TypeNumber))
		})

		It("should convert an array schema", func() {
			type TestStruct struct {
				Tags []string `json:"tags"`
			}

			reflector := jsonschema.Reflector{}
			schema := reflector.Reflect(&TestStruct{})

			jsSchema := llms.ConvertToJsonstreamSchema(schema)

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Type).To(Equal(jsonstream.TypeObject))
			Expect(jsSchema.Root.Children).To(HaveKey("tags"))
			Expect(jsSchema.Root.Children["tags"].Type).To(Equal(jsonstream.TypeArray))
			Expect(jsSchema.Root.Children["tags"].ItemConfig).NotTo(BeNil())
			Expect(jsSchema.Root.Children["tags"].ItemConfig.Type).To(Equal(jsonstream.TypeString))
		})

		It("should convert nested object schema", func() {
			type Inner struct {
				Value string `json:"value"`
			}
			type Outer struct {
				Inner Inner `json:"inner"`
			}

			reflector := jsonschema.Reflector{}
			schema := reflector.Reflect(&Outer{})

			jsSchema := llms.ConvertToJsonstreamSchema(schema)

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Type).To(Equal(jsonstream.TypeObject))
			Expect(jsSchema.Root.Children).To(HaveKey("inner"))
			Expect(jsSchema.Root.Children["inner"].Type).To(Equal(jsonstream.TypeObject))
			Expect(jsSchema.Root.Children["inner"].Children).To(HaveKey("value"))
		})

		It("should handle nil schema", func() {
			jsSchema := llms.ConvertToJsonstreamSchema(nil)

			Expect(jsSchema).NotTo(BeNil())
			Expect(jsSchema.Root.Type).To(Equal(jsonstream.TypeAny))
		})
	})

	Describe("ConvertToGenaiSchema", func() {
		It("should convert a simple object schema", func() {
			type TestStruct struct {
				Name        string `json:"name"`
				Description string `json:"description"`
			}

			reflector := jsonschema.Reflector{}
			schema := reflector.Reflect(&TestStruct{})

			genaiSchema := llms.ConvertToGenaiSchema(schema)

			Expect(genaiSchema).NotTo(BeNil())
			Expect(genaiSchema.Type).To(Equal(genai.TypeObject))
			Expect(genaiSchema.Properties).To(HaveKey("name"))
			Expect(genaiSchema.Properties).To(HaveKey("description"))
			Expect(genaiSchema.Properties["name"].Type).To(Equal(genai.TypeString))
			Expect(genaiSchema.Properties["description"].Type).To(Equal(genai.TypeString))
		})

		It("should convert numeric types", func() {
			type TestStruct struct {
				Count   int     `json:"count"`
				Price   float64 `json:"price"`
				Enabled bool    `json:"enabled"`
			}

			reflector := jsonschema.Reflector{}
			schema := reflector.Reflect(&TestStruct{})

			genaiSchema := llms.ConvertToGenaiSchema(schema)

			Expect(genaiSchema).NotTo(BeNil())
			Expect(genaiSchema.Properties["count"].Type).To(Equal(genai.TypeInteger))
			Expect(genaiSchema.Properties["price"].Type).To(Equal(genai.TypeNumber))
			Expect(genaiSchema.Properties["enabled"].Type).To(Equal(genai.TypeBoolean))
		})

		It("should convert array types", func() {
			type TestStruct struct {
				Items []string `json:"items"`
			}

			reflector := jsonschema.Reflector{}
			schema := reflector.Reflect(&TestStruct{})

			genaiSchema := llms.ConvertToGenaiSchema(schema)

			Expect(genaiSchema).NotTo(BeNil())
			Expect(genaiSchema.Properties["items"].Type).To(Equal(genai.TypeArray))
			Expect(genaiSchema.Properties["items"].Items).NotTo(BeNil())
			Expect(genaiSchema.Properties["items"].Items.Type).To(Equal(genai.TypeString))
		})

		It("should handle nil schema", func() {
			genaiSchema := llms.ConvertToGenaiSchema(nil)
			Expect(genaiSchema).To(BeNil())
		})
	})
})
