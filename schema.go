package llms

import (
	"reflect"
	"strings"

	jsonstream "github.com/aholstenson/llms-go/jsonstream"
	"github.com/invopop/jsonschema"
	"google.golang.org/genai"
)

// JSON Schema type constants.
const (
	jsonTypeString  = "string"
	jsonTypeNumber  = "number"
	jsonTypeInteger = "integer"
	jsonTypeBoolean = "boolean"
	jsonTypeArray   = "array"
	jsonTypeObject  = "object"
)

// SubParserConfig configures a sub-parser for a streaming field.
type SubParserConfig interface {
	CreateSubParser() jsonstream.SubParser
}

// StreamSchemaOptions holds configuration options for structured streaming schema generation.
type StreamSchemaOptions struct {
	// SubParsers maps field paths to their sub-parser configurations.
	// Field paths use dot notation for nested fields (e.g., "response.content").
	// Programmatic configuration overrides struct tag configuration.
	SubParsers map[string]SubParserConfig

	// registry maps tag names to sub-parser configs, used to resolve
	// jsonstream struct tags dynamically. Set internally by the Manager.
	registry map[string]SubParserConfig
}

// StructuredStreamingOption configures options for structured streaming.
type StructuredStreamingOption func(*StreamSchemaOptions)

// ConfigureSubParser sets a sub-parser config for a specific field.
// This overrides any struct tag configuration for that field.
// Field paths use dot notation for nested fields (e.g., "response.content").
func ConfigureSubParser(fieldPath string, config SubParserConfig) StructuredStreamingOption {
	return func(opts *StreamSchemaOptions) {
		if opts.SubParsers == nil {
			opts.SubParsers = make(map[string]SubParserConfig)
		}
		opts.SubParsers[fieldPath] = config
	}
}

// ConvertToJsonstreamSchemaFromType generates a jsonstream.Schema from a Go type T
// with streaming enabled for all string fields. It reads jsonstream struct tags
// and resolves them using the provided registry.
//
// Struct tag format: `jsonstream:"<name>"` where name matches a registered sub-parser.
//
// Example:
//
//	type Response struct {
//	    Topic string `json:"topic"`
//	    Text  string `json:"text" jsonstream:"markdown"`
//	}
//
//	// The "markdown" tag is resolved via the registry provided by the Manager.
//	schema := ConvertToJsonstreamSchemaFromType[Response](registry,
//	    ConfigureSubParser("text", customConfig), // optional programmatic override
//	)
func ConvertToJsonstreamSchemaFromType[T any](registry map[string]SubParserConfig, opts ...StructuredStreamingOption) *jsonstream.Schema {
	// Collect options
	options := &StreamSchemaOptions{
		SubParsers: make(map[string]SubParserConfig),
		registry:   registry,
	}
	for _, opt := range opts {
		opt(options)
	}

	// Parse struct tags to find jsonstream configurations
	t := reflect.TypeOf((*T)(nil)).Elem()
	tagConfigs := parseJsonstreamTags(t, "", options.registry)

	// Merge tag configs with programmatic configs (programmatic wins)
	for path, config := range tagConfigs {
		if _, exists := options.SubParsers[path]; !exists {
			options.SubParsers[path] = config
		}
	}

	// Generate the schema using jsonschema reflector, then convert
	schema := jsonSchemaReflector.Reflect(new(T))
	root := convertFieldConfigWithStreaming(schema, schema.Definitions, "", options)

	return &jsonstream.Schema{
		Root:       root,
		StrictMode: false,
	}
}

// parseJsonstreamTags recursively parses struct tags to find jsonstream configurations.
// It resolves tag values using the provided registry and returns a map of field paths
// to their sub-parser configurations.
func parseJsonstreamTags(t reflect.Type, prefix string, registry map[string]SubParserConfig) map[string]SubParserConfig {
	configs := make(map[string]SubParserConfig)

	// Handle pointer types
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return configs
	}

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Get the JSON field name
		jsonTag := field.Tag.Get("json")
		jsonName := strings.Split(jsonTag, ",")[0]
		if jsonName == "" || jsonName == "-" {
			jsonName = field.Name
		}

		// Build the full path
		fieldPath := jsonName
		if prefix != "" {
			fieldPath = prefix + "." + jsonName
		}

		// Check for jsonstream tag and resolve via registry
		jsTag := field.Tag.Get("jsonstream")
		if jsTag != "" {
			if config, ok := registry[jsTag]; ok {
				configs[fieldPath] = config
			}
		}

		// Recursively process nested structs
		fieldType := field.Type
		if fieldType.Kind() == reflect.Ptr {
			fieldType = fieldType.Elem()
		}

		switch fieldType.Kind() { //nolint:exhaustive // Only struct and slice types need recursive processing
		case reflect.Struct:
			nestedConfigs := parseJsonstreamTags(fieldType, fieldPath, registry)
			for path, config := range nestedConfigs {
				configs[path] = config
			}
		case reflect.Slice:
			elemType := fieldType.Elem()
			if elemType.Kind() == reflect.Ptr {
				elemType = elemType.Elem()
			}
			if elemType.Kind() == reflect.Struct {
				// For arrays, we use the same path prefix (items inherit parent path)
				nestedConfigs := parseJsonstreamTags(elemType, fieldPath, registry)
				for path, config := range nestedConfigs {
					configs[path] = config
				}
			}
		}
	}

	return configs
}

// convertFieldConfigWithStreaming converts a jsonschema.Schema to a jsonstream.FieldConfig
// with streaming enabled for string fields.
func convertFieldConfigWithStreaming(schema *jsonschema.Schema, definitions jsonschema.Definitions, path string, options *StreamSchemaOptions) jsonstream.FieldConfig {
	if schema == nil {
		return jsonstream.FieldConfig{Type: jsonstream.TypeAny}
	}

	// Handle $ref - resolve the reference to the actual schema
	if schema.Ref != "" {
		resolved := resolveRef(schema.Ref, definitions)
		if resolved != nil {
			return convertFieldConfigWithStreaming(resolved, definitions, path, options)
		}
		return jsonstream.FieldConfig{Type: jsonstream.TypeAny}
	}

	config := jsonstream.FieldConfig{}

	// Handle type
	switch schema.Type {
	case jsonTypeString:
		config.Type = jsonstream.TypeString
		config.Streaming = true // Enable streaming for all string fields

		// Check if there's a sub-parser configured for this path
		if subParserConfig, exists := options.SubParsers[path]; exists {
			config.SubParser = subParserConfig.CreateSubParser()
		}

	case jsonTypeNumber, jsonTypeInteger:
		config.Type = jsonstream.TypeNumber

	case jsonTypeBoolean:
		config.Type = jsonstream.TypeBoolean

	case jsonTypeArray:
		config.Type = jsonstream.TypeArray
		if schema.Items != nil {
			itemConfig := convertFieldConfigWithStreaming(schema.Items, definitions, path, options)
			config.ItemConfig = &itemConfig
		}

	case jsonTypeObject:
		config.Type = jsonstream.TypeObject
		if schema.Properties != nil {
			config.Children = make(map[string]jsonstream.FieldConfig)
			for pair := schema.Properties.Oldest(); pair != nil; pair = pair.Next() {
				childPath := pair.Key
				if path != "" {
					childPath = path + "." + pair.Key
				}
				config.Children[pair.Key] = convertFieldConfigWithStreaming(pair.Value, definitions, childPath, options)
			}
		}

	default:
		config.Type = jsonstream.TypeAny
	}

	return config
}

// ConvertToJsonstreamSchema converts a jsonschema.Schema to a jsonstream.Schema.
// This enables incremental JSON parsing for structured streaming output.
func ConvertToJsonstreamSchema(schema *jsonschema.Schema) *jsonstream.Schema {
	if schema == nil {
		return &jsonstream.Schema{
			Root:       jsonstream.FieldConfig{Type: jsonstream.TypeAny},
			StrictMode: false,
		}
	}
	root := convertFieldConfig(schema, schema.Definitions)
	return &jsonstream.Schema{
		Root:       root,
		StrictMode: false,
	}
}

// convertFieldConfig converts a jsonschema.Schema to a jsonstream.FieldConfig.
func convertFieldConfig(schema *jsonschema.Schema, definitions jsonschema.Definitions) jsonstream.FieldConfig {
	if schema == nil {
		return jsonstream.FieldConfig{Type: jsonstream.TypeAny}
	}

	// Handle $ref - resolve the reference to the actual schema
	if schema.Ref != "" {
		resolved := resolveRef(schema.Ref, definitions)
		if resolved != nil {
			return convertFieldConfig(resolved, definitions)
		}
		return jsonstream.FieldConfig{Type: jsonstream.TypeAny}
	}

	config := jsonstream.FieldConfig{}

	// Handle type
	switch schema.Type {
	case jsonTypeString:
		config.Type = jsonstream.TypeString
	case jsonTypeNumber, jsonTypeInteger:
		config.Type = jsonstream.TypeNumber
	case jsonTypeBoolean:
		config.Type = jsonstream.TypeBoolean
	case jsonTypeArray:
		config.Type = jsonstream.TypeArray
		if schema.Items != nil {
			itemConfig := convertFieldConfig(schema.Items, definitions)
			config.ItemConfig = &itemConfig
		}
	case jsonTypeObject:
		config.Type = jsonstream.TypeObject
		if schema.Properties != nil {
			config.Children = make(map[string]jsonstream.FieldConfig)
			for pair := schema.Properties.Oldest(); pair != nil; pair = pair.Next() {
				config.Children[pair.Key] = convertFieldConfig(pair.Value, definitions)
			}
		}
	default:
		config.Type = jsonstream.TypeAny
	}

	return config
}

// resolveRef resolves a JSON schema $ref to the actual schema definition.
func resolveRef(ref string, definitions jsonschema.Definitions) *jsonschema.Schema {
	// Handle both #/$defs/Name and #/definitions/Name formats
	if strings.HasPrefix(ref, "#/$defs/") {
		name := strings.TrimPrefix(ref, "#/$defs/")
		if definitions != nil {
			if def, ok := definitions[name]; ok {
				return def
			}
		}
	} else if strings.HasPrefix(ref, "#/definitions/") {
		name := strings.TrimPrefix(ref, "#/definitions/")
		if definitions != nil {
			if def, ok := definitions[name]; ok {
				return def
			}
		}
	}
	return nil
}

// ConvertToGenaiSchema converts a jsonschema.Schema to Google's genai.Schema.
func ConvertToGenaiSchema(schema *jsonschema.Schema) *genai.Schema {
	if schema == nil {
		return nil
	}
	return convertToGenaiSchemaWithDefs(schema, schema.Definitions)
}

// convertToGenaiSchemaWithDefs converts a jsonschema.Schema to Google's genai.Schema,
// resolving $ref references using the provided definitions.
func convertToGenaiSchemaWithDefs(schema *jsonschema.Schema, definitions jsonschema.Definitions) *genai.Schema {
	if schema == nil {
		return nil
	}

	// Handle $ref - resolve the reference to the actual schema
	if schema.Ref != "" {
		resolved := resolveRef(schema.Ref, definitions)
		if resolved != nil {
			return convertToGenaiSchemaWithDefs(resolved, definitions)
		}
		return nil
	}

	result := &genai.Schema{
		Description: schema.Description,
	}

	// Handle type
	switch schema.Type {
	case jsonTypeString:
		result.Type = genai.TypeString
		if len(schema.Enum) > 0 {
			for _, e := range schema.Enum {
				if s, ok := e.(string); ok {
					result.Enum = append(result.Enum, s)
				}
			}
		}
	case jsonTypeNumber:
		result.Type = genai.TypeNumber
	case jsonTypeInteger:
		result.Type = genai.TypeInteger
	case jsonTypeBoolean:
		result.Type = genai.TypeBoolean
	case jsonTypeArray:
		result.Type = genai.TypeArray
		if schema.Items != nil {
			result.Items = convertToGenaiSchemaWithDefs(schema.Items, definitions)
		}
	case jsonTypeObject:
		result.Type = genai.TypeObject
		if schema.Properties != nil {
			result.Properties = make(map[string]*genai.Schema)
			for pair := schema.Properties.Oldest(); pair != nil; pair = pair.Next() {
				result.Properties[pair.Key] = convertToGenaiSchemaWithDefs(pair.Value, definitions)
			}
		}
		// Handle required fields
		if len(schema.Required) > 0 {
			result.Required = schema.Required
		}
	default:
		// Default to string for unknown types
		result.Type = genai.TypeString
	}

	return result
}
