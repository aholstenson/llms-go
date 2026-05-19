package llms

// Toolkit is a composable collection of tools.
type Toolkit interface {
	Tools() []ToolDef
}

// WithToolkits sets tools from one or more toolkits.
func WithToolkits(toolkits ...Toolkit) GenerateOption {
	totalLen := 0
	for _, tk := range toolkits {
		totalLen += len(tk.Tools())
	}

	tools := make([]ToolDef, 0, totalLen)
	for _, tk := range toolkits {
		tools = append(tools, tk.Tools()...)
	}
	return WithTools(tools...)
}
