package llms

import (
	"context"
	"errors"
	"reflect"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// condArgs is a no-op input shape for conditional-tool tests.
type condArgs struct{}

// plainTool is a Tool[*condArgs, string] that does not opt in to ConditionalTool.
type plainTool struct{ name string }

func (t plainTool) Name() string                                           { return t.name }
func (t plainTool) Description() string                                    { return "plain" }
func (t plainTool) Schema() *condArgs                                      { return &condArgs{} }
func (t plainTool) Render(s string) ToolResult                             { return TextToolResult(s) }
func (t plainTool) Execute(_ context.Context, _ *condArgs) (string, error) { return t.name, nil }

// condTool is a Tool[*condArgs, string] that opts in to ConditionalTool.
type condTool struct {
	name  string
	avail bool
	err   error
}

func (t condTool) Name() string                                           { return t.name }
func (t condTool) Description() string                                    { return "conditional" }
func (t condTool) Schema() *condArgs                                      { return &condArgs{} }
func (t condTool) Render(s string) ToolResult                             { return TextToolResult(s) }
func (t condTool) Execute(_ context.Context, _ *condArgs) (string, error) { return t.name, nil }
func (t condTool) IsAvailable(_ context.Context) (bool, error)            { return t.avail, t.err }

var _ = Describe("filterAvailableTools", func() {
	ctx := context.Background()

	It("drops conditional tools that report unavailable", func() {
		keep := NewToolDef[*condArgs, string](condTool{name: "keep", avail: true})
		drop := NewToolDef[*condArgs, string](condTool{name: "drop", avail: false})
		in := []ToolDef{keep, drop}

		out, err := filterAvailableTools(ctx, in)
		Expect(err).NotTo(HaveOccurred())
		Expect(out).To(HaveLen(1))
		Expect(out[0].Name()).To(Equal("keep"))
	})

	It("propagates errors from IsAvailable with the tool name", func() {
		boom := errors.New("boom")
		bad := NewToolDef[*condArgs, string](condTool{name: "bad", err: boom})

		out, err := filterAvailableTools(ctx, []ToolDef{bad})
		Expect(out).To(BeNil())
		Expect(err).To(HaveOccurred())
		Expect(errors.Is(err, boom)).To(BeTrue())
		Expect(err.Error()).To(ContainSubstring(`"bad"`))
	})

	It("returns the input slice unchanged (zero alloc) when nothing is dropped", func() {
		a := NewToolDef[*condArgs, string](plainTool{name: "a"})
		b := NewToolDef[*condArgs, string](condTool{name: "b", avail: true})
		in := []ToolDef{a, b}

		out, err := filterAvailableTools(ctx, in)
		Expect(err).NotTo(HaveOccurred())
		Expect(reflect.ValueOf(out).Pointer()).To(Equal(reflect.ValueOf(in).Pointer()))
	})

	It("preserves the order of remaining tools", func() {
		a := NewToolDef[*condArgs, string](plainTool{name: "a"})
		drop := NewToolDef[*condArgs, string](condTool{name: "drop", avail: false})
		c := NewToolDef[*condArgs, string](plainTool{name: "c"})
		d := NewToolDef[*condArgs, string](condTool{name: "d", avail: true})

		out, err := filterAvailableTools(ctx, []ToolDef{a, drop, c, d})
		Expect(err).NotTo(HaveOccurred())
		names := make([]string, len(out))
		for i, t := range out {
			names[i] = t.Name()
		}
		Expect(names).To(Equal([]string{"a", "c", "d"}))
	})

	It("is a no-op on nil/empty input", func() {
		out, err := filterAvailableTools(ctx, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(out).To(BeNil())

		empty := []ToolDef{}
		out, err = filterAvailableTools(ctx, empty)
		Expect(err).NotTo(HaveOccurred())
		Expect(out).To(HaveLen(0))
	})
})

var _ = Describe("ToolResult builders", func() {
	It("TextToolResult carries text and no attachments", func() {
		r := TextToolResult("done")
		Expect(r.Text).To(Equal("done"))
		Expect(r.Attachments).To(BeEmpty())
	})

	It("WithImage and WithBinary append attachments in order", func() {
		r := TextToolResult("chart").
			WithImage("https://example.com/a.png").
			WithBinary("image/png", []byte{1, 2, 3})

		Expect(r.Text).To(Equal("chart"))
		Expect(r.Attachments).To(Equal([]MessagePart{
			NewImagePart("https://example.com/a.png"),
			NewBinaryPart("image/png", []byte{1, 2, 3}),
		}))
	})

	It("supports an attachment-only result off a bare ToolResult", func() {
		r := ToolResult{}.WithImage("https://example.com/b.png")
		Expect(r.Text).To(BeEmpty())
		Expect(r.Attachments).To(HaveLen(1))
	})

	It("does not mutate the receiver (value semantics)", func() {
		base := TextToolResult("base")
		base.WithImage("https://example.com/c.png")
		Expect(base.Attachments).To(BeEmpty())
	})
})

var _ = Describe("toolWrapper.IsAvailable", func() {
	ctx := context.Background()

	It("returns (true, nil) for tools that don't implement ConditionalTool", func() {
		td := NewToolDef[*condArgs, string](plainTool{name: "plain"})
		c, ok := td.(ConditionalTool)
		Expect(ok).To(BeTrue(), "toolWrapper should always satisfy ConditionalTool")
		ok2, err := c.IsAvailable(ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(ok2).To(BeTrue())
	})

	It("delegates to the wrapped tool when it implements ConditionalTool", func() {
		td := NewToolDef[*condArgs, string](condTool{name: "x", avail: false})
		c := td.(ConditionalTool)
		ok, err := c.IsAvailable(ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(ok).To(BeFalse())
	})
})
