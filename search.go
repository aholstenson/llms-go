package llms

type SearchToolArgs interface {
	Query() string
}

type SearchToolResult interface {
	Count() int
	Items() []SearchToolResultItem
}

type SearchToolResultItem interface {
	Title() string
	URL() string
}

type searchToolArgs struct {
	RawQuery string `json:"query"`
}

func (s *searchToolArgs) Query() string {
	return s.RawQuery
}

type searchToolResult struct {
	count int
	items []SearchToolResultItem
}

func (s *searchToolResult) Count() int {
	return s.count
}

func (s *searchToolResult) Items() []SearchToolResultItem {
	return s.items
}
