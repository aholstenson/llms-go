package jsonstream

import "context"

// Handler is an interface for processing parser events.
type Handler interface {
	HandleEvent(ctx context.Context, event Event) error
}

// HandlerFunc is a function that implements Handler.
type HandlerFunc func(ctx context.Context, event Event) error

// HandleEvent implements Handler.
func (f HandlerFunc) HandleEvent(ctx context.Context, event Event) error {
	return f(ctx, event)
}

// FilteredHandler wraps a Handler and only forwards events matching specified paths.
type FilteredHandler struct {
	paths   map[string]bool
	handler Handler
}

// NewFilteredHandler creates a new FilteredHandler that forwards events
// for the specified paths to the wrapped handler.
func NewFilteredHandler(handler Handler, paths ...string) *FilteredHandler {
	pathMap := make(map[string]bool)
	for _, path := range paths {
		pathMap[path] = true
	}
	return &FilteredHandler{
		paths:   pathMap,
		handler: handler,
	}
}

// HandleEvent implements Handler, filtering events by path.
func (f *FilteredHandler) HandleEvent(ctx context.Context, event Event) error {
	if f.paths[event.Path()] {
		return f.handler.HandleEvent(ctx, event)
	}
	return nil
}

// MultiHandler wraps multiple handlers and forwards events to all of them.
type MultiHandler struct {
	handlers []Handler
}

// NewMultiHandler creates a new MultiHandler that forwards events to all handlers.
func NewMultiHandler(handlers ...Handler) *MultiHandler {
	return &MultiHandler{handlers: handlers}
}

// HandleEvent implements Handler, forwarding to all wrapped handlers.
func (m *MultiHandler) HandleEvent(ctx context.Context, event Event) error {
	for _, h := range m.handlers {
		if err := h.HandleEvent(ctx, event); err != nil {
			return err
		}
	}
	return nil
}

// ChannelHandler sends events to a channel.
type ChannelHandler struct {
	ch chan<- Event
}

// NewChannelHandler creates a new ChannelHandler that sends events to the given channel.
func NewChannelHandler(ch chan<- Event) *ChannelHandler {
	return &ChannelHandler{ch: ch}
}

// HandleEvent implements Handler, sending events to the channel.
// It respects context cancellation, returning ctx.Err() if the context is canceled
// before the event can be sent.
func (c *ChannelHandler) HandleEvent(ctx context.Context, event Event) error {
	select {
	case c.ch <- event:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
