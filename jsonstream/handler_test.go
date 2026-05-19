package jsonstream_test

import (
	"context"
	"time"

	jsonstream "github.com/aholstenson/llms-go/jsonstream"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ChannelHandler", func() {
	It("should send events to the channel", func() {
		ch := make(chan jsonstream.Event, 1)
		handler := jsonstream.NewChannelHandler(ch)

		event := jsonstream.EventObjectStart{}
		err := handler.HandleEvent(context.Background(), event)
		Expect(err).NotTo(HaveOccurred())

		select {
		case received := <-ch:
			Expect(received).To(Equal(event))
		case <-time.After(100 * time.Millisecond):
			Fail("event not received")
		}
	})

	It("should return context error when context is canceled", func() {
		ch := make(chan jsonstream.Event) // unbuffered, will block
		handler := jsonstream.NewChannelHandler(ch)

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		event := jsonstream.EventObjectStart{}
		err := handler.HandleEvent(ctx, event)
		Expect(err).To(Equal(context.Canceled))
	})

	It("should return deadline exceeded when context times out", func() {
		ch := make(chan jsonstream.Event) // unbuffered, will block
		handler := jsonstream.NewChannelHandler(ch)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()

		event := jsonstream.EventObjectStart{}
		err := handler.HandleEvent(ctx, event)
		Expect(err).To(Equal(context.DeadlineExceeded))
	})

	It("should not block when channel has capacity", func() {
		ch := make(chan jsonstream.Event, 10)
		handler := jsonstream.NewChannelHandler(ch)

		for i := 0; i < 10; i++ {
			err := handler.HandleEvent(context.Background(), jsonstream.EventObjectStart{})
			Expect(err).NotTo(HaveOccurred())
		}

		Expect(ch).To(HaveLen(10))
	})
})

var _ = Describe("FilteredHandler", func() {
	It("should only forward events matching specified paths", func() {
		var received []jsonstream.Event
		inner := jsonstream.HandlerFunc(func(_ context.Context, event jsonstream.Event) error {
			received = append(received, event)
			return nil
		})

		handler := jsonstream.NewFilteredHandler(inner, "name", "items[0]")

		events := []jsonstream.Event{
			jsonstream.EventStringComplete{Value: "test"}, // path ""
			jsonstream.EventNumber{Value: 42},             // path ""
		}

		// These events don't have paths set, so they won't match
		for _, e := range events {
			err := handler.HandleEvent(context.Background(), e)
			Expect(err).NotTo(HaveOccurred())
		}

		// Only events with matching paths should be forwarded
		Expect(received).To(BeEmpty())
	})
})

var _ = Describe("MultiHandler", func() {
	It("should forward events to all handlers", func() {
		var received1, received2 []jsonstream.Event

		handler1 := jsonstream.HandlerFunc(func(_ context.Context, event jsonstream.Event) error {
			received1 = append(received1, event)
			return nil
		})
		handler2 := jsonstream.HandlerFunc(func(_ context.Context, event jsonstream.Event) error {
			received2 = append(received2, event)
			return nil
		})

		multi := jsonstream.NewMultiHandler(handler1, handler2)

		event := jsonstream.EventObjectStart{}
		err := multi.HandleEvent(context.Background(), event)
		Expect(err).NotTo(HaveOccurred())

		Expect(received1).To(HaveLen(1))
		Expect(received2).To(HaveLen(1))
	})

	It("should stop on first error", func() {
		var callCount int

		handler1 := jsonstream.HandlerFunc(func(_ context.Context, event jsonstream.Event) error {
			callCount++
			return context.Canceled
		})
		handler2 := jsonstream.HandlerFunc(func(_ context.Context, event jsonstream.Event) error {
			callCount++
			return nil
		})

		multi := jsonstream.NewMultiHandler(handler1, handler2)

		err := multi.HandleEvent(context.Background(), jsonstream.EventObjectStart{})
		Expect(err).To(Equal(context.Canceled))
		Expect(callCount).To(Equal(1)) // Only first handler was called
	})

	It("should handle empty handler list", func() {
		multi := jsonstream.NewMultiHandler()

		err := multi.HandleEvent(context.Background(), jsonstream.EventObjectStart{})
		Expect(err).NotTo(HaveOccurred())
	})
})

var _ = Describe("FilteredHandler matching paths", func() {
	It("should forward events that match specified paths", func() {
		// Use the parser to generate events with real paths
		schema := &jsonstream.Schema{
			Root: jsonstream.FieldConfig{
				Type: jsonstream.TypeObject,
				Children: map[string]jsonstream.FieldConfig{
					"name":  {Type: jsonstream.TypeString},
					"count": {Type: jsonstream.TypeNumber},
				},
			},
		}
		p := jsonstream.New(schema)

		events, err := p.Feed(`{"name": "test", "count": 42}`)
		Expect(err).NotTo(HaveOccurred())

		var received []jsonstream.Event
		inner := jsonstream.HandlerFunc(func(_ context.Context, event jsonstream.Event) error {
			received = append(received, event)
			return nil
		})

		// Only listen for "name" path
		handler := jsonstream.NewFilteredHandler(inner, "name")

		for _, e := range events {
			err := handler.HandleEvent(context.Background(), e)
			Expect(err).NotTo(HaveOccurred())
		}

		// Should have received events for "name" path only
		Expect(received).NotTo(BeEmpty())
		for _, e := range received {
			Expect(e.Path()).To(Equal("name"))
		}
	})
})
