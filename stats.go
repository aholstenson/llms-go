package llms

import (
	"context"
	"math"
	"sync"
)

// MetricsRecorder aggregates statistics from multiple service calls.
type MetricsRecorder interface {
	RecordSuccess(serviceName string, collector Collector)
	RecordFailure(serviceName string, collector Collector)
}

// Collector accumulates local statistics for a single operation.
type Collector interface {
	Counter(name string) Counter
	GetCounters() map[string]int
}

// Counter represents a single metric that can be incremented.
type Counter interface {
	Add(value int)
}

// CallStats represents the aggregated statistics for an entire call.
type CallStats struct {
	Success map[string]map[string]int `json:"success"` // service -> counter -> value
	Failure map[string]map[string]int `json:"failure"` // service -> counter -> value
	Costs   map[string]float64        `json:"costs"`   // service -> total cost in USD
}

func (c CallStats) Add(other CallStats) CallStats {
	success := copyMap(c.Success)
	failure := copyMap(c.Failure)
	costs := make(map[string]float64)

	// Copy existing costs
	for service, cost := range c.Costs {
		costs[service] = cost
	}

	for service, counters := range other.Success {
		if _, ok := success[service]; !ok {
			success[service] = make(map[string]int)
		}

		for name, value := range counters {
			success[service][name] += value
		}
	}

	for service, counters := range other.Failure {
		if _, ok := failure[service]; !ok {
			failure[service] = make(map[string]int)
		}

		for name, value := range counters {
			failure[service][name] += value
		}
	}

	// Aggregate costs
	for service, cost := range other.Costs {
		totalCost := costs[service] + cost
		// Round the cost to 6 decimal places
		costs[service] = math.Round(totalCost*1000000) / 1000000
	}

	return CallStats{
		Success: success,
		Failure: failure,
		Costs:   costs,
	}
}

func copyMap(m map[string]map[string]int) map[string]map[string]int {
	result := make(map[string]map[string]int)
	for service, counters := range m {
		result[service] = make(map[string]int)
		for name, value := range counters {
			result[service][name] = value
		}
	}
	return result
}

// DefaultMetricsRecorder is the default implementation of MetricsRecorder.
type DefaultMetricsRecorder struct {
	mu      sync.RWMutex
	success map[string]map[string]int
	failure map[string]map[string]int
}

// NewMetricsRecorder creates a new DefaultMetricsRecorder.
func NewMetricsRecorder() *DefaultMetricsRecorder {
	return &DefaultMetricsRecorder{
		success: make(map[string]map[string]int),
		failure: make(map[string]map[string]int),
	}
}

// RecordSuccess records successful operation statistics for a service.
func (d *DefaultMetricsRecorder) RecordSuccess(serviceName string, collector Collector) {
	d.record(d.success, serviceName, collector)
}

// RecordFailure records failed operation statistics for a service.
func (d *DefaultMetricsRecorder) RecordFailure(serviceName string, collector Collector) {
	d.record(d.failure, serviceName, collector)
}

// GetStats returns a copy of the current aggregated statistics.
func (d *DefaultMetricsRecorder) GetStats() CallStats {
	d.mu.RLock()
	defer d.mu.RUnlock()

	success := make(map[string]map[string]int)
	for service, counters := range d.success {
		success[service] = make(map[string]int)
		for name, value := range counters {
			success[service][name] = value
		}
	}

	failure := make(map[string]map[string]int)
	for service, counters := range d.failure {
		failure[service] = make(map[string]int)
		for name, value := range counters {
			failure[service][name] = value
		}
	}

	return CallStats{
		Success: success,
		Failure: failure,
	}
}

func (d *DefaultMetricsRecorder) record(target map[string]map[string]int, serviceName string, collector Collector) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if target[serviceName] == nil {
		target[serviceName] = make(map[string]int)
	}

	counters := collector.GetCounters()
	for name, value := range counters {
		target[serviceName][name] += value
	}
}

// localCollector is the default implementation of Collector.
type localCollector struct {
	mu       sync.RWMutex
	counters map[string]int
}

// NewCollector creates a new local collector for accumulating statistics.
func NewCollector() Collector {
	return &localCollector{
		counters: make(map[string]int),
	}
}

// Counter returns a counter for the given metric name.
func (l *localCollector) Counter(name string) Counter {
	return &localCounter{
		collector: l,
		name:      name,
	}
}

// GetCounters returns a copy of all accumulated counters.
func (l *localCollector) GetCounters() map[string]int {
	l.mu.RLock()
	defer l.mu.RUnlock()

	result := make(map[string]int, len(l.counters))
	for name, value := range l.counters {
		result[name] = value
	}
	return result
}

// localCounter implements Counter for a specific metric.
type localCounter struct {
	collector *localCollector
	name      string
}

// Add increments the counter by the given value.
func (c *localCounter) Add(value int) {
	c.collector.mu.Lock()
	defer c.collector.mu.Unlock()
	c.collector.counters[c.name] += value
}

type metricsKey struct{}

// WithMetrics adds a MetricsRecorder to the context.
func WithMetrics(ctx context.Context, recorder MetricsRecorder) context.Context {
	return context.WithValue(ctx, metricsKey{}, recorder)
}

// GetMetrics retrieves the MetricsRecorder from the context, or nil if not present.
func GetMetrics(ctx context.Context) MetricsRecorder {
	if recorder, ok := ctx.Value(metricsKey{}).(MetricsRecorder); ok {
		return recorder
	}
	return nil
}
