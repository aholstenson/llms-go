package llms

import (
	_ "embed"
	"encoding/json"
	"sync"
)

//go:embed modelinfo_data.json
var modelInfoRaw []byte

// modelInfoData lazily parses the embedded model info data exactly once.
var modelInfoData = sync.OnceValue(func() map[string]ModelInfo {
	var data map[string]ModelInfo
	if err := json.Unmarshal(modelInfoRaw, &data); err != nil {
		// The embedded artifact is generated and committed; a parse failure
		// is a build-time mistake, not a runtime condition.
		panic("llms: failed to parse embedded modelinfo_data.json: " + err.Error())
	}
	return data
})
