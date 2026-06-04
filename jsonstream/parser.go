package jsonstream

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// isIncompleteUTF8Start checks if the byte at the start of data could be the
// beginning of an incomplete multi-byte UTF-8 sequence given the available bytes.
// Returns true if data starts with a valid multi-byte sequence leader but doesn't
// have enough bytes to complete the sequence.
//
// UTF-8 encoding:
//   - 0x00-0x7F: 1-byte (ASCII) - always complete
//   - 0xC0-0xDF: 2-byte sequence start
//   - 0xE0-0xEF: 3-byte sequence start
//   - 0xF0-0xF7: 4-byte sequence start
//   - 0x80-0xBF: Continuation bytes
func isIncompleteUTF8Start(data string) bool {
	if len(data) == 0 {
		return false
	}

	b := data[0]

	// Determine expected sequence length from first byte
	var expected int
	if b&0x80 == 0 {
		// ASCII (0xxxxxxx) - always complete
		return false
	} else if b&0xC0 == 0x80 {
		// Continuation byte (10xxxxxx) without leader - invalid, not incomplete
		return false
	} else if b&0xE0 == 0xC0 {
		expected = 2 // 110xxxxx
	} else if b&0xF0 == 0xE0 {
		expected = 3 // 1110xxxx
	} else if b&0xF8 == 0xF0 {
		expected = 4 // 11110xxx
	} else {
		// Invalid start byte (0xF8-0xFF)
		return false
	}

	// Check if we have fewer bytes than needed
	return len(data) < expected
}

// parseState represents the current state of the parser state machine.
type parseState int

const (
	stateRoot parseState = iota
	stateValue
	stateObject
	stateObjectKey
	stateObjectColon
	stateObjectValue
	stateObjectComma
	stateArray
	stateArrayValue
	stateArrayComma
	stateString
	stateStringEscape
	stateStringUnicode
	stateStringSurrogateBackslash // Expecting '\' after high surrogate
	stateStringSurrogateU         // Expecting 'u' after '\'
	stateStringSurrogateLow       // Parsing low surrogate hex digits
	stateNumber
	stateLiteral // true, false, null
)

// stateFrame represents a level in the parsing stack.
type stateFrame struct {
	state         parseState
	pathComponent string // Field name or array index as string
	config        *FieldConfig
	buffer        strings.Builder
	arrayIdx      int
	subParser     SubParser
	objectValue   map[string]any // For building object values
	arrayValue    []any          // For building array values
	currentKey    string         // Current object key being parsed
	unicodeBuffer string         // Buffer for unicode escape sequences
	highSurrogate rune           // High surrogate for UTF-16 surrogate pairs
}

// Parser is an incremental JSON parser that emits events based on a schema.
type Parser struct {
	schema       *Schema
	stack        []*stateFrame
	buffer       string
	offset       int
	events       []Event
	pendingChunk string // Buffer for incomplete string chunks
}

// New creates a new Parser with the given schema.
func New(schema *Schema) *Parser {
	p := &Parser{
		schema: schema,
	}
	p.Reset()
	return p
}

// Reset clears the parser state for reuse.
func (p *Parser) Reset() {
	p.stack = []*stateFrame{{
		state:  stateRoot,
		config: &p.schema.Root,
	}}
	p.buffer = ""
	p.offset = 0
	p.events = nil
	p.pendingChunk = ""
}

// Feed processes a chunk of JSON and returns any events generated.
func (p *Parser) Feed(chunk string) ([]Event, error) {
	p.buffer += chunk
	p.events = nil

	if err := p.parse(); err != nil {
		return p.events, err
	}

	return p.events, nil
}

// Flush signals the end of input and returns any final events.
func (p *Parser) Flush() ([]Event, error) {
	p.events = nil

	// Flush any remaining buffered content
	if err := p.flushPending(); err != nil {
		return p.events, err
	}

	return p.events, nil
}

func (p *Parser) parse() error {
	for len(p.buffer) > 0 {
		if len(p.stack) == 0 {
			// Parsing complete, ignore any trailing content
			p.buffer = ""
			return nil
		}

		frame := p.stack[len(p.stack)-1]
		prevState := frame.state
		prevStackLen := len(p.stack)

		consumed, err := p.processState(frame)
		if err != nil {
			return err
		}

		// Check if we made progress
		stateChanged := len(p.stack) != prevStackLen || (len(p.stack) > 0 && p.stack[len(p.stack)-1].state != prevState)
		if consumed == 0 && !stateChanged {
			// No progress made, need more input
			break
		}

		if consumed > 0 {
			p.buffer = p.buffer[consumed:]
			p.offset += consumed
		}
	}
	return nil
}

func (p *Parser) processState(frame *stateFrame) (int, error) {
	switch frame.state {
	case stateRoot:
		return p.processRoot(frame)
	case stateValue:
		return p.processValue(frame)
	case stateObject:
		return p.processObject(frame)
	case stateObjectKey:
		return p.processObjectKey(frame)
	case stateObjectColon:
		return p.processObjectColon(frame)
	case stateObjectValue:
		return p.processObjectValue(frame)
	case stateObjectComma:
		return p.processObjectComma(frame)
	case stateArray:
		return p.processArray(frame)
	case stateArrayValue:
		return p.processArrayValue(frame)
	case stateArrayComma:
		return p.processArrayComma(frame)
	case stateString:
		return p.processString(frame)
	case stateStringEscape:
		return p.processStringEscape(frame)
	case stateStringUnicode:
		return p.processStringUnicode(frame)
	case stateStringSurrogateBackslash:
		return p.processStringSurrogateBackslash(frame)
	case stateStringSurrogateU:
		return p.processStringSurrogateU(frame)
	case stateStringSurrogateLow:
		return p.processStringSurrogateLow(frame)
	case stateNumber:
		return p.processNumber(frame)
	case stateLiteral:
		return p.processLiteral(frame)
	default:
		return 0, fmt.Errorf("unknown state: %d", frame.state)
	}
}

func (p *Parser) processRoot(frame *stateFrame) (int, error) {
	consumed := p.skipWhitespace()
	if consumed == len(p.buffer) {
		return consumed, nil
	}

	frame.state = stateValue
	return consumed, nil
}

func (p *Parser) processValue(frame *stateFrame) (int, error) {
	consumed := p.skipWhitespace()
	if consumed >= len(p.buffer) {
		return consumed, nil
	}

	ch := p.buffer[consumed]
	switch ch {
	case '{':
		frame.state = stateObject
		frame.objectValue = make(map[string]any)
		p.emitEvent(EventObjectStart{path: p.currentPath()})
		return consumed + 1, nil

	case '[':
		frame.state = stateArray
		frame.arrayValue = nil
		frame.arrayIdx = 0
		p.emitEvent(EventArrayStart{path: p.currentPath()})
		return consumed + 1, nil

	case '"':
		frame.state = stateString
		frame.buffer.Reset()
		if frame.config != nil && frame.config.SubParser != nil {
			frame.subParser = frame.config.SubParser
			frame.subParser.Reset()
		}
		p.emitEvent(EventFieldStart{path: p.currentPath(), FieldType: TypeString})
		return consumed + 1, nil

	case 't', 'f', 'n':
		frame.state = stateLiteral
		frame.buffer.Reset()
		return consumed, nil

	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		frame.state = stateNumber
		frame.buffer.Reset()
		return consumed, nil

	default:
		return consumed, newParseError(p.currentPath(), p.offset+consumed, "unexpected character", ErrInvalidJSON)
	}
}

func (p *Parser) processObject(frame *stateFrame) (int, error) {
	consumed := p.skipWhitespace()
	if consumed >= len(p.buffer) {
		return consumed, nil
	}

	ch := p.buffer[consumed]
	switch ch {
	case '}':
		// End of object
		p.emitEvent(EventObjectComplete{path: p.currentPath(), Value: frame.objectValue})
		p.popFrame(frame.objectValue)
		return consumed + 1, nil

	case '"':
		// Start of key
		frame.state = stateObjectKey
		frame.buffer.Reset()
		return consumed + 1, nil

	default:
		return consumed, newParseError(p.currentPath(), p.offset+consumed, "expected '\"' or '}'", ErrInvalidJSON)
	}
}

func (p *Parser) processObjectKey(frame *stateFrame) (int, error) {
	consumed := 0
	for consumed < len(p.buffer) {
		ch := p.buffer[consumed]
		switch ch {
		case '"':
			frame.currentKey = frame.buffer.String()
			frame.state = stateObjectColon
			return consumed + 1, nil
		case '\\':
			if consumed+1 >= len(p.buffer) {
				return consumed, nil // Need more input
			}
			consumed++
			nextCh := p.buffer[consumed]
			if nextCh == 'u' {
				// Unicode escape: \uXXXX
				consumed++
				if consumed+4 > len(p.buffer) {
					// Not enough hex digits yet, back up to the backslash
					return consumed - 2, nil
				}
				hex := p.buffer[consumed : consumed+4]
				for _, h := range hex {
					if !isHexDigit(byte(h)) {
						return consumed, newParseError(p.currentPath(), p.offset+consumed, "invalid unicode escape", ErrInvalidJSON)
					}
				}
				codePoint, _ := strconv.ParseInt(hex, 16, 32)
				frame.buffer.WriteRune(rune(codePoint))
				consumed += 3 // +1 happens at end of loop
			} else {
				escaped, err := p.unescapeChar(nextCh)
				if err != nil {
					return consumed, err
				}
				frame.buffer.WriteRune(escaped)
			}
		default:
			frame.buffer.WriteByte(ch)
		}
		consumed++
	}
	return consumed, nil
}

func (p *Parser) processObjectColon(frame *stateFrame) (int, error) {
	consumed := p.skipWhitespace()
	if consumed >= len(p.buffer) {
		return consumed, nil
	}

	if p.buffer[consumed] != ':' {
		return consumed, newParseError(p.currentPath(), p.offset+consumed, "expected ':'", ErrInvalidJSON)
	}

	frame.state = stateObjectValue

	// Push a new frame for the value
	var valueConfig *FieldConfig
	if frame.config != nil && frame.config.Children != nil {
		if cfg, ok := frame.config.Children[frame.currentKey]; ok {
			valueConfig = &cfg
		} else if p.schema.StrictMode {
			return consumed, newParseError(p.currentPath(), p.offset+consumed,
				fmt.Sprintf("unexpected field %q", frame.currentKey), ErrUnexpectedField)
		}
	}

	p.stack = append(p.stack, &stateFrame{
		state:         stateValue,
		pathComponent: frame.currentKey,
		config:        valueConfig,
	})

	return consumed + 1, nil
}

func (p *Parser) processObjectValue(frame *stateFrame) (int, error) {
	// This state is reached after a value frame completes
	consumed := p.skipWhitespace()
	if consumed >= len(p.buffer) {
		return consumed, nil
	}

	ch := p.buffer[consumed]
	switch ch {
	case ',':
		frame.state = stateObjectComma
		return consumed + 1, nil
	case '}':
		p.emitEvent(EventObjectComplete{path: p.currentPath(), Value: frame.objectValue})
		p.popFrame(frame.objectValue)
		return consumed + 1, nil
	default:
		return consumed, newParseError(p.currentPath(), p.offset+consumed, "expected ',' or '}'", ErrInvalidJSON)
	}
}

func (p *Parser) processObjectComma(frame *stateFrame) (int, error) {
	consumed := p.skipWhitespace()
	if consumed >= len(p.buffer) {
		return consumed, nil
	}

	if p.buffer[consumed] == '"' {
		frame.state = stateObjectKey
		frame.buffer.Reset()
		return consumed + 1, nil
	}

	// Allow trailing comma (lenient mode)
	if p.buffer[consumed] == '}' {
		p.emitEvent(EventObjectComplete{path: p.currentPath(), Value: frame.objectValue})
		p.popFrame(frame.objectValue)
		return consumed + 1, nil
	}

	return consumed, newParseError(p.currentPath(), p.offset+consumed, "expected '\"' or '}'", ErrInvalidJSON)
}

func (p *Parser) processArray(frame *stateFrame) (int, error) {
	consumed := p.skipWhitespace()
	if consumed >= len(p.buffer) {
		return consumed, nil
	}

	ch := p.buffer[consumed]
	if ch == ']' {
		// Empty array
		p.emitEvent(EventArrayEnd{path: p.currentPath(), Count: 0})
		p.popFrame(frame.arrayValue)
		return consumed + 1, nil
	}

	frame.state = stateArrayValue

	// Push a new frame for the first value
	var itemConfig *FieldConfig
	if frame.config != nil && frame.config.ItemConfig != nil {
		itemConfig = frame.config.ItemConfig
	}

	p.stack = append(p.stack, &stateFrame{
		state:         stateValue,
		pathComponent: strconv.Itoa(frame.arrayIdx),
		config:        itemConfig,
	})

	return consumed, nil
}

func (p *Parser) processArrayValue(frame *stateFrame) (int, error) {
	// This state is reached after a value frame completes
	consumed := p.skipWhitespace()
	if consumed >= len(p.buffer) {
		return consumed, nil
	}

	ch := p.buffer[consumed]
	switch ch {
	case ',':
		frame.state = stateArrayComma
		frame.arrayIdx++
		return consumed + 1, nil
	case ']':
		p.emitEvent(EventArrayEnd{path: p.currentPath(), Count: len(frame.arrayValue)})
		p.popFrame(frame.arrayValue)
		return consumed + 1, nil
	default:
		return consumed, newParseError(p.currentPath(), p.offset+consumed, "expected ',' or ']'", ErrInvalidJSON)
	}
}

func (p *Parser) processArrayComma(frame *stateFrame) (int, error) {
	consumed := p.skipWhitespace()
	if consumed >= len(p.buffer) {
		return consumed, nil
	}

	// Allow trailing comma (lenient mode)
	if p.buffer[consumed] == ']' {
		p.emitEvent(EventArrayEnd{path: p.currentPath(), Count: len(frame.arrayValue)})
		p.popFrame(frame.arrayValue)
		return consumed + 1, nil
	}

	frame.state = stateArrayValue

	// Push a new frame for the next value
	var itemConfig *FieldConfig
	if frame.config != nil && frame.config.ItemConfig != nil {
		itemConfig = frame.config.ItemConfig
	}

	p.stack = append(p.stack, &stateFrame{
		state:         stateValue,
		pathComponent: strconv.Itoa(frame.arrayIdx),
		config:        itemConfig,
	})

	return consumed, nil
}

func (p *Parser) processString(frame *stateFrame) (int, error) {
	consumed := 0
	streaming := frame.config != nil && frame.config.Streaming
	var chunkBuilder strings.Builder

	for consumed < len(p.buffer) {
		ch := p.buffer[consumed]
		if ch == '"' {
			// End of string
			if streaming {
				if chunkBuilder.Len() > 0 || p.pendingChunk != "" {
					chunk := p.pendingChunk + chunkBuilder.String()
					p.pendingChunk = ""
					if frame.subParser != nil {
						events, ferr := frame.subParser.Feed(chunk)
						for _, ev := range events {
							p.emitSubParserEvent(ev)
						}
						if ferr != nil {
							return consumed, newParseError(p.currentPath(), p.offset+consumed, "sub-parser feed failed", ferr)
						}
					} else {
						p.emitEvent(EventStringChunk{path: p.currentPath(), Chunk: chunk})
					}
				}
				if frame.subParser != nil {
					flushEvents, flerr := frame.subParser.Flush()
					for _, ev := range flushEvents {
						p.emitSubParserEvent(ev)
					}
					if flerr != nil {
						return consumed, newParseError(p.currentPath(), p.offset+consumed, "sub-parser flush failed", flerr)
					}
				}
			}

			value := frame.buffer.String()
			p.emitEvent(EventStringComplete{path: p.currentPath(), Value: value})
			p.emitEvent(EventFieldEnd{path: p.currentPath()})
			p.popFrame(value)
			return consumed + 1, nil
		} else if ch == '\\' {
			// Emit accumulated content before transitioning to escape state
			if streaming && chunkBuilder.Len() > 0 {
				chunk := p.pendingChunk + chunkBuilder.String()
				p.pendingChunk = ""
				if frame.subParser != nil {
					events, ferr := frame.subParser.Feed(chunk)
					for _, ev := range events {
						p.emitSubParserEvent(ev)
					}
					if ferr != nil {
						return consumed, newParseError(p.currentPath(), p.offset+consumed, "sub-parser feed failed", ferr)
					}
				} else {
					p.emitEvent(EventStringChunk{path: p.currentPath(), Chunk: chunk})
				}
			}
			frame.state = stateStringEscape
			return consumed + 1, nil
		} else if ch < 0x20 {
			// Control characters are not allowed in strings
			return consumed, newParseError(p.currentPath(), p.offset+consumed, "control character in string", ErrInvalidJSON)
		} else {
			// Regular character
			remaining := p.buffer[consumed:]
			r, size := utf8.DecodeRuneInString(remaining)
			if r == utf8.RuneError && size == 1 {
				// Check if this could be an incomplete UTF-8 sequence at the end of the buffer
				// If so, we should wait for more data rather than erroring
				if isIncompleteUTF8Start(remaining) {
					// Incomplete UTF-8 sequence - stop here and wait for more data
					// The incomplete bytes will remain in the buffer for the next Feed()
					break
				}
				return consumed, newParseError(p.currentPath(), p.offset+consumed, "invalid UTF-8", ErrInvalidJSON)
			}
			frame.buffer.WriteRune(r)
			if streaming {
				chunkBuilder.WriteRune(r)
			}
			consumed += size
		}
	}

	// If we're streaming and have accumulated content, emit it
	if streaming && chunkBuilder.Len() > 0 {
		chunk := p.pendingChunk + chunkBuilder.String()
		p.pendingChunk = ""
		if frame.subParser != nil {
			events, ferr := frame.subParser.Feed(chunk)
			for _, ev := range events {
				p.emitSubParserEvent(ev)
			}
			if ferr != nil {
				return consumed, newParseError(p.currentPath(), p.offset+consumed, "sub-parser feed failed", ferr)
			}
		} else {
			p.emitEvent(EventStringChunk{path: p.currentPath(), Chunk: chunk})
		}
	}

	return consumed, nil
}

func (p *Parser) processStringEscape(frame *stateFrame) (int, error) {
	if len(p.buffer) == 0 {
		return 0, nil
	}

	ch := p.buffer[0]
	streaming := frame.config != nil && frame.config.Streaming

	if ch == 'u' {
		frame.state = stateStringUnicode
		frame.unicodeBuffer = ""
		return 1, nil
	}

	escaped, err := p.unescapeChar(ch)
	if err != nil {
		return 0, err
	}

	frame.buffer.WriteRune(escaped)
	if streaming {
		p.pendingChunk += string(escaped)
	}
	frame.state = stateString
	return 1, nil
}

func (p *Parser) processStringUnicode(frame *stateFrame) (int, error) {
	consumed := 0
	for consumed < len(p.buffer) && len(frame.unicodeBuffer) < 4 {
		ch := p.buffer[consumed]
		if !isHexDigit(ch) {
			return consumed, newParseError(p.currentPath(), p.offset+consumed, "invalid unicode escape", ErrInvalidJSON)
		}
		frame.unicodeBuffer += string(ch)
		consumed++
	}

	if len(frame.unicodeBuffer) < 4 {
		return consumed, nil // Need more input
	}

	codePoint, _ := strconv.ParseInt(frame.unicodeBuffer, 16, 32)
	r := rune(codePoint)

	// Check for UTF-16 high surrogate (U+D800 to U+DBFF)
	if r >= 0xD800 && r <= 0xDBFF {
		// Store high surrogate and expect low surrogate
		frame.highSurrogate = r
		frame.state = stateStringSurrogateBackslash
		return consumed, nil
	}

	// Check for unpaired low surrogate (invalid)
	if r >= 0xDC00 && r <= 0xDFFF {
		// Unpaired low surrogate - emit replacement character
		r = utf8.RuneError
	}

	streaming := frame.config != nil && frame.config.Streaming
	frame.buffer.WriteRune(r)
	if streaming {
		p.pendingChunk += string(r)
	}
	frame.state = stateString
	return consumed, nil
}

func (p *Parser) processStringSurrogateBackslash(frame *stateFrame) (int, error) {
	if len(p.buffer) == 0 {
		return 0, nil // Need more input
	}

	ch := p.buffer[0]
	if ch == '\\' {
		frame.state = stateStringSurrogateU
		return 1, nil
	}

	// Not a backslash - emit the unpaired high surrogate as replacement character
	// and reprocess this character
	streaming := frame.config != nil && frame.config.Streaming
	frame.buffer.WriteRune(utf8.RuneError)
	if streaming {
		p.pendingChunk += string(utf8.RuneError)
	}
	frame.highSurrogate = 0
	frame.state = stateString
	return 0, nil // Don't consume - let stateString handle it
}

func (p *Parser) processStringSurrogateU(frame *stateFrame) (int, error) {
	if len(p.buffer) == 0 {
		return 0, nil // Need more input
	}

	ch := p.buffer[0]
	if ch == 'u' {
		frame.unicodeBuffer = ""
		frame.state = stateStringSurrogateLow
		return 1, nil
	}

	// Not 'u' - emit the unpaired high surrogate as replacement character
	// and handle this as a regular escape sequence
	streaming := frame.config != nil && frame.config.Streaming
	frame.buffer.WriteRune(utf8.RuneError)
	if streaming {
		p.pendingChunk += string(utf8.RuneError)
	}
	frame.highSurrogate = 0
	frame.state = stateStringEscape
	return 0, nil // Don't consume - let stateStringEscape handle the escape char
}

func (p *Parser) processStringSurrogateLow(frame *stateFrame) (int, error) {
	consumed := 0
	for consumed < len(p.buffer) && len(frame.unicodeBuffer) < 4 {
		ch := p.buffer[consumed]
		if !isHexDigit(ch) {
			return consumed, newParseError(p.currentPath(), p.offset+consumed, "invalid unicode escape", ErrInvalidJSON)
		}
		frame.unicodeBuffer += string(ch)
		consumed++
	}

	if len(frame.unicodeBuffer) < 4 {
		return consumed, nil // Need more input
	}

	codePoint, _ := strconv.ParseInt(frame.unicodeBuffer, 16, 32)
	lowSurrogate := rune(codePoint)

	var r rune
	// Check if this is a valid low surrogate (U+DC00 to U+DFFF)
	if lowSurrogate >= 0xDC00 && lowSurrogate <= 0xDFFF {
		// Combine surrogate pair into the actual code point
		// Formula: 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
		r = 0x10000 + ((frame.highSurrogate - 0xD800) << 10) + (lowSurrogate - 0xDC00)
	} else {
		// Not a valid low surrogate - emit replacement for high surrogate
		// and then handle this code point normally
		streaming := frame.config != nil && frame.config.Streaming
		frame.buffer.WriteRune(utf8.RuneError)
		if streaming {
			p.pendingChunk += string(utf8.RuneError)
		}
		// The low surrogate code point is just a regular character (or invalid)
		if lowSurrogate >= 0xD800 && lowSurrogate <= 0xDBFF {
			// It's another high surrogate - start waiting for its low surrogate
			frame.highSurrogate = lowSurrogate
			frame.state = stateStringSurrogateBackslash
			return consumed, nil
		}
		r = lowSurrogate
	}

	frame.highSurrogate = 0
	streaming := frame.config != nil && frame.config.Streaming
	frame.buffer.WriteRune(r)
	if streaming {
		p.pendingChunk += string(r)
	}
	frame.state = stateString
	return consumed, nil
}

func (p *Parser) processNumber(frame *stateFrame) (int, error) {
	consumed := 0
	for consumed < len(p.buffer) {
		ch := p.buffer[consumed]
		if isNumberChar(ch) {
			frame.buffer.WriteByte(ch)
			consumed++
		} else {
			break
		}
	}

	// Check if we might need more input
	if consumed == len(p.buffer) && len(p.buffer) > 0 {
		// Number might continue, need more input
		return consumed, nil
	}

	// Parse the number
	numStr := frame.buffer.String()
	if numStr == "" || numStr == "-" {
		return 0, nil // Need more input
	}

	value, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return consumed, newParseError(p.currentPath(), p.offset, "invalid number", ErrInvalidJSON)
	}

	p.emitEvent(EventNumber{path: p.currentPath(), Value: value})
	p.popFrame(value)
	return consumed, nil
}

func (p *Parser) processLiteral(frame *stateFrame) (int, error) {
	consumed := 0
	for consumed < len(p.buffer) {
		ch := p.buffer[consumed]
		if ch >= 'a' && ch <= 'z' {
			frame.buffer.WriteByte(ch)
			consumed++
		} else {
			break
		}
	}

	lit := frame.buffer.String()

	// Check if we have a complete literal
	switch lit {
	case "true":
		p.emitEvent(EventBoolean{path: p.currentPath(), Value: true})
		p.popFrame(true)
		return consumed, nil
	case "false":
		p.emitEvent(EventBoolean{path: p.currentPath(), Value: false})
		p.popFrame(false)
		return consumed, nil
	case "null":
		p.emitEvent(EventNull{path: p.currentPath()})
		p.popFrame(nil)
		return consumed, nil
	}

	// If we stopped because of a non-letter character (not end-of-buffer),
	// the literal is terminated and can't grow. A prefix at this point is
	// malformed (e.g., "tru " can never become "true").
	if consumed < len(p.buffer) {
		return consumed, newParseError(p.currentPath(), p.offset, "invalid literal", ErrInvalidJSON)
	}

	// Ran out of buffer — check if the accumulated text is a valid prefix
	// that could still complete with more input.
	if isLiteralPrefix(lit) {
		return consumed, nil
	}

	return consumed, newParseError(p.currentPath(), p.offset, "invalid literal", ErrInvalidJSON)
}

func (p *Parser) skipWhitespace() int {
	consumed := 0
	for consumed < len(p.buffer) {
		ch := p.buffer[consumed]
		if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
			consumed++
		} else {
			break
		}
	}
	return consumed
}

func (p *Parser) unescapeChar(ch byte) (rune, error) {
	switch ch {
	case '"':
		return '"', nil
	case '\\':
		return '\\', nil
	case '/':
		return '/', nil
	case 'b':
		return '\b', nil
	case 'f':
		return '\f', nil
	case 'n':
		return '\n', nil
	case 'r':
		return '\r', nil
	case 't':
		return '\t', nil
	default:
		return 0, newParseError(p.currentPath(), p.offset, fmt.Sprintf("invalid escape character: %c", ch), ErrInvalidJSON)
	}
}

func (p *Parser) currentPath() string {
	var parts []string
	for _, frame := range p.stack {
		if frame.pathComponent != "" {
			// Check if it's an array index
			if _, err := strconv.Atoi(frame.pathComponent); err == nil {
				parts = append(parts, "["+frame.pathComponent+"]")
			} else {
				parts = append(parts, frame.pathComponent)
			}
		}
	}

	// Build path with proper separators
	var result strings.Builder
	for i, part := range parts {
		if strings.HasPrefix(part, "[") {
			result.WriteString(part)
		} else {
			if i > 0 {
				result.WriteString(".")
			}
			result.WriteString(part)
		}
	}
	return result.String()
}

func (p *Parser) popFrame(value any) {
	if len(p.stack) == 0 {
		return
	}

	p.stack = p.stack[:len(p.stack)-1]

	if len(p.stack) == 0 {
		return
	}

	parent := p.stack[len(p.stack)-1]

	// Store value in parent
	switch parent.state { //nolint:exhaustive
	case stateObjectValue:
		if parent.objectValue != nil {
			parent.objectValue[parent.currentKey] = value
		}
	case stateArrayValue:
		parent.arrayValue = append(parent.arrayValue, value)
		// Emit array item event
		p.emitEvent(EventArrayItem{
			path:  p.currentPath(),
			Index: len(parent.arrayValue) - 1,
			Value: value,
		})
	}
}

func (p *Parser) emitEvent(event Event) {
	p.events = append(p.events, event)
}

// emitSubParserEvent wraps a sub-parser event with path context and emits it.
func (p *Parser) emitSubParserEvent(event any) {
	path := p.currentPath()
	p.emitEvent(EventParsedStringChunk{path: path, Chunk: event})
}

func (p *Parser) flushPending() error {
	// Handle any remaining buffered content
	if len(p.stack) == 0 {
		return nil
	}
	frame := p.stack[len(p.stack)-1]
	if frame.state != stateString || frame.config == nil || !frame.config.Streaming {
		return nil
	}
	if p.pendingChunk == "" && frame.subParser == nil {
		return nil
	}
	if frame.subParser != nil {
		var feedErr error
		if p.pendingChunk != "" {
			var events []any
			events, feedErr = frame.subParser.Feed(p.pendingChunk)
			for _, ev := range events {
				p.emitSubParserEvent(ev)
			}
		}
		p.pendingChunk = ""
		if feedErr != nil {
			return newParseError(p.currentPath(), p.offset, "sub-parser feed failed", feedErr)
		}
		flushEvents, flushErr := frame.subParser.Flush()
		for _, ev := range flushEvents {
			p.emitSubParserEvent(ev)
		}
		if flushErr != nil {
			return newParseError(p.currentPath(), p.offset, "sub-parser flush failed", flushErr)
		}
		return nil
	}
	p.emitEvent(EventStringChunk{path: p.currentPath(), Chunk: p.pendingChunk})
	p.pendingChunk = ""
	return nil
}

func isHexDigit(ch byte) bool {
	return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F')
}

func isNumberChar(ch byte) bool {
	return (ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E'
}

func isLiteralPrefix(s string) bool {
	literals := []string{"true", "false", "null"}
	for _, lit := range literals {
		if strings.HasPrefix(lit, s) {
			return true
		}
	}
	return false
}
