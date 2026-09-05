package memory

import (
	"bufio"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// WordPieceTokenizer is a BERT-style uncased tokenizer.
type WordPieceTokenizer struct {
	vocab   map[string]int64
	unkID   int64
	clsID   int64
	sepID   int64
	padID   int64
	maxLen  int
	maxWord int
}

// LoadWordPiece reads a vocab.txt file.
func LoadWordPiece(path string, maxLen int) (*WordPieceTokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	t := &WordPieceTokenizer{vocab: map[string]int64{}, maxLen: maxLen, maxWord: 100}
	scanner := bufio.NewScanner(f)
	var i int64
	for scanner.Scan() {
		tok := strings.TrimRight(scanner.Text(), "\r\n")
		if _, dup := t.vocab[tok]; !dup {
			t.vocab[tok] = i
		}
		i++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	t.unkID = t.lookup("[UNK]", 100)
	t.clsID = t.lookup("[CLS]", 101)
	t.sepID = t.lookup("[SEP]", 102)
	t.padID = t.lookup("[PAD]", 0)
	return t, nil
}

// NewWordPieceFromVocab builds a tokenizer from an in-memory vocab list.
func NewWordPieceFromVocab(tokens []string, maxLen int) *WordPieceTokenizer {
	t := &WordPieceTokenizer{vocab: map[string]int64{}, maxLen: maxLen, maxWord: 100}
	for i, tok := range tokens {
		t.vocab[tok] = int64(i)
	}
	t.unkID = t.lookup("[UNK]", 100)
	t.clsID = t.lookup("[CLS]", 101)
	t.sepID = t.lookup("[SEP]", 102)
	t.padID = t.lookup("[PAD]", 0)
	return t
}

func (t *WordPieceTokenizer) lookup(tok string, fallback int64) int64 {
	if id, ok := t.vocab[tok]; ok {
		return id
	}
	return fallback
}

// stripAccents lowercases and removes combining marks.
func stripAccents(s string) string {
	s = strings.ToLower(s)
	decomposed := norm.NFD.String(s)
	var b strings.Builder
	for _, r := range decomposed {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func isPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

// basicTokenize splits on whitespace and punctuation (BERT BasicTokenizer).
func basicTokenize(text string) []string {
	text = stripAccents(text)
	var tokens []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case isPunct(r) || unicode.Is(unicode.Han, r):
			flush()
			tokens = append(tokens, string(r))
		case unicode.IsControl(r):
			// drop
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return tokens
}

// Encode returns input ids (with [CLS]/[SEP]) truncated to maxLen.
func (t *WordPieceTokenizer) Encode(text string) []int64 {
	ids := []int64{t.clsID}
	for _, word := range basicTokenize(text) {
		ids = append(ids, t.wordPiece(word)...)
		if len(ids) >= t.maxLen-1 {
			ids = ids[:t.maxLen-1]
			break
		}
	}
	return append(ids, t.sepID)
}

func (t *WordPieceTokenizer) wordPiece(word string) []int64 {
	runes := []rune(word)
	if len(runes) > t.maxWord {
		return []int64{t.unkID}
	}
	var out []int64
	start := 0
	for start < len(runes) {
		end := len(runes)
		var found int64 = -1
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := t.vocab[sub]; ok {
				found = id
				break
			}
			end--
		}
		if found < 0 {
			return []int64{t.unkID}
		}
		out = append(out, found)
		start = end
	}
	return out
}

// PadID returns the padding id.
func (t *WordPieceTokenizer) PadID() int64 { return t.padID }
