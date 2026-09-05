package memory

import "testing"

func testTokenizer() *WordPieceTokenizer {
	return NewWordPieceFromVocab([]string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "un", "##affable", "##able", ",", "!"}, 16)
}

func TestWordPieceEncodeBasic(t *testing.T) {
	tok := testTokenizer()

	ids := tok.Encode("Hello, world!")

	// [CLS] hello , world ! [SEP]
	want := []int64{2, 4, 9, 5, 10, 3}
	if len(ids) != len(want) {
		t.Fatalf("got %v", ids)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("got %v want %v", ids, want)
		}
	}
}

func TestWordPieceSubwordsAndUnknown(t *testing.T) {
	tok := testTokenizer()

	ids := tok.Encode("unaffable zzz")

	// [CLS] un ##affable [UNK] [SEP]
	want := []int64{2, 6, 7, 1, 3}
	if len(ids) != len(want) {
		t.Fatalf("got %v", ids)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("got %v want %v", ids, want)
		}
	}
}

func TestWordPieceStripsAccentsAndTruncates(t *testing.T) {
	tok := NewWordPieceFromVocab([]string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"}, 4)

	ids := tok.Encode("héllo hello hello hello")

	if len(ids) != 4 || ids[0] != 2 || ids[1] != 4 || ids[3] != 3 {
		t.Fatalf("got %v", ids)
	}
}
