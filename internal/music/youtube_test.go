package music

import "testing"

func TestParseYtDlpTracks(t *testing.T) {
	data := []byte(`{"id":"abc","title":"Song","artists":["A","B"],"duration":201.7,"album":"Alb"}
not json
{"id":"","title":"skipped"}
{"id":"def","title":"Other","uploader":"Uploader - Topic","duration":10}
{"id":"ghi","title":"Third","artist":"Solo","channel":"Chan"}
`)

	tracks := parseYtDlpTracks(data)

	if len(tracks) != 3 {
		t.Fatalf("got %d tracks: %+v", len(tracks), tracks)
	}
	if tracks[0].VideoID != "abc" || tracks[0].Artist != "A, B" || tracks[0].DurationSeconds != 201 || tracks[0].Album != "Alb" || tracks[0].Service != "youtube" {
		t.Fatalf("track0 = %+v", tracks[0])
	}
	if tracks[1].Artist != "Uploader" || tracks[2].Artist != "Solo" {
		t.Fatalf("artists = %q / %q", tracks[1].Artist, tracks[2].Artist)
	}
}

func TestParseYtDlpTracksEmpty(t *testing.T) {
	if len(parseYtDlpTracks(nil)) != 0 {
		t.Fatal("expected no tracks")
	}
}
