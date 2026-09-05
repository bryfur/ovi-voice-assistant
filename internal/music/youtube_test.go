package music

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestParseTracks(t *testing.T) {
	data := []byte(`{"id":"abc","title":"Song","artists":["A","B"],"duration":201.7,"album":"Alb"}
not json
{"id":"","title":"skipped"}
{"id":"def","title":"Other","uploader":"Uploader - Topic","duration":10}
{"id":"ghi","title":"Third","artist":"Solo","channel":"Chan"}
`)

	tracks := parseTracks(data)

	if len(tracks) != 3 {
		t.Fatalf("got %d tracks: %+v", len(tracks), tracks)
	}
	if got := tracks[0]; got.ID != "abc" || got.Artist != "A, B" || got.Duration != 201 || got.Album != "Alb" || got.Service != "youtube" {
		t.Fatalf("track0 = %+v", got)
	}
	if tracks[1].Artist != "Uploader" || tracks[2].Artist != "Solo" {
		t.Fatalf("artists = %q / %q", tracks[1].Artist, tracks[2].Artist)
	}
	if len(parseTracks(nil)) != 0 {
		t.Fatal("expected no tracks")
	}
}

func TestPlayThroughStubTools(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell script stubs")
	}
	dir := t.TempDir()
	stub := func(name, body string) string {
		path := filepath.Join(dir, name)
		os.WriteFile(path, []byte("#!/bin/sh\n"+body+"\n"), 0o755)
		return path
	}
	y := youtube{stub("yt-dlp", "echo http://audio"), stub("ffmpeg", "head -c 3850 /dev/zero")}
	s := &sink{}

	err := y.Play(context.Background(), Track{ID: "x"}, s)

	if _, n := s.got(); err != nil || n != 3850 {
		t.Fatalf("err=%v bytes=%d", err, n)
	}
}
