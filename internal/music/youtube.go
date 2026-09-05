package music

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/url"
	"os/exec"
	"strconv"
	"strings"
)

// YtDlpPath is the yt-dlp binary used for YouTube Music search and stream
// URL extraction.
var YtDlpPath = "yt-dlp"

// ExtractAudioURL uses yt-dlp to get a direct audio stream URL for a
// YouTube video.
func ExtractAudioURL(ctx context.Context, videoID string) (string, error) {
	cmd := exec.CommandContext(ctx, YtDlpPath,
		"-f", "bestaudio/best", "-g", "--no-warnings", "--no-playlist",
		"https://music.youtube.com/watch?v="+videoID)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("yt-dlp: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return "", fmt.Errorf("yt-dlp returned no URL for %s", videoID)
	}
	return lines[0], nil
}

// ytdlpEntry is the subset of yt-dlp's flat-playlist JSON we care about.
type ytdlpEntry struct {
	ID       string          `json:"id"`
	Title    string          `json:"title"`
	Duration float64         `json:"duration"`
	Artist   string          `json:"artist"`
	Artists  []string        `json:"artists"`
	Uploader string          `json:"uploader"`
	Channel  string          `json:"channel"`
	Creator  string          `json:"creator"`
	Album    string          `json:"album"`
	Creators json.RawMessage `json:"creators"`
}

// ParseYtDlpTracks parses newline-delimited yt-dlp JSON into tracks.
func ParseYtDlpTracks(data []byte) []MusicTrack {
	var tracks []MusicTrack
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 0, 64<<10), 16<<20)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var e ytdlpEntry
		if err := json.Unmarshal(line, &e); err != nil || e.ID == "" {
			continue
		}
		artist := e.Artist
		if artist == "" && len(e.Artists) > 0 {
			artist = strings.Join(e.Artists, ", ")
		}
		if artist == "" {
			artist = firstNonEmpty(e.Creator, e.Uploader, e.Channel)
		}
		artist = strings.TrimSuffix(artist, " - Topic")
		tracks = append(tracks, MusicTrack{
			Title:           e.Title,
			Artist:          artist,
			Album:           e.Album,
			DurationSeconds: int(e.Duration),
			VideoID:         e.ID,
			Service:         "youtube",
		})
	}
	return tracks
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// SearchYouTube searches YouTube Music and returns tracks with video IDs
// for streaming.
func SearchYouTube(ctx context.Context, query string, limit int) ([]MusicTrack, error) {
	if limit <= 0 {
		limit = 20
	}
	run := func(target string) ([]MusicTrack, error) {
		cmd := exec.CommandContext(ctx, YtDlpPath,
			"-j", "--flat-playlist", "--no-warnings", "--ignore-errors",
			"--playlist-end", strconv.Itoa(limit), target)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil && len(out) == 0 {
			return nil, fmt.Errorf("yt-dlp search: %w: %s", err, strings.TrimSpace(stderr.String()))
		}
		return ParseYtDlpTracks(out), nil
	}
	tracks, err := run("https://music.youtube.com/search?q=" + url.QueryEscape(query) + "#songs")
	if err != nil || len(tracks) == 0 {
		// Fall back to a plain YouTube search.
		tracks, err = run(fmt.Sprintf("ytsearch%d:%s", limit, query))
		if err != nil {
			return nil, err
		}
	}
	slog.Info("YouTube Music search", "query", query, "results", len(tracks))
	return tracks, nil
}
