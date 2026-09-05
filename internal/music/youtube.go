package music

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os/exec"
	"strconv"
	"strings"

	"github.com/bryfur/ovi-voice-assistant/internal/device"
)

// youtube plays YouTube Music: yt-dlp finds tracks and resolves stream
// URLs, ffmpeg decodes them to PCM.
type youtube struct{ ytdlp, ffmpeg string }

// Search asks YouTube Music first and plain YouTube if that finds nothing.
func (y youtube) Search(ctx context.Context, query string, limit int) ([]Track, error) {
	tracks, err := y.list(ctx, limit, "https://music.youtube.com/search?q="+url.QueryEscape(query)+"#songs")
	if err != nil || len(tracks) == 0 {
		if tracks, err = y.list(ctx, limit, fmt.Sprintf("ytsearch%d:%s", limit, query)); err != nil {
			return nil, err
		}
	}
	slog.Info("YouTube Music search", "query", query, "results", len(tracks))
	return tracks, nil
}

func (y youtube) list(ctx context.Context, limit int, target string) ([]Track, error) {
	out, err := y.run(ctx, "-j", "--flat-playlist", "--no-warnings", "--ignore-errors",
		"--playlist-end", strconv.Itoa(limit), target)
	if err != nil && len(out) == 0 {
		return nil, err
	}
	return parseTracks(out), nil
}

// Play resolves a direct audio URL and pipes it through ffmpeg as s16le PCM.
func (y youtube) Play(ctx context.Context, track Track, out device.Output) error {
	u, err := y.run(ctx, "-f", "bestaudio/best", "-g", "--no-warnings", "--no-playlist",
		"https://music.youtube.com/watch?v="+track.ID)
	if err != nil {
		return err
	}
	src, _, _ := strings.Cut(strings.TrimSpace(string(u)), "\n")
	if src == "" {
		return fmt.Errorf("yt-dlp returned no URL for %s", track.ID)
	}
	cmd := exec.CommandContext(ctx, y.ffmpeg,
		"-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
		"-i", src, "-f", "s16le", "-ar", strconv.Itoa(Rate), "-ac", strconv.Itoa(Channels),
		"-loglevel", "error", "pipe:1")
	pcm, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start ffmpeg: %w", err)
	}
	defer func() { _ = cmd.Process.Kill(); _ = cmd.Wait() }()

	buf := make([]byte, Rate*Channels*2*20/1000) // 20 ms
	for {
		n, err := io.ReadFull(pcm, buf)
		if n > 0 {
			if err := out.SendAudio(ctx, buf[:n]); err != nil {
				return err
			}
		}
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil
		} else if err != nil {
			return err
		}
	}
}

func (y youtube) run(ctx context.Context, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, y.ytdlp, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return out, fmt.Errorf("yt-dlp: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}

// parseTracks reads yt-dlp's newline-delimited JSON.
func parseTracks(data []byte) []Track {
	var tracks []Track
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(nil, 16<<20)
	for sc.Scan() {
		var e struct {
			ID, Title, Artist, Album, Creator, Uploader, Channel string
			Artists                                              []string
			Duration                                             float64
		}
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil || e.ID == "" {
			continue
		}
		artist := cmp.Or(e.Artist, strings.Join(e.Artists, ", "), e.Creator, e.Uploader, e.Channel)
		tracks = append(tracks, Track{
			Title: e.Title, Artist: strings.TrimSuffix(artist, " - Topic"), Album: e.Album,
			Duration: int(e.Duration), ID: e.ID, Service: "youtube",
		})
	}
	return tracks
}
