package music

import (
	"context"
	"errors"
	"fmt"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"io"
	"log/slog"
	"os/exec"
	"strconv"
	"sync"
)

// BrowserMusic is a browser-based music provider (Spotify, Apple Music).
type BrowserMusic interface {
	Search(ctx context.Context, query string, limit int) ([]MusicTrack, error)
	StreamTrack(ctx context.Context, track MusicTrack, output device.Output) error
	StopPlayback(ctx context.Context) error
}

// MusicPlayer manages a music queue and streams audio via yt-dlp + ffmpeg.
//
// The player holds queue state (tracks, position, active flag). Actual
// streaming is driven by DeviceConnection which calls Stream and manages
// cancellation on wake-word.
type MusicPlayer struct {
	mu           sync.Mutex
	queue        []MusicTrack
	currentIndex int
	active       bool

	sampleRate int
	channels   int
	browsers   map[string]BrowserMusic

	// FFmpegPath allows tests to substitute the binary.
	FFmpegPath string
	// ExtractURL resolves a YouTube video ID to a direct audio URL.
	ExtractURL func(ctx context.Context, videoID string) (string, error)
}

// NewMusicPlayer creates a player producing PCM at sampleRate/channels.
func NewMusicPlayer(sampleRate, channels int, browsers map[string]BrowserMusic) *MusicPlayer {
	if channels <= 0 {
		channels = 1
	}
	if browsers == nil {
		browsers = map[string]BrowserMusic{}
	}
	return &MusicPlayer{
		sampleRate: sampleRate,
		channels:   channels,
		browsers:   browsers,
		FFmpegPath: "ffmpeg",
		ExtractURL: extractAudioURL,
	}
}

// QueueLen returns the number of queued tracks.
func (p *MusicPlayer) QueueLen() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.queue)
}

// CurrentIndex returns the position in the queue.
func (p *MusicPlayer) CurrentIndex() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.currentIndex
}

// IsActive reports whether music is (or should be) playing.
func (p *MusicPlayer) IsActive() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.active
}

// SetQueue replaces the queue and marks music as active.
func (p *MusicPlayer) SetQueue(tracks []MusicTrack, startIndex int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.queue = append([]MusicTrack(nil), tracks...)
	p.currentIndex = startIndex
	p.active = true
}

// Pause marks playback inactive.
func (p *MusicPlayer) Pause() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.active = false
}

// Resume marks playback active if there is something to play.
func (p *MusicPlayer) Resume() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.queue) > 0 && p.currentIndex < len(p.queue) {
		p.active = true
	}
}

// Skip advances to the next track. Returns the new track, or nil at the end.
func (p *MusicPlayer) Skip() *MusicTrack {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.currentIndex+1 < len(p.queue) {
		p.currentIndex++
		p.active = true
		t := p.queue[p.currentIndex]
		return &t
	}
	p.active = false
	return nil
}

// Stop clears the queue.
func (p *MusicPlayer) Stop() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.active = false
	p.queue = nil
	p.currentIndex = 0
}

// GetCurrent returns the current track, or nil.
func (p *MusicPlayer) GetCurrent() *MusicTrack {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.queue) > 0 && p.currentIndex < len(p.queue) {
		t := p.queue[p.currentIndex]
		return &t
	}
	return nil
}

// Stream plays tracks starting at the current index until the queue ends
// or ctx is cancelled.
func (p *MusicPlayer) Stream(ctx context.Context, output device.Output) error {
	for {
		p.mu.Lock()
		if !p.active || p.currentIndex >= len(p.queue) {
			p.active = false
			p.mu.Unlock()
			return nil
		}
		track := p.queue[p.currentIndex]
		p.mu.Unlock()

		slog.Info("Streaming", "artist", track.Artist, "title", track.Title)
		if err := p.streamTrack(ctx, output, track); err != nil {
			if errors.Is(err, context.Canceled) || ctx.Err() != nil {
				return ctx.Err()
			}
			slog.Error("Track streaming failed", "title", track.Title, "err", err)
		}
		p.mu.Lock()
		p.currentIndex++
		p.mu.Unlock()
	}
}

// streamTrack streams a track via browser capture or yt-dlp + ffmpeg.
func (p *MusicPlayer) streamTrack(ctx context.Context, output device.Output, track MusicTrack) error {
	if b, ok := p.browsers[track.Service]; ok {
		return b.StreamTrack(ctx, track, output)
	}

	audioURL, err := p.ExtractURL(ctx, track.VideoID)
	if err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, p.FFmpegPath,
		"-reconnect", "1",
		"-reconnect_streamed", "1",
		"-reconnect_delay_max", "5",
		"-i", audioURL,
		"-f", "s16le",
		"-ar", strconv.Itoa(p.sampleRate),
		"-ac", strconv.Itoa(p.channels),
		"-loglevel", "error",
		"pipe:1",
	)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start ffmpeg: %w", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()

	// Read ~20ms chunks of 16-bit PCM
	chunkBytes := p.sampleRate * p.channels * 2 * 20 / 1000
	buf := make([]byte, chunkBytes)
	for {
		n, err := io.ReadFull(stdout, buf)
		if n > 0 {
			data := make([]byte, n)
			copy(data, buf[:n])
			if err := output.SendAudio(ctx, data); err != nil {
				return err
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil
			}
			return err
		}
	}
}
