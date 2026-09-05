// Package browser plays Spotify and Apple Music through a Chromium tab
// whose audio is captured and streamed to the devices.
package browser

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	"github.com/coder/websocket"
)

// ProfileRoot holds persistent browser profiles.
var ProfileRoot = "~/.config/ovi"

const captureJS = `
async (wsPort) => {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      suppressLocalAudioPlayback: true,
    },
    video: true,
    preferCurrentTab: true,
  });
  stream.getVideoTracks().forEach(t => t.stop());

  const audioTrack = stream.getAudioTracks()[0];
  const settings = audioTrack?.getSettings?.() || {};
  const channels = settings.channelCount || 2;
  console.log('[ovi] audio track:', audioTrack?.label,
              'channels:', channels, 'sampleRate:', settings.sampleRate);

  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, channels, channels);

  const ws = new WebSocket('ws://127.0.0.1:' + wsPort);
  ws.binaryType = 'arraybuffer';
  await new Promise((res, rej) => {
    ws.addEventListener('open', res);
    ws.addEventListener('error', () => rej(new Error('WebSocket connection failed')));
  });

  ws.send(JSON.stringify({
    sampleRate: ctx.sampleRate,
    channels: channels,
  }));

  proc.onaudioprocess = (e) => {
    const len = e.inputBuffer.getChannelData(0).length;
    const pcm = new Int16Array(len * channels);
    for (let ch = 0; ch < channels; ch++) {
      const data = e.inputBuffer.getChannelData(ch);
      for (let i = 0; i < len; i++)
        pcm[i * channels + ch] = Math.max(-32768, Math.min(32767, data[i] * 32768));
    }
    if (ws.readyState === 1) ws.send(pcm.buffer);
  };
  source.connect(proc);
  proc.connect(ctx.destination);
  window.__oviCapture = { ctx, ws, stream };
  return true;
}
`

// Session is the shared machinery for streaming music via browser
// tab audio capture.
//
// It launches Chromium (chromedp), navigates to a music service, captures
// tab audio with getDisplayMedia, and streams s16le PCM over a local
// WebSocket back to Go. Providers supply service-specific search / play
// logic.
type Session struct {
	URL         string
	ProfileName string

	sampleRate int

	allocCancel context.CancelFunc
	ctxCancel   context.CancelFunc
	pageCtx     context.Context

	httpServer *http.Server
	wsPort     int

	mu                sync.Mutex
	audioQueue        chan []byte
	captureActive     bool
	browserSampleRate int
	browserChannels   int
}

// NewSession creates an unstarted session.
func NewSession(url, profileName string, sampleRate int) *Session {
	return &Session{
		URL:               url,
		ProfileName:       profileName,
		sampleRate:        sampleRate,
		audioQueue:        make(chan []byte, 256),
		browserSampleRate: 48000,
		browserChannels:   2,
	}
}

// Start launches the browser, opens the music service and starts the
// WebSocket audio bridge.
func (b *Session) Start(ctx context.Context) error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	b.wsPort = ln.Addr().(*net.TCPAddr).Port
	b.httpServer = &http.Server{Handler: http.HandlerFunc(b.wsHandler)}
	go func() { _ = b.httpServer.Serve(ln) }()
	slog.Info("Audio bridge listening", "url", fmt.Sprintf("ws://127.0.0.1:%d", b.wsPort))

	profile := filepath.Join(expandUser(ProfileRoot), b.ProfileName)
	if err := os.MkdirAll(profile, 0o755); err != nil {
		return err
	}
	opts := []chromedp.ExecAllocatorOption{
		chromedp.NoFirstRun,
		chromedp.NoDefaultBrowserCheck,
		chromedp.UserDataDir(profile),
		chromedp.Flag("headless", false),
		chromedp.Flag("auto-accept-this-tab-capture", true),
		chromedp.Flag("autoplay-policy", "no-user-gesture-required"),
	}
	allocCtx, allocCancel := chromedp.NewExecAllocator(context.Background(), opts...)
	pageCtx, pageCancel := chromedp.NewContext(allocCtx)
	b.allocCancel = allocCancel
	b.ctxCancel = pageCancel
	b.pageCtx = pageCtx

	if err := chromedp.Run(pageCtx,
		page.SetBypassCSP(true),
		chromedp.Navigate(b.URL),
	); err != nil {
		b.Close()
		return fmt.Errorf("launch browser: %w", err)
	}
	return nil
}

// Close shuts down the browser and WebSocket server.
func (b *Session) Close() {
	if b.ctxCancel != nil {
		b.ctxCancel()
		b.ctxCancel = nil
	}
	if b.allocCancel != nil {
		b.allocCancel()
		b.allocCancel = nil
	}
	if b.httpServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_ = b.httpServer.Shutdown(ctx)
		cancel()
		b.httpServer = nil
	}
}

// Evaluate runs a JS function expression with JSON-encoded arguments and
// decodes the awaited result into out (which may be nil).
func (b *Session) Evaluate(ctx context.Context, fn string, out any, args ...any) error {
	if b.pageCtx == nil {
		return errors.New("browser not started")
	}
	argJSON := make([]string, len(args))
	for i, a := range args {
		j, err := json.Marshal(a)
		if err != nil {
			return err
		}
		argJSON[i] = string(j)
	}
	expr := "(" + fn + ")(" + joinStrings(argJSON, ",") + ")"
	runCtx, cancel := context.WithCancel(b.pageCtx)
	defer cancel()
	stop := context.AfterFunc(ctx, cancel)
	defer stop()
	var raw json.RawMessage
	err := chromedp.Run(runCtx, chromedp.Evaluate(expr, &raw,
		func(p *runtime.EvaluateParams) *runtime.EvaluateParams {
			return p.WithAwaitPromise(true).WithReturnByValue(true)
		}))
	if err != nil {
		return err
	}
	if out != nil && len(raw) > 0 {
		return json.Unmarshal(raw, out)
	}
	return nil
}

func joinStrings(parts []string, sep string) string {
	s := ""
	for i, p := range parts {
		if i > 0 {
			s += sep
		}
		s += p
	}
	return s
}

// StreamTrack plays a track (via play) in the browser and forwards captured
// PCM to output until play returns.
func (b *Session) StreamTrack(ctx context.Context, output device.Output, play func(ctx context.Context) error) error {
	if err := b.ensureCapture(ctx); err != nil {
		return err
	}
	b.drain()

	playCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	playDone := make(chan error, 1)
	go func() { playDone <- play(playCtx) }()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-playDone:
			// Forward any remaining captured audio.
			b.forwardPending(ctx, output)
			return err
		case data, ok := <-b.audioQueue:
			if !ok || data == nil {
				<-playDone
				return nil
			}
			if err := output.SendAudio(ctx, data); err != nil {
				return err
			}
		}
	}
}

func (b *Session) drain() {
	for {
		select {
		case <-b.audioQueue:
		default:
			return
		}
	}
}

func (b *Session) forwardPending(ctx context.Context, output device.Output) {
	for {
		select {
		case data := <-b.audioQueue:
			if data != nil {
				_ = output.SendAudio(ctx, data)
			}
		default:
			return
		}
	}
}

func (b *Session) wsHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
	if err != nil {
		return
	}
	defer conn.CloseNow()
	conn.SetReadLimit(4 << 20)
	slog.Info("Browser audio capture connected")
	first := true
	for {
		typ, data, err := conn.Read(r.Context())
		if err != nil {
			break
		}
		if first && typ == websocket.MessageText {
			var cfg struct {
				SampleRate int `json:"sampleRate"`
				Channels   int `json:"channels"`
			}
			if json.Unmarshal(data, &cfg) == nil {
				b.mu.Lock()
				if cfg.SampleRate > 0 {
					b.browserSampleRate = cfg.SampleRate
				}
				if cfg.Channels > 0 {
					b.browserChannels = cfg.Channels
				}
				b.mu.Unlock()
				slog.Info("Browser audio", "rate", cfg.SampleRate, "channels", cfg.Channels)
			}
			first = false
			continue
		}
		first = false
		if typ == websocket.MessageBinary {
			select {
			case b.audioQueue <- data:
			default:
				slog.Debug("Browser audio queue full, dropping chunk")
			}
		}
	}
	select {
	case b.audioQueue <- nil:
	default:
	}
}

func (b *Session) ensureCapture(ctx context.Context) error {
	b.mu.Lock()
	active := b.captureActive
	b.mu.Unlock()
	if active {
		return nil
	}
	if err := b.Evaluate(ctx, captureJS, nil, b.wsPort); err != nil {
		return fmt.Errorf("start tab capture: %w", err)
	}
	b.mu.Lock()
	b.captureActive = true
	b.mu.Unlock()
	slog.Info("Tab audio capture active")
	return nil
}

func expandUser(p string) string {
	if len(p) > 0 && p[0] == '~' {
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, p[1:])
		}
	}
	return p
}
