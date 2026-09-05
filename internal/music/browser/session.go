// Package browser plays Spotify and Apple Music through a Chromium tab
// whose audio is captured and streamed to the devices.
package browser

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	"github.com/coder/websocket"

	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// captureJS grabs the tab's own audio and ships it as s16le PCM over a
// local WebSocket, after one JSON line announcing the format.
const captureJS = `
async (wsPort) => {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false,
             suppressLocalAudioPlayback: true },
    video: true,
    preferCurrentTab: true,
  });
  stream.getVideoTracks().forEach(t => t.stop());
  const track = stream.getAudioTracks()[0];
  const channels = track?.getSettings?.().channelCount || 2;

  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, channels, channels);
  const ws = new WebSocket('ws://127.0.0.1:' + wsPort);
  ws.binaryType = 'arraybuffer';
  await new Promise((res, rej) => {
    ws.addEventListener('open', res);
    ws.addEventListener('error', () => rej(new Error('WebSocket connection failed')));
  });
  ws.send(JSON.stringify({ sampleRate: ctx.sampleRate, channels }));

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

// Session is a Chromium tab on a music service plus the audio bridge that
// receives its captured sound.
type Session struct {
	url, profile, stopJS string

	tab       context.Context // chromedp context; nil until launched
	quit      func()
	server    *http.Server
	port      int
	audio     chan []byte // captured PCM; nil marks the end of a capture
	capturing atomic.Bool
}

func newSession(url, profile, stopJS string) *Session {
	return &Session{url: url, profile: profile, stopJS: stopJS, audio: make(chan []byte, 256)}
}

// launch starts the audio bridge and opens the service in a persistent
// browser profile, so logins survive restarts.
func (s *Session) launch() error {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	s.port = ln.Addr().(*net.TCPAddr).Port
	s.server = &http.Server{Handler: http.HandlerFunc(s.bridge)}
	go func() { _ = s.server.Serve(ln) }()

	dir, err := os.UserConfigDir()
	if err != nil {
		return err
	}
	profile := filepath.Join(dir, "ovi", s.profile)
	if err := os.MkdirAll(profile, 0o755); err != nil {
		return err
	}
	alloc, cancelAlloc := chromedp.NewExecAllocator(context.Background(),
		chromedp.NoFirstRun, chromedp.NoDefaultBrowserCheck, chromedp.UserDataDir(profile),
		chromedp.Flag("headless", false),
		chromedp.Flag("auto-accept-this-tab-capture", true),
		chromedp.Flag("autoplay-policy", "no-user-gesture-required"))
	tab, cancelTab := chromedp.NewContext(alloc)
	s.tab, s.quit = tab, func() { cancelTab(); cancelAlloc() }
	if err := chromedp.Run(tab, page.SetBypassCSP(true), chromedp.Navigate(s.url)); err != nil {
		s.Close()
		return fmt.Errorf("launch browser: %w", err)
	}
	return nil
}

// Close quits the browser and the audio bridge.
func (s *Session) Close() {
	if s.quit != nil {
		s.quit()
		s.quit, s.tab = nil, nil
	}
	if s.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = s.server.Shutdown(ctx)
		s.server = nil
	}
}

// eval calls a JS function expression with JSON-encoded args, awaits it
// and decodes the result into out (which may be nil).
func (s *Session) eval(ctx context.Context, fn string, out any, args ...any) error {
	if s.tab == nil {
		return errors.New("browser not started")
	}
	encoded := make([]string, len(args))
	for i, a := range args {
		j, err := json.Marshal(a)
		if err != nil {
			return err
		}
		encoded[i] = string(j)
	}
	run, cancel := context.WithCancel(s.tab)
	defer cancel()
	defer context.AfterFunc(ctx, cancel)()
	var raw json.RawMessage
	err := chromedp.Run(run, chromedp.Evaluate("("+fn+")("+strings.Join(encoded, ",")+")", &raw,
		func(p *runtime.EvaluateParams) *runtime.EvaluateParams {
			return p.WithAwaitPromise(true).WithReturnByValue(true)
		}))
	if err != nil || out == nil || len(raw) == 0 {
		return err
	}
	return json.Unmarshal(raw, out)
}

// play runs playJS(arg) in the tab and forwards captured audio to out
// until it returns. A cancelled ctx stops playback in the tab too.
func (s *Session) play(ctx context.Context, out device.Output, playJS string, arg any) error {
	if err := s.capture(ctx); err != nil {
		return err
	}
	s.forward(ctx, nil) // drop stale audio
	finished := make(chan error, 1)
	go func() { finished <- s.eval(ctx, playJS, nil, arg) }()
	for {
		select {
		case <-ctx.Done():
			_ = s.eval(tail(), s.stopJS, nil)
			return ctx.Err()
		case err := <-finished:
			s.forward(ctx, out)
			return err
		case pcm := <-s.audio:
			if pcm == nil {
				return <-finished
			}
			if err := out.SendAudio(ctx, pcm); err != nil {
				return err
			}
		}
	}
}

// forward passes already-captured audio to out (nil discards it).
func (s *Session) forward(ctx context.Context, out device.Output) {
	for {
		select {
		case pcm := <-s.audio:
			if pcm != nil && out != nil {
				_ = out.SendAudio(ctx, pcm)
			}
		default:
			return
		}
	}
}

func (s *Session) capture(ctx context.Context) error {
	if s.capturing.Load() {
		return nil
	}
	if err := s.eval(ctx, captureJS, nil, s.port); err != nil {
		return fmt.Errorf("start tab capture: %w", err)
	}
	s.capturing.Store(true)
	return nil
}

// bridge receives the tab's audio over WebSocket.
func (s *Session) bridge(w http.ResponseWriter, r *http.Request) {
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
	if err != nil {
		return
	}
	defer conn.CloseNow()
	conn.SetReadLimit(4 << 20)
	for {
		typ, data, err := conn.Read(r.Context())
		if err != nil {
			break
		}
		if typ == websocket.MessageText {
			var f struct{ SampleRate, Channels int }
			_ = json.Unmarshal(data, &f)
			slog.Info("Browser audio capture", "rate", f.SampleRate, "channels", f.Channels)
			if f.SampleRate != music.Rate || f.Channels != music.Channels {
				slog.Warn("Browser audio format differs from the music format; playback speed will be off",
					"want_rate", music.Rate, "want_channels", music.Channels)
			}
			continue
		}
		select {
		case s.audio <- data:
		default:
			slog.Debug("Browser audio queue full, dropping chunk")
		}
	}
	select {
	case s.audio <- nil:
	default:
	}
}

// search runs searchJS({query, limit}) and labels the results with service.
func (s *Session) search(ctx context.Context, searchJS, service, query string, limit int) ([]music.Track, error) {
	var found []struct {
		ID, Title, Artist, Album string
		Duration                 int
	}
	if err := s.eval(ctx, searchJS, &found, map[string]any{"query": query, "limit": limit}); err != nil {
		return nil, err
	}
	tracks := make([]music.Track, len(found))
	for i, f := range found {
		tracks[i] = music.Track{Title: f.Title, Artist: f.Artist, Album: f.Album, Duration: f.Duration, ID: f.ID, Service: service}
	}
	return tracks, nil
}

func tail() context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	context.AfterFunc(ctx, cancel)
	return ctx
}
