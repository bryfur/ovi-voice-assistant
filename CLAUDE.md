# Ovi — Open Voice Assistant

AI-powered voice assistant with streaming STT, TTS, and an OpenAI-compatible tool-calling agent. Written in Go. Connects directly to ESPHome devices over WiFi/BLE — no Home Assistant.

## Architecture

Streaming voice pipeline: ESPHome device → server → device.

1. **Transport** (`internal/transport/`) — receives audio from ESPHome devices over WiFi (TCP) or BLE (tinygo.org/x/bluetooth). Codec-agnostic: PCM, LC3, or Opus.
2. **DeviceConnection** (`internal/device/device_connection.go`) — bridges transport to pipeline; handles codec encode/decode and audio pacing.
3. **DeviceManager** (`internal/device/device_manager.go`) — manages multiple devices; arbitrates competing wake words (0.5s window).
4. **VoiceAssistant** (`internal/pipeline/voice_assistant.go`) — core pipeline: STT → Agent → TTS. `EncodingOutput` paces encoded audio to real time; `SpeechQueue` serialises utterances.
5. **STT** (`internal/stt/`) — Nemotron Speech 600M (ONNX, local, default) or Whisper via any OpenAI-compatible `/audio/transcriptions` endpoint. Both use Silero VAD.
6. **Agent** (`internal/agent/`) — tool-calling loop over `internal/llm` (OpenAI-compatible chat completions, streaming) with built-in tools, MCP servers (`internal/mcp`, stdio JSON-RPC) and sub-agents exposed as tools. In-memory session history per wake session.
7. **TTS** (`internal/tts/`) — Kokoro 82M ONNX int8 (default) or Piper. Both phonemize with `espeak-ng`. Streams audio back to device.
8. **Music** (`internal/music/`) — YouTube Music via `yt-dlp` + `ffmpeg`; Spotify / Apple Music via browser tab capture (chromedp). Multi-room sync via `MusicGroup`.
9. **Memory** (`internal/memory/`) — persistent fact extraction and recall (SQLite via modernc.org/sqlite + all-MiniLM-L6-v2 ONNX embeddings).
10. **Scheduler** (`internal/scheduler/`) — cron-based proactive automations.
11. **ONNX Runtime** (`internal/ort/`) — loads `libonnxruntime` via `github.com/yalue/onnxruntime_go`; downloads the matching release (1.29.0) into `~/.cache/ovi/onnxruntime` on first use.

## Configuration

- `internal/config` — layered: defaults → `~/.ovi/config.yaml` → `.env` → `OVI_` env vars (`__` nesting) → CLI flags
- CLI flags in `cmd/ovi/main.go` override config for model/provider selection
- Device-specific config (codec, sample rate) in `DeviceConfig`
- ESPHome device YAML configs live in `esphome/` (voice-pe, atom-echo, s3-box-3, crowpanel-9, crowpanel-s3-5)

## Build & Run

Requires Go 1.24+, a C compiler, and the native libs `liblc3`, `libopus`, `libopusfile` (cgo, found via `pkg-config`). Runtime tools: `espeak-ng` (TTS), `ffmpeg` + `yt-dlp` (music).

```bash
go build ./...                 # build everything
go run ./cmd/ovi               # run the assistant
go install ./cmd/ovi           # install the `ovi` binary
go test ./...                  # run all tests
go test ./internal/codec/      # run one package
OVI_TEST_MODELS=1 go test ./internal/stt/ -run RealModel   # real ONNX Runtime + Silero smoke test (downloads)
```

## Code Organization

- **One type per file, file named after the type**: `DeviceConnection` → `device_connection.go`, `MusicPlayer` → `music_player.go`, `WiFiTransport` → `wifi_transport.go`, etc.
- **Packages are small and single-purpose** under `internal/`; the only binary is `cmd/ovi`.
- **`internal/audio`** holds the `PipelineOutput` interface shared by pipeline, music and device to avoid import cycles.
- Concurrency: goroutines + `context.Context` cancellation. Long-running tasks (device sessions, music streaming, MCP servers) are cancelled through their context, never abandoned.

## Testing Standards

- **Framework**: standard `testing` package. No external assertion libraries.
- **Colocated tests**: every test file lives next to the source file it tests — `foo.go` → `foo_test.go`, same package (internal tests).
- **One test file per source file** where practical; shared fakes live in the test file that first needs them.
- **AAA pattern**: Arrange, Act, Assert with blank lines between phases:
  ```go
  func TestPCMCodecIdentity(t *testing.T) {
      c := NewPCMCodec(16000, 1)       // Arrange

      enc, _ := c.Encode(data)         // Act

      if !bytes.Equal(enc, data) {     // Assert
          t.Fatal("...")
      }
  }
  ```
- **No real models or network in unit tests**: ONNX sessions, LLM endpoints, MCP servers, transports and subprocesses are faked (in-process `httptest` servers, a self-re-executing fake MCP server, stub shell scripts for ffmpeg). Model-backed tests are gated behind `OVI_TEST_MODELS=1`.
- **Codec tests use the real native libraries** (liblc3/libopus) since they are build-time dependencies.
- **After any code change**: `go test ./...`

## Linting

- **Formatter**: `gofmt` — all code must be formatted (`gofmt -l internal cmd` must print nothing).
- **Vet**: `go vet ./...` must pass.
- **After any code change**: run both.

## ESPHome Build & Flash

ESPHome is still a Python tool. Install it separately (`pipx install esphome` or `uv tool install esphome`); `ovi --flash` shells out to `esphome` on PATH (falling back to `uv run esphome`). Components are loaded as local `external_components` in each device YAML — no install step needed.

```bash
# Build + flash over USB
esphome run esphome/voice-pe.yaml

# Build only (no flash)
esphome compile esphome/voice-pe.yaml
```

Components use ESP-IDF (not Arduino). The `espressif/esp_audio_codec` IDF component (v2.4.1) is pulled automatically.

## ESPHome Components

Custom ESPHome components live in `esphome/components/`. They replace the stock `voice_assistant` component entirely — no Home Assistant dependency.

### `ovi_voice_assistant`
The main device-side component. A state machine that streams mic audio to the server and plays back TTS/music audio.

- **Transport**: WiFi (plain TCP server on port 6055) or BLE (GATT service). Selected at compile time via `transport: wifi|ble`.
- **Codec**: Configurable `codec: pcm|lc3|opus` (default LC3). Encodes mic audio before sending, decodes speaker audio on receive.
- **Wire protocol**: Length-prefixed binary frames `[2B LE length][payload]`. Payload byte 0 is message type — control events (`0x01`–`0x0B`) or audio frames (`0x20` mic, `0x21` speaker).
- **Mic**: Fixed 16kHz 16-bit mono. Ring buffer (16KB / ~512ms).
- **microWakeWord integration**: Optional `micro_wake_word:` config. Pauses MWW during TTS on shared audio bus devices (ATOM Echo).
- **Automation triggers**: `on_start`, `on_listening`, `on_stt_vad_start/end`, `on_stt_end`, `on_tts_start`, `on_tts_stream_start`, `on_end`, `on_error`, `on_client_connected/disconnected`.
- **`shared_audio_bus`**: Set `true` when mic and speaker share one I2S peripheral (stops MWW during playback).

### `ovi_audio_codec`
Pluggable audio encoder/decoder abstraction. Auto-loaded by `ovi_voice_assistant`.

- **Interface**: `AudioEncoder` / `AudioDecoder` base classes with `open()`, `encode()`/`decode()`, `close()`, `reset()`.
- **Implementations**: PCM passthrough, LC3 (10ms frames, 40 bytes), Opus (20ms frames, ~32kbps VBR).
- **Backend**: Uses `espressif/esp_audio_codec` IDF component (v2.4.1).

### Device configs (`esphome/`)
- `voice-pe.yaml` — ESP32-S3 Voice PE (WiFi, on-device MWW, LC3)
- `voice-pe-ble.yaml` — Voice PE over BLE
- `atom-echo.yaml` — M5Stack ATOM Echo (WiFi, shared audio bus, on-device MWW)
- `s3-box-3.yaml` — ESP32-S3-BOX-3 (WiFi, on-device MWW)
- `crowpanel-9.yaml` — Elecrow CrowPanel 9" ESP32-P4 (WiFi via C6, PDM mic, on-device MWW, LC3)
- `crowpanel-s3-5.yaml` — Elecrow CrowPanel Advance 5"/4.3"/7" ESP32-S3 (WiFi, I2S mic, on-device MWW, LC3)
