# Ovi — Open Voice Assistant

Go voice assistant for ESPHome devices. Streaming STT and TTS run locally through sherpa-onnx; the agent uses the official OpenAI Go SDK against any OpenAI-compatible endpoint. No Home Assistant.

Guiding principles: keep it simple, keep it short, keep it fast. Prefer deleting code to adding abstractions.

## Architecture

ESPHome device → server → device, all streaming. Seven top-level domains; the big ones have subpackages. Dependencies flow downward only.

1. **device** — the wire: `Event` payloads, a `Transport` (WiFi TCP or BLE GATT) that delivers to a `Handler`, and `Speaker`, the `Output` that encodes PCM, paces it to real time and orders events behind playback. `device/codec` holds PCM/LC3/Opus behind one `Format` value. Depends on nothing internal.
2. **speech** — everything sherpa-onnx. `speech/models` downloads packs to `~/.cache/ovi/models`; `speech/stt` is the VAD-driven `listen` loop feeding Nemotron (online transducer, ~0 ms tail) or Whisper (offline); `speech/tts` is Kokoro (fp32 pack; int8 is 3× slower on x86) or Piper plus sentence streaming (the first chunk ends at a clause boundary so audio starts early). `stt.silence` is the main latency knob.
3. **agent** — OpenAI SDK streaming loop with 19 built-in tools and sub-agents as tools; tools act on an `Env` (announce, music, scheduler, timers); `agent/mcp` is the stdio MCP client; `agent/scheduler` the cron automations. History is per wake session, in memory.
4. **music** — one `Player` streams a queue of `Track`s to every device at once (SYNC_PLAY); each source is a `Service`: YouTube via `yt-dlp` + `ffmpeg` built in, `music/browser` adds Spotify / Apple Music through a captured Chromium tab, enabled with `music.services`. The pipeline calls `Interrupt` on a wake word and `Continue` when the session ends, so music ducks under speech and comes back.
5. **pipeline** — `VoiceAssistant` runs STT → Agent → TTS per utterance, `Connection` is the per-device state machine, `Manager` arbitrates wake words across devices (0.5 s window).
6. **cli** — terminal prompts, mDNS discovery, config-file editing, the ESPHome flash flow and the setup wizard.
7. **config** — the settings struct and its layered loading.

`cmd/ovi` is the binary: flag parsing and the serving loop.

## Configuration

`internal/config`: defaults → `~/.ovi/config.yaml` → `.env` → `OVI_SECTION__KEY` env → CLI `--section-key`. Device YAMLs live in `esphome/`.

## Build, test, lint

Native code: sherpa-onnx (prebuilt shared libraries inside its Go module) and liblc3 (C sources bundled in `github.com/caitunai/lc3`, compiled by cgo). Opus is pure Go (`github.com/tphakala/go-opus`). No system audio libraries.

```bash
go build ./... && go vet ./... && gofmt -l internal cmd   # must be clean
go test ./...                                              # no network, no models
OVI_TEST_MODELS=1 go test ./internal/speech/... -run Real   # real models (downloads)
```

Run both lint and tests after every change.

## Code conventions

- One type per file, file named after the type in snake_case.
- Small interfaces at the point of use (`pipeline.Voice`, `stt.vad`) so tests use fakes; no mocking libraries. Keep packages few and cohesive, subpackages only where a domain has a clearly separable part; unexport anything that does not cross a package line.
- Goroutines are cancelled via `context.Context` or a stop channel, never abandoned.
- Tests: standard `testing`, colocated `foo_test.go`, Arrange / Act / Assert separated by blank lines, fakes over real models or network. Model-backed tests are gated by `OVI_TEST_MODELS=1`.
- Error strings are lowercase; `slog` for logging; keep comments to what the code cannot say.

## ESPHome

Firmware in `esphome/components/` (`ovi_voice_assistant`, `ovi_audio_codec`) replaces the stock `voice_assistant` component. Wire protocol: `[2B LE length][payload]`, payload byte 0 is the message type (control events `0x01`–`0x0B`, mic audio `0x20`, speaker audio `0x21`). Mic is fixed 16 kHz 16-bit mono. Build with `esphome run esphome/<device>.yaml` (ESP-IDF; pulls `espressif/esp_audio_codec`).
