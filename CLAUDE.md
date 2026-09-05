# Ovi — Open Voice Assistant

Go voice assistant for ESPHome devices. Streaming STT and TTS run locally through sherpa-onnx; the agent uses the official OpenAI Go SDK against any OpenAI-compatible endpoint. No Home Assistant.

Guiding principles: keep it simple, keep it short, keep it fast. Prefer deleting code to adding abstractions.

## Architecture

ESPHome device → server → device, all streaming. Seven packages, dependencies flow downward only:

1. **device** — the wire: events, WiFi TCP and BLE GATT transports, PCM/LC3/Opus codecs (cgo), the `Output`/`Speaker` interfaces and `EncodingOutput`, which paces encoded audio to real time and orders events with playback. Depends on nothing internal.
2. **speech** — everything sherpa-onnx: model pack downloads (`~/.cache/ovi/models`), the VAD-driven `listen` loop, Nemotron (online transducer, decodes while you talk; ~0 ms tail) and Whisper (offline), Kokoro (fp32 pack; int8 is 3× slower on x86) and Piper TTS, sentence streaming (`SplitSentences` ends the first chunk at a clause boundary so audio starts early). `stt.silence` is the main latency knob.
3. **agent** — OpenAI SDK streaming loop with 19 built-in tools, the MCP stdio client, sub-agents as tools, and the cron scheduler for automations. History is per wake session, in memory.
4. **music** — YouTube via `yt-dlp` + `ffmpeg`; Spotify / Apple Music via chromedp tab capture, enabled with `music.services`; `MusicGroup` for multi-room sync. Depends on `device`.
5. **pipeline** — `VoiceAssistant` runs STT → Agent → TTS per utterance, `SpeechQueue` serializes speech, `DeviceConnection` is the per-device state machine, `DeviceManager` arbitrates wake words across devices (0.5 s window).
6. **cli** — terminal prompts, mDNS discovery, config-file editing, the ESPHome flash flow and the setup wizard.
7. **config** — the settings struct and its layered loading.

`cmd/ovi` is the binary: flag parsing and the serving loop.

## Configuration

`internal/config`: defaults → `~/.ovi/config.yaml` → `.env` → `OVI_SECTION__KEY` env → CLI `--section-key`. Device YAMLs live in `esphome/`.

## Build, test, lint

Native deps: `liblc3`, `libopus`, `libopusfile` (pkg-config). sherpa-onnx is prebuilt inside the Go module.

```bash
go build ./... && go vet ./... && gofmt -l internal cmd   # must be clean
go test ./...                                              # no network, no models
OVI_TEST_MODELS=1 go test ./internal/speech/ -run Real   # real models (downloads)
```

Run both lint and tests after every change.

## Code conventions

- One type per file, file named after the type in snake_case.
- Small interfaces at the point of use (`pipeline.Runner`, `speech.vad`) so tests use fakes; no mocking libraries. Keep packages few and cohesive; unexport anything that does not cross a package line.
- Goroutines are cancelled via `context.Context` or a stop channel, never abandoned.
- Tests: standard `testing`, colocated `foo_test.go`, Arrange / Act / Assert separated by blank lines, fakes over real models or network. Model-backed tests are gated by `OVI_TEST_MODELS=1`.
- Error strings are lowercase; `slog` for logging; keep comments to what the code cannot say.

## ESPHome

Firmware in `esphome/components/` (`ovi_voice_assistant`, `ovi_audio_codec`) replaces the stock `voice_assistant` component. Wire protocol: `[2B LE length][payload]`, payload byte 0 is the message type (control events `0x01`–`0x0B`, mic audio `0x20`, speaker audio `0x21`). Mic is fixed 16 kHz 16-bit mono. Build with `esphome run esphome/<device>.yaml` (ESP-IDF; pulls `espressif/esp_audio_codec`).
