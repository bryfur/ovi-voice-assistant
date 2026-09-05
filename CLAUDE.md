# Ovi — Open Voice Assistant

Go voice assistant for ESPHome devices. Streaming STT and TTS run locally through sherpa-onnx; the agent uses the official OpenAI Go SDK against any OpenAI-compatible endpoint. No Home Assistant.

Guiding principles: keep it simple, keep it short, keep it fast. Prefer deleting code to adding abstractions.

## Architecture

ESPHome device → server → device, all streaming.

1. **transport** — WiFi TCP or BLE GATT. Length-prefixed frames; codec-agnostic.
2. **device** — `DeviceConnection` bridges transport to pipeline (codec, mic queue, session task); `DeviceManager` handles many devices and wake-word arbitration (0.5 s window).
3. **pipeline** — `VoiceAssistant` runs STT → Agent → TTS for one utterance. `EncodingOutput` paces encoded audio to real time and orders events with playback. `SpeechQueue` serializes utterances.
4. **stt** — one VAD-driven `listen` loop (Silero via sherpa-onnx) feeding either Nemotron (online transducer, decodes while you talk) or Whisper (offline, decodes the segment).
5. **tts** — Kokoro or Piper through sherpa-onnx `OfflineTts`; `MaxNumSentences: 1` so audio streams per sentence.
6. **agent** — OpenAI SDK streaming loop with 19 built-in tools, MCP stdio servers, sub-agents as tools. History is per wake session, in memory.
7. **music** — YouTube via `yt-dlp` + `ffmpeg`; Spotify / Apple Music via chromedp tab capture, enabled with `music.services`; `MusicGroup` for multi-room sync.
8. **scheduler** — cron automations; **models** — downloads sherpa-onnx packs to `~/.cache/ovi/models`.

`internal/audio` holds the `PipelineOutput` interface shared by pipeline, music and device (avoids an import cycle).

## Configuration

`internal/config`: defaults → `~/.ovi/config.yaml` → `.env` → `OVI_SECTION__KEY` env → CLI `--section-key`. Device YAMLs live in `esphome/`.

## Build, test, lint

Native deps: `liblc3`, `libopus`, `libopusfile` (pkg-config). sherpa-onnx is prebuilt inside the Go module.

```bash
go build ./... && go vet ./... && gofmt -l internal cmd   # must be clean
go test ./...                                              # no network, no models
OVI_TEST_MODELS=1 go test ./internal/tts/ ./internal/stt/ -run Real   # real models (downloads)
```

Run both lint and tests after every change.

## Code conventions

- One type per file, file named after the type in snake_case.
- Small interfaces at the point of use (`device.Pipeline`, `stt.vad`) so tests use fakes; no mocking libraries.
- Goroutines are cancelled via `context.Context` or a stop channel, never abandoned.
- Tests: standard `testing`, colocated `foo_test.go`, Arrange / Act / Assert separated by blank lines, fakes over real models or network. Model-backed tests are gated by `OVI_TEST_MODELS=1`.
- Error strings are lowercase; `slog` for logging; keep comments to what the code cannot say.

## ESPHome

Firmware in `esphome/components/` (`ovi_voice_assistant`, `ovi_audio_codec`) replaces the stock `voice_assistant` component. Wire protocol: `[2B LE length][payload]`, payload byte 0 is the message type (control events `0x01`–`0x0B`, mic audio `0x20`, speaker audio `0x21`). Mic is fixed 16 kHz 16-bit mono. Build with `esphome run esphome/<device>.yaml` (ESP-IDF; pulls `espressif/esp_audio_codec`).
