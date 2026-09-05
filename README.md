# Ovi — Open Voice Assistant

A standalone AI voice assistant that connects directly to ESPHome devices over WiFi or BLE. No Home Assistant required. One Go binary plus the sherpa-onnx runtime.

Handles the full voice pipeline: wake word detection (on-device) → speech-to-text → AI agent → text-to-speech, with audio streamed back to the device speaker.

## How it works

```
ESPHome Device (wake word) ──► mic audio over WiFi/BLE ──► Ovi Server
                                                              │
                                                        STT (Nemotron / Whisper)
                                                              │
                                                        Agent (OpenAI SDK, tool calling)
                                                              │
                                                        TTS (Kokoro / Piper)
                                                              │
ESPHome Device (speaker)   ◄── encoded audio ◄───────────────┘
```

- **STT**: [Nemotron Speech 600M](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) streaming (default) or Whisper, both local via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Silero VAD decides when you stopped talking.
- **Agent**: the official OpenAI Go SDK against any OpenAI-compatible endpoint (OpenAI, ollama, vLLM, LM Studio, …). Supports MCP tools and sub-agents.
- **TTS**: [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) 82M int8 (default) or [Piper](https://github.com/rhasspy/piper) voices, local via sherpa-onnx.
- **Transport**: WiFi (plain TCP) or BLE (GATT). Audio codecs: PCM, LC3, Opus.
- **Music**: YouTube Music, plus Spotify and Apple Music through a browser window. Multi-room synchronized playback.
- **Automations**: Cron-based proactive announcements.

## Supported devices

| Device | Transport | Codec | Notes |
|--------|-----------|-------|-------|
| [Voice Preview Edition](https://www.home-assistant.io/voice-pe/) | WiFi, BLE | LC3 | 12-LED ring, rotary volume, mute switch |
| [M5Stack ATOM Echo](https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit) | WiFi | PCM | Shared audio bus, single LED |
| [ESP32-S3-BOX-3](https://www.espressif.com/en/dev-board/esp32-s3-box-3) | WiFi | LC3 | Dual mics, ES8311 DAC |
| [Elecrow CrowPanel 9"](https://www.elecrow.com/crowpanel-advance-9-0-hmi-esp32-p4-ai-display.html) | WiFi | LC3 | ESP32-P4 + C6, touchscreen |
| [Elecrow CrowPanel Advance 5"](https://www.elecrow.com/crowpanel-advance-5-0-hmi-esp32-s3-ai-display.html) | WiFi | LC3 | ESP32-S3, touchscreen |

## Requirements

- Go 1.24+ and a C compiler (cgo)
- `liblc3`, `libopus`, `libopusfile` with `pkg-config` files
  - Arch: `pacman -S liblc3 opus opusfile` · Debian/Ubuntu: `apt install liblc3-dev libopus-dev libopusfile-dev` · macOS: `brew install liblc3 opus opusfile`
- For music: `yt-dlp` and `ffmpeg` (YouTube Music); Chromium for Spotify / Apple Music
- An ESPHome-compatible device (see above) and an OpenAI-compatible LLM endpoint
- [ESPHome](https://esphome.io) only if you flash firmware with `ovi --flash`

sherpa-onnx ships prebuilt as part of the Go module; speech models download into `~/.cache/ovi/models` on first use (Nemotron ~460 MB, Kokoro ~350 MB, Piper voices ~70 MB).

## Quick start

### 1. Build

```bash
go install ./cmd/ovi
```

The binary links two shared libraries from the Go module cache (`libsherpa-onnx-c-api.so`, `libonnxruntime.so`). To run it elsewhere, copy them from `$(go env GOMODCACHE)/github.com/k2-fsa/sherpa-onnx-go-linux@*/lib/<arch>/` next to the binary and set `LD_LIBRARY_PATH` (or `DYLD_LIBRARY_PATH` on macOS).

### 2. Flash a device

Edit `esphome/secrets.yaml` with your WiFi credentials, then:

```bash
esphome run esphome/voice-pe.yaml
```

Or use the guided flow, which also writes `secrets.yaml` and adds the device to your config:

```bash
ovi --flash
```

### 3. Configure

On first run, Ovi launches an interactive setup wizard for LLM, STT, TTS, devices, codec and music services:

```bash
ovi
```

It saves to `~/.ovi/config.yaml`; re-run anytime with `ovi --setup` or edit the file:

```yaml
llm:
  base_url: http://localhost:11434/v1   # ollama, LM Studio, ...
  model: llama3.2

stt:
  provider: nemotron         # nemotron | whisper
  model: 560ms               # nemotron chunk: 80ms | 160ms | 560ms | 1120ms
                             # whisper: tiny.en | base.en | small.en | medium.en | turbo | distil-large-v3
  silence: 0.75              # seconds of silence that end your turn; lower = snappier, higher = tolerates pauses

tts:
  provider: kokoro           # kokoro | piper
  model: af_heart            # kokoro voice, or a piper voice like en_US-lessac-medium
  speed: 1.0

devices: voice-pe-XXXX.local

transport:
  codec: lc3                 # lc3 | opus | pcm

music:
  services: [spotify]        # optional browser services; youtube always works
```

### 4. Run

```bash
ovi                          # devices from config
ovi voice-pe-XXXX.local      # or pass devices
ovi --scan                   # discover devices on the network
ovi --transport ble          # single BLE device
```

## Configuration layers

Later sources override earlier ones:

1. `~/.ovi/config.yaml`
2. `.env` in the working directory
3. Environment variables — `OVI_` prefix, `__` for nesting: `OVI_LLM__MODEL=gpt-4o-mini`, `OVI_STT__PROVIDER=whisper`, `OVI_MUSIC__SERVICES=spotify,apple`
4. CLI flags — `ovi --llm-model gpt-4o --transport-codec opus --stt-model base.en` (any `section.key` as `--section-key`)

| Path | Contents |
|---|---|
| `~/.ovi/config.yaml` | Configuration |
| `~/.ovi/automations.json` | Scheduled automations |
| `~/.cache/ovi/models/` | Speech models |
| `~/.config/ovi/` | Browser profiles for Spotify / Apple Music |

## Encryption

```bash
ovi --gen-key
```

writes a key to `esphome/secrets.yaml`. Uncomment the `api.encryption` block in the device YAML, reflash, and connect with `ovi voice-pe-XXXX.local::KEY`.

## Agent tools

19 built-in tools: timers (`set_timer`, `check_timer`, `cancel_timer`), `get_current_time`, `calculate`, `unit_convert`, `roll_dice`, `random_number`, `flip_coin`, music (`play_music` with a `service` of youtube / spotify / apple, `pause_music`, `resume_music`, `skip_track`, `stop_music`, `now_playing`) and automations (`create_automation`, `list_automations`, `delete_automation`, `toggle_automation`). Add more through MCP servers (`llm.mcp_servers`) and sub-agents (`llm.agents`); see `mcp.json` and `agents.json`.

## Latency

Everything streams: mic audio is decoded while you talk, the transcript is ready as soon as the VAD closes your turn, LLM tokens are spoken chunk by chunk, and encoded audio is paced to the device 300 ms ahead of playback. After you stop talking the fixed costs are the `stt.silence` window, the LLM's first tokens, and synthesis of the first chunk. The first chunk ends at the first clause boundary so speech starts before the model finishes its first sentence.

Measured with the process pinned to four 3.7 GHz cores of a Core Ultra 9 388H (a sandboxed dev shell; the full chip will be faster), synthesis of a 12-word sentence: Kokoro fp32 0.5 s, Piper medium 0.1 s. The Kokoro int8 pack was three times slower than fp32 on that CPU and is not used. Pick Piper when latency matters more than voice quality. Re-measure on your own hardware with `OVI_TEST_MODELS=1 go test ./internal/tts/ -run Latency -v` (and `OVI_TEST_PIPER=1` for Piper).

## Device features

- On-device wake word detection (microWakeWord) and a "stop" wake word
- LED state feedback, volume control, hardware mute
- Conversation follow-up (`[LISTEN]` token)
- Multi-device wake word arbitration (closest device wins)
- Synchronized multi-room music (NTP-based)

## Development

```bash
go build ./... && go vet ./... && gofmt -l internal cmd     # lint
go test ./...                                                # unit tests, no network or models
OVI_TEST_MODELS=1 go test ./internal/tts/ ./internal/stt/ -run Real   # real Kokoro + Silero (downloads)
OVI_TEST_NEMOTRON=1 go test ./internal/stt/ -run Nemotron    # real Nemotron (460 MB download)
```

## Project structure

```
cmd/ovi/               CLI, serving loop
internal/
    agent/             OpenAI SDK tool-calling loop, built-in tools, sub-agents
    audio/             PipelineOutput interface
    codec/             PCM, LC3 (cgo liblc3), Opus (cgo libopus)
    config/            Layered settings, config file helpers
    device/            Device connection state machine, wake arbitration
    discovery/         mDNS
    dsp/               PCM conversion, resampling
    mcp/               MCP stdio client
    models/            sherpa-onnx model pack downloads
    music/             YouTube (yt-dlp), Spotify/Apple (browser capture), sync group
    pipeline/          STT → Agent → TTS, paced encoded output, speech queue
    scheduler/         Cron automations
    setup/ flash/ console/   Wizard, ESPHome flashing, prompts
    stt/               Silero VAD listen loop, Nemotron (streaming), Whisper (offline)
    transport/         WiFi (TCP) and BLE (GATT)
    tts/               Kokoro / Piper via sherpa-onnx, sentence streaming
esphome/               Device firmware: custom components and per-device YAMLs
```
