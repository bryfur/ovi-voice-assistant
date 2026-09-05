# Ovi — Open Voice Assistant

A standalone AI voice assistant that connects directly to ESPHome devices over WiFi or BLE. No Home Assistant required. Written in Go — a single static-ish binary plus a handful of native audio libraries.

Handles the full voice pipeline: wake word detection (on-device) → speech-to-text → AI agent → text-to-speech, with audio streamed back to the device speaker.

## How it works

```
ESPHome Device (wake word) ──► mic audio over WiFi/BLE ──► Ovi Server
                                                              │
                                                        STT (Nemotron / Whisper)
                                                              │
                                                        Agent (OpenAI-compatible, tool calling)
                                                              │
                                                        TTS (Kokoro / Piper)
                                                              │
ESPHome Device (speaker)   ◄── encoded audio ◄───────────────┘
```

- **STT**: [Nemotron Speech 600M](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) streaming RNNT (default, local ONNX inference) or Whisper via any OpenAI-compatible `/audio/transcriptions` endpoint (OpenAI, faster-whisper-server, whisper.cpp server, LocalAI…). Both use Silero VAD.
- **Agent**: built-in tool-calling loop over the OpenAI chat completions API — works with any OpenAI-compatible endpoint (OpenAI, ollama, vLLM, LM Studio, etc.). Supports MCP tools (stdio), sub-agents, and per-session memory.
- **TTS**: [Kokoro](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) 82M ONNX int8 (default) or [Piper](https://github.com/rhasspy/piper) ONNX voices. CPU inference; phonemization via `espeak-ng`.
- **Transport**: WiFi (plain TCP) or BLE (GATT). Audio codecs: PCM, LC3, Opus.
- **Music**: YouTube Music (`yt-dlp` + `ffmpeg`), Spotify and Apple Music via browser tab capture. Multi-room synchronized playback.
- **Memory**: Persistent fact extraction and recall (SQLite + local embeddings).
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
- Native libraries: `liblc3`, `libopus`, `libopusfile` (with `pkg-config` files)
  - Arch: `pacman -S liblc3 opus opusfile`
  - Debian/Ubuntu: `apt install liblc3-dev libopus-dev libopusfile-dev`
  - macOS: `brew install liblc3 opus opusfile`
- Runtime tools: `espeak-ng` (TTS phonemization), `ffmpeg` and `yt-dlp` (music)
- ONNX Runtime 1.29.0 — downloaded automatically into `~/.cache/ovi/onnxruntime` on first run (or point `OVI_ONNXRUNTIME_LIB` at an existing `libonnxruntime.so`)
- An ESPHome-compatible device (see above)
- An OpenAI-compatible LLM endpoint
- [ESPHome](https://esphome.io) (Python) only if you want to flash firmware from `ovi --flash`

## Quick start

### 1. Build

```bash
go install ./cmd/ovi
```

### 2. Flash a device

Edit `esphome/secrets.yaml` with your WiFi credentials:

```yaml
wifi_ssid: "YourNetwork"
wifi_password: "YourPassword"
```

Flash the device (needs `esphome` installed, e.g. `pipx install esphome`):

```bash
esphome run esphome/voice-pe.yaml
```

Or use the guided flow, which also writes `secrets.yaml` and adds the device to your config:

```bash
ovi --flash
```

### 3. Configure

On first run, Ovi launches an interactive setup wizard that walks you through LLM, STT, TTS, device, and codec selection:

```bash
ovi
```

The wizard saves configuration to `~/.ovi/config.yaml`. You can re-run it anytime:

```bash
ovi --setup
```

Or edit the YAML directly:

```yaml
# ~/.ovi/config.yaml

llm:
  base_url: http://localhost:11434/v1   # ollama, LM Studio, etc.
  model: llama3.2

stt:
  provider: nemotron         # nemotron or whisper
  model: int8-dynamic

tts:
  provider: kokoro           # kokoro or piper
  model: af_heart

devices: voice-pe-XXXX.local

transport:
  codec: lc3                 # lc3, opus, or pcm
```

### 4. Run

```bash
ovi
```

Or pass devices on the command line:

```bash
ovi voice-pe-XXXX.local
```

Discover devices on the network:

```bash
ovi --scan
```

BLE mode (single device):

```bash
ovi --transport ble
```

## Configuration

Ovi uses a layered configuration system. Sources are applied in this order (later overrides earlier):

1. **YAML config** — `~/.ovi/config.yaml` (created by `ovi --setup`)
2. **`.env` file** — dotenv in the working directory
3. **Environment variables** — `OVI_` prefix with `__` for nesting
4. **CLI arguments** — `--agent-model`, `--codec`, etc.

### YAML config

The config file lives at `~/.ovi/config.yaml` and uses nested sections:

```yaml
llm:
  api_key: sk-...
  base_url: http://localhost:11434/v1
  model: llama3.2
  mcp_servers: '@~/.ovi/mcp.json'
  agents: '@~/.ovi/agents.json'

stt:
  provider: nemotron    # nemotron, whisper
  model: int8-dynamic   # nemotron: int8-dynamic, int8-static, fp16, fp32
                        # whisper: model name for the transcription endpoint (e.g. whisper-1)
  device: cpu           # cpu, cuda
  base_url: ""          # whisper only: transcription endpoint (empty = llm.base_url)
  api_key: ""           # whisper only (empty = llm.api_key)

tts:
  provider: kokoro      # kokoro, piper
  model: af_heart       # kokoro voice, or piper voice like en_US-lessac-medium

devices: voice-pe-XXXX.local

transport:
  type: wifi            # wifi, ble
  codec: lc3            # lc3, opus, pcm

memory:
  enabled: true
```

### Environment variable overrides

Override any config value using the `OVI_` prefix and `__` as the nesting delimiter:

```bash
OVI_LLM__MODEL=gpt-4o-mini ovi             # override LLM model
OVI_STT__PROVIDER=whisper ovi              # switch STT provider
OVI_TRANSPORT__CODEC=opus ovi               # change codec
OVI_LLM__API_KEY=sk-... ovi                 # set API key without saving to file
```

### CLI overrides

CLI arguments take highest priority:

```bash
ovi --agent-model gpt-4o --codec opus --stt-model whisper-1
```

Run `ovi --help` for all options.

### Storage paths

| Path | Contents |
|---|---|
| `~/.ovi/config.yaml` | Configuration file |
| `~/.ovi/memory.db` | Persistent memory (SQLite) |
| `~/.ovi/automations.json` | Scheduled automations |
| `~/.cache/ovi/onnxruntime/` | ONNX Runtime shared library |
| `~/.cache/ovi/nemotron/`, `~/.cache/ovi/silero/` | STT and VAD models |
| `~/.cache/ovi/kokoro/`, `~/.cache/ovi/piper/` | TTS models and voices |
| `~/.cache/ovi/fastembed/` | Embedding model |
| `~/.config/ovi/` | Browser profiles for Spotify / Apple Music |

## Encryption

Generate an encryption key:

```bash
ovi --gen-key
```

This writes the key to `esphome/secrets.yaml`. Uncomment the encryption block in the device YAML:

```yaml
api:
  encryption:
    key: !secret api_encryption_key
```

Reflash and connect:

```bash
ovi voice-pe-XXXX.local::KEY
```

## Agent tools

The voice assistant includes 19 built-in tools:

- **Timers**: `set_timer`, `check_timer`, `cancel_timer`
- **Time**: `get_current_time`
- **Math**: `calculate`, `unit_convert`
- **Random**: `roll_dice`, `random_number`, `flip_coin`
- **Music**: `play_music`, `pause_music`, `resume_music`, `skip_track`, `stop_music`, `now_playing`
- **Automations**: `create_automation`, `list_automations`, `delete_automation`, `toggle_automation`

Additional tools via MCP servers (`mcp.json`) and sub-agents (`agents.json`).

## Device features

The ESPHome firmware provides:

- On-device wake word detection (microWakeWord)
- "Stop" wake word to interrupt responses
- LED state feedback (listening, thinking, replying, error, muted)
- Volume control (rotary encoder or software)
- Hardware mute switch
- Conversation follow-up (`[LISTEN]` token)
- Multi-device wake word arbitration (closest device wins)
- Synchronized multi-room music playback (NTP-based)

## Development

```bash
go build ./...                                        # build
go vet ./... && gofmt -l internal cmd                 # lint
go test ./...                                         # unit tests (no network, no models)
OVI_TEST_MODELS=1 go test ./internal/stt/ -run RealModel   # downloads ONNX Runtime + Silero and runs them
```

## Project structure

```
cmd/ovi/                     CLI entry point (ovi command)
internal/
    config/                  Layered settings (YAML, .env, OVI_ env, CLI)
    transport/               WiFi (TCP) and BLE (GATT) transports, wire events
    codec/                   PCM, LC3 (cgo liblc3), Opus (cgo libopus)
    audio/                   PipelineOutput interface
    pipeline/                STT → Agent → TTS, paced encoding output, speech queue
    device/                  Device connection state machine + wake arbitration
    stt/                     Silero VAD, Nemotron (ONNX), Whisper (API)
    tts/                     Sentence streaming, espeak-ng phonemizer, Kokoro, Piper
    agent/                   Tool-calling agent, built-in tools, sub-agents
    llm/                     OpenAI-compatible chat/transcription client
    mcp/                     MCP stdio client
    memory/                  SQLite fact store, embeddings, recall/retain
    music/                   YouTube (yt-dlp), Spotify/Apple (browser capture), sync group
    scheduler/               Cron automations
    discovery/               mDNS device discovery
    setup/, flash/, console/ Interactive wizard, ESPHome flashing, prompts
    ort/, dsp/, download/    ONNX Runtime loader, resampling/FFT, model downloads
esphome/
    components/              Custom ESPHome components
        ovi_voice_assistant/ Device-side voice pipeline
        ovi_audio_codec/     LC3/Opus codec for ESP32
    voice-pe.yaml            Voice PE (WiFi)
    voice-pe-ble.yaml        Voice PE (BLE)
    atom-echo.yaml           ATOM Echo
    s3-box-3.yaml            S3-BOX-3
    crowpanel-9.yaml         CrowPanel 9" (ESP32-P4)
    crowpanel-s3-5.yaml      CrowPanel Advance 5" (ESP32-S3)
```

## Migrating from the Python version

- Configuration files (`~/.ovi/config.yaml`, `automations.json`, `memory.db`) are compatible.
- The Whisper provider now talks to a transcription API instead of running faster-whisper in-process; run a local server (e.g. faster-whisper-server or whisper.cpp) or use OpenAI, and set `stt.base_url` if it differs from your LLM endpoint.
- The experimental Qwen3 TTS provider was not ported.
- `espeak-ng`, `ffmpeg` and `yt-dlp` must be installed on the system; previously the Python packages bundled equivalents.
