# Ovi — Open Voice Assistant

A standalone AI voice assistant that connects directly to ESPHome devices over WiFi or BLE. No Home Assistant required.

Handles the full voice pipeline: wake word detection (on-device) → speech-to-text → AI agent → text-to-speech, with audio streamed back to the device speaker.

## How it works

```
ESPHome Device (wake word) ──► mic audio over WiFi/BLE ──► Ovi Server
                                                              │
                                                        STT (faster-whisper / Nemotron)
                                                              │
                                                        Agent (OpenAI Agents SDK)
                                                              │
                                                        TTS (Piper / pocket-tts)
                                                              │
ESPHome Device (speaker)   ◄── encoded audio ◄───────────────┘
```

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Silero VAD, or [Nemotron](https://build.nvidia.com/nvidia/nemotron-speech-streaming-en) (ONNX streaming). CPU inference.
- **Agent**: [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) — works with any OpenAI-compatible endpoint (OpenAI, ollama, vLLM, LM Studio, etc.). Supports MCP tools, sub-agents, and session memory.
- **TTS**: [Piper](https://github.com/rhasspy/piper) ONNX voices (default) or [pocket-tts](https://github.com/kyutai-labs/pocket-tts) (Kyutai). CPU inference.
- **Transport**: WiFi (plain TCP) or BLE (GATT). Audio codecs: PCM, LC3, Opus.
- **Music**: YouTube Music, Spotify, Apple Music via browser automation. Multi-room synchronized playback.
- **Memory**: Persistent fact extraction and recall (SQLite + embeddings).
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

- Python 3.14+, [uv](https://docs.astral.sh/uv/)
- An ESPHome-compatible device (see above)
- An OpenAI-compatible LLM endpoint

## Quick start

### 1. Install

```bash
uv sync --group dev
```

### 2. Flash a device

Edit `esphome/secrets.yaml` with your WiFi credentials:

```yaml
wifi_ssid: "YourNetwork"
wifi_password: "YourPassword"
```

Flash the device:

```bash
uv run esphome run esphome/voice-pe.yaml
```

### 3. Configure

Edit `.env`:

```bash
OVI_DEVICES=voice-pe-XXXX.local
OVI_OPENAI_BASE_URL=http://localhost:11434/v1   # ollama example
OVI_AGENT_MODEL=llama3.2
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

All settings use the `OVI_` environment variable prefix or the `.env` file.

| Variable | Default | Description |
|---|---|---|
| `OVI_DEVICES` | | Comma-separated devices: `host[:port[:key]]` |
| `OVI_OPENAI_BASE_URL` | | LLM endpoint |
| `OVI_OPENAI_API_KEY` | | API key for the LLM endpoint |
| `OVI_AGENT_MODEL` | `gpt-4o-mini` | Model name |
| `OVI_STT_MODEL` | `base.en` | Whisper model |
| `OVI_STT_PROVIDER` | `whisper` | STT provider (`whisper`, `nemotron`) |
| `OVI_TTS_MODEL` | `en_US-lessac-medium` | Piper voice / pocket-tts voice |
| `OVI_TTS_PROVIDER` | `piper` | TTS provider (`piper`, `pocket`) |
| `OVI_TRANSPORT` | `wifi` | Transport (`wifi`, `ble`) |
| `OVI_CODEC` | `lc3` | Audio codec (`pcm`, `lc3`, `opus`) |
| `OVI_MCP_SERVERS` | | MCP tool servers JSON or `@path/to/file.json` |
| `OVI_AGENTS` | | Sub-agents JSON or `@path/to/agents.json` |
| `OVI_MEMORY_ENABLED` | `true` | Enable persistent memory |

CLI arguments override `.env` values. Run `ovi --help` for all options.

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

- **Speech**: `say` (immediate speech)
- **Timers**: `set_timer`, `check_timer`, `cancel_timer`
- **Time**: `get_current_time`
- **Math**: `calculate`, `unit_convert`
- **Random**: `roll_dice`, `random_number`, `flip_coin`
- **Music**: `play_music`, `pause_music`, `resume_music`, `skip_track`, `stop_music`, `now_playing`
- **Automations**: `create_automation`, `list_automations`, `delete_automation`, `toggle_automation`

Additional tools via MCP servers and sub-agents.

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

## Project structure

```
src/ovi_voice_assistant/
    __main__.py              CLI entry point (ovi command)
    config.py                Settings (OVI_ env prefix)
    voice_assistant.py       STT → Agent → TTS pipeline
    device_connection.py     Device transport bridge + codec
    device_manager.py        Multi-device management + wake arbitration
    discovery.py             mDNS device discovery
    scheduler.py             Cron automations
    pipeline_output.py       Pipeline event types
    codec/                   PCM, LC3, Opus encoding/decoding
    transport/               WiFi (TCP) and BLE (GATT) transports
    stt/                     Whisper + Nemotron speech-to-text
    tts/                     Piper + Pocket text-to-speech
    agent/                   OpenAI Agents SDK + MCP tools
    memory/                  SQLite fact store + embeddings
    music/                   YouTube, Spotify, Apple Music
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
