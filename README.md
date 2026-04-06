# Open Voice Assistant

A standalone AI voice assistant for the [Home Assistant Voice Preview Edition](https://www.home-assistant.io/voice-pe/) (Voice PE). Runs entirely on your local network with no Home Assistant dependency.

The server connects directly to the Voice PE over the ESPHome native API, handling the full voice pipeline: wake word detection (on-device), speech-to-text, AI agent, and text-to-speech, with audio streamed back to the device speaker.

## How it works

```
Voice PE (wake word) --> mic audio over ESPHome API --> Server
                                                         |
                                                   faster-whisper (STT)
                                                         |
                                                   OpenAI Agents SDK (LLM)
                                                         |
                                                   TTS (Piper / pocket-tts)
                                                         |
Voice PE (speaker)   <-- audio over ESPHome API  <-------+
```

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Silero VAD for end-of-speech detection. Runs on CPU.
- **Agent**: [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) with any OpenAI-compatible endpoint (OpenAI, ollama, vLLM, LM Studio, etc.)
- **TTS**: [Piper](https://github.com/rhasspy/piper) ONNX voices (default) or [pocket-tts](https://github.com/kyutai-labs/pocket-tts) (Kyutai). Runs on CPU.
- **Device**: Connects via [aioesphomeapi](https://github.com/esphome/aioesphomeapi). Supports multiple devices, auto-reconnect, and encrypted connections.

## Requirements

- Python 3.14+, [uv](https://docs.astral.sh/uv/)
- A Home Assistant Voice Preview Edition device
- An OpenAI-compatible LLM endpoint

## Quick start

### 1. Install

```bash
uv sync --group dev
```

This installs the project and all dependencies (including ESPHome for flashing).

### 2. Flash the Voice PE

Edit `esphome/secrets.yaml` with your WiFi credentials:

```yaml
wifi_ssid: "YourNetwork"
wifi_password: "YourPassword"
```

Flash the device:

```bash
esphome run esphome/voice-pe.yaml
```

### 3. Configure the server

Edit `.env` and uncomment/set your LLM endpoint and device:

```bash
OVA_DEVICES=voice-pe-XXXX.local
OVA_OPENAI_BASE_URL=http://localhost:11434/v1   # ollama example
OVA_AGENT_MODEL=llama3.2
```

### 4. Run

```bash
ova
```

Or pass devices on the command line:

```bash
ova voice-pe-XXXX.local
```

You can also discover devices on the network:

```bash
ova --scan
```

## Configuration

All settings can be configured via environment variables (prefix `OVA_`) or the `.env` file.

| Variable | Default | Description |
|---|---|---|
| `OVA_DEVICES` | | Comma-separated devices: `host[:port[:key]]` |
| `OVA_OPENAI_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint |
| `OVA_OPENAI_API_KEY` | | API key for the LLM endpoint |
| `OVA_AGENT_MODEL` | `gpt-4o-mini` | Model name |
| `OVA_STT_MODEL` | `base.en` | Whisper: `tiny.en`, `base.en`, `small.en`, `medium.en` |
| `OVA_TTS_MODEL` | `en_US-lessac-medium` | Piper voice name / pocket-tts voice prompt |
| `OVA_STT_PROVIDER` | `whisper` | STT provider |
| `OVA_TTS_PROVIDER` | `piper` | TTS provider (`piper`, `pocket`) |

CLI arguments override `.env` values. Run `ova --help` for all options.

> **Note**: All commands assume you've run `uv sync --group dev` first. If you prefer not to install, you can use `uv run ova` instead.

## Encryption

Generate an encryption key and add it to your ESPHome secrets:

```bash
ova --gen-key
```

This writes the key to `esphome/secrets.yaml`. Uncomment the encryption block in `esphome/voice-pe.yaml`:

```yaml
api:
  encryption:
    key: !secret api_encryption_key
```

Reflash the device, then include the key in your device config:

```
ova voice-pe-XXXX.local::KEY

or

OVA_DEVICES=voice-pe-XXXX.local::KEY
ova
```

## Device features

The included ESPHome configuration provides:

- On-device wake word detection (Okay Nabu, Hey Jarvis, Hey Mycroft)
- "Stop" wake word to interrupt responses
- 12-LED RGB ring with animated state feedback (listening, thinking, replying, error, muted)
- Rotary dial volume control with LED arc display
- Hardware mute switch
- Center button (single press: trigger/stop, double press: toggle mute)
- Speaker with mixer (voice + media player for wake sounds)
- Conversation follow-up (agent can request the mic to reactivate)

## Project structure

```
src/open_voice_assistant/
    __main__.py              CLI entry point
    config.py                Settings (env/CLI)
    device.py                ESPHome device connection
    device_manager.py        Multi-device management
    voice_assistant.py           STT -> Agent -> TTS orchestration
    discovery.py             mDNS device discovery
    agent/assistant.py       OpenAI Agents SDK wrapper
    stt/                     STT interface + faster-whisper
    tts/                     TTS interface + Piper / pocket-tts
esphome/
    voice-pe.yaml            ESPHome firmware for Voice PE
    secrets.yaml             WiFi + encryption credentials (gitignored)
```
