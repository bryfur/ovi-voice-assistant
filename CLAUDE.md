# Ovi — Open Voice Assistant

AI-powered voice assistant with streaming STT, TTS, and OpenAI Agents SDK. Connects directly to ESPHome devices over WiFi/BLE — no Home Assistant.

## Architecture

Streaming voice pipeline: ESPHome device → server → device.

1. **Transport** (`transport/`) — receives audio from ESPHome devices over WiFi or BLE (bleak). Codec-agnostic: PCM, LC3, or Opus.
2. **DeviceConnection** (`device_connection.py`) — bridges transport to pipeline; handles codec encode/decode and audio pacing.
3. **DeviceManager** (`device_manager.py`) — manages multiple devices; arbitrates competing wake words (0.5s window).
4. **VoiceAssistant** (`voice_assistant.py`) — core pipeline: STT → Agent → TTS, lazy-loads providers.
5. **STT** (`stt/`) — speech-to-text via faster-whisper.
6. **Agent** (`agent/`) — OpenAI Agents SDK with MCP tool support, sub-agents, SQLite session memory.
7. **TTS** (`tts/`) — text-to-speech via Kokoro 82M ONNX int8 (default) or Piper. Streams audio back to device.
8. **Music** (`music/`) — music playback via YouTube Music, Spotify, Apple Music (browser automation + yt-dlp).
9. **Memory** (`memory/`) — persistent fact extraction and recall (SQLite + fastembed embeddings).
10. **Scheduler** (`scheduler.py`) — cron-based proactive automations.

## Configuration

- Pydantic BaseSettings with `OVI_` env prefix + `.env` file
- CLI args in `__main__.py` override env vars for model/provider selection
- Device-specific config (codec, sample rate) in `DeviceConfig`
- ESPHome device YAML configs live in `esphome/` (voice-pe, atom-echo, s3-box-3, crowpanel-9, crowpanel-s3-5)

## Build & Run

```bash
uv sync --group dev          # install all deps including test
uv run ovi                   # run the assistant
uv run pytest src/           # run all tests
uv run pytest src/ovi_voice_assistant/codec/pcm_test.py  # run one file
```

## Code Organization

- **No classes in `__init__.py`**: `__init__.py` files are thin re-export layers only. All classes live in their own files.
- **Class names match file names**: `AudioCodec` → `audio_codec.py`, `MusicPlayer` → `music_player.py`, `DeviceTransport` → `device_transport.py`, etc.

## Testing Standards

- **Framework**: pytest + pytest-asyncio
- **Config**: `pyproject.toml` `[tool.pytest.ini_options]` — `testpaths = ["src"]`, `python_files = ["*_test.py"]`, `asyncio_mode = "auto"`
- **Colocated tests**: every test file lives next to the source file it tests
  - `foo.py` → `foo_test.py` in the same directory
  - `__init__.py` → `__init___test.py` in the same directory
- **One test file per source file**: never combine tests for multiple source files
- **No shared conftest.py**: each test file defines its own fixtures inline
- **Non-unit tests** (integration, e2e) go in a top-level `tests/` directory
- **AAA pattern**: every test uses Arrange, Act, Assert with blank lines between phases:
  ```python
  def test_example(self):
      codec = PcmCodec(16000)          # Arrange

      result = codec.encode(b"\x00")   # Act

      assert result == b"\x00"         # Assert
  ```
- **No real models in tests**: mock heavy dependencies (Whisper, Piper, OpenAI, device transports)
- **Async tests**: use `@pytest.mark.asyncio` and `unittest.mock.AsyncMock`
- **Settings fixture pattern** (inline, not conftest):
  ```python
  @pytest.fixture
  def settings():
      return Settings(_env_file=None, devices="", openai_api_key="test-key")
  ```
- **After any code change**: run the test agent (`/run-agent test`) or `uv run pytest src/` to verify

## Linting

- **Linter/formatter**: ruff (config in `pyproject.toml`)
- **Check**: `uv run ruff check src/` and `uv run ruff format --check src/`
- **Fix**: `uv run ruff check src/ --fix` and `uv run ruff format src/`
- **After any code change**: run the lint agent (`/run-agent lint`) or the commands above

## ESPHome Build & Flash

ESPHome is a dev dependency (`uv sync --group dev`). Components are loaded as local `external_components` in each device YAML — no install step needed.

```bash
# Build + flash over USB
uv run esphome run esphome/voice-pe.yaml

# Build only (no flash)
uv run esphome compile esphome/voice-pe.yaml
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
