# Ovi BLE Voice Assistant Server Specification

This document specifies the BLE GATT protocol used between an Ovi ESPHome device (GATT server/peripheral) and an Ovi voice assistant server (GATT client/central). It contains everything needed to implement a compatible server in any language.

---

## 1. Roles

| Role | Runs on | BLE Role |
|------|---------|----------|
| **Device** | ESP32 (ESPHome firmware) | GATT Server (Peripheral) — advertises service, accepts connections |
| **Server** | PC / Raspberry Pi / etc. | GATT Client (Central) — scans, connects, subscribes to notifications |

The ESPHome device is the BLE **peripheral**. The voice assistant server is the BLE **central** that connects to it.

---

## 2. GATT Service

### Service UUID

```
BA5E0001-FADA-4C14-A34C-1AE0F0A0A0A0
```

The device advertises this UUID. The server scans for devices advertising this service.

### Characteristics

| Name | UUID | Properties | Direction | Purpose |
|------|------|------------|-----------|---------|
| **AUDIO_TX** | `BA5E0002-FADA-4C14-A34C-1AE0F0A0A0A0` | NOTIFY | Device → Server | Mic audio frames |
| **AUDIO_RX** | `BA5E0003-FADA-4C14-A34C-1AE0F0A0A0A0` | WRITE_NO_RESPONSE | Server → Device | Speaker audio frames |
| **CONTROL** | `BA5E0004-FADA-4C14-A34C-1AE0F0A0A0A0` | READ, WRITE, NOTIFY | Bidirectional | Control events |

Both AUDIO_TX and CONTROL have a BLE 2902 descriptor (CCCD) for enabling notifications.

The GATT service is created with `15` handles (enough for 3 characteristics + descriptors):
```cpp
this->service_ = this->ble_server_->create_service(SERVICE_UUID, true, 15);
```

---

## 3. Discovery

The server discovers devices by scanning for BLE peripherals that advertise the service UUID `BA5E0001-FADA-4C14-A34C-1AE0F0A0A0A0`.

Optionally, the server can filter by:
- **Device name** (substring match, case-insensitive)
- **Device address** (exact MAC match)

If scanning by name and no match is found, fall back to the first device advertising the service UUID.

**Scan timeout**: 10 seconds (recommended).

---

## 4. Connection Lifecycle

### 4.1 Initial Connection

```
1. Server scans for device advertising SERVICE_UUID
2. Server connects (BleakClient or equivalent)
3. Server subscribes to AUDIO_TX notifications (mic audio)
4. Server subscribes to CONTROL notifications (events)
5. Server sends AUDIO_CONFIG event (speaker codec parameters)
6. Device receives AUDIO_CONFIG, configures decoder, replies with MIC_CONFIG
7. Server receives MIC_CONFIG, configures mic decoder
8. Ready — device shows "idle" state, awaiting wake word
```

### 4.2 Disconnection

When the BLE connection drops:
- **Device side**: Stops mic and speaker, returns state machine to IDLE, fires `on_client_disconnected` trigger.
- **Server side**: Fires disconnect callback, schedules reconnect.

### 4.3 Auto-Reconnect

The server should auto-reconnect on disconnect:
1. Wait **2 seconds** before attempting reconnect
2. Re-scan for the device (MAC may change with random addressing)
3. Re-establish connection (subscribe, send AUDIO_CONFIG, etc.)
4. Retry indefinitely until connected or stopped

---

## 5. Control Event Protocol

Control events are sent via the **CONTROL** characteristic. The format is:

```
[1 byte: event_type][0+ bytes: payload]
```

Both directions (server→device write, device→server notify) use the same format.

### 5.1 Event Types

| Value | Name | Direction | Description |
|-------|------|-----------|-------------|
| `0x01` | WAKE_WORD | Device → Server | Wake word detected, start pipeline |
| `0x02` | VAD_START | Server → Device | Speech detected (user started talking) |
| `0x03` | MIC_STOP | Server → Device | Stop recording mic audio |
| `0x04` | TTS_START | Server → Device | Speaker audio stream starting |
| `0x05` | TTS_END | Server → Device | Speaker audio stream complete |
| `0x06` | CONTINUE | Server → Device | Keep listening (follow-up conversation) |
| `0x07` | ERROR | Server → Device | Error occurred |
| `0x08` | AUDIO_CONFIG | Bidirectional | Speaker/decoder codec configuration |
| `0x09` | MIC_CONFIG | Bidirectional | Mic/encoder codec configuration |
| `0x0A` | WAKE_ABORT | Server → Device | Abort wake (another device won arbitration) |
| `0x0B` | SYNC_PLAY | Server → Device | Start playback at NTP timestamp |

### 5.2 Event Payloads

#### WAKE_WORD (0x01) — Device → Server

```
[2 bytes LE: peak_energy]
[2 bytes LE: ambient_energy]
[N bytes: wake_word UTF-8 string]
```

- `peak_energy` — Fast EMA (~80ms window, alpha ~0.25) of absolute sample values. Tracks speech peaks.
- `ambient_energy` — Slow EMA (~1.3s window, alpha ~1/64) of absolute sample values. Tracks ambient noise floor.
- `wake_word` — UTF-8 string, **not** null-terminated. Length is inferred from remaining payload bytes.

The energy values are updated continuously from mic data, even during IDLE. The EMA formulas on the device:
```cpp
// chunk_energy = mean(abs(samples)) for each mic callback chunk
mic_energy_ = (mic_energy_ * 3 + chunk_energy) / 4;           // fast EMA
mic_energy_ambient_ = (mic_energy_ambient_ * 63 + chunk_energy) / 64;  // slow EMA
```

**Wake word arbitration score** (computed on server):
```python
score = peak * 1000 // max(ambient, 1)
```

Higher score = closer to the device. The ratio normalizes across different mic gains so the **closest** device wins, not the loudest mic.

**Empty wake word payload**: During a follow-up (continue conversation), the device sends WAKE_WORD with no peak/ambient/wake_word payload (0 bytes) to signal a new listening cycle.

#### VAD_START (0x02) — Server → Device

No payload. Informs the device that the server's VAD has detected the start of speech.

#### MIC_STOP (0x03) — Server → Device

No payload. Tells the device to stop streaming mic audio. The device transitions from `STREAMING_MICROPHONE` to `STOP_MICROPHONE` → `AWAITING_RESPONSE`.

#### TTS_START (0x04) — Server → Device

No payload. Signals that speaker audio frames will follow.

**Important**: Before sending TTS_START, the server should send an `AUDIO_CONFIG` event with the speaker codec parameters (sample rate, nbyte, codec, channels). The device uses this to configure its audio decoder. The `_EncodingOutput` class does this automatically.

#### TTS_END (0x05) — Server → Device

No payload. Signals that all speaker audio has been sent. The device drains its speaker buffer, then transitions to `STOP`.

#### CONTINUE (0x06) — Server → Device

No payload. Tells the device to keep listening after TTS playback completes (follow-up conversation). The device sets an internal flag. When TTS finishes and the speaker buffer is drained, instead of returning to IDLE, the device automatically:
1. Sends a new `WAKE_WORD` event (empty payload)
2. Restarts the microphone
3. Begins streaming mic audio again

#### ERROR (0x07) — Server → Device

```
[N bytes: error_code UTF-8, null-terminated]
[M bytes: error_message UTF-8]
```

Example: `"stt-no-text-recognized\0User did not speak clearly"`

On error, the device stops mic/speaker and returns to IDLE (via `STOP` state).

#### AUDIO_CONFIG (0x08) — Bidirectional

```
[4 bytes LE: sample_rate]       (uint32)
[2 bytes LE: encoded_frame_bytes] (uint16, per-channel nbyte)
[1 byte: codec_id]              (0=PCM, 1=LC3, 2=Opus)
[1 byte: channels]              (1=mono, 2=stereo; optional, defaults to 1 if missing)
```

**Server → Device**: Sent before TTS to configure the device's audio decoder. The `channels` field (byte 8) is optional — if the payload is only 7 bytes, the device defaults to 1 channel.

**Device → Server**: The device responds with MIC_CONFIG after receiving AUDIO_CONFIG (not with another AUDIO_CONFIG).

The `encoded_frame_bytes` field is the **per-channel** byte count. For LC3 stereo, the actual on-wire frame size is `encoded_frame_bytes * channels`.

struct format string (Python): `"<IHBB"` — little-endian uint32 + uint16 + uint8 + uint8.

#### MIC_CONFIG (0x09) — Bidirectional

```
[4 bytes LE: sample_rate]       (uint32)
[2 bytes LE: encoded_frame_bytes] (uint16, per-channel nbyte)
[1 byte: codec_id]              (0=PCM, 1=LC3, 2=Opus)
```

**Device → Server**: Sent as a response to AUDIO_CONFIG. Tells the server the mic's codec parameters.

**Server → Device**: Optional. The server can request a mic codec change by sending MIC_CONFIG with preferred parameters. The device will reconfigure its encoder and reply with a new MIC_CONFIG.

struct format string (Python): `"<IHB"` — little-endian uint32 + uint16 + uint8.

#### WAKE_ABORT (0x0A) — Server → Device

No payload. Sent when another device won wake-word arbitration. The device stops mic/speaker, restarts wake word detection if applicable, and returns to IDLE.

#### SYNC_PLAY (0x0B) — Server → Device

```
[8 bytes LE: epoch_ms]  (uint64, NTP epoch milliseconds)
```

For multi-room synchronized playback. The device buffers incoming audio into its ring buffer until the NTP timestamp is reached, then starts the speaker and drains the buffer. Requires the device to have NTP time configured.

---

## 6. Audio Protocol

### 6.1 Mic Audio (Device → Server via AUDIO_TX)

Each GATT notification on AUDIO_TX contains one or more encoded audio frames. There is **no framing header** — the raw encoded bytes are the notification payload.

- **With codec (LC3/Opus)**: Each notification is exactly one encoded frame. For LC3 mono at 40 nbyte: 40 bytes per notification. For LC3 stereo: `nbyte * channels` bytes.
- **With PCM passthrough**: Each notification is a chunk of raw PCM, up to `SEND_BUFFER_SIZE` (244) bytes, 16-bit sample aligned.

### 6.2 Speaker Audio (Server → Device via AUDIO_RX)

Written to AUDIO_RX using write-without-response. Each write contains raw encoded audio bytes — **no framing header**.

- Each write must fit within the BLE MTU. The server should chunk audio to fit within `mtu_payload` (default 512 bytes, safe for most BLE stacks).
- The device calls `handle_audio_rx_()` for each write, decoding if a codec is configured.

### 6.3 BLE vs WiFi Framing Difference

On **WiFi (TCP)**, all messages use length-prefixed frames with a type byte:
```
[2 bytes LE: payload_length]
[1 byte: frame_type (0x01-0x0B for events, 0x20 for mic, 0x21 for speaker)]
[N bytes: frame payload]
```

On **BLE**, there are **no length prefixes or type bytes** for audio. Each characteristic has a dedicated purpose:
- AUDIO_TX = mic audio (raw codec frames)
- AUDIO_RX = speaker audio (raw codec frames)
- CONTROL = events (first byte is event_type)

This is the key architectural difference: BLE uses separate characteristics instead of multiplexing message types over one stream.

---

## 7. Audio Codec Details

### 7.1 Codec IDs

| ID | Name | Description |
|----|------|-------------|
| 0 | PCM | Raw 16-bit signed LE samples, no compression |
| 1 | LC3 | Low Complexity Communication Codec (Bluetooth LE Audio standard) |
| 2 | Opus | Opus codec (variable bitrate) |

### 7.2 Microphone Parameters (Fixed)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Bit depth | 16-bit signed LE |
| Channels | 1 (mono) |

These are hardcoded on the device and cannot be changed via protocol.

### 7.3 Speaker Parameters (Configurable)

Configured via AUDIO_CONFIG before each TTS stream. Common configurations:

| Mode | Sample Rate | Channels | Codec | nbyte | Bitrate |
|------|-------------|----------|-------|-------|---------|
| TTS (voice) | 16,000 Hz | 1 (mono) | LC3 | 40 | 32 kbps |
| TTS (voice) | 22,050 Hz | 1 (mono) | LC3 | 40 | 32 kbps |
| Music | 48,000 Hz | 2 (stereo) | LC3 | 40 | 128 kbps |

The device creates/reconfigures its decoder on every AUDIO_CONFIG, so the server can switch between voice and music codec settings freely.

### 7.4 LC3 Codec

- **Frame duration**: 10 ms (fixed)
- **Default nbyte**: 40 bytes per channel
- **Valid sample rates**: 8000, 16000, 24000, 32000, 48000 Hz
- **Frame samples**: `sample_rate * frame_duration_ms / 1000`
  - 16 kHz: 160 samples/frame
  - 48 kHz: 480 samples/frame
- **PCM bytes per frame**: `frame_samples * channels * 2` (16-bit)
  - 16 kHz mono: 320 bytes
  - 48 kHz stereo: 1920 bytes
- **Encoded bytes per frame (on wire)**: `nbyte * channels`
  - Mono (nbyte=40): 40 bytes
  - Stereo (nbyte=40): 80 bytes
- **`encoded_frame_bytes` in AUDIO_CONFIG**: Per-channel value (40), NOT total (80). Matches ESP32's `esp_lc3_dec_cfg_t.nbyte` field.

### 7.5 Opus Codec

- **Frame duration**: 20 ms (fixed)
- **Default nbyte**: 80 bytes (~32 kbps VBR at 16 kHz)
- **Valid sample rates**: 8000, 12000, 16000, 24000, 48000 Hz
- **Frame samples**: `sample_rate * frame_duration_ms / 1000`
  - 16 kHz: 320 samples/frame
- **PCM bytes per frame**: `frame_samples * channels * 2`
  - 16 kHz mono: 640 bytes
- **Encoded bytes per frame**: Variable (VBR), but nbyte is configured for target bitrate

### 7.6 PCM Passthrough

- No encoding/decoding
- **Send chunk size**: 244 bytes per BLE notification (device side)
- **Alignment**: Must be 2-byte aligned (16-bit samples)
- **PCM send duration**: 20 ms equivalent chunks

---

## 8. Device State Machine

```
IDLE ──────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │ request_start(wake_word)                                           │
  │   → send WAKE_WORD event                                           │
  ▼                                                                    │
START_MICROPHONE                                                       │
  │ start mic, reset ring buffer & encoder                             │
  │ fire on_listening trigger                                          │
  ▼                                                                    │
STREAMING_MICROPHONE                                                   │
  │ continuously read mic ring buffer → encode → send via AUDIO_TX     │
  │                                                                    │
  │ on MIC_STOP event from server:                                     │
  ▼                                                                    │
STOP_MICROPHONE                                                        │
  │ stop mic source                                                    │
  │ (shared_audio_bus: also stop microWakeWord, wait for I2S release)  │
  ▼                                                                    │
AWAITING_RESPONSE                                                      │
  │ waiting for TTS_START from server                                  │
  │                                                                    │
  │ on TTS_START event:                                                │
  ▼                                                                    │
STREAMING_RESPONSE                                                     │
  │ receive speaker audio via AUDIO_RX → decode → play on speaker      │
  │ (sync mode: buffer until NTP timestamp, then start speaker)        │
  │                                                                    │
  │ on TTS_END event: set stream_ended flag                            │
  │ wait for speaker buffer to drain                                   │
  ▼                                                                    │
STOP                                                                   │
  │ stop speaker                                                       │
  │ (shared_audio_bus: restart microWakeWord)                          │
  │                                                                    │
  │ if continue_conversation flag set:                                 │
  │   → send WAKE_WORD (empty), restart mic → START_MICROPHONE         │
  │ else:                                                              │
  └────────────────────────────────────────────────────────────────────┘
```

### State Transitions Triggered by Events

| Current State | Event Received | New State |
|---------------|----------------|-----------|
| IDLE | (request_start called) | START_MICROPHONE |
| STREAMING_MICROPHONE | MIC_STOP | STOP_MICROPHONE |
| AWAITING_RESPONSE | TTS_START | STREAMING_RESPONSE |
| STREAMING_RESPONSE | TTS_END + buffer drained | STOP |
| STOP | (continue flag) | START_MICROPHONE |
| STOP | (no continue) | IDLE |
| Any | ERROR | STOP → IDLE |
| Any | WAKE_ABORT | IDLE |
| Any | (BLE disconnect) | IDLE |

---

## 9. Voice Pipeline Sequence

A complete voice interaction over BLE:

```
Server                              Device
  │                                    │
  │◄── BLE connect ────────────────────│
  │                                    │
  │── subscribe AUDIO_TX notify ──────►│
  │── subscribe CONTROL notify ───────►│
  │                                    │
  │── AUDIO_CONFIG(16kHz,40,LC3,1) ──►│ configure decoder
  │◄── MIC_CONFIG(16kHz,40,LC3) ──────│ report mic codec
  │                                    │
  │         ... idle, waiting for wake word ...
  │                                    │
  │◄── WAKE_WORD(peak,ambient,"hey") ─│ wake word detected
  │                                    │ → START_MICROPHONE
  │                                    │ → STREAMING_MICROPHONE
  │◄── AUDIO_TX (encoded mic frame) ──│
  │◄── AUDIO_TX (encoded mic frame) ──│
  │◄── AUDIO_TX (encoded mic frame) ──│
  │    ... (decode, feed to STT) ...   │
  │                                    │
  │── VAD_START ──────────────────────►│ (optional UI feedback)
  │                                    │
  │    ... (STT detects end of speech) │
  │                                    │
  │── MIC_STOP ───────────────────────►│ → STOP_MICROPHONE
  │                                    │ → AWAITING_RESPONSE
  │                                    │
  │    ... (run agent, generate TTS) ...
  │                                    │
  │── AUDIO_CONFIG(22050,40,LC3,1) ──►│ reconfigure decoder for TTS rate
  │── TTS_START ──────────────────────►│ → STREAMING_RESPONSE
  │                                    │
  │── AUDIO_RX (encoded TTS frame) ──►│ decode, play on speaker
  │── AUDIO_RX (encoded TTS frame) ──►│
  │── AUDIO_RX (encoded TTS frame) ──►│
  │    ... (paced to real-time) ...    │
  │                                    │
  │── TTS_END ────────────────────────►│ stream_ended = true
  │                                    │ drain speaker buffer
  │                                    │ → STOP → IDLE
```

### With Follow-up Conversation

```
  │── CONTINUE ───────────────────────►│ set continue flag
  │── TTS_END ────────────────────────►│ stream_ended = true
  │                                    │ → STOP
  │                                    │ continue flag set, so:
  │◄── WAKE_WORD (empty payload) ─────│ → START_MICROPHONE
  │◄── AUDIO_TX (mic frames) ─────────│ → STREAMING_MICROPHONE
  │    ... repeat pipeline ...         │
```

### Announcement (Server-Initiated TTS)

The server can push TTS to the device at any time, even from IDLE:

```
  │── AUDIO_CONFIG(rate,nbyte,codec) ►│ configure decoder
  │── TTS_START ──────────────────────►│ → STREAMING_RESPONSE (from IDLE)
  │── AUDIO_RX (audio frames) ───────►│
  │── TTS_END ────────────────────────►│ → STOP → IDLE
```

---

## 10. Audio Pacing (Server Side)

The server must pace speaker audio to real-time to avoid overflowing the device's speaker buffer. The algorithm:

```python
LEAD_TIME = 0.3  # seconds ahead of real-time

frame_count = 0
t0 = None

for each encoded_frame:
    if frame_count == 0:
        t0 = time.monotonic()
    frame_count += 1

    send_audio(encoded_frame)

    # How far ahead of real-time are we?
    ahead = (frame_count * frame_duration_sec) - (time.monotonic() - t0)

    # Only sleep if we're more than LEAD_TIME ahead
    if ahead > LEAD_TIME:
        sleep(ahead - LEAD_TIME)
```

This keeps a 300ms audio buffer on the device to absorb BLE jitter while preventing buffer overflow.

### Flushing

At the end of TTS, if there's a partial PCM frame in the buffer, pad it with zero bytes to fill a complete codec frame, encode, and send.

---

## 11. Multi-Device Wake Word Arbitration

When multiple BLE devices detect the same wake word within a short window:

1. Each device sends `WAKE_WORD` with peak/ambient energy values
2. The server buffers incoming wake events for **0.5 seconds**
3. Server computes score for each device: `score = peak * 1000 // max(ambient, 1)`
4. **Highest score wins** (closest device — high peak/ambient ratio means speaker is near this mic)
5. Server sends to winner: starts the voice pipeline
6. Server sends to losers: `WAKE_ABORT` event

The peak/ambient ratio normalizes across different mic hardware gains, so the selection is based on proximity, not raw volume.

---

## 12. Buffer Sizes and Timing

| Buffer | Size | Duration |
|--------|------|----------|
| Mic ring buffer (device) | 16,384 bytes | ~512 ms @ 16 kHz 16-bit mono |
| Speaker ring buffer (device) | 65,536 bytes (default) | Configurable via `speaker_buffer_size` |
| BLE send buffer / MTU payload | 244 bytes (device), 512 bytes (server) | Per-notification/write |
| Max encoded frame | 244 bytes | Fits in one BLE notification |
| Max PCM frame (decode buffer) | 3,840 bytes | 48 kHz * 20 ms * 2 bytes * 2 channels |

### BLE MTU Considerations

- **BLE ATT MTU**: Typically 247 bytes after negotiation (244 bytes usable payload after 3-byte ATT header)
- **Device sends**: Up to 244 bytes per notification (one encoded frame, always fits)
- **Server sends**: Chunks audio writes to `mtu_payload` (default 512). If the BLE stack supports larger MTU, a single write can contain multiple codec frames.
- The server should discover/negotiate MTU after connection for optimal throughput

---

## 13. Error Handling

### Error Event Format

```
[1 byte: 0x07]
[N bytes: error_code (UTF-8, null-terminated)]
[M bytes: error_message (UTF-8)]
```

The null byte (`\0`) separates the code from the message. If no null byte is present, the entire payload is the error code with no message.

### Common Error Codes

| Code | Meaning |
|------|---------|
| `stt-no-text-recognized` | STT ran but produced no text (user was silent or unintelligible) |

### Device Behavior on Error

The device stops streaming (mic or speaker), transitions through `STOP` to `IDLE`, and fires the `on_error` automation trigger with the code and message.

---

## 14. Configuration Constants Summary

```
# GATT UUIDs
SERVICE_UUID       = "BA5E0001-FADA-4C14-A34C-1AE0F0A0A0A0"
AUDIO_TX_UUID      = "BA5E0002-FADA-4C14-A34C-1AE0F0A0A0A0"  # notify
AUDIO_RX_UUID      = "BA5E0003-FADA-4C14-A34C-1AE0F0A0A0A0"  # write-no-response
CONTROL_UUID       = "BA5E0004-FADA-4C14-A34C-1AE0F0A0A0A0"  # read/write/notify

# Event type bytes
EVT_WAKE_WORD      = 0x01
EVT_VAD_START      = 0x02
EVT_MIC_STOP       = 0x03
EVT_TTS_START      = 0x04
EVT_TTS_END        = 0x05
EVT_CONTINUE       = 0x06
EVT_ERROR          = 0x07
EVT_AUDIO_CONFIG   = 0x08
EVT_MIC_CONFIG     = 0x09
EVT_WAKE_ABORT     = 0x0A
EVT_SYNC_PLAY      = 0x0B

# Codec IDs
CODEC_PCM          = 0
CODEC_LC3          = 1
CODEC_OPUS         = 2

# Mic parameters (fixed on device)
MIC_SAMPLE_RATE    = 16000  # Hz
MIC_CHANNELS       = 1
MIC_BITS           = 16

# LC3 defaults
LC3_FRAME_DURATION_MS = 10
LC3_DEFAULT_NBYTE     = 40   # 32 kbps per channel
LC3_MUSIC_NBYTE       = 40   # 64 kbps per channel (stereo = 128 kbps)

# Opus defaults
OPUS_FRAME_DURATION_MS = 20
OPUS_DEFAULT_NBYTE     = 80  # ~32 kbps VBR

# Timing
LEAD_TIME           = 0.3   # seconds of audio buffer ahead of real-time
RECONNECT_DELAY     = 2.0   # seconds between reconnect attempts
SCAN_TIMEOUT        = 10.0  # seconds for BLE device scan
ARBITRATION_WINDOW  = 0.5   # seconds to collect wake events from all devices

# Buffer sizes
MIC_RING_BUFFER     = 16384   # bytes (~512 ms @ 16 kHz mono 16-bit)
SPK_BUFFER_DEFAULT  = 65536   # bytes
BLE_MTU_PAYLOAD     = 512     # safe default for write-without-response
MAX_ENCODED_FRAME   = 244     # bytes (fits in one BLE notification)
```

---

## 15. Comparison: BLE vs WiFi Transport

| Aspect | WiFi (TCP) | BLE (GATT) |
|--------|-----------|-------------|
| Device role | TCP server (port 6055) | GATT server (peripheral) |
| Server role | TCP client | GATT client (central) |
| Framing | Length-prefixed: `[2B LE len][payload]` | Per-characteristic (no length prefix) |
| Audio type byte | `0x20` (mic), `0x21` (speaker) in payload | Not needed — separate characteristics |
| Event type byte | First byte of payload | First byte of CONTROL write/notify |
| Audio delivery | Payload bytes after type byte | Full notification/write payload |
| Discovery | mDNS on local network | BLE scan for service UUID |
| Reconnect | TCP reconnect to IP:port | BLE re-scan + connect |
| Latency | Lower (TCP is fast on LAN) | Higher (BLE connection intervals) |
| Range | LAN (unlimited with routing) | ~10-30 meters |
| Multi-client | One TCP client at a time | One BLE central at a time |

### Key Protocol Difference

On WiFi, **all** data (audio + events) flows over one TCP stream, multiplexed with type bytes (`0x01`-`0x0B` for events, `0x20`/`0x21` for audio).

On BLE, data is **separated by characteristic**:
- AUDIO_TX → mic audio only
- AUDIO_RX → speaker audio only
- CONTROL → events only

No type byte is needed for audio on BLE since each characteristic has a single purpose.

---

## 16. Implementation Checklist

A complete BLE server implementation needs:

- [ ] **BLE scanning** — Discover devices advertising `BA5E0001-...`
- [ ] **GATT client** — Connect, subscribe to AUDIO_TX + CONTROL notifications
- [ ] **Event parser** — Parse `[event_type][payload]` from CONTROL notifications
- [ ] **Event sender** — Write `[event_type][payload]` to CONTROL characteristic
- [ ] **Audio receiver** — Decode AUDIO_TX notifications (raw codec frames)
- [ ] **Audio sender** — Write encoded frames to AUDIO_RX (chunked to MTU)
- [ ] **Codec support** — LC3 encoder/decoder (required), Opus (optional), PCM (trivial)
- [ ] **AUDIO_CONFIG exchange** — Send speaker config on connect, parse MIC_CONFIG response
- [ ] **Audio pacing** — Rate-limit speaker audio writes with 300ms lead time
- [ ] **State tracking** — Track device state for proper event sequencing
- [ ] **Wake word handling** — Parse peak/ambient energy, compute arbitration score
- [ ] **Multi-device arbitration** — 0.5s window, highest score wins, WAKE_ABORT to losers
- [ ] **Continue conversation** — Send CONTINUE before TTS_END for follow-ups
- [ ] **Error handling** — Format and send ERROR events with code + message
- [ ] **Auto-reconnect** — Re-scan and reconnect on BLE disconnect
- [ ] **Flush on TTS end** — Pad partial PCM frames to full codec frame before final send

### Library Recommendations by Language

| Language | BLE Library | LC3 Library |
|----------|-------------|-------------|
| Python | bleak | lc3-python (liblc3 bindings) |
| Rust | btleplug | lc3-codec |
| Go | tinygo bluetooth | (C bindings to liblc3) |
| C/C++ | BlueZ D-Bus API | liblc3 |
| JavaScript/Node | noble | (WebAssembly liblc3) |
| Swift | CoreBluetooth | (C bindings to liblc3) |
| Kotlin | Android BLE API | android.bluetooth.le (built-in LC3 on Android 13+) |
