/**
 * Ovi Voice Assistant — Unified ESPHome component
 *
 * Supports both WiFi (plain TCP server) and BLE transport via compile-time
 * #ifdef USE_OVI_WIFI / USE_OVI_BLE.  Audio codec is pluggable via
 * ovi_audio_codec (LC3, Opus, or raw PCM passthrough).
 *
 * WiFi mode opens a TCP server on a configurable port (default 6055).
 * The Python server connects as a TCP client.  Messages are length-prefixed
 * binary frames:  [2 bytes LE length][payload].
 * Payload byte 0 is the message type (control event or audio).
 *
 * BLE mode uses a GATT service with audio TX/RX and control characteristics.
 */

#pragma once

#include "esphome/core/component.h"
#include "esphome/core/automation.h"
#include "esphome/core/ring_buffer.h"
#include "esphome/components/microphone/microphone_source.h"
#include "esphome/components/speaker/speaker.h"
#include "esphome/components/ovi_audio_codec/audio_codec.h"

#ifdef USE_TIME
#include "esphome/components/time/real_time_clock.h"
#endif

#ifdef USE_OVI_WIFI
#include "esphome/components/socket/socket.h"
#endif

#ifdef USE_OVI_BLE
#include "esphome/components/esp32_ble_server/ble_server.h"
#include "esphome/components/esp32_ble_server/ble_characteristic.h"
#include "esphome/components/esp32_ble_server/ble_service.h"
#endif

#ifdef USE_MICRO_WAKE_WORD
#include "esphome/components/micro_wake_word/micro_wake_word.h"
#endif

namespace esphome {
namespace ovi_voice_assistant {

// Codec type passed from Python codegen (mirrors ovi_audio_codec::CodecType values)
enum CodecTypeConfig : uint8_t {
  CODEC_PCM = 0,
  CODEC_LC3 = 1,
  CODEC_OPUS = 2,
};

// Control event types (shared with server, used by both WiFi and BLE)
static const uint8_t EVT_WAKE_WORD     = 0x01;  // device→server
static const uint8_t EVT_VAD_START    = 0x02;  // server→device: speech detected
static const uint8_t EVT_MIC_STOP      = 0x03;  // server→device
static const uint8_t EVT_TTS_START     = 0x04;  // server→device
static const uint8_t EVT_TTS_END       = 0x05;  // server→device
static const uint8_t EVT_CONTINUE      = 0x06;  // server→device
static const uint8_t EVT_ERROR         = 0x07;  // server→device
static const uint8_t EVT_AUDIO_CONFIG  = 0x08;  // bidirectional
static const uint8_t EVT_MIC_CONFIG    = 0x09;  // bidirectional
static const uint8_t EVT_WAKE_ABORT    = 0x0A;  // server→device: abort wake (another device won)
static const uint8_t EVT_SYNC_PLAY     = 0x0B;  // server→device: start playback at NTP timestamp (8B LE ms)

// Audio frame type bytes (used in TCP framing and BLE)
static const uint8_t FRAME_TYPE_MIC_AUDIO     = 0x20;
static const uint8_t FRAME_TYPE_SPEAKER_AUDIO = 0x21;

// Buffer sizes
static const size_t MIC_RING_BUFFER_SIZE = 16384;   // 512ms at 16kHz 16-bit mono
static const size_t DEFAULT_SPK_BUFFER_SIZE = 65536;
static const size_t SEND_BUFFER_SIZE = 244;          // BLE MTU-3 or WiFi chunk
static const size_t MAX_ENCODED_FRAME = 244;
static const size_t MAX_PCM_FRAME_BYTES = 3840;      // 48kHz * 20ms * 2 bytes * 2 channels

// Mic fixed parameters
static const uint32_t MIC_SAMPLE_RATE = 16000;
static const uint8_t MIC_CHANNELS = 1;
static const uint8_t MIC_BITS = 16;

// Default encoder parameters
static const uint16_t LC3_FRAME_DURATION_MS = 10;
static const uint16_t LC3_NBYTE = 40;
static const uint16_t OPUS_FRAME_DURATION_MS = 20;
static const uint16_t OPUS_NBYTE = 80;  // ~32kbps VBR
static const uint16_t PCM_SEND_DURATION_MS = 20;

#ifdef USE_OVI_WIFI
// TCP receive buffer size (enough for length header + max encoded frame + type byte)
static const size_t TCP_RECV_BUF_SIZE = 2048;
#endif

enum OVIState : uint8_t {
  OVI_STATE_IDLE,
  OVI_STATE_START_MICROPHONE,
  OVI_STATE_STREAMING_MICROPHONE,
  OVI_STATE_STOP_MICROPHONE,
  OVI_STATE_AWAITING_RESPONSE,
  OVI_STATE_STREAMING_RESPONSE,
  OVI_STATE_STOP,
};

class OVIVoiceAssistant : public Component {
 public:

  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  // Configuration setters (called from Python codegen)
  void set_microphone_source(microphone::MicrophoneSource *mic) { this->mic_source_ = mic; }
  void set_speaker(speaker::Speaker *spk) { this->speaker_ = spk; }
  void set_codec_type(CodecTypeConfig type) { this->codec_type_config_ = type; }
  void set_speaker_buffer_size(uint32_t size) { this->spk_buffer_size_ = size; }

#ifdef USE_OVI_WIFI
  void set_port(uint16_t port) { this->port_ = port; }
#endif

#ifdef USE_OVI_BLE
  void set_ble_server(esp32_ble_server::BLEServer *server) { this->ble_server_ = server; }
#endif

#ifdef USE_MICRO_WAKE_WORD
  void set_micro_wake_word(micro_wake_word::MicroWakeWord *mww) { this->micro_wake_word_ = mww; }
#endif

  void set_shared_audio_bus(bool shared) { this->shared_audio_bus_ = shared; }
#ifdef USE_TIME
  void set_time(time::RealTimeClock *time) { this->time_ = time; }
#endif

  // Pipeline control
  void request_start(const std::string &wake_word = "");
  void request_stop();
  bool is_running() const { return this->ovi_state_ != OVI_STATE_IDLE; }
  bool is_connected() const { return this->client_connected_; }

  // Automation trigger getters
  Trigger<> *get_start_trigger() { return &this->ovi_start_trigger_; }
  Trigger<> *get_listening_trigger() { return &this->ovi_listening_trigger_; }
  Trigger<> *get_stt_vad_start_trigger() { return &this->ovi_stt_vad_start_trigger_; }
  Trigger<> *get_stt_vad_end_trigger() { return &this->ovi_stt_vad_end_trigger_; }
  Trigger<std::string> *get_stt_end_trigger() { return &this->ovi_stt_end_trigger_; }
  Trigger<std::string> *get_tts_start_trigger() { return &this->ovi_tts_start_trigger_; }
  Trigger<> *get_tts_stream_start_trigger() { return &this->ovi_tts_stream_start_trigger_; }
  Trigger<> *get_end_trigger() { return &this->ovi_end_trigger_; }
  Trigger<std::string, std::string> *get_error_trigger() { return &this->ovi_error_trigger_; }
  Trigger<> *get_client_connected_trigger() { return &this->ovi_client_connected_trigger_; }
  Trigger<> *get_client_disconnected_trigger() { return &this->ovi_client_disconnected_trigger_; }

 protected:
  void set_ovi_state_(OVIState state);
  void configure_decoder_(uint32_t sample_rate, uint16_t encoded_frame_bytes, uint8_t channels = 1);

  // Control event handling (shared logic, used by both WiFi and BLE)
  void send_control_event_(uint8_t event_type, const uint8_t *data = nullptr, size_t len = 0);
  void send_mic_config_();
  void handle_control_event_(const uint8_t *data, size_t len);

  // Audio receive handler (speaker audio from server)
  void handle_audio_rx_(const uint8_t *data, size_t len);

  // Mic audio send (reads from ring buffer, encodes, sends via transport)
  void send_mic_audio_();

#ifdef USE_OVI_WIFI
  // TCP server methods
  void setup_tcp_server_();
  void tcp_accept_();
  void tcp_read_();
  void tcp_disconnect_client_();
  void process_incoming_(const uint8_t *data, size_t len);
  void send_frame_(const uint8_t *payload, size_t len);
#endif

  // Microphone and speaker
  microphone::MicrophoneSource *mic_source_{nullptr};
  speaker::Speaker *speaker_{nullptr};
  bool shared_audio_bus_{false};  // true when mic and speaker share one I2S bus

#ifdef USE_MICRO_WAKE_WORD
  micro_wake_word::MicroWakeWord *micro_wake_word_{nullptr};
#endif

  // Codec
  CodecTypeConfig codec_type_config_{CODEC_LC3};
  ovi_audio_codec::CodecType codec_type_{ovi_audio_codec::CodecType::LC3};
  ovi_audio_codec::AudioEncoder *encoder_{nullptr};
  ovi_audio_codec::AudioDecoder *decoder_{nullptr};

  // Ring buffers
  std::unique_ptr<RingBuffer> mic_ring_buffer_;
  std::unique_ptr<RingBuffer> spk_ring_buffer_;
  uint32_t spk_buffer_size_{DEFAULT_SPK_BUFFER_SIZE};

  // Speaker playback state
  uint32_t spk_sample_rate_{16000};
  uint8_t spk_channels_{1};
  bool stream_ended_{false};
  bool spk_started_{false};

#ifdef USE_TIME
  // SNTP-synced playback — buffer audio until target timestamp, then start
  time::RealTimeClock *time_{nullptr};
  uint64_t sync_play_at_ms_{0};  // NTP epoch ms to start playback (0 = immediate)
  bool sync_waiting_{false};     // true while buffering audio before sync start
#endif

  // Connection state
  bool client_connected_{false};
  bool continue_conversation_{false};

  // Audio energy tracking (for multi-device wake word arbitration).
  // Both are updated continuously from the mic data callback.
  uint16_t mic_energy_{0};          // fast EMA (~80ms) — recent level including wake word
  uint16_t mic_energy_ambient_{0};  // slow EMA (~1.3s) — ambient noise floor

  // Ovi state machine
  OVIState ovi_state_{OVI_STATE_IDLE};

  // Encoder PCM accumulation buffer
  uint8_t enc_pcm_buf_[MAX_PCM_FRAME_BYTES];
  size_t enc_pcm_buf_len_{0};
  uint8_t enc_out_buf_[MAX_ENCODED_FRAME];

#ifdef USE_OVI_WIFI
  // TCP server
  std::unique_ptr<socket::ListenSocket> server_socket_;
  std::unique_ptr<socket::Socket> client_socket_;
  uint16_t port_{6055};
  // Receive buffer for TCP framing (accumulates partial frames)
  uint8_t recv_buf_[TCP_RECV_BUF_SIZE];
  size_t recv_buf_len_{0};
#endif

#ifdef USE_OVI_BLE
  // BLE GATT
  esp32_ble_server::BLEServer *ble_server_{nullptr};
  esp32_ble_server::BLEService *service_{nullptr};
  esp32_ble_server::BLECharacteristic *audio_tx_{nullptr};
  esp32_ble_server::BLECharacteristic *audio_rx_{nullptr};
  esp32_ble_server::BLECharacteristic *control_{nullptr};
#endif

  // Automation triggers
  Trigger<> ovi_start_trigger_;
  Trigger<> ovi_listening_trigger_;
  Trigger<> ovi_stt_vad_start_trigger_;
  Trigger<> ovi_stt_vad_end_trigger_;
  Trigger<std::string> ovi_stt_end_trigger_;
  Trigger<std::string> ovi_tts_start_trigger_;
  Trigger<> ovi_tts_stream_start_trigger_;
  Trigger<> ovi_end_trigger_;
  Trigger<std::string, std::string> ovi_error_trigger_;
  Trigger<> ovi_client_connected_trigger_;
  Trigger<> ovi_client_disconnected_trigger_;
};

}  // namespace ovi_voice_assistant
}  // namespace esphome
