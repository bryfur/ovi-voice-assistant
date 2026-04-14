/**
 * Ovi Voice Assistant — Unified implementation
 *
 * State machine:
 *   IDLE -> START_MICROPHONE -> STREAMING_MICROPHONE -> STOP_MICROPHONE
 *     -> AWAITING_RESPONSE -> STREAMING_RESPONSE -> STOP -> IDLE
 *
 * WiFi transport: plain TCP server with length-prefixed binary framing.
 * BLE transport: GATT service with audio TX/RX and control characteristics.
 */

#include "ovi_voice_assistant.h"
#include "esphome/components/audio/audio.h"
#include "esphome/core/log.h"

#include <cerrno>
#include <cstring>

#ifdef USE_OVI_BLE
#include "esphome/components/esp32_ble/ble_uuid.h"
#include "esphome/components/esp32_ble_server/ble_2902.h"
#endif

namespace esphome {
namespace ovi_voice_assistant {

static const char *const TAG = "ovi_voice_assistant";

#ifdef USE_OVI_BLE
// GATT UUIDs
static const char *SERVICE_UUID  = "BA5E0001-FADA-4C14-A34C-1AE0F0A0A0A0";
static const char *AUDIO_TX_UUID = "BA5E0002-FADA-4C14-A34C-1AE0F0A0A0A0";
static const char *AUDIO_RX_UUID = "BA5E0003-FADA-4C14-A34C-1AE0F0A0A0A0";
static const char *CONTROL_UUID  = "BA5E0004-FADA-4C14-A34C-1AE0F0A0A0A0";
#endif

// ---------------------------------------------------------------------------
// setup()
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::setup() {
  ESP_LOGI(TAG, "Setting up Ovi Voice Assistant...");

  // Map config enum to codec type
  switch (this->codec_type_config_) {
    case CODEC_LC3:
      this->codec_type_ = ovi_audio_codec::CodecType::LC3;
      break;
    case CODEC_OPUS:
      this->codec_type_ = ovi_audio_codec::CodecType::OPUS;
      break;
    default:
      this->codec_type_ = ovi_audio_codec::CodecType::PCM;
      break;
  }

  // Create mic ring buffer
  this->mic_ring_buffer_ = RingBuffer::create(MIC_RING_BUFFER_SIZE);
  if (!this->mic_ring_buffer_) {
    ESP_LOGE(TAG, "Failed to allocate mic ring buffer");
    this->mark_failed();
    return;
  }

  // Create speaker ring buffer
  this->spk_ring_buffer_ = RingBuffer::create(this->spk_buffer_size_);
  if (!this->spk_ring_buffer_) {
    ESP_LOGE(TAG, "Failed to allocate speaker ring buffer");
    this->mark_failed();
    return;
  }

  // Register mic data callback -> ring buffer + energy tracking
  this->mic_source_->add_data_callback([this](const std::vector<uint8_t> &data) {
    // Always track audio energy (even during IDLE) so we have a fresh
    // value when a wake word fires — used for multi-device arbitration.
    if (data.size() >= 2) {
      const int16_t *samples = reinterpret_cast<const int16_t *>(data.data());
      size_t n = data.size() / 2;
      uint32_t sum = 0;
      for (size_t i = 0; i < n; i++) {
        int32_t s = samples[i];
        sum += static_cast<uint32_t>(s < 0 ? -s : s);
      }
      uint16_t chunk_energy = static_cast<uint16_t>(sum / n);
      // Fast EMA (alpha ≈ 0.25, ~80ms) — tracks speech peaks
      this->mic_energy_ = (this->mic_energy_ * 3 + chunk_energy) / 4;
      // Slow EMA (alpha ≈ 1/64, ~1.3s) — tracks ambient noise floor
      this->mic_energy_ambient_ = (this->mic_energy_ambient_ * 63 + chunk_energy) / 64;
    }

    if (this->ovi_state_ == OVI_STATE_STREAMING_MICROPHONE) {
      this->mic_ring_buffer_->write((const void *) data.data(), data.size());
    }
  });

  // Create encoder (mic PCM -> compressed), skip for PCM passthrough
  if (this->codec_type_ != ovi_audio_codec::CodecType::PCM) {
    this->encoder_ = ovi_audio_codec::create_encoder(this->codec_type_);
    if (this->encoder_ == nullptr) {
      ESP_LOGE(TAG, "Failed to create audio encoder");
      this->mark_failed();
      return;
    }

    ovi_audio_codec::CodecConfig enc_cfg = {};
    enc_cfg.sample_rate = MIC_SAMPLE_RATE;
    enc_cfg.channels = MIC_CHANNELS;
    enc_cfg.bits_per_sample = MIC_BITS;
    if (this->codec_type_ == ovi_audio_codec::CodecType::LC3) {
      enc_cfg.frame_duration_ms = LC3_FRAME_DURATION_MS;
      enc_cfg.encoded_frame_bytes = LC3_NBYTE;
    } else {
      enc_cfg.frame_duration_ms = OPUS_FRAME_DURATION_MS;
      enc_cfg.encoded_frame_bytes = OPUS_NBYTE;
    }

    if (!this->encoder_->open(enc_cfg)) {
      ESP_LOGE(TAG, "Failed to open audio encoder");
      this->mark_failed();
      return;
    }
  }

  // Decoder is created on-demand when audio config is received
  this->decoder_ = nullptr;
  this->enc_pcm_buf_len_ = 0;

#ifdef USE_OVI_WIFI
  this->setup_tcp_server_();
#endif

#ifdef USE_OVI_BLE
  // Create GATT service on the shared BLE server
  this->service_ = this->ble_server_->create_service(
      esp32_ble::ESPBTUUID::from_raw(SERVICE_UUID), true, 15);

  // Audio TX: mic -> server (notify, needs CCCD for subscription)
  this->audio_tx_ = this->service_->create_characteristic(
      esp32_ble::ESPBTUUID::from_raw(AUDIO_TX_UUID),
      esp32_ble_server::BLECharacteristic::PROPERTY_NOTIFY);
  this->audio_tx_->add_descriptor(new esp32_ble_server::BLE2902());

  // Audio RX: server -> speaker (write without response)
  this->audio_rx_ = this->service_->create_characteristic(
      esp32_ble::ESPBTUUID::from_raw(AUDIO_RX_UUID),
      esp32_ble_server::BLECharacteristic::PROPERTY_WRITE_NR);

  this->audio_rx_->on_write([this](std::span<const uint8_t> data, uint16_t conn_id) {
    if (data.size() == 0) return;
    this->handle_audio_rx_(data.data(), data.size());
  });

  // Control: bidirectional events (read/write/notify, needs CCCD)
  this->control_ = this->service_->create_characteristic(
      esp32_ble::ESPBTUUID::from_raw(CONTROL_UUID),
      esp32_ble_server::BLECharacteristic::PROPERTY_READ |
      esp32_ble_server::BLECharacteristic::PROPERTY_WRITE |
      esp32_ble_server::BLECharacteristic::PROPERTY_NOTIFY);
  this->control_->add_descriptor(new esp32_ble_server::BLE2902());

  this->control_->on_write([this](std::span<const uint8_t> data, uint16_t conn_id) {
    if (data.size() > 0) {
      this->handle_control_event_(data.data(), data.size());
    }
  });

  // Enqueue service for starting
  this->ble_server_->enqueue_start_service(this->service_);

  // Track connect/disconnect
  this->ble_server_->on_connect([this](uint16_t conn_id) {
    ESP_LOGI(TAG, "BLE client connected (conn_id=%d)", conn_id);
    this->client_connected_ = true;
    this->ovi_client_connected_trigger_.trigger();
  });

  this->ble_server_->on_disconnect([this](uint16_t conn_id) {
    ESP_LOGI(TAG, "BLE client disconnected (conn_id=%d)", conn_id);
    this->client_connected_ = false;

    if (this->ovi_state_ != OVI_STATE_IDLE) {
      this->mic_source_->stop();
      if (this->speaker_ != nullptr) {
        this->speaker_->stop();
      }
      this->spk_started_ = false;
      this->set_ovi_state_(OVI_STATE_IDLE);
    }

    this->ovi_client_disconnected_trigger_.trigger();
  });
#endif

  ESP_LOGI(TAG, "Ovi Voice Assistant ready (codec=%d)", static_cast<int>(this->codec_type_config_));
}

// ---------------------------------------------------------------------------
// WiFi: TCP server setup
// ---------------------------------------------------------------------------

#ifdef USE_OVI_WIFI
void OVIVoiceAssistant::setup_tcp_server_() {
  ESP_LOGI(TAG, "Setting up TCP server on port %u", this->port_);

  this->server_socket_ = socket::socket_ip_loop_monitored(SOCK_STREAM, 0);
  if (!this->server_socket_) {
    ESP_LOGE(TAG, "Failed to create server socket: errno %d", errno);
    this->mark_failed();
    return;
  }

  int enable = 1;
  int err = this->server_socket_->setsockopt(SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
  if (err != 0) {
    ESP_LOGW(TAG, "Socket SO_REUSEADDR failed: errno %d", errno);
  }

  err = this->server_socket_->setblocking(false);
  if (err != 0) {
    ESP_LOGE(TAG, "Socket setblocking failed: errno %d", errno);
    this->mark_failed();
    return;
  }

  struct sockaddr_storage server_addr;
  socklen_t sl = socket::set_sockaddr_any((struct sockaddr *) &server_addr, sizeof(server_addr), this->port_);
  if (sl == 0) {
    ESP_LOGE(TAG, "Failed to set sockaddr");
    this->mark_failed();
    return;
  }

  err = this->server_socket_->bind((struct sockaddr *) &server_addr, sl);
  if (err != 0) {
    ESP_LOGE(TAG, "Socket bind failed: errno %d", errno);
    this->mark_failed();
    return;
  }

  err = this->server_socket_->listen(1);
  if (err != 0) {
    ESP_LOGE(TAG, "Socket listen failed: errno %d", errno);
    this->mark_failed();
    return;
  }

  ESP_LOGI(TAG, "TCP server listening on port %u", this->port_);
}

void OVIVoiceAssistant::tcp_accept_() {
  if (this->client_socket_) return;
  if (!this->server_socket_->ready()) return;

  struct sockaddr_storage source_addr;
  socklen_t addr_len = sizeof(source_addr);

  auto client = this->server_socket_->accept((struct sockaddr *) &source_addr, &addr_len);
  if (!client) return;

  client->setblocking(false);

  // Enable TCP_NODELAY for low-latency audio
  int enable = 1;
  client->setsockopt(IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(int));

  this->client_socket_ = std::move(client);
  this->recv_buf_len_ = 0;
  this->client_connected_ = true;

  ESP_LOGI(TAG, "TCP client connected");

  this->send_mic_config_();
  this->ovi_client_connected_trigger_.trigger();
}

void OVIVoiceAssistant::tcp_read_() {
  if (!this->client_socket_) return;

  // Read into temporary buffer
  uint8_t tmp_buf[512];
  ssize_t n = this->client_socket_->read(tmp_buf, sizeof(tmp_buf));

  if (n > 0) {
    this->process_incoming_(tmp_buf, n);
  } else if (n == 0) {
    // Client closed connection gracefully
    this->tcp_disconnect_client_();
  } else {
    // n < 0: check errno
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      ESP_LOGW(TAG, "TCP read error: errno %d", errno);
      this->tcp_disconnect_client_();
    }
    // EAGAIN/EWOULDBLOCK: no data available, just return
  }
}

void OVIVoiceAssistant::tcp_disconnect_client_() {
  if (!this->client_socket_) return;

  ESP_LOGI(TAG, "TCP client disconnected");
  this->client_socket_ = nullptr;
  this->client_connected_ = false;
  this->recv_buf_len_ = 0;

  if (this->ovi_state_ != OVI_STATE_IDLE) {
    this->mic_source_->stop();
    if (this->speaker_ != nullptr) {
      this->speaker_->stop();
    }
    this->spk_started_ = false;
    this->set_ovi_state_(OVI_STATE_IDLE);
  }

  this->ovi_client_disconnected_trigger_.trigger();
}

void OVIVoiceAssistant::process_incoming_(const uint8_t *data, size_t len) {
  // Append to receive buffer
  size_t space = TCP_RECV_BUF_SIZE - this->recv_buf_len_;
  size_t to_copy = std::min(len, space);
  if (to_copy < len) {
    ESP_LOGW(TAG, "TCP recv buffer overflow, dropping %zu bytes", len - to_copy);
  }
  memcpy(this->recv_buf_ + this->recv_buf_len_, data, to_copy);
  this->recv_buf_len_ += to_copy;

  // Parse complete frames: [2 bytes LE length][payload]
  while (this->recv_buf_len_ >= 2) {
    uint16_t payload_len = this->recv_buf_[0] | (this->recv_buf_[1] << 8);

    if (payload_len == 0) {
      // Empty frame, skip the 2-byte header
      memmove(this->recv_buf_, this->recv_buf_ + 2, this->recv_buf_len_ - 2);
      this->recv_buf_len_ -= 2;
      continue;
    }

    // Check if we have the full frame
    size_t frame_total = 2 + payload_len;
    if (this->recv_buf_len_ < frame_total) {
      break;  // Wait for more data
    }

    // Extract payload
    const uint8_t *payload = this->recv_buf_ + 2;
    uint8_t frame_type = payload[0];

    if (frame_type == FRAME_TYPE_SPEAKER_AUDIO) {
      // Speaker audio from server
      if (payload_len > 1) {
        this->handle_audio_rx_(payload + 1, payload_len - 1);
      }
    } else if (frame_type == FRAME_TYPE_MIC_AUDIO) {
      // Mic audio echoed back (not expected from server, ignore)
      ESP_LOGW(TAG, "Unexpected mic audio frame from server");
    } else {
      // Control event: frame_type is the event type, rest is event payload
      this->handle_control_event_(payload, payload_len);
    }

    // Consume the frame from the buffer
    size_t remaining = this->recv_buf_len_ - frame_total;
    if (remaining > 0) {
      memmove(this->recv_buf_, this->recv_buf_ + frame_total, remaining);
    }
    this->recv_buf_len_ = remaining;
  }
}

void OVIVoiceAssistant::send_frame_(const uint8_t *payload, size_t len) {
  if (!this->client_socket_ || len == 0) return;

  // Combine header + payload into a single write to avoid partial sends
  uint8_t frame[2 + 512];  // header + max payload
  if (len > sizeof(frame) - 2) {
    ESP_LOGW(TAG, "Frame too large: %zu bytes", len);
    return;
  }
  frame[0] = len & 0xFF;
  frame[1] = (len >> 8) & 0xFF;
  memcpy(frame + 2, payload, len);

  ssize_t written = this->client_socket_->write(frame, 2 + len);
  if (written < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
    ESP_LOGW(TAG, "TCP write failed: errno %d", errno);
    this->tcp_disconnect_client_();
  }
}
#endif

// ---------------------------------------------------------------------------
// loop() state machine
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::loop() {
#ifdef USE_OVI_WIFI
  this->tcp_accept_();
  this->tcp_read_();
#endif

  switch (this->ovi_state_) {
    case OVI_STATE_IDLE:
      break;

    case OVI_STATE_START_MICROPHONE: {
      ESP_LOGD(TAG, "Starting microphone");
      this->mic_source_->start();
      this->mic_ring_buffer_->reset();
      this->enc_pcm_buf_len_ = 0;
      if (this->encoder_ != nullptr) {
        this->encoder_->reset();
      }

      this->set_ovi_state_(OVI_STATE_STREAMING_MICROPHONE);
      this->ovi_listening_trigger_.trigger();
      break;
    }

    case OVI_STATE_STREAMING_MICROPHONE: {
      this->send_mic_audio_();
      break;
    }

    case OVI_STATE_STOP_MICROPHONE: {
      static bool mic_stop_initiated = false;
      if (!mic_stop_initiated) {
        ESP_LOGD(TAG, "Stopping microphone");
        this->mic_source_->stop();
#ifdef USE_MICRO_WAKE_WORD
        // On devices with a shared I2S bus (e.g. ATOM Echo), the microphone
        // must be fully released before the speaker can claim the bus.
        // microWakeWord keeps the mic running; stop it so the I2S driver can
        // tear down.  We restart it in OVI_STATE_STOP after TTS finishes.
        // On devices with separate buses (Voice PE) we leave mww running so
        // "stop" wake word interruption works during TTS playback.
        if (this->shared_audio_bus_ && this->micro_wake_word_ != nullptr) {
          this->micro_wake_word_->stop();
        }
#endif
        mic_stop_initiated = true;
      }
      if (this->shared_audio_bus_) {
        // Poll until the microphone's I2S driver has fully released.
        if (this->mic_source_ != nullptr && this->mic_source_->is_running()) {
          break;  // Still tearing down — wait
        }
        ESP_LOGD(TAG, "Microphone released, ready for speaker");
      }
      mic_stop_initiated = false;
      this->set_ovi_state_(OVI_STATE_AWAITING_RESPONSE);
      break;
    }

    case OVI_STATE_AWAITING_RESPONSE:
      break;

    case OVI_STATE_STREAMING_RESPONSE: {
      if (this->speaker_ == nullptr) {
        if (this->stream_ended_) {
          this->set_ovi_state_(OVI_STATE_STOP);
        }
        break;
      }

#ifdef USE_TIME
      // Sync mode: wait until NTP timestamp, then start speaker and drain buffer
      if (this->sync_waiting_ && this->time_ != nullptr) {
        auto now = this->time_->now();
        if (now.is_valid()) {
          uint64_t now_ms = (uint64_t) now.timestamp * 1000;
          if (now_ms >= this->sync_play_at_ms_) {
            ESP_LOGI(TAG, "Sync playback starting (delta=%lld ms)",
                     (long long)(now_ms - this->sync_play_at_ms_));
            this->sync_waiting_ = false;

            // Start speaker and drain buffered audio
            this->speaker_->start();
            this->spk_started_ = true;
            uint8_t drain_buf[512];
            while (this->spk_ring_buffer_->available() > 0) {
              size_t n = this->spk_ring_buffer_->read(drain_buf, sizeof(drain_buf), 0);
              if (n == 0) break;
              this->speaker_->play(drain_buf, n);
            }
          }
        }
      }
#endif

      // Normal flow: check if stream is done and speaker has finished
      if (this->stream_ended_
#ifdef USE_TIME
          && !this->sync_waiting_
#endif
      ) {
        if (this->spk_started_ && this->speaker_->has_buffered_data()) {
          break;  // Wait for speaker to finish
        }
        this->set_ovi_state_(OVI_STATE_STOP);
      }
      break;
    }

    case OVI_STATE_STOP: {
      ESP_LOGD(TAG, "Pipeline complete");
      if (this->speaker_ != nullptr && this->spk_started_) {
        this->speaker_->stop();
        this->spk_started_ = false;
      }
#ifdef USE_MICRO_WAKE_WORD
      // On shared-bus devices we stopped mww to release the I2S bus for
      // the speaker.  Restart it now that TTS is done so the mic can
      // listen for the next wake word.
      if (this->shared_audio_bus_ && this->micro_wake_word_ != nullptr) {
        this->micro_wake_word_->start();
      }
#endif
      this->ovi_end_trigger_.trigger();

      if (this->continue_conversation_) {
        // Follow-up: restart mic without requiring a new wake word
        ESP_LOGI(TAG, "Follow-up: restarting pipeline");
        this->continue_conversation_ = false;
        this->stream_ended_ = false;
        this->mic_ring_buffer_->reset();
        this->enc_pcm_buf_len_ = 0;
        if (this->encoder_ != nullptr) {
          this->encoder_->reset();
        }
        this->send_control_event_(EVT_WAKE_WORD, nullptr, 0);
        this->ovi_start_trigger_.trigger();
        this->set_ovi_state_(OVI_STATE_START_MICROPHONE);
      } else {
        this->set_ovi_state_(OVI_STATE_IDLE);
      }
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// send_mic_audio_() — read mic ring buffer, encode, send via transport
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::send_mic_audio_() {
  if (this->encoder_ != nullptr) {
    // Codec mode: accumulate PCM until one full frame, then encode and send
    size_t pcm_frame_bytes = this->encoder_->pcm_frame_bytes();

    while (this->enc_pcm_buf_len_ < pcm_frame_bytes) {
      size_t bytes_needed = pcm_frame_bytes - this->enc_pcm_buf_len_;
      size_t bytes_available = this->mic_ring_buffer_->available();
      size_t to_read = std::min(bytes_needed, bytes_available);
      to_read = (to_read / 2) * 2;  // align to 16-bit samples
      if (to_read == 0) break;

      size_t read_bytes = this->mic_ring_buffer_->read(
          (void *) (this->enc_pcm_buf_ + this->enc_pcm_buf_len_), to_read, 0);
      this->enc_pcm_buf_len_ += read_bytes;
    }

    if (this->enc_pcm_buf_len_ >= pcm_frame_bytes) {
      int encoded = this->encoder_->encode(
          this->enc_pcm_buf_, pcm_frame_bytes,
          this->enc_out_buf_, sizeof(this->enc_out_buf_));

      if (encoded > 0) {
#ifdef USE_OVI_WIFI
        if (this->client_connected_) {
          // Send: [2 byte len][0x20][codec frame]
          uint8_t frame[1 + MAX_ENCODED_FRAME];
          frame[0] = FRAME_TYPE_MIC_AUDIO;
          memcpy(frame + 1, this->enc_out_buf_, encoded);
          this->send_frame_(frame, 1 + encoded);
        }
#endif
#ifdef USE_OVI_BLE
        if (this->client_connected_) {
          this->audio_tx_->set_value(std::vector<uint8_t>(
              this->enc_out_buf_, this->enc_out_buf_ + encoded));
          this->audio_tx_->notify();
        }
#endif
      }
      this->enc_pcm_buf_len_ = 0;
    }
  } else {
    // PCM passthrough: read a chunk and send directly
    uint8_t send_buf[SEND_BUFFER_SIZE];
    size_t available = this->mic_ring_buffer_->available();
    size_t to_read = std::min(available, (size_t) SEND_BUFFER_SIZE);
    to_read = (to_read / 2) * 2;

    if (to_read > 0) {
      size_t read_bytes = this->mic_ring_buffer_->read(send_buf, to_read, 0);
      if (read_bytes > 0) {
#ifdef USE_OVI_WIFI
        if (this->client_connected_) {
          uint8_t frame[1 + SEND_BUFFER_SIZE];
          frame[0] = FRAME_TYPE_MIC_AUDIO;
          memcpy(frame + 1, send_buf, read_bytes);
          this->send_frame_(frame, 1 + read_bytes);
        }
#endif
#ifdef USE_OVI_BLE
        if (this->client_connected_) {
          this->audio_tx_->set_value(std::vector<uint8_t>(send_buf, send_buf + read_bytes));
          this->audio_tx_->notify();
        }
#endif
      }
    }
  }
}

// ---------------------------------------------------------------------------
// handle_audio_rx_() — decode incoming speaker audio (shared WiFi/BLE)
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::handle_audio_rx_(const uint8_t *data, size_t len) {
  if (this->speaker_ == nullptr || len == 0) return;
  if (this->ovi_state_ != OVI_STATE_STREAMING_RESPONSE && this->ovi_state_ != OVI_STATE_AWAITING_RESPONSE) return;

  // Decode the incoming frame to PCM
  const uint8_t *pcm_data = data;
  size_t pcm_len = len;
  uint8_t pcm_buf[MAX_PCM_FRAME_BYTES];

  if (this->decoder_ != nullptr) {
    int decoded = this->decoder_->decode(data, len, pcm_buf, sizeof(pcm_buf));
    if (decoded <= 0) return;
    pcm_data = pcm_buf;
    pcm_len = decoded;
  }

#ifdef USE_TIME
  if (this->sync_waiting_) {
    // Sync mode: buffer PCM into ring buffer until the target NTP timestamp
    if (!this->spk_started_) {
      this->speaker_->set_audio_stream_info(
          audio::AudioStreamInfo(MIC_BITS, MIC_CHANNELS, this->spk_sample_rate_));
    }
    this->spk_ring_buffer_->write((const uint8_t *) pcm_data, pcm_len);
  } else
#endif
  {
    // Immediate mode (normal TTS or post-sync): play directly
    if (!this->spk_started_) {
      this->speaker_->set_audio_stream_info(
          audio::AudioStreamInfo(MIC_BITS, MIC_CHANNELS, this->spk_sample_rate_));
      this->speaker_->start();
      this->spk_started_ = true;
    }
    this->speaker_->play(pcm_data, pcm_len);
  }
}

// ---------------------------------------------------------------------------
// send_control_event_() — send event via transport
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::send_control_event_(uint8_t event_type, const uint8_t *data, size_t len) {
  if (!this->client_connected_) return;

#ifdef USE_OVI_WIFI
  uint8_t payload[1 + 256];
  payload[0] = event_type;
  if (data != nullptr && len > 0) {
    size_t copy_len = std::min(len, (size_t) 256);
    memcpy(payload + 1, data, copy_len);
    len = copy_len;
  }
  this->send_frame_(payload, 1 + len);
#endif

#ifdef USE_OVI_BLE
  std::vector<uint8_t> payload;
  payload.push_back(event_type);
  if (data != nullptr && len > 0) {
    payload.insert(payload.end(), data, data + len);
  }
  this->control_->set_value(std::move(payload));
  this->control_->notify();
#endif
}

void OVIVoiceAssistant::send_mic_config_() {
  uint8_t mic_cfg[7];
  uint32_t rate = MIC_SAMPLE_RATE;
  uint16_t nbyte = 0;
  uint8_t codec = static_cast<uint8_t>(this->codec_type_config_);
  if (this->codec_type_ == ovi_audio_codec::CodecType::LC3) {
    nbyte = LC3_NBYTE;
  } else if (this->codec_type_ == ovi_audio_codec::CodecType::OPUS) {
    nbyte = OPUS_NBYTE;
  }
  mic_cfg[0] = rate & 0xFF;
  mic_cfg[1] = (rate >> 8) & 0xFF;
  mic_cfg[2] = (rate >> 16) & 0xFF;
  mic_cfg[3] = (rate >> 24) & 0xFF;
  mic_cfg[4] = nbyte & 0xFF;
  mic_cfg[5] = (nbyte >> 8) & 0xFF;
  mic_cfg[6] = codec;
  this->send_control_event_(EVT_MIC_CONFIG, mic_cfg, sizeof(mic_cfg));
}

// ---------------------------------------------------------------------------
// handle_control_event_() — process incoming control event (shared WiFi/BLE)
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::handle_control_event_(const uint8_t *data, size_t len) {
  if (len < 1) return;
  uint8_t event_type = data[0];
  const uint8_t *payload = data + 1;
  size_t payload_len = len - 1;

  switch (event_type) {
    case EVT_VAD_START:
      ESP_LOGD(TAG, "Server: VAD_START (speech detected)");
      this->ovi_stt_vad_start_trigger_.trigger();
      break;

    case EVT_MIC_STOP:
      ESP_LOGD(TAG, "Server: MIC_STOP");
      if (this->ovi_state_ == OVI_STATE_STREAMING_MICROPHONE) {
        this->set_ovi_state_(OVI_STATE_STOP_MICROPHONE);
      }
      this->ovi_stt_vad_end_trigger_.trigger();
      break;

    case EVT_TTS_START:
      ESP_LOGD(TAG, "Server: TTS_START");
      this->stream_ended_ = false;
      this->spk_started_ = false;
#ifdef USE_TIME
      this->sync_waiting_ = false;
      this->sync_play_at_ms_ = 0;
#endif
      this->spk_ring_buffer_->reset();
      if (this->ovi_state_ == OVI_STATE_IDLE) {
        ESP_LOGI(TAG, "Announcement stream starting");
      }
      this->set_ovi_state_(OVI_STATE_STREAMING_RESPONSE);
      this->ovi_tts_start_trigger_.trigger("");
      this->ovi_tts_stream_start_trigger_.trigger();
      break;

    case EVT_TTS_END:
      ESP_LOGD(TAG, "Server: TTS_END");
      this->stream_ended_ = true;
      break;

    case EVT_CONTINUE:
      ESP_LOGD(TAG, "Server: CONTINUE");
      this->continue_conversation_ = true;
      break;

    case EVT_ERROR: {
      std::string full((const char *) payload, payload_len);
      size_t sep = full.find('\0');
      std::string code = (sep != std::string::npos) ? full.substr(0, sep) : full;
      std::string message = (sep != std::string::npos) ? full.substr(sep + 1) : "";
      ESP_LOGW(TAG, "Server: ERROR code=%s msg=%s", code.c_str(), message.c_str());
      this->ovi_error_trigger_.trigger(code, message);
      if (this->ovi_state_ != OVI_STATE_IDLE) {
        if (this->ovi_state_ == OVI_STATE_STREAMING_MICROPHONE) {
          this->mic_source_->stop();
        }
        this->set_ovi_state_(OVI_STATE_STOP);
      }
      break;
    }

    case EVT_AUDIO_CONFIG: {
      // Server tells us the TTS sample rate, encoded frame bytes, codec, and channels:
      // [4B rate LE][2B encoded_frame_bytes LE][1B codec_id][1B channels]
      if (payload_len >= 6) {
        uint32_t rate = payload[0] | (payload[1] << 8) | (payload[2] << 16) | (payload[3] << 24);
        uint16_t nbyte = payload[4] | (payload[5] << 8);
        uint8_t channels = (payload_len >= 8) ? payload[7] : 1;
        ESP_LOGI(TAG, "Server: AUDIO_CONFIG rate=%lu nbyte=%u ch=%u", (unsigned long) rate, nbyte, channels);
        this->configure_decoder_(rate, nbyte, channels);

        // Reply with our mic codec config so the server knows how to
        // decode.  Sending here (instead of on_connect) guarantees the
        // server has already subscribed to notifications.
        this->send_mic_config_();
      }
      break;
    }

    case EVT_MIC_CONFIG: {
      // Server requests a mic codec change: [4B rate LE][2B nbyte LE][1B codec_type]
      if (payload_len >= 7) {
        uint32_t rate = payload[0] | (payload[1] << 8) | (payload[2] << 16) | (payload[3] << 24);
        uint16_t nbyte = payload[4] | (payload[5] << 8);
        uint8_t codec = payload[6];
        ESP_LOGI(TAG, "Server: MIC_CONFIG rate=%lu nbyte=%u codec=%u", (unsigned long) rate, nbyte, codec);

        // Reconfigure encoder
        ovi_audio_codec::CodecType new_type;
        switch (codec) {
          case 0: new_type = ovi_audio_codec::CodecType::PCM; break;
          case 1: new_type = ovi_audio_codec::CodecType::LC3; break;
          case 2: new_type = ovi_audio_codec::CodecType::OPUS; break;
          default: new_type = ovi_audio_codec::CodecType::PCM; break;
        }

        // Close old encoder
        if (this->encoder_ != nullptr) {
          this->encoder_->close();
          delete this->encoder_;
          this->encoder_ = nullptr;
        }

        this->codec_type_ = new_type;

        if (new_type != ovi_audio_codec::CodecType::PCM) {
          this->encoder_ = ovi_audio_codec::create_encoder(new_type);
          if (this->encoder_ != nullptr) {
            ovi_audio_codec::CodecConfig enc_cfg = {};
            enc_cfg.sample_rate = rate;
            enc_cfg.channels = MIC_CHANNELS;
            enc_cfg.bits_per_sample = MIC_BITS;
            enc_cfg.frame_duration_ms = (new_type == ovi_audio_codec::CodecType::LC3)
                                            ? LC3_FRAME_DURATION_MS : OPUS_FRAME_DURATION_MS;
            enc_cfg.encoded_frame_bytes = nbyte;
            if (!this->encoder_->open(enc_cfg)) {
              ESP_LOGE(TAG, "Failed to open new mic encoder");
              delete this->encoder_;
              this->encoder_ = nullptr;
            }
          }
        }
        this->enc_pcm_buf_len_ = 0;

        // Acknowledge by sending back our actual mic config
        this->send_mic_config_();
      }
      break;
    }

    case EVT_WAKE_ABORT:
      // Another device won the wake-word arbitration — return to idle.
      ESP_LOGI(TAG, "Server: WAKE_ABORT (another device won)");
      if (this->ovi_state_ != OVI_STATE_IDLE) {
        if (this->ovi_state_ == OVI_STATE_STREAMING_MICROPHONE ||
            this->ovi_state_ == OVI_STATE_START_MICROPHONE) {
          this->mic_source_->stop();
        }
        if (this->speaker_ != nullptr && this->spk_started_) {
          this->speaker_->stop();
          this->spk_started_ = false;
        }
#ifdef USE_MICRO_WAKE_WORD
        // Restart micro_wake_word if it was stopped for a shared audio bus
        if (this->shared_audio_bus_ && this->micro_wake_word_ != nullptr) {
          this->micro_wake_word_->start();
        }
#endif
        this->ovi_end_trigger_.trigger();
        this->set_ovi_state_(OVI_STATE_IDLE);
      }
      break;

#ifdef USE_TIME
    case EVT_SYNC_PLAY: {
      // Synchronized playback: buffer audio until NTP timestamp, then start.
      // Payload: 8 bytes LE uint64 — epoch milliseconds.
      if (payload_len >= 8) {
        uint64_t ts = 0;
        for (int i = 0; i < 8; i++) {
          ts |= ((uint64_t) payload[i]) << (i * 8);
        }
        ESP_LOGI(TAG, "Server: SYNC_PLAY at epoch_ms=%llu", (unsigned long long) ts);
        this->sync_play_at_ms_ = ts;
        this->sync_waiting_ = true;
      }
      break;
    }
#endif

    default:
      ESP_LOGW(TAG, "Unknown control event: 0x%02X", event_type);
      break;
  }
}

// ---------------------------------------------------------------------------
// request_start / request_stop
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::request_start(const std::string &wake_word) {
  if (this->ovi_state_ != OVI_STATE_IDLE) return;
  if (!this->client_connected_) return;

  ESP_LOGI(TAG, "Starting pipeline (wake_word=%s, peak=%u, ambient=%u)",
           wake_word.c_str(), this->mic_energy_, this->mic_energy_ambient_);

  this->stream_ended_ = false;
  this->spk_started_ = false;
  this->continue_conversation_ = false;
  this->mic_ring_buffer_->reset();
  this->enc_pcm_buf_len_ = 0;
  if (this->encoder_ != nullptr) {
    this->encoder_->reset();
  }

  // Pack [2B LE peak][2B LE ambient][wake_word UTF-8] so the server
  // can arbitrate when multiple devices hear the same wake word.
  // Using peak/ambient ratio normalises across different mic gains.
  uint8_t wake_payload[4 + 256];
  wake_payload[0] = this->mic_energy_ & 0xFF;
  wake_payload[1] = (this->mic_energy_ >> 8) & 0xFF;
  wake_payload[2] = this->mic_energy_ambient_ & 0xFF;
  wake_payload[3] = (this->mic_energy_ambient_ >> 8) & 0xFF;
  size_t ww_len = std::min(wake_word.size(), (size_t) 252);
  memcpy(wake_payload + 4, wake_word.c_str(), ww_len);
  this->send_control_event_(EVT_WAKE_WORD, wake_payload, 4 + ww_len);

  this->ovi_start_trigger_.trigger();
  this->set_ovi_state_(OVI_STATE_START_MICROPHONE);
}

void OVIVoiceAssistant::request_stop() {
  if (this->ovi_state_ == OVI_STATE_IDLE) return;

  ESP_LOGI(TAG, "Stopping pipeline");

  if (this->ovi_state_ == OVI_STATE_STREAMING_MICROPHONE) {
    this->set_ovi_state_(OVI_STATE_STOP_MICROPHONE);
  } else {
    if (this->speaker_ != nullptr && this->spk_started_) {
      this->speaker_->stop();
      this->spk_started_ = false;
    }
    this->set_ovi_state_(OVI_STATE_STOP);
  }
}

// ---------------------------------------------------------------------------
// State management
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::set_ovi_state_(OVAState state) {
  OVAState old = this->ovi_state_;
  this->ovi_state_ = state;
  ESP_LOGD(TAG, "Ovi State: %d -> %d", old, state);
}

// ---------------------------------------------------------------------------
// configure_decoder_() — create or reconfigure the TTS audio decoder
// ---------------------------------------------------------------------------

void OVIVoiceAssistant::configure_decoder_(uint32_t sample_rate, uint16_t encoded_frame_bytes, uint8_t channels) {
  // Close existing decoder
  if (this->decoder_ != nullptr) {
    this->decoder_->close();
    delete this->decoder_;
    this->decoder_ = nullptr;
  }

  this->spk_sample_rate_ = sample_rate;
  this->spk_channels_ = channels;

  // No decoder needed for PCM
  if (this->codec_type_ == ovi_audio_codec::CodecType::PCM) {
    ESP_LOGI(TAG, "PCM mode: decoder not needed, sample_rate=%lu", (unsigned long) sample_rate);
    if (this->speaker_ != nullptr) {
      this->speaker_->set_audio_stream_info(
          audio::AudioStreamInfo(MIC_BITS, channels, sample_rate));
    }
    return;
  }

  this->decoder_ = ovi_audio_codec::create_decoder(this->codec_type_);
  if (this->decoder_ == nullptr) {
    ESP_LOGE(TAG, "Failed to create decoder");
    return;
  }

  ovi_audio_codec::CodecConfig dec_cfg = {};
  dec_cfg.sample_rate = sample_rate;
  dec_cfg.channels = channels;
  dec_cfg.bits_per_sample = MIC_BITS;
  if (this->codec_type_ == ovi_audio_codec::CodecType::LC3) {
    dec_cfg.frame_duration_ms = LC3_FRAME_DURATION_MS;
  } else {
    dec_cfg.frame_duration_ms = OPUS_FRAME_DURATION_MS;
  }
  dec_cfg.encoded_frame_bytes = encoded_frame_bytes;

  if (!this->decoder_->open(dec_cfg)) {
    ESP_LOGE(TAG, "Failed to open decoder (rate=%lu, nbyte=%u, ch=%u)",
             (unsigned long) sample_rate, encoded_frame_bytes, channels);
    delete this->decoder_;
    this->decoder_ = nullptr;
    return;
  }

  // Tell the speaker the audio format so the resampler knows the input rate
  if (this->speaker_ != nullptr) {
    this->speaker_->set_audio_stream_info(
        audio::AudioStreamInfo(MIC_BITS, channels, sample_rate));
  }

  ESP_LOGI(TAG, "Decoder configured: %luHz, %u bytes/frame, %u ch",
           (unsigned long) sample_rate, encoded_frame_bytes, channels);
}

}  // namespace ovi_voice_assistant
}  // namespace esphome
