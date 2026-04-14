#include "audio_codec.h"

#include <cstring>

#include "esphome/core/log.h"

extern "C" {
#include <esp_lc3_enc.h>
#include <esp_lc3_dec.h>
#include <esp_opus_enc.h>
#include <esp_opus_dec.h>
}

namespace esphome {
namespace ovi_audio_codec {

static const char *const TAG = "ovi_audio_codec";

// ---------------------------------------------------------------------------
// LC3 Encoder
// ---------------------------------------------------------------------------

class LC3Encoder : public AudioEncoder {
 public:
  ~LC3Encoder() override { close(); }

  bool open(const CodecConfig &config) override {
    close();
    config_ = config;
    pcm_frame_bytes_ = config.sample_rate * config.channels * (config.bits_per_sample / 8)
                        * config.frame_duration_ms / 1000;

    esp_lc3_enc_config_t cfg = {};
    cfg.sample_rate = config.sample_rate;
    cfg.bits_per_sample = config.bits_per_sample;
    cfg.channel = config.channels;
    cfg.frame_dms = config.frame_duration_ms * 10;  // ms to deciseconds
    cfg.nbyte = config.encoded_frame_bytes;
    cfg.len_prefixed = 0;

    esp_audio_err_t err = esp_lc3_enc_open(&cfg, sizeof(cfg), &handle_);
    if (err != ESP_AUDIO_ERR_OK || handle_ == nullptr) {
      ESP_LOGE(TAG, "LC3 encoder open failed: %d", err);
      handle_ = nullptr;
      return false;
    }
    return true;
  }

  void close() override {
    if (handle_ != nullptr) {
      esp_lc3_enc_close(handle_);
      handle_ = nullptr;
    }
  }

  void reset() override {
    if (handle_ != nullptr) {
      esp_lc3_enc_reset(handle_);
    }
  }

  int encode(const uint8_t *pcm_in, size_t pcm_len, uint8_t *out, size_t out_capacity) override {
    if (handle_ == nullptr) return -1;

    esp_audio_enc_in_frame_t in_frame = {};
    in_frame.buffer = const_cast<uint8_t *>(pcm_in);
    in_frame.len = static_cast<uint32_t>(pcm_len);

    esp_audio_enc_out_frame_t out_frame = {};
    out_frame.buffer = out;
    out_frame.len = static_cast<uint32_t>(out_capacity);

    esp_audio_err_t err = esp_lc3_enc_process(handle_, &in_frame, &out_frame);
    if (err != ESP_AUDIO_ERR_OK) {
      ESP_LOGE(TAG, "LC3 encode failed: %d", err);
      return -1;
    }
    return static_cast<int>(out_frame.encoded_bytes);
  }

  size_t pcm_frame_bytes() const override { return pcm_frame_bytes_; }

 private:
  void *handle_{nullptr};
  CodecConfig config_{};
  size_t pcm_frame_bytes_{0};
};

// ---------------------------------------------------------------------------
// LC3 Decoder
// ---------------------------------------------------------------------------

class LC3Decoder : public AudioDecoder {
 public:
  ~LC3Decoder() override { close(); }

  bool open(const CodecConfig &config) override {
    close();
    config_ = config;

    esp_lc3_dec_cfg_t cfg = {};
    cfg.sample_rate = config.sample_rate;
    cfg.channel = config.channels;
    cfg.bits_per_sample = config.bits_per_sample;
    cfg.frame_dms = config.frame_duration_ms * 10;  // ms to deciseconds
    cfg.nbyte = config.encoded_frame_bytes;
    cfg.is_cbr = 1;
    cfg.len_prefixed = 0;
    cfg.enable_plc = 1;

    esp_audio_err_t err = esp_lc3_dec_open(&cfg, sizeof(cfg), &handle_);
    if (err != ESP_AUDIO_ERR_OK || handle_ == nullptr) {
      ESP_LOGE(TAG, "LC3 decoder open failed: %d", err);
      handle_ = nullptr;
      return false;
    }
    return true;
  }

  void close() override {
    if (handle_ != nullptr) {
      esp_lc3_dec_close(handle_);
      handle_ = nullptr;
    }
  }

  void reset() override {
    if (handle_ != nullptr) {
      esp_lc3_dec_reset(handle_);
    }
  }

  int decode(const uint8_t *in, size_t in_len, uint8_t *pcm_out, size_t pcm_capacity) override {
    if (handle_ == nullptr) return -1;

    esp_audio_dec_in_raw_t raw = {};
    raw.buffer = const_cast<uint8_t *>(in);
    raw.len = static_cast<uint32_t>(in_len);
    raw.frame_recover = ESP_AUDIO_DEC_RECOVERY_NONE;

    esp_audio_dec_out_frame_t frame = {};
    frame.buffer = pcm_out;
    frame.len = static_cast<uint32_t>(pcm_capacity);

    esp_audio_dec_info_t dec_info = {};

    esp_audio_err_t err = esp_lc3_dec_decode(handle_, &raw, &frame, &dec_info);
    if (err != ESP_AUDIO_ERR_OK) {
      ESP_LOGE(TAG, "LC3 decode failed: %d", err);
      return -1;
    }
    return static_cast<int>(frame.decoded_size);
  }

 private:
  void *handle_{nullptr};
  CodecConfig config_{};
};

// ---------------------------------------------------------------------------
// Opus Encoder
// ---------------------------------------------------------------------------

class OpusEncoder : public AudioEncoder {
 public:
  ~OpusEncoder() override { close(); }

  bool open(const CodecConfig &config) override {
    close();
    config_ = config;
    pcm_frame_bytes_ = config.sample_rate * config.channels * (config.bits_per_sample / 8)
                        * config.frame_duration_ms / 1000;

    esp_opus_enc_config_t cfg = {};
    cfg.sample_rate = config.sample_rate;
    cfg.channel = config.channels;
    cfg.bits_per_sample = config.bits_per_sample;
    cfg.frame_duration = ESP_OPUS_ENC_FRAME_DURATION_20_MS;
    cfg.application_mode = ESP_OPUS_ENC_APPLICATION_VOIP;
    cfg.complexity = 5;
    cfg.enable_fec = false;
    cfg.enable_dtx = false;
    cfg.enable_vbr = true;

    // Choose a reasonable bitrate based on sample rate
    if (config.sample_rate <= 16000) {
      cfg.bitrate = 32000;
    } else {
      cfg.bitrate = 48000;
    }

    esp_audio_err_t err = esp_opus_enc_open(&cfg, sizeof(cfg), &handle_);
    if (err != ESP_AUDIO_ERR_OK || handle_ == nullptr) {
      ESP_LOGE(TAG, "Opus encoder open failed: %d", err);
      handle_ = nullptr;
      return false;
    }
    return true;
  }

  void close() override {
    if (handle_ != nullptr) {
      esp_opus_enc_close(handle_);
      handle_ = nullptr;
    }
  }

  void reset() override {
    if (handle_ != nullptr) {
      esp_opus_enc_reset(handle_);
    }
  }

  int encode(const uint8_t *pcm_in, size_t pcm_len, uint8_t *out, size_t out_capacity) override {
    if (handle_ == nullptr) return -1;

    esp_audio_enc_in_frame_t in_frame = {};
    in_frame.buffer = const_cast<uint8_t *>(pcm_in);
    in_frame.len = static_cast<uint32_t>(pcm_len);

    esp_audio_enc_out_frame_t out_frame = {};
    out_frame.buffer = out;
    out_frame.len = static_cast<uint32_t>(out_capacity);

    esp_audio_err_t err = esp_opus_enc_process(handle_, &in_frame, &out_frame);
    if (err != ESP_AUDIO_ERR_OK) {
      ESP_LOGE(TAG, "Opus encode failed: %d", err);
      return -1;
    }
    return static_cast<int>(out_frame.encoded_bytes);
  }

  size_t pcm_frame_bytes() const override { return pcm_frame_bytes_; }

 private:
  void *handle_{nullptr};
  CodecConfig config_{};
  size_t pcm_frame_bytes_{0};
};

// ---------------------------------------------------------------------------
// Opus Decoder
// ---------------------------------------------------------------------------

class OpusDecoder : public AudioDecoder {
 public:
  ~OpusDecoder() override { close(); }

  bool open(const CodecConfig &config) override {
    close();
    config_ = config;

    esp_opus_dec_cfg_t cfg = {};
    cfg.sample_rate = config.sample_rate;
    cfg.channel = config.channels;
    cfg.frame_duration = ESP_OPUS_DEC_FRAME_DURATION_20_MS;
    cfg.self_delimited = false;

    esp_audio_err_t err = esp_opus_dec_open(&cfg, sizeof(cfg), &handle_);
    if (err != ESP_AUDIO_ERR_OK || handle_ == nullptr) {
      ESP_LOGE(TAG, "Opus decoder open failed: %d", err);
      handle_ = nullptr;
      return false;
    }
    return true;
  }

  void close() override {
    if (handle_ != nullptr) {
      esp_opus_dec_close(handle_);
      handle_ = nullptr;
    }
  }

  void reset() override {
    if (handle_ != nullptr) {
      esp_opus_dec_reset(handle_);
    }
  }

  int decode(const uint8_t *in, size_t in_len, uint8_t *pcm_out, size_t pcm_capacity) override {
    if (handle_ == nullptr) return -1;

    esp_audio_dec_in_raw_t raw = {};
    raw.buffer = const_cast<uint8_t *>(in);
    raw.len = static_cast<uint32_t>(in_len);
    raw.frame_recover = ESP_AUDIO_DEC_RECOVERY_NONE;

    esp_audio_dec_out_frame_t frame = {};
    frame.buffer = pcm_out;
    frame.len = static_cast<uint32_t>(pcm_capacity);

    esp_audio_dec_info_t dec_info = {};

    esp_audio_err_t err = esp_opus_dec_decode(handle_, &raw, &frame, &dec_info);
    if (err != ESP_AUDIO_ERR_OK) {
      ESP_LOGE(TAG, "Opus decode failed: %d", err);
      return -1;
    }
    return static_cast<int>(frame.decoded_size);
  }

 private:
  void *handle_{nullptr};
  CodecConfig config_{};
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

AudioEncoder *create_encoder(CodecType type) {
  switch (type) {
    case CodecType::LC3:
      return new LC3Encoder();
    case CodecType::OPUS:
      return new OpusEncoder();
    default:
      ESP_LOGE(TAG, "Unsupported encoder codec type: %d", static_cast<int>(type));
      return nullptr;
  }
}

AudioDecoder *create_decoder(CodecType type) {
  switch (type) {
    case CodecType::LC3:
      return new LC3Decoder();
    case CodecType::OPUS:
      return new OpusDecoder();
    default:
      ESP_LOGE(TAG, "Unsupported decoder codec type: %d", static_cast<int>(type));
      return nullptr;
  }
}

}  // namespace ovi_audio_codec
}  // namespace esphome
