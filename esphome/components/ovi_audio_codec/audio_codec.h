#pragma once

#include <cstddef>
#include <cstdint>

namespace esphome {
namespace ovi_audio_codec {

enum class CodecType : uint8_t {
  PCM = 0,
  LC3 = 1,
  OPUS = 2,
};

struct CodecConfig {
  uint32_t sample_rate;         // Hz
  uint8_t channels;             // 1 = mono
  uint8_t bits_per_sample;      // 16
  uint16_t frame_duration_ms;   // 10 for LC3, 20 for Opus
  uint16_t encoded_frame_bytes; // nbyte for LC3 (e.g. 40), bitrate-derived for Opus
};

class AudioEncoder {
 public:
  virtual ~AudioEncoder() = default;
  virtual bool open(const CodecConfig &config) = 0;
  virtual void close() = 0;
  virtual void reset() = 0;
  // Returns encoded bytes written, or -1 on error
  virtual int encode(const uint8_t *pcm_in, size_t pcm_len, uint8_t *out, size_t out_capacity) = 0;
  // PCM bytes needed for one codec frame
  virtual size_t pcm_frame_bytes() const = 0;
};

class AudioDecoder {
 public:
  virtual ~AudioDecoder() = default;
  virtual bool open(const CodecConfig &config) = 0;
  virtual void close() = 0;
  virtual void reset() = 0;
  // Returns decoded PCM bytes written, or -1 on error
  virtual int decode(const uint8_t *in, size_t in_len, uint8_t *pcm_out, size_t pcm_capacity) = 0;
};

// Factory functions
AudioEncoder *create_encoder(CodecType type);
AudioDecoder *create_decoder(CodecType type);

}  // namespace ovi_audio_codec
}  // namespace esphome
