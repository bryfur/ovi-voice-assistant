"""LC3 codec wrapper using lc3py."""

from ovi_voice_assistant.codec.audio_codec import AudioCodec, CodecType

LC3_FRAME_DURATION_US = 10_000  # 10ms
LC3_DEFAULT_NBYTE = 40  # 32 kbps per channel (voice, mono)
LC3_MUSIC_NBYTE = 40  # 64 kbps per channel (music, stereo = 128 kbps total)


class Lc3Codec(AudioCodec):
    """LC3 audio codec (10ms frames).

    ``nbyte`` is the per-channel encoded byte count — matching the ESP32
    ``esp_lc3_dec_cfg_t.nbyte`` definition. The wire frame size is
    ``nbyte * channels`` (channels are concatenated in the encoded frame).
    """

    codec_type = CodecType.LC3
    codec_id = 1

    def __init__(
        self, sample_rate: int, channels: int = 1, nbyte: int = LC3_DEFAULT_NBYTE
    ) -> None:
        from lc3 import Decoder as LC3Decoder
        from lc3 import Encoder as LC3Encoder

        self._sample_rate = sample_rate
        self._channels = channels
        self._nbyte_per_channel = nbyte

        self._encoder = LC3Encoder(
            frame_duration_us=LC3_FRAME_DURATION_US,
            sample_rate_hz=sample_rate,
            num_channels=channels,
        )
        self._decoder = LC3Decoder(
            frame_duration_us=LC3_FRAME_DURATION_US,
            sample_rate_hz=sample_rate,
            num_channels=channels,
        )

        self._frame_samples = self._encoder.get_frame_samples()

    def encode(self, pcm: bytes) -> bytes:
        # Python lc3 library takes the TOTAL output size across all channels
        return self._encoder.encode(
            pcm, num_bytes=self._nbyte_per_channel * self._channels, bit_depth=16
        )

    def decode(self, data: bytes) -> bytes:
        return self._decoder.decode(data, bit_depth=16)

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_duration_ms(self) -> int:
        return LC3_FRAME_DURATION_US // 1000

    @property
    def pcm_frame_bytes(self) -> int:
        return self._frame_samples * self._channels * 2

    @property
    def encoded_frame_bytes(self) -> int:
        # Per-channel byte count — matches esp_lc3_dec_cfg_t.nbyte on the
        # device. The on-wire frame size is encoded_frame_bytes * channels.
        return self._nbyte_per_channel
