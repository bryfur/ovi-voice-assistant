"""Configuration for the voice assistant."""

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DeviceConfig(BaseModel):
    """Configuration for a single ESPHome Voice PE device."""

    host: str
    port: int = 6053
    encryption_key: str | None = None


def parse_devices(raw: str) -> list[DeviceConfig]:
    """Parse comma-separated device strings: host[:port[:key]]"""
    devices = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", maxsplit=2)
        devices.append(DeviceConfig(
            host=parts[0],
            port=int(parts[1]) if len(parts) > 1 and parts[1] else 6053,
            encryption_key=parts[2] if len(parts) > 2 else None,
        ))
    return devices


class Settings(BaseSettings):
    model_config = {"env_prefix": "OVA_", "env_file": ".env"}

    # Devices — comma-separated: host[:port[:key]]
    devices: str = ""

    # Provider selection
    stt_provider: str = "whisper"
    tts_provider: str = "piper"

    # STT (faster-whisper)
    stt_model: str = "base.en"
    stt_language: str = "en"
    stt_beam_size: int = 1
    stt_compute_type: str = "int8"

    # TTS (piper)
    tts_model: str = "en_US-lessac-medium"
    tts_speaker_id: int | None = None
    tts_length_scale: float = 1.0
    tts_sentence_silence: float = 0.1

    # Agent (OpenAI Agents SDK)
    openai_api_key: str = ""
    openai_base_url: str = ""
    agent_model: str = "gpt-4o-mini"
    agent_instructions: str = (
        "You are a voice assistant. Your responses will be spoken aloud. "
        "Rules: Reply in 1-2 short sentences maximum. Never explain your reasoning. "
        "Never use markdown, bullet points, or lists. Never include internal thoughts. "
        "Just give a direct, natural spoken answer. "
        "If your response asks a question or requires a follow-up from the user, "
        "end your response with [LISTEN]. Only use [LISTEN] when you need the user to respond."
    )

    # Mic audio format (from Voice PE: 16-bit PCM, 16kHz, mono)
    mic_sample_rate: int = 16000
    mic_sample_width: int = 2  # bytes (16-bit)
    mic_channels: int = 1

    def get_devices(self) -> list[DeviceConfig]:
        return parse_devices(self.devices) if self.devices else []
