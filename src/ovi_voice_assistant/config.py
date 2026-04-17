"""Configuration for the voice assistant."""

from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

CONFIG_DIR = Path.home() / ".ovi"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
CACHE_DIR = Path.home() / ".cache" / "ovi"


class DeviceConfig(BaseModel):
    """Configuration for a single ESPHome voice device."""

    host: str
    port: int = 6055
    encryption_key: str | None = None


def parse_devices(raw: str) -> list[DeviceConfig]:
    """Parse comma-separated device strings.

    Format: ``host[:port[:key]]``
    """
    devices = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", maxsplit=2)
        devices.append(
            DeviceConfig(
                host=parts[0],
                port=int(parts[1]) if len(parts) > 1 and parts[1] else 6055,
                encryption_key=parts[2] if len(parts) > 2 and parts[2] else None,
            )
        )
    return devices


# ── Nested config sections ───────────────────────────────────


class LlmConfig(BaseModel):
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    instructions: str = (
        "You are a voice assistant. Your responses will be spoken aloud. "
        "Rules: Reply in 8-9 short sentences if possible. Never explain your reasoning. "
        "Never use markdown, bullet points, or lists. Never include internal thoughts. "
        "Just give a direct, natural spoken answer. "
        "If your response asks a question or requires a follow-up from the user, "
        "end your response with [LISTEN]. Only use [LISTEN] when you need the user to respond."
    )
    mcp_servers: str = ""
    agents: str = ""


class SttConfig(BaseModel):
    provider: str = "nemotron"
    model: str = "int8-dynamic"
    device: str = "cpu"  # "cpu" or "cuda"
    language: str = "en"
    beam_size: int = 1
    compute_type: str = "int8"


class TtsConfig(BaseModel):
    provider: str = "kokoro"
    model: str = "af_heart"
    speaker_id: int | None = None
    length_scale: float = 1.0
    sentence_silence: float = 0.1
    # qwen3 only:
    language: str = "english"
    reference_audio: str = ""  # path to ref WAV/FLAC/MP3; empty = built-in shadowheart


class TransportConfig(BaseModel):
    type: str = "wifi"  # "wifi" or "ble"
    codec: str = "lc3"  # "pcm", "lc3", "opus"
    speaker_sample_rate: int = 0  # 0 = TTS native rate


class MicConfig(BaseModel):
    sample_rate: int = 16000
    sample_width: int = 2  # bytes (16-bit)
    channels: int = 1


class BleConfig(BaseModel):
    device_name: str | None = None
    device_address: str | None = None


class MemoryConfig(BaseModel):
    enabled: bool = True
    db_path: str = "~/.ovi/memory.db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bank_id: str = "voice-assistant"


class AutomationsConfig(BaseModel):
    path: str = "~/.ovi/automations.json"


# ── Main settings ────────────────────────────────────────────


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OVI_",
        env_file=".env",
        env_nested_delimiter="__",
    )

    llm: LlmConfig = LlmConfig()
    stt: SttConfig = SttConfig()
    tts: TtsConfig = TtsConfig()
    transport: TransportConfig = TransportConfig()
    mic: MicConfig = MicConfig()
    ble: BleConfig = BleConfig()
    memory: MemoryConfig = MemoryConfig()
    automations: AutomationsConfig = AutomationsConfig()

    # Devices — comma-separated string or YAML list: host[:port[:key]]
    devices: str = ""

    @field_validator("devices", mode="before")
    @classmethod
    def _coerce_devices(cls, v):
        if isinstance(v, list):
            return ",".join(str(item) for item in v)
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Priority (highest to lowest): init kwargs > env vars > .env > yaml > defaults
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=CONFIG_PATH),
            file_secret_settings,
        )

    def get_devices(self) -> list[DeviceConfig]:
        return parse_devices(self.devices) if self.devices else []
