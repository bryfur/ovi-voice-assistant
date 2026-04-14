"""ESPHome component: Ovi Voice Assistant (unified WiFi + BLE).

WiFi mode: plain TCP server on configurable port (no VoiceAssistant inheritance).
BLE mode: GATT service (unchanged from before).
"""

import esphome.codegen as cg
import esphome.config_validation as cv
from esphome import automation
from esphome.const import CONF_ID, CONF_MICROPHONE, CONF_SPEAKER
from esphome.components import microphone, speaker
from esphome.components.esp32 import add_idf_component

try:
    from esphome.components import micro_wake_word

    HAS_MWW = True
except ImportError:
    HAS_MWW = False

CODEOWNERS = ["@bryanfuria"]
AUTO_LOAD = ["ovi_audio_codec"]  # NO voice_assistant!
DEPENDENCIES = ["microphone"]

CONF_TIME_ID = "time_id"

CONF_TRANSPORT = "transport"
CONF_CODEC = "codec"
CONF_PORT = "port"
CONF_BLE_SERVER = "ble_server"
CONF_SPEAKER_BUFFER_SIZE = "speaker_buffer_size"
CONF_MICRO_WAKE_WORD = "micro_wake_word"
CONF_SHARED_AUDIO_BUS = "shared_audio_bus"
CONF_ON_START = "on_start"
CONF_ON_LISTENING = "on_listening"
CONF_ON_STT_VAD_START = "on_stt_vad_start"
CONF_ON_STT_VAD_END = "on_stt_vad_end"
CONF_ON_STT_END = "on_stt_end"
CONF_ON_TTS_START = "on_tts_start"
CONF_ON_TTS_STREAM_START = "on_tts_stream_start"
CONF_ON_END = "on_end"
CONF_ON_ERROR = "on_error"
CONF_ON_CLIENT_CONNECTED = "on_client_connected"
CONF_ON_CLIENT_DISCONNECTED = "on_client_disconnected"

ovi_voice_assistant_ns = cg.esphome_ns.namespace("ovi_voice_assistant")
OVIVoiceAssistant = ovi_voice_assistant_ns.class_("OVIVoiceAssistant", cg.Component)

CodecTypeEnum = ovi_voice_assistant_ns.enum("CodecTypeConfig")
CODEC_TYPES = {
    "pcm": CodecTypeEnum.CODEC_PCM,
    "lc3": CodecTypeEnum.CODEC_LC3,
    "opus": CodecTypeEnum.CODEC_OPUS,
}


def _validate_transport(config):
    if config[CONF_TRANSPORT] == "ble" and CONF_BLE_SERVER not in config:
        raise cv.Invalid("ble_server is required when transport is 'ble'")
    return config


_SCHEMA_FIELDS = {
    cv.GenerateID(): cv.declare_id(OVIVoiceAssistant),
    cv.Required(CONF_TRANSPORT): cv.one_of("wifi", "ble", lower=True),
    cv.Optional(CONF_CODEC, default="lc3"): cv.one_of("pcm", "lc3", "opus", lower=True),
    cv.Optional(CONF_PORT, default=6055): cv.port,
    cv.Optional(CONF_MICROPHONE, default={}): microphone.microphone_source_schema(
        min_bits_per_sample=16,
        max_bits_per_sample=16,
        min_channels=1,
        max_channels=1,
    ),
    cv.Optional(CONF_SPEAKER): cv.use_id(speaker.Speaker),
    cv.Optional(CONF_SPEAKER_BUFFER_SIZE, default=65536): cv.uint32_t,
    cv.Optional(CONF_SHARED_AUDIO_BUS, default=False): cv.boolean,
    cv.Optional(CONF_TIME_ID): cv.use_id(cg.esphome_ns.namespace("time").class_("RealTimeClock", cg.Component)),
    cv.Optional(CONF_BLE_SERVER): cv.use_id(
        cg.esphome_ns.namespace("esp32_ble_server").class_("BLEServer", cg.Component)
    ),
    cv.Optional(CONF_ON_START): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_LISTENING): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_STT_VAD_START): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_STT_VAD_END): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_STT_END): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_TTS_START): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_TTS_STREAM_START): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_END): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_ERROR): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_CLIENT_CONNECTED): automation.validate_automation(single=True),
    cv.Optional(CONF_ON_CLIENT_DISCONNECTED): automation.validate_automation(single=True),
}

if HAS_MWW:
    _SCHEMA_FIELDS[cv.Optional(CONF_MICRO_WAKE_WORD)] = cv.use_id(
        micro_wake_word.MicroWakeWord
    )

CONFIG_SCHEMA = cv.All(
    cv.Schema(_SCHEMA_FIELDS).extend(cv.COMPONENT_SCHEMA),
    _validate_transport,
)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    transport = config[CONF_TRANSPORT]

    # Transport-specific defines
    if transport == "wifi":
        cg.add_define("USE_OVI_WIFI")
        cg.add(var.set_port(config[CONF_PORT]))
    elif transport == "ble":
        cg.add_define("USE_OVI_BLE")

    # Codec type
    codec = config[CONF_CODEC]
    cg.add(var.set_codec_type(CODEC_TYPES[codec]))

    # Microphone source
    mic_source = await microphone.microphone_source_to_code(config[CONF_MICROPHONE])
    cg.add(var.set_microphone_source(mic_source))

    # Speaker
    if CONF_SPEAKER in config:
        spk = await cg.get_variable(config[CONF_SPEAKER])
        cg.add(var.set_speaker(spk))

    # Speaker buffer size
    cg.add(var.set_speaker_buffer_size(config[CONF_SPEAKER_BUFFER_SIZE]))

    # Shared audio bus — set true when mic and speaker share one I2S peripheral
    # (e.g. ATOM Echo).  This stops mww during TTS to release the bus.
    cg.add(var.set_shared_audio_bus(config[CONF_SHARED_AUDIO_BUS]))

    # SNTP time component — enables synchronized multi-device playback
    if CONF_TIME_ID in config:
        time_comp = await cg.get_variable(config[CONF_TIME_ID])
        cg.add(var.set_time(time_comp))

    # BLE server (BLE transport only)
    if transport == "ble" and CONF_BLE_SERVER in config:
        from esphome.components import esp32_ble_server  # noqa: F401

        server = await cg.get_variable(config[CONF_BLE_SERVER])
        cg.add(var.set_ble_server(server))

    # Micro wake word
    if HAS_MWW and CONF_MICRO_WAKE_WORD in config:
        mww = await cg.get_variable(config[CONF_MICRO_WAKE_WORD])
        cg.add(var.set_micro_wake_word(mww))

    # No-arg triggers
    for conf_key, getter in [
        (CONF_ON_START, "get_start_trigger"),
        (CONF_ON_LISTENING, "get_listening_trigger"),
        (CONF_ON_STT_VAD_START, "get_stt_vad_start_trigger"),
        (CONF_ON_STT_VAD_END, "get_stt_vad_end_trigger"),
        (CONF_ON_TTS_STREAM_START, "get_tts_stream_start_trigger"),
        (CONF_ON_END, "get_end_trigger"),
        (CONF_ON_CLIENT_CONNECTED, "get_client_connected_trigger"),
        (CONF_ON_CLIENT_DISCONNECTED, "get_client_disconnected_trigger"),
    ]:
        if conf_key in config:
            await automation.build_automation(
                getattr(var, getter)(), [], config[conf_key]
            )

    # String-arg triggers
    for conf_key, getter in [
        (CONF_ON_STT_END, "get_stt_end_trigger"),
        (CONF_ON_TTS_START, "get_tts_start_trigger"),
    ]:
        if conf_key in config:
            await automation.build_automation(
                getattr(var, getter)(),
                [(cg.std_string, "x")],
                config[conf_key],
            )

    # Error trigger (code, message)
    if CONF_ON_ERROR in config:
        await automation.build_automation(
            var.get_error_trigger(),
            [(cg.std_string, "code"), (cg.std_string, "message")],
            config[CONF_ON_ERROR],
        )

    # Audio codec library
    add_idf_component(
        name="espressif/esp_audio_codec",
        ref="2.4.1",
    )
