import esphome.codegen as cg
from esphome.components.esp32 import add_idf_component

CODEOWNERS = ["@bryanfuria"]


def to_code(config):
    add_idf_component(
        name="espressif/esp_audio_codec",
        ref="2.4.1",
    )
