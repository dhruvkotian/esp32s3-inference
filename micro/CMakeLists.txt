cmake_minimum_required(VERSION 3.16)

# ESP-DSP is an external component for this benchmark only. It is not linked
# into the paper firmware.
if(DEFINED ENV{ESP_DSP_PATH})
    set(ESP_DSP_PATH $ENV{ESP_DSP_PATH})
elseif(NOT DEFINED ESP_DSP_PATH)
    set(ESP_DSP_PATH ${CMAKE_CURRENT_LIST_DIR}/components/esp-dsp)
endif()

if(NOT EXISTS ${ESP_DSP_PATH}/CMakeLists.txt)
    message(FATAL_ERROR
        "ESP-DSP not found at ${ESP_DSP_PATH}. Clone it and set ESP_DSP_PATH.")
endif()

list(APPEND EXTRA_COMPONENT_DIRS ${ESP_DSP_PATH})

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(micro)
