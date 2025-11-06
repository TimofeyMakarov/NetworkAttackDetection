cmake_minimum_required(VERSION 3.15)
project(NetworkAttackDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Поиск ONNX Runtime, должна быть установлена переменная окружения ONNXRUNTIME_ROOT
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_ROOT}")
else()
    message(FATAL_ERROR "Please set ONNXRUNTIME_ROOT environment variable")
endif()

include_directories(
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_SOURCE_DIR}/include
)

# Исполняемый файл
add_executable(network_ids 
    src/main.cpp
    src/ONNXModel.cpp
)

# Добавляем переменную для исходного кода
target_compile_definitions(network_ids PRIVATE 
    PROJECT_DIR="${CMAKE_SOURCE_DIR}"
)

# Линковка
if(WIN32)
    target_link_libraries(network_ids ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
else()
    target_link_libraries(network_ids ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

# Копируем DLL
add_custom_command(TARGET network_ids POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:network_ids>"
)
