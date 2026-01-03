set(DIR_SRC_SRCS
  inference.cpp
  model.cpp
  model_settings.cpp
)

include_directories(../lib/pico-tflmicro)

add_library(src ${DIR_SRC_SRCS})
target_include_directories(
  src
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/../models
)
target_link_libraries(
  src 
  PUBLIC 
  pico_stdlib 
  pico-tflmicro
  models
)
