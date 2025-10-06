add_library(data_preprocessing STATIC
    fft.c
    quantization.c
    statistic.c
)

target_include_directories(data_preprocessing PUBLIC ${CMAKE_CURRENT_LIST_DIR})
