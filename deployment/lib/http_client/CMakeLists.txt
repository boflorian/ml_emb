aux_source_directory(. DIR_HTTP_CLIENT_SRCS)

add_library(http_client ${DIR_HTTP_CLIENT_SRCS})

target_include_directories(http_client PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_SOURCE_DIR})

target_link_libraries(http_client PUBLIC
	pico_stdlib
	pico_cyw43_arch_lwip_poll
	pico_lwip_http
)

# HTTP only; TLS disabled for lower RAM usage.
