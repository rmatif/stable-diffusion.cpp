include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cli)
# add_subdirectory(server-openaiapi)  # Disabled: httplib API incompatibility
add_subdirectory(sd-server)
