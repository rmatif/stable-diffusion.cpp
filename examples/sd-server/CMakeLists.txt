set(TARGET sd-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)

find_package(OpenSSL QUIET)
if (OpenSSL_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    message(STATUS "sd-server: enabling HTTPS support via OpenSSL")
else()
    message(STATUS "sd-server: OpenSSL not found, HTTPS image downloads are disabled")
endif()

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
