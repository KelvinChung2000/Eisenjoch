function(eisenjoch_add_nextpnr_bridge)
    if(NOT TARGET nextpnr-himbaechel-core OR BUILD_PYTHON OR BUILD_GUI OR BUILD_RUST)
        message(FATAL_ERROR "Build the bridge with ARCH=himbaechel and BUILD_PYTHON, BUILD_GUI, BUILD_RUST disabled")
    endif()

    add_library(eisenjoch-nextpnr SHARED "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/bridge.cc")
    target_link_libraries(eisenjoch-nextpnr PRIVATE nextpnr-himbaechel-core)
    set_target_properties(eisenjoch-nextpnr PROPERTIES CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
endfunction()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR AND PROJECT_NAME STREQUAL "nextpnr")
    cmake_language(DEFER CALL eisenjoch_add_nextpnr_bridge)
endif()
