#####################################################################
####################### High-Level CMake Setup ######################
#####################################################################
cmake_minimum_required(VERSION 3.1)
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

# Includes for installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

#####################################################################
####################### Misc. Project Setup #########################
#####################################################################
project(jpc VERSION 0.1 LANGUAGES CXX)

### OPTIONS
option(JPC_BUILD_TESTS "Build tests" ON)
option(JPC_ENABLE_CUDA "Enable CUDA as a parallelization backend" ON)

if(${JPC_BUILD_TESTS})
    include(GoogleTest)
    enable_testing()
endif()
if(${JPC_ENABLE_CUDA})
    enable_language(CUDA)
    message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
else()
    message(STATUS "CUDA Disabled.")
endif()

add_library(${PROJECT_NAME} INTERFACE)

# We do this instead of compile_features() becase cxx_std_17 feature is not happy with CUDA
target_compile_options(${PROJECT_NAME} INTERFACE -std=c++17)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
)

#####################################################################
####################### Executables #################################
#####################################################################
add_executable(jpc_main src/main.cpp src/test.cu)
target_link_libraries(jpc_main ${PROJECT_NAME})

#####################################################################
####################### Tests #######################################
#####################################################################
if(${JCONTROLS_BUILD_TESTS})
endif()

#####################################################################
####################### Installation ################################
#####################################################################
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
    FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME} DESTINATION include)
