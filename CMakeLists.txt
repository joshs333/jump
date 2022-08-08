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
project(jump VERSION 0.1 LANGUAGES CXX)

### OPTIONS
option(JUMP_BUILD_TESTS "Build tests" ON)
option(JUMP_BUILD_EXAMPLES "Build examples" ON)
option(JUMP_ENABLE_CUDA "Enable CUDA as a parallelization backend" ON)

if(${JUMP_BUILD_TESTS})
    include(GoogleTest)
    enable_testing()
endif()

add_library(${PROJECT_NAME} INTERFACE)

# We do this instead of compile_features() becase cxx_std_17 feature is not happy with CUDA
target_compile_options(${PROJECT_NAME} INTERFACE -std=c++17)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
)

# Goal of these functions is to setup a target with libaries and definitions to
# enable a particular backend, and no other links or includes
# (wrt jump and it's dependencies) should be needed
function(jump_target_cuda target_name)
    enable_language(CUDA)
    message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    find_package(CUDA REQUIRED)
    message(STATUS "CUDA Version: ${CUDA_VERSION}")

    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(${target_name} PUBLIC JUMP_ENABLE_CUDA)
    target_link_libraries(${target_name} jump ${CUDA_LIBRARIES})
    target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIRS})
endfunction()

#####################################################################
####################### Executables #################################
#####################################################################
add_executable(jump_main src/main.cpp)
jump_target_cuda(jump_main)

if(${JUMP_BUILD_EXAMPLES})
    add_executable(jump_array_examples src/examples/array_examples.cpp)
    jump_target_cuda(jump_array_examples)

    # add_executable(jump_shared_ptr_examples src/examples/shared_ptr_examples.cpp)
    # jump_target_cuda(jump_shared_ptr_examples)
endif()

# set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)
# set(NVCC_ARGS ${NVCC_ARGS} --expt-relaxed-constexpr -Xcudafe "--diag_suppress=20012 --display_error_number")
# target_compile_options(jump_main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_ARGS}>)

# set(NVCC_ARGS ${NVCC_ARGS} --expt-relaxed-constexpr -Xcudafe "--diag_suppress=20012 --display_error_number")
# set_property(TARGET jump_main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_compile_options(jump_main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_ARGS}>)
# set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

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
