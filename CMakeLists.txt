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
set(JUMP_VERSION 0.1)

### OPTIONS
option(JUMP_BUILD_TESTS "Build tests" ON)
option(JUMP_BUILD_EXAMPLES "Build examples" ON)
option(JUMP_ENABLE_CUDA "Enable CUDA as a parallelization backend" ON)

if(${JUMP_ENABLE_CUDA})
    project(jump VERSION ${JUMP_VERSION} LANGUAGES CXX CUDA)
else()
    project(jump VERSION ${JUMP_VERSION} LANGUAGES CXX)
endif()

find_package(Eigen3 REQUIRED)


include(CheckLanguage)
check_language(CUDA)

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
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

# Goal of these functions is to setup a target with libaries and definitions to
# enable a particular backend, and no other links or includes
# (wrt jump and it's dependencies) should be needed
function(jump_cuda_target target_name)
    enable_language(CUDA)
    message(STATUS "[JUMP][${target_name}] CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    find_package(CUDA REQUIRED)
    message(STATUS "[JUMP][${target_name}] CUDA Version: ${CUDA_VERSION}")

    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(${target_name} PUBLIC JUMP_ENABLE_CUDA)
    target_link_libraries(${target_name} PUBLIC jump ${CUDA_LIBRARIES})
    target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIRS})

    set(NVCC_ARGS ${NVCC_ARGS} --expt-relaxed-constexpr -Xcudafe "--diag_suppress=20012 --display_error_number")
    target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_ARGS}>)
endfunction()

# Mark sources as cuda even if the extension is not .cuh
function(jump_cuda_sources)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
endfunction()

# Create a new executable and call jump_cuda_target and jump_cuda_sources on it
function(jump_cuda_executable target_name)
    add_executable(${target_name} ${ARGN})
    jump_cuda_target(${target_name})
    jump_cuda_sources(${ARGN})
endfunction()

# Mark an executable as a test (link gtest / deps & call gtest_discover_tests on it)
function(jump_test target_name)
    message(STATUS "[JUMP][${target_name}] adding test")
    target_link_libraries(${target_name} PUBLIC gtest gtest_main pthread jump)
    gtest_discover_tests(${target_name})
endfunction()

#####################################################################
####################### Executables #################################
#####################################################################
# add_executable(jump_main src/main.cpp)
# jump_cuda_target(jump_main)

if(${JUMP_BUILD_EXAMPLES})
    message(STATUS "[JUMP] Building Examples")

    if(${JUMP_ENABLE_CUDA})
        jump_cuda_executable(jump_memory_buffer_example src/examples/memory_buffer.cpp)
        jump_cuda_executable(jump_array_example src/examples/array.cpp)
        jump_cuda_executable(jump_multi_array_example src/examples/multi_array.cpp)
        jump_cuda_executable(jump_parallel_example src/examples/parallel.cpp)
        jump_cuda_executable(jump_octree_example src/examples/octree.cpp)
    endif()
endif()

#####################################################################
####################### Tests #######################################
#####################################################################
if(${JUMP_BUILD_TESTS})
    message(STATUS "[JUMP] Building Tests")
    if(${JUMP_ENABLE_CUDA})
        jump_cuda_executable(jump_memory_buffer_test src/tests/memory_buffer.cpp)
        jump_test(jump_memory_buffer_test)

        jump_cuda_executable(jump_array_test src/tests/array.cpp)
        jump_test(jump_array_test)

        jump_cuda_executable(jump_shared_ptr_test src/tests/shared_ptr.cpp)
        jump_test(jump_shared_ptr_test)

        jump_cuda_executable(jump_multi_array_test src/tests/multi_array.cpp)
        jump_test(jump_multi_array_test)
    endif()

    add_executable(jump_memory_buffer_test_no_cuda src/tests/memory_buffer.cpp)
    jump_test(jump_memory_buffer_test_no_cuda)

    add_executable(jump_array_test_no_cuda src/tests/array.cpp)
    jump_test(jump_array_test_no_cuda)

    add_executable(jump_shared_ptr_test_no_cuda src/tests/shared_ptr.cpp)
    jump_test(jump_shared_ptr_test_no_cuda)
endif()

#####################################################################
####################### Installation ################################
#####################################################################
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/jump
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/jump
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/jump
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
