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

#####################################################################
####################### Misc. Project Setup #########################
#####################################################################
set(JUMP_VERSION 0.1)

project(jump VERSION ${JUMP_VERSION} LANGUAGES CXX)

### OPTIONS
option(JUMP_BUILD_TESTS "Build tests" ON)
option(JUMP_BUILD_EXAMPLES "Build examples" ON)
option(JUMP_ENABLE_CUDA "Enable CUDA as a parallelization backend" ON)
option(JUMP_BUILD_DOCS "Build documentation with doxygen" ON)

if(${JUMP_ENABLE_CUDA})
    # include(CheckLanguage)
    # check_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    enable_language(CUDA)
endif()

if(${JUMP_BUILD_TESTS})
    include(GoogleTest)
    enable_testing()
endif()

# Includes for target definition
include(cmake/jumpMacros.cmake)

# Includes for installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

#####################################################################
####################### Documentation Building ######################
#####################################################################
if(${JUMP_BUILD_DOCS})
    # Thanks to https://vicrucann.github.io/tutorials/quick-cmake-doxygen/
    find_package(Doxygen)

    if(DOXYGEN_FOUND)
        add_custom_target(jump_docs ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/.doxygen
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Generating JUMP API Documentation with Doxygen"
            VERBATIM
        )
    else()
        message(SEND_ERROR "Doxygen must be installed to generate doxygen documentation")
    endif()
endif()

#####################################################################
####################### Library Definition ##########################
#####################################################################
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
# We do this instead of compile_features() becase cxx_std_17 feature is not happy with CUDA
target_compile_options(${PROJECT_NAME} INTERFACE -std=c++17 -O3 -g)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

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
        jump_cuda_executable(jump_threadpool_example src/examples/threadpool.cpp)
        jump_cuda_executable(jump_random_example src/examples/random.cpp)
    endif()
endif()

#####################################################################
####################### Test Target macros ##########################
#####################################################################
# Mark an executable as a test (link gtest / deps & call gtest_discover_tests on it)
function(jump_test target_name)
    message(STATUS "[JUMP][${target_name}] adding test")
    target_link_libraries(${target_name} PUBLIC gtest gtest_main pthread jump gcov)
    target_compile_options(${target_name} PUBLIC -O0 --coverage -fno-elide-constructors)
    gtest_discover_tests(${target_name})
endfunction()

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

    add_custom_target(jump_lcov
        COMMAND lcov --no-external --directory . --capture --output-file build/coverage.info
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating lcov report for JUMP"
        VERBATIM
    )
    add_custom_target(jump_coverage
        COMMAND genhtml --demangle-cpp --ignore-errors source --legend -o build/coverage build/coverage.info --prefix ${PROJECT_SOURCE_DIR}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating html report for JUMP"
        DEPENDS jump_lcov
        VERBATIM
    )
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
        "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Macros.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME} DESTINATION include)
