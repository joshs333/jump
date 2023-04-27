##
# Macros that help define jump targets.
#
# Usage:
#   # Already included with a find_package(jump)
#   include(cmake/jumpMacros.cmake)
#
#   # Suppose you have a target that must use CUDA, instead of
#   add_executable(my_executable src/main.cpp src/source_file_a.cpp src/source_file_b.cpp)
#   target_link_libraries(my_executable PUBLIC jump ${CUDA_LIBRARIES})
#   target_compile_options(my_executable PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
#   target_compile_definitions(my_executable PUBLIC JUMP_ENABLE_CUDA)
#   set_source_files_properties(src/main.cpp src/source_file_a.cpp src/source_file_b.cpp PROPERTIES LANGUAGE CUDA)
#   (or some variations...)
#
#   # you can do
#   jump_cuda_executable(my_executable src/main.cpp src/source_file_a.cpp src/source_file_b.cpp)
#
#   # Suppose you have a target that can use CUDA if it's enabled as a lannguage, instead of
#   add_executable(my_executable src/main.cpp src/source_file_a.cpp src/source_file_b.cpp)
#   if(DEFINED CMAKE_CUDA_COMPILER_VERSION)
#       target_link_libraries(my_executable PUBLIC jump ${CUDA_LIBRARIES})
#       target_compile_options(my_executable PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
#       target_compile_definitions(my_executable PUBLIC JUMP_ENABLE_CUDA)
#       set_source_files_properties(src/main.cpp src/source_file_a.cpp src/source_file_b.cpp PROPERTIES LANGUAGE CUDA)
#   endif()
#   (or some variations...)
#
#   # you can do
#   jump_executable(my_executable src/main.cpp src/source_file_a.cpp src/source_file_b.cpp)
#
# Take a look below for more detail, it should (hopefully) not be that confused.
##

##
# Adds compile definitions to a target to enable cuda functionality with jump
#
#   !! Fails if CUDA is not enabled
##
function(jump_cuda_target target_name)
    if(NOT DEFINED CMAKE_CUDA_COMPILER_VERSION)
        message(FATAL_ERROR "Ensure CUDA language is enabled before making a jump_cuda_target")
        return()
    endif()

    message(STATUS "[JUMP][${target_name}] CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    find_package(CUDA REQUIRED)
    message(STATUS "[JUMP][${target_name}] CUDA Version: ${CUDA_VERSION}")

    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(${target_name} PUBLIC JUMP_ENABLE_CUDA)
    target_link_libraries(${target_name} PUBLIC jump ${CUDA_LIBRARIES})
    target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIRS})

    set(NVCC_ARGS ${NVCC_ARGS} --expt-relaxed-constexpr  -Xptxas -v -Xcudafe "--diag_suppress=20012 --display_error_number")
    target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_ARGS}>)
endfunction()

##
# Mark some list of sources as being the CUDA language. This is useful if
#   (like me) you dislike the .cu and .cuh extensions. My personal reasons
#   for disliking them revolve around a preference for interopable code and
#   believing in a future where it's all c++ (.cpp).
#
#   For anyone who disagrees, I acknowledge this is probably
#   not good practice. But it works, and I'm stubborn. So until
#   I scew someone else over I'm happy with this. :)
#
#   !! Fails if CUDA is not enabled
##
function(jump_cuda_sources)
    if(NOT DEFINED CMAKE_CUDA_COMPILER_VERSION)
        message(FATAL_ERROR "Ensure CUDA language is enabled before setting jump_cuda_sources")
        return()
    endif()

    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
endfunction()

##
# Functionally serves as a replacement for add_executable, except it's intended
#   to define targets that use the cuda backend with jump.
#
#   !! Fails if CUDA is not enabled
##
function(jump_cuda_executable target_name)
    if(NOT DEFINED CMAKE_CUDA_COMPILER_VERSION)
        message(FATAL_ERROR "Ensure CUDA language is enabled before making a jump_cuda_executable")
        return()
    endif()

    add_executable(${target_name} ${ARGN})
    jump_cuda_target(${target_name})
    jump_cuda_sources(${ARGN})
endfunction()

##
# Functionally serves as a replacement for add_library, except it's intended
#   to define libraries that use the cuda backend with jump.
#
#   !! Fails if CUDA is not enabled
##
function(jump_cuda_library target_name)
    if(NOT DEFINED CMAKE_CUDA_COMPILER_VERSION)
        message(FATAL_ERROR "Ensure CUDA language is enabled before making a jump_cuda_executable")
        return()
    endif()

    add_library(${target_name} ${ARGN})
    jump_cuda_target(${target_name})
    jump_cuda_sources(${ARGN})
endfunction()

##
# Functionally serves as a replacement for add_executable, except
#   if CUDA is enabled as a langage, it performs the same actions
#   as jump_cuda_executable.
##
function(jump_executable target_name)
    add_executable(${target_name} ${ARGN})

    if(DEFINED CMAKE_CUDA_COMPILER_VERSION)
        jump_cuda_target(${target_name})
        jump_cuda_sources(${ARGN})
    endif()
endfunction()

##
# Functionally serves as a replacement for add_library, except
#   if CUDA is enabled as a langage, it performs the same actions
#   as jump_cuda_executable.
##
function(jump_library target_name)
    add_library(${target_name} ${ARGN})

    if(DEFINED CMAKE_CUDA_COMPILER_VERSION)
        jump_cuda_target(${target_name})
        jump_cuda_sources(${ARGN})
    endif()
endfunction()
