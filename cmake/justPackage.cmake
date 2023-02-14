##
# Supposed to help find / include / link against other packages in
# the Just Suite. This will be designed around the following setups:
# - targets are already added as subdirectories to a parent CMakeLists.txt, eg:
#       parent_ws/CMakeLists.txt
#           ```
#           add_subdirectory(sub_project_a)
#           add_subdirectory(sub_project_b)
#           ```
#       parent_ws/sub_project_a/...
#       parent_ws/sub_project_b/CMakeLists.txt
#           ```
#           just_find_package(sub_project_a)
#           ```
#       parent_ws/sub_project_b/...
#   we expect in the parent CMakeLists.txt they are ordered correctly (sub_project_b
#   depends on sub_project_a so add_subdirectory(sub_project_a) is first)
#
# - dependencies are in the libs/ folder under the main project, eg:
#       sub_project_b/libs/sub_project_a/...
#       sub_project_b/CMakeLists.txt
#           ```
#           just_find_package(sub_project_a)
#           ```
#       sub_project_b/...
#   sub_project_b/libs/sub_project_a is added as a subdirectory to sub_project_b
#
# - targets are all added in the same directory, although no parent CMakeLists.txt exists, eg:
#       parent_ws/sub_project_a/...
#       parent_ws/sub_project_b/CMakeLists.txt
#           ```
#           just_find_package(sub_project_a)
#           ```
#       parent_ws/sub_project_b/...
#   we then add sub_project_a as a 'subdirectory' to sub_project_b
#
# - dependencies are installed and can be found with find_package(), additional arguments to
#   just_find_package() are processed as arguments to find_package, this is the only setup
#   where version or additional arguments are processed and not assumed to be correct.
#
# just_find_package checks for the above setups in the order listed above and stops when
# any of the above setups work (target exists, folder is found, find_package succeeds). All
# targets are assumed to be required.
##
function(just_find_package package_name)
    # Scenario 1: target exists
    if(TARGET ${package_name})
        message(STATUS "${PROJECT_NAME}:${package_name} target exists (1)")
        return()
    endif()

    # Scenario 2: libs/${package_name}
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libs/${package_name})
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/${package_name})
        message(STATUS "${PROJECT_NAME}:${package_name} adding ${CMAKE_CURRENT_SOURCE_DIR}/libs/${package_name} (2)")
        return()
    endif()

    # Scenario 3: ../${package_name}
    get_filename_component(PARENT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    if(EXISTS ${PARENT_DIRECTORY}/${package_name})
        add_subdirectory(${PARENT_DIRECTORY}/${package_name} ${CMAKE_CURRENT_BINARY_DIR}/${package_name})
        message(STATUS "${PROJECT_NAME}:${package_name} adding ${PARENT_DIRECTORY}/${package_name} (3)")
        return()
    endif()

    # Scenario 4: find_package(${package_name})
    find_package(${package_name} ${ARGN})
    if(${package_name}_FOUND)
        message(STATUS "${PROJECT_NAME}:${package_name} found. (4)")
        return()
    endif()

    message(FATAL_ERROR "${PROJECT_NAME}:${package_name} not found")
endfunction()
