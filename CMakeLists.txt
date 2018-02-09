# Force CMake version 3.1 or above
cmake_minimum_required (VERSION 3.1)

# This project has the name: tw3720tu.2017
project (tw3720tu.2017)

# Prevent in-source builds 
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are not allowed.")
endif()

# Output message
message("Build all build targets of the course tw3720tu from 2017")

# Macro for generating log files for submitting assignments 
macro(add_assignment target src)
  add_custom_target(
    "${target}_assignment"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${src}" "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "=== Start Environment " >> "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "PROJECT_NAME = ${PROJECT_NAME}" >> "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "CMAKE_SYSTEM = ${CMAKE_SYSTEM}" >> "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "CMAKE_VERSION = ${CMAKE_VERSION}" >> "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}" >> "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}" >> "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "=== End Environment " >> "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "=== Start of output from execution of binary ${target}" >> "${target}.log"
    COMMAND ${target} >> "${target}.log"
    COMMAND ${CMAKE_COMMAND} -E echo "=== End of output from execution of binary ${target}" >> "${target}.log"
    DEPENDS ${target}
    VERBATIM
    )
endmacro()

# Enable CMake testing
include (CTest)
enable_testing()

# Enable Doxygen documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile @ONLY)
    add_custom_target(doc
      ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
    )
endif()

