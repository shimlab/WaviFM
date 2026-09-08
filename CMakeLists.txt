cmake_minimum_required(VERSION 3.20)

project(WaveFactor LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Subdirectories for libraries
add_subdirectory(src)
add_subdirectory(lib/googletest)
add_subdirectory(lib/eigen)

# Enable testing for C++ GoogleTest suite
enable_testing()
add_subdirectory(test)

# Headers and Sources for Pybind11 extension module
set(Headers
    src/cavi.hpp
    src/elbo.hpp
    src/init.hpp
    src/parameters.hpp
    src/tensor.hpp
    src/updates.hpp
    src/utilities.hpp
    src/bindings.hpp
)

set(Sources
    src/cavi.cpp
    src/elbo.cpp
    src/init.cpp
    src/parameters.cpp
    src/tensor.cpp
    src/updates.cpp
    src/utilities.cpp
    src/bindings.cpp
)

# Pybind11 module build configuration
option(BUILD_PYTHON_MODULE "Build Python Pybind11 extension module" ON)

if(BUILD_PYTHON_MODULE)
    set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use FindPython mode in pybind11" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11/CMakeLists.txt")
        add_subdirectory(lib/pybind11)
    else()
        find_package(pybind11 CONFIG QUIET)
    endif()

    if(TARGET pybind11::module OR TARGET pybind11::pybind11)
        pybind11_add_module(WaveFactor SHARED ${Sources} ${Headers})
        target_link_libraries(WaveFactor PRIVATE eigen)
        target_include_directories(WaveFactor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

        # Optimization flags (GCC)
        target_compile_options(WaveFactor PRIVATE -O3 -march=native -funroll-loops -ffast-math)

        # Create __init__.py in binary directory for testing direct imports from build
        set(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
        if(NOT EXISTS ${INIT_PY})
            file(WRITE ${INIT_PY} "")
        endif()
    endif()
endif()