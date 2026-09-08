cmake_minimum_required(VERSION 3.20)

set(This WaveFactorSrc)

set(Headers
    cavi.hpp
    elbo.hpp
    init.hpp
    parameters.hpp
    tensor.hpp
    updates.hpp
    utilities.hpp
)

set(Sources
    cavi.cpp
    elbo.cpp
    init.cpp
    parameters.cpp
    tensor.cpp
    updates.cpp
    utilities.cpp
)

add_library(${This} STATIC ${Sources} ${Headers})

target_link_libraries(${This} PUBLIC eigen)
target_include_directories(${This} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(${This} PRIVATE -O3 -march=native -funroll-loops -ffast-math)