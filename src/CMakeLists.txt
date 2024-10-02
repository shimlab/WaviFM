# Uncomment when doing testing
cmake_minimum_required(VERSION 3.30)

set(This WaviFMSrc)

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

target_link_libraries(${This} eigen)

set_target_properties(${This} PROPERTIES
    COMPILE_FLAGS "-Ofast -march=native -flto -funroll-loops -fomit-frame-pointer -funsafe-math-optimizations -ffast-math -fno-math-errno -fno-trapping-math -ftree-vectorize -fprefetch-loop-arrays -falign-functions=32 -finline-functions -fipa-pta -fstrict-aliasing -fwhole-program"
    LINK_FLAGS "-Wl,--gc-sections -Wl,-O2 -Wl,--as-needed"
)