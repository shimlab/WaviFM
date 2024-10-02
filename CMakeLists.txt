cmake_minimum_required(VERSION 3.30)

set(This WaviFM)

project(${This})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(src)
add_subdirectory(googletest)
add_subdirectory(eigen)

enable_testing()
add_subdirectory(test)

# For compiling into a python module (comment out all of the following when doing testing)
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

# These PYTHON settings will need to be changed depending on one's setup
set(PYBIND11_PYTHON_VERSION 3.7.12)
set(PYTHON_EXECUTABLE "C:/Users/yw/miniconda3/envs/py3712/python.exe")
include_directories(pybind11)
add_subdirectory(pybind11)
pybind11_add_module(${This} SHARED ${Sources} ${Headers})
target_link_libraries(${This} PRIVATE eigen)

# Needed for importing the python module from the build directory
set(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/__init__.py")
if(NOT EXISTS ${INIT_PY})
    file(WRITE ${INIT_PY} "")
endif()

# Set optimization flags for Pybind11 target (below has two alternatives; can also comment out both or use other sets of flags for less aggressively optimised code (when numerical accuracy is considered vital, to the extent that longer runtimes are considered acceptable))
# Should update compile flags changes both here and in the CMakeLists.txt in \src so to let testing be performed on the binaries optimised the same way as in the pybind bindings for consistency.

# Extremely aggressive optimization (potentially can optimize further using profile guided optimization (involving -fprofile-generate, -fprofile-use flags); this may slightly affect numerical accuracy
set_target_properties(${This} PROPERTIES
    COMPILE_FLAGS "-Ofast -march=native -flto -funroll-loops -fomit-frame-pointer -funsafe-math-optimizations -ffast-math -fno-math-errno -fno-trapping-math -ftree-vectorize -fprefetch-loop-arrays -falign-functions=32 -finline-functions -fipa-pta -fstrict-aliasing -fwhole-program"
    LINK_FLAGS "-Wl,--gc-sections -Wl,-O2 -Wl,--as-needed"
)

# # Aggresive optimization
# set_target_properties(${This} PROPERTIES COMPILE_FLAGS "-O3 -march=native -flto -funroll-loops -fomit-frame-pointer")