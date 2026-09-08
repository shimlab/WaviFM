cmake_minimum_required(VERSION 3.20)

set(This WaveFactorTests)

set(Headers
    mocks.hpp
    testing_utilities.hpp
)

set(Sources
    mocks.cpp
    cavi_test.cpp
    elbo_test.cpp
    init_test.cpp
    parameters_test.cpp
    tensor_test.cpp
    updates_test.cpp
    utilities_test.cpp
    testing_utilities.cpp
)

add_executable(${This} ${Sources} ${Headers})

target_link_libraries(${This} PUBLIC
    gtest_main
    WaveFactorSrc
    eigen
)

add_test(
    NAME ${This}
    COMMAND ${This}
)

target_include_directories(${This} PRIVATE ${CMAKE_SOURCE_DIR}/src)
