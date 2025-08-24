project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

file(GLOB HEADERS src/*.h)
file(GLOB HEADERS_HPP src/*.hpp)

set(sources src/particle_filter.cpp src/main.cpp ${HEADERS} ${HEADERS_HPP})



if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

# Detect Homebrew prefix (works for both Intel and ARM Macs)
execute_process(COMMAND brew --prefix OUTPUT_VARIABLE HOMEBREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

include_directories(${HOMEBREW_PREFIX}/include)
include_directories(${HOMEBREW_PREFIX}/opt/openssl@3/include)

link_directories(${HOMEBREW_PREFIX}/lib)
link_directories(${HOMEBREW_PREFIX}/opt/openssl@3/lib)
link_directories(${HOMEBREW_PREFIX}/opt/libuv/lib)

endif() 


add_executable(particle_filter ${sources})


target_link_libraries(particle_filter z ssl uv uWS)

