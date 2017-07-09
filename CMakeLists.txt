cmake_minimum_required(VERSION 3.6)
project(rnn)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp src/lstm/lstm_paras.h src/lstm/lstm_paras.cpp src/lstm/lstm.h src/lstm/lstm.cpp src/utils/math.h src/utils/math.cpp)
include_directories(./include/eigen-3.3.4)
add_executable(rnn ${SOURCE_FILES})