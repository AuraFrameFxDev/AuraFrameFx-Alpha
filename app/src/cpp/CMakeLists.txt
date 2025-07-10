# app/src/main/cpp/CMakeLists.txt

# Sets the minimum version of CMake required.
cmake_minimum_required(VERSION 3.22.1)

# Declares your project and the languages used.
project("auraframesfx" LANGUAGES CXX)

# Adds your C++ code as a shared library (.so file).
# "aura-native-lib" is the name of your library.
# "native-lib.cpp" is your source code file.
add_library(aura-native-lib SHARED native-lib.cpp)

# Finds the standard Android logging library.
find_library(log-lib log)

# Links the logging library to your native library so you can print to logcat from C++.
target_link_libraries(aura-native-lib ${log-lib})