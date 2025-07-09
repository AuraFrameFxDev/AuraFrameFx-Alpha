cmake_minimum_required(VERSION 3.18.1) # Or 3.22.1 as specified in build.gradle

# Set project name
project(aura_native_lib LANGUAGES C CXX) # Changed project name

# Set library name
set(LIBRARY_NAME aura-native-lib) # Explicitly set library name

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add JNI includes - This might not be needed if not using native_app_glue directly
# include_directories(${CMAKE_ANDROID_NDK}/sources/android/native_app_glue)

# Find log library
find_library(log-lib log)

# Create the shared library
add_library(
        ${LIBRARY_NAME}
        SHARED
        language_id_l2c_jni.cpp # Assumes this file is in the same directory (app/src/main/cpp)
        ../native-lib.cpp       # Path to native-lib.cpp from app/src/main/cpp/ to app/src/cpp/
)

# Link libraries
target_link_libraries(
        ${LIBRARY_NAME}
        ${log-lib}
        android # Common Android native APIs
)

# Set compile definitions (optional, but good practice)
target_compile_definitions(${LIBRARY_NAME} PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        -DANDROID
)

# Output directories are usually handled by Gradle, but explicit setting can be kept if needed.
# Ensure these paths are correct if kept, or remove if Gradle manages them well.
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/intermediates/cxx/Debug/${ANDROID_ABI})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/intermediates/cxx/Debug/${ANDROID_ABI})

# Set output name to match Android naming conventions (lib<name>.so)
# Gradle usually handles this, but explicit control can be useful.
# The OUTPUT_NAME should be just "aura-native-lib", Gradle prepends "lib" and appends ".so".
set_target_properties(${LIBRARY_NAME} PROPERTIES
        OUTPUT_NAME ${LIBRARY_NAME}
        # PREFIX "lib" # Typically handled by Gradle/CMake default
        # SUFFIX ".so" # Typically handled by Gradle/CMake default
)
