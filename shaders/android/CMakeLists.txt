cmake_minimum_required(VERSION 3.4.1)

add_subdirectory(../../../../../../../ ${CMAKE_CURRENT_BINARY_DIR}/kompute_build)

set(VK_ANDROID_INCLUDE_DIR ${ANDROID_NDK}/sources/third_party/vulkan/src/include)
set("OpenCV_DIR" "E:/research/AndroidStudioProjects/kompute/examples/android/android-simple/build-android/install/sdk/native/jni/abi-armeabi-v7a")

add_library(kompute-jni SHARED
        RunEyeshield.cpp
        KomputeShaders.cpp)

find_package(OpenCV REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${VK_ANDROID_COMMON_DIR}
        ${VK_ANDROID_INCLUDE_DIR}
        ../../../../../../../single_include/
        ../../../../../../../vk_ndk_wrapper_include/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 \
                     -DVK_USE_PLATFORM_ANDROID_KHR=1 \
                     -DKOMPUTE_DISABLE_VK_DEBUG_LAYERS=1")

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

target_link_libraries(kompute-jni
        # Libraries from kompute build
        kompute
        kompute_vk_ndk_wrapper
        ${OpenCV_LIBS}
        ${log-lib}
        # Libraries from android build
        log
        android)