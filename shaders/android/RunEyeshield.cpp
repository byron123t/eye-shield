// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// Includes the Jni utilities for Android to be able to create the 
// relevant bindings for java, including JNIEXPORT, JNICALL , and 
// other "j-variables".
#include <jni.h>

// The ML class exposing the Kompute ML workflow for training and 
// prediction of inference data.
#include "KomputeShaders.hpp"

// Allows us to use the C++ sleep function to wait when loading the 
// Vulkan library in android
#include <unistd.h>
#include <chrono>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifndef KOMPUTE_VK_INIT_RETRIES
#define KOMPUTE_VK_INIT_RETRIES 5
#endif

using namespace std;
using namespace cv;
using namespace std::chrono;

float* createGrid(const int width, const int height, int gridsize) {
    float* arr3 = new float[width * height];
    float* ones = new float[gridsize];
    float* arr1 = new float[width];
    float* arr2 = new float[width];
    fill_n(ones, gridsize, 1);
    fill_n(arr1, width, 0);
    fill_n(arr2, width, 0);
    for (int i = 0; i < width; i += (gridsize * 2)) {
        copy(ones, ones + min(gridsize, width - (1 + gridsize)), arr1 + i);
    }
    for (int i = gridsize; i < width; i += (gridsize * 2)) {
        copy(ones, ones + min(gridsize, width - (1 + gridsize)), arr2 + i);
    }
    for (int i = 0; i < height; i++) {
        if (static_cast<int>(floor(i / gridsize)) % 2) {
            copy(arr1, arr1 + width, arr3 + i * width);
        } else {
            copy(arr2, arr2 + width, arr3 + i * width);
        }
    }
    return arr3;
}

extern "C" {
void Java_com_example_kompute_KomputeJni_kompute(
        JNIEnv *env,
        jobject thiz,
        jfloatArray xiJFloatArr,
        jfloatArray xjJFloatArr,
        jfloatArray yJFloatArr) {

    KP_LOG_INFO("Creating manager");

//    std::vector<float> xiVector = jfloatArrayToVector(env, xiJFloatArr);
//    std::vector<float> xjVector = jfloatArrayToVector(env, xjJFloatArr);
//    std::vector<float> yVector = jfloatArrayToVector(env, yJFloatArr);

    Mat img, target;
    int width = img.size().width;
    int height = img.size().height;
    imread("/storage/emulated/0/Android/data/com.example.kompute/files/penguin1600.png", IMREAD_COLOR).copyTo(img);
    int gridsize = 1;
    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    ostringstream out;
    string s = out.str();
    out.clear();
    ofstream outfile("/storage/emulated/0/Android/data/com.example.kompute/files/out-penguin1280.txt");
    vector<double> durations;
    float *tempgrid;

    for(int i = 0; i < 100; i++) {
        start = high_resolution_clock::now();
        tempgrid = createGrid(width, height, gridsize);
        stop = high_resolution_clock::now();
        durations.emplace_back(duration_cast<microseconds>(stop - start).count());
    }
    double average = accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
    durations.clear();
    out << average;
    s = out.str();
    outfile << s;
    outfile << "microseconds";

    vector<float> grid(tempgrid, tempgrid + img.size().width + img.size().height);
    __android_log_print(ANDROID_LOG_VERBOSE, "testjni", "grid");

    for(int i = 0; i < 100; i++) {
        start = high_resolution_clock::now();
        GaussianBlur(img, target, Size(7, 7), 1.5);
        stop = high_resolution_clock::now();
        durations.emplace_back(duration_cast<microseconds>(stop - start).count());
    }
    average = accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
    durations.clear();
    out << average;
    s = out.str();
    out.clear();
    outfile << s;
    outfile << "microseconds";
    __android_log_print(ANDROID_LOG_VERBOSE, "testjni", "blur");

    Mat flat;
    vector<float> targvec;
    vector<float> imgvec;
    for(int i = 0; i < 100; i++) {
        start = high_resolution_clock::now();
        flat = img.reshape(1,1);
        flat.copyTo(imgvec);
        flat = target.reshape(1,1);
        flat.copyTo(targvec);
        stop = high_resolution_clock::now();
        durations.emplace_back(duration_cast<microseconds>(stop - start).count());
    }
    average = accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
    durations.clear();
    out << average;
    s = out.str();
    out.clear();
    outfile << s;
    outfile << "microseconds";
    outfile.close();

    __android_log_print(ANDROID_LOG_VERBOSE, "testjni", "flat");

    KomputeShaders kml;
    Mat output = Mat(kml.runAlg(imgvec, grid, targvec));
    __android_log_print(ANDROID_LOG_VERBOSE, "testjni", "run");
    output.reshape(3, vector<int>(width, height));
}
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_kompute_KomputeJni_initVulkan(JNIEnv *env, jobject thiz) {

    KP_LOG_INFO("Initialising vulkan");

    uint32_t totalRetries = 0;

    while (totalRetries < KOMPUTE_VK_INIT_RETRIES) {
        KP_LOG_INFO("VULKAN LOAD TRY NUMBER: %u", totalRetries);
        if (InitVulkan()) {
            break;
        }
        sleep(1);
        totalRetries++;
    }

    return totalRetries < KOMPUTE_VK_INIT_RETRIES;
}

}
