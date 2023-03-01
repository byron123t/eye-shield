#include "KomputeShaders.hpp"
#include <chrono>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <vector>


using namespace std;
using namespace std::chrono;

KomputeShaders::KomputeShaders() {

}

KomputeShaders::~KomputeShaders() {

}

std::vector<float> KomputeShaders::runAlg(std::vector<float> img, std::vector<float> grid, std::vector<float> targ) {

    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    ostringstream out;
    string s = out.str();
    out.clear();
    ofstream outfile("/storage/emulated/0/Android/data/com.example.kompute/files/out1-penguin1280.txt");
    vector<double> durations;

    std::vector<float> zerosData;

    for (size_t i = 0; i < img.size(); i++) {
        zerosData.push_back(0);
    }

    {

        kp::Manager mgr(0);

        std::shared_ptr<kp::TensorT<float>> xImg = mgr.tensor(img);
        std::shared_ptr<kp::TensorT<float>> xGrid = mgr.tensor(grid);
        std::shared_ptr<kp::TensorT<float>> xTarg = mgr.tensor(targ);

        std::shared_ptr<kp::TensorT<float>> squared = mgr.tensor(zerosData);
        std::shared_ptr<kp::TensorT<float>> newimg = mgr.tensor(zerosData);

        std::shared_ptr<kp::TensorT<float>> pert = mgr.tensor(zerosData);
        std::shared_ptr<kp::TensorT<float>> outimg = mgr.tensor(zerosData);

        std::vector<std::shared_ptr<kp::Tensor>> params = { xImg, xGrid, xTarg,
                                                            squared, newimg, pert,
                                                            outimg };

        std::vector<uint32_t> spirv = std::vector<uint32_t>(
                (uint32_t*)shaders_glsl_runimg,
                (uint32_t*)(shaders_glsl_runimg +
                        shaders_glsl_runimg_len));


        std::shared_ptr<kp::Algorithm> algorithm = mgr.algorithm(
                params, spirv, kp::Workgroup({ 256 }), std::vector<float>({ 256.0 }));

        mgr.sequence()->eval<kp::OpTensorSyncDevice>(params);

        std::shared_ptr<kp::Sequence> sq = mgr.sequence()
            ->record<kp::OpTensorSyncDevice>({ xImg, xGrid, xTarg })
            ->record<kp::OpAlgoDispatch>(algorithm)
            ->record<kp::OpTensorSyncLocal>({ outimg });

        for(int i = 0; i < 100; i++) {
            start = high_resolution_clock::now();
            sq->eval();
            stop = high_resolution_clock::now();
            durations.emplace_back(duration_cast<microseconds>(stop - start).count());
        }
        double average = accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
        durations.clear();
        out << average;
        s = out.str();
        outfile << s;
        outfile << "microseconds";
        outfile.close();
        return outimg->vector();

    }
}
