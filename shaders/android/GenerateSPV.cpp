#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <stdio.h>
#include <inttypes.h>

static std::vector<uint32_t>
compileSource(
  const std::string& source)
{
    if (system(std::string("glslangValidator --stdin -S comp -V -o tmp_kp_shader.comp.spv << END\n" + source + "\nEND").c_str()))
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return {(uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size())};
}

int main() {
static std::string LR_SHADER = R"(
#version 450

layout (constant_id = 0) const uint M = 0;

layout (local_size_x = 1) in;

layout(set = 0, binding = 0) buffer bxImg { float xImg[]; };
layout(set = 0, binding = 1) buffer bxGrid { float xGrid[]; };
layout(set = 0, binding = 2) buffer bxTarg { float xTarg[]; };
layout(set = 0, binding = 3) buffer boutimg { float outimg[]; };

float m = float(M);

void main() {
    uint index = gl_GlobalInvocationID.x;

    outimg[index] = sqrt((xImg[index] * xImg[index]) + ((((xTarg[index] * xTarg[index]) * 2) - (xImg[index] * xImg[index])) - (xImg[index] * xImg[index])) * xGrid[index]);
}
)";
    std::vector<uint32_t> out = compileSource(LR_SHADER);
    for (uint32_t i: out) {
        printf("%" PRIu32 "\n", i);
    }
    return 0;
}