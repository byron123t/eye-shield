//
// Created by btang on 4/5/2022.
//
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


void initVulkan() {
    createCommandPool();
    createTextureImage();
    createVertexBuffer();
}


void createTextureImage() {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }
}