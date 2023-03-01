# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import os
import cv2
from src.algorithm import eyeshield_rmse
from src.algorithm_cuda import cupy_eyeshield_rmse
from src.utils import pixelate, blur, create_grid_mask, decrease_contrast
from src.utils import open_video, save_image, convert_to_gpu


def protect_image(img, mode='blur', strength='strong', resolution=1920, gpu=False):
    """
    It takes a numpy image, and returns a new image that is a shoulder-surfing protected image
    
    :param img: the numpy image to be protected
    :param mode: 'blur' or 'pixelate', defaults to blur (optional)
    :param strength: how strong of a protection for the image, defaults to strong (optional)
    :param resolution: the resolution of the image. This is used to determine the size of the grid, defaults to 1920 (optional)
    :param gpu: whether to use the GPU or not, defaults to False (optional)
    :return: the protected image
    """
    if resolution >= 4000:
        gridsize = 4
    elif resolution >= 3000:
        gridsize = 3
    elif resolution >= 2000:
        gridsize = 2
    else:
        gridsize = 1

    if strength == 'full':
        sigma = 24
        num_blocks_height = 8
        contrast = 80
    elif strength == 'strong':
        sigma = 20
        num_blocks_height = 16
        contrast = 100
    elif strength == 'moderate':
        sigma = 16
        num_blocks_height = 24
        contrast = 115
    elif strength == 'weak':
        sigma = 8
        num_blocks_height = 32
        contrast = 127
    else:
        raise Exception('Only strengths of "full", "strong", "moderate", and "weak" are available.')

    if gpu:
        grid_img = img.download()
    else:
        grid_img = img

    if mode == 'blur':
        target = blur(img, sigma, gpu)
    elif mode == 'pixelate':
        target = pixelate(img, num_blocks_height, gpu)
    else:
        raise Exception('Only "blur" and "pixelate" are defined as modes.')

    grid = create_grid_mask(grid_img, gridsize, gpu)
    target = decrease_contrast(target, contrast=contrast)
    grid_img = decrease_contrast(grid_img, contrast=contrast)
    if gpu:
        import cupy as cp
        grid_img = cp.array(grid_img).astype(cp.int32)
        grid = cp.array(grid)
        target = cp.array(target).astype(cp.int32)
        protected = cp.asnumpy(cupy_eyeshield_rmse(grid_img, grid, target))
    else:
        protected = eyeshield_rmse(grid_img, grid, target)
    return protected
