# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import cupy as cp


def cupy_eyeshield_rmse(img, grid, pixelated):
    """
    It takes the original image, the grid, and the pixelated or blurred image, and computes the complementary pixels with the grid applied that average with the original pixels to equal the target image. Uses CUDA instead of numpy.
    
    :param img: the original image
    :param grid: the grid mask of the image size
    :param target: the pixelated or blurred image
    :return: the protected image
    """
    squared = cp.power(img, 2)
    newimg = (cp.power(pixelated, 2) * 2) - squared
    perturbation = (newimg - squared) * grid
    img = cp.clip(cp.sqrt(squared + perturbation), 0, 255).astype(cp.uint8)
    return img
