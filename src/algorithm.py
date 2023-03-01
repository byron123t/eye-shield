# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import numpy as np
from numba import njit


@njit
def eyeshield_rmse(img, grid, target):
    """
    It takes the original image, the grid, and the pixelated or blurred image, and computes the complementary pixels with the grid applied that average with the original pixels to equal the target image.
    
    :param img: the original image
    :param grid: the grid mask of the image size
    :param target: the pixelated or blurred image
    :return: the protected image
    """
    squared = np.power(img, 2)
    newimg = (np.power(target, 2) * 2) - squared
    perturbation = (newimg - squared) * grid
    img = np.clip(np.sqrt(squared + perturbation), 0, 255).astype(np.uint8)
    return img
