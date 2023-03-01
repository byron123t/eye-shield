# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

from timeit import default_timer as timer
import numpy as np
import cv2
from math import ceil
from glob import glob


class Benchmark:
    """
    It's a class that can be used to benchmark the performance of a function
    """

    def __init__(self):
        self.start = -1
        self.end = -1
        self.verbose = False
        
    def set_verbose(self):
        self.verbose = True

    def mark(self, message=''):
        """
        The first call sets self.start to the current time. Subsequent calls cause return (and print if verbose is set to True) the runtime.
        
        :param message: The message to print
        :return: The time it took to run the code
        """
        if self.start == -1:
            self.start = timer()
        else:
            if self.end == -1:
                self.end = timer()
            self.time = self.end - self.start
            if self.verbose:
                print('{message:{fill}{align}{width}}-{time}'
                      .format(message=message, fill='-', align='<', width=30, time=self.time))
            self.start = -1
            self.end = -1
            return self.time


BM = Benchmark()


def decrease_contrast(img, brightness=255, contrast=127):
    """
    It takes an image, and returns a new image with the brightness and contrast adjusted
    
    :param img: The image to be processed
    :param brightness: 0-255, defaults to 255 (optional)
    :param contrast: 0-127, defaults to 127 (optional)
    :return: A numpy array of the image
    """
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            maxs = 255
        else:
            shadow = 0
            maxs = 255 + brightness
        al_pha = (maxs - shadow) / 255
        ga_mma = shadow
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
    else:
        cal = img
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)
    return cal


def pixelate(img, num_blocks_height, gpu=False):
    """
    It takes an image, resizes it to a smaller size, then resizes it back to its original size, but with a different interpolation method to pixelate the image
    
    :param img: the image to be pixelated
    :param num_blocks_height: The number of blocks in the height of the image
    :param gpu: If you want to use the GPU, set this to True, defaults to False (optional)
    :return: The pixelated image is being returned
    """
    new_size = (img.shape[1], img.shape[0])
    aspect_ratio = img.shape[1] / img.shape[0]
    num_blocks_width = round(num_blocks_height * aspect_ratio)
    dim = (num_blocks_width, num_blocks_height)
    if gpu:
        imgsmall = cv2.cuda.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        img = cv2.cuda.resize(imgsmall, new_size, interpolation=cv2.INTER_NEAREST).download()
    else:
        imgsmall = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(imgsmall, new_size, interpolation=cv2.INTER_NEAREST)
    return img


def blur(img, sigma, gpu=False):
    """
    If the GPU flag is set, create a Gaussian filter with the given sigma and apply it to the image. Otherwise, use the CPU version of the Gaussian blur
    
    :param img: The image to be blurred
    :param sigma: The standard deviation of the Gaussian filter
    :param gpu: If True, will use the GPU to blur the image, defaults to False (optional)
    :return: The blurred image is being returned
    """
    if gpu:
        ksize = min((sigma * 2) - 1, 31)
        fil = cv2.cuda.createGaussianFilter(img.type(), img.type(), (ksize, ksize), sigma, sigma)
        img = fil.apply(img).download()
    else:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    return img


def create_grid_mask(img, grid_halfsize, gpu=False):
    """
    It creates a grid mask that is the same size as the image, and has a grid size of `grid_halfsize`
    
    :param img: the image you want to create the grid mask for
    :param grid_halfsize: the size of the grid block in pixels
    :param gpu: boolean, whether to use cupy or numpy, defaults to False (optional)
    :return: A grid mask
    """
    if gpu:
        import cupy as cp
        width = img.shape[1]
        height = img.shape[0]
        tile = cp.array([[0,1],[1,0]]).repeat(grid_halfsize, axis=0).repeat(grid_halfsize, axis=1)
        thing = ceil(height/(2*grid_halfsize)+1), ceil(width/(2*grid_halfsize)+1)
        grid = cp.tile(tile, thing)
        grid = cp.dstack((grid[:height,:width], grid[:height,:width], grid[:height,:width]))
    else:
        width = img.shape[1]
        height = img.shape[0]
        tile = np.array([[0,1],[1,0]]).repeat(grid_halfsize, axis=0).repeat(grid_halfsize, axis=1)
        thing = ceil(height/(2*grid_halfsize)+1), ceil(width/(2*grid_halfsize)+1)
        grid = np.tile(tile, thing)
        grid = np.dstack((grid[:height,:width], grid[:height,:width], grid[:height,:width]))
    return grid


def downscale(img, dim, resample):
    img = cv2.resize(img, dim, interpolation=resample)
    return img


def uint8_float(img):
    return (img * 255).astype(np.uint8)


def uint8_clip(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def get_ssim(img1, img2):
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, channel_axis=2)


def get_mse(img1, img2):
    from skimage.metrics import mean_squared_error
    return mean_squared_error(img1, img2)


def get_l2(img1, img2):
    return np.linalg.norm(img1 - img2)


def get_entropy(img):
    from skimage.color import rgb2gray
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    best_disk = disk(5)
    gray_entropy = entropy(rgb2gray(img), best_disk)
    return threshold_checker(gray_entropy)


def threshold_checker(entropies):
    """
    It takes the entropy image and returns the areas of the image that are above the threshold
    
    :param entropies: the entropy values of the image
    :return: The area ratio of the image
    """
    th = 0.7
    scaled_entropy = entropies / entropies.max()
    thresh = scaled_entropy > th
    pixels = len(np.column_stack(np.where(thresh > 0)))
    image_area = entropies.shape[0] * entropies.shape[1]
    area_ratio = (pixels / image_area) * 100
    return area_ratio


def get_shannon_entropy(img):
    from skimage.measure.entropy import shannon_entropy
    return shannon_entropy(img)


def open_image(file, gpu=False):
    if gpu:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(cv2.imread(file))
        return cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)


def open_video(file, gpu=False, verbose=False):
    """
    It opens a video file, reads the frames, and returns a list of the frames and the fps of the video
    
    :param file: The path to the video file
    :param gpu: If you have a GPU, set this to True, defaults to False (optional)
    :return: A list of images and the fps of the video
    """
    vidcap = cv2.VideoCapture(file)
    images = []
    pos_frame = 0
    while not vidcap.isOpened():
        vidcap = cv2.VideoCapture(file)
        cv2.waitKey(1000)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    while vidcap.isOpened():
        hasFrames, image = vidcap.read()
        if hasFrames:
            # cv2.imshow('Frame', image)
            images.append(image)
            pos_frame += 1
            cv2.waitKey(25)
        else:
            break
        if verbose:
            print('                          ', end='\r')
            print('Frames Processed: {}'.format(pos_frame), end='\r')
    vidcap.release()
    cv2.destroyAllWindows()
    print(fps)
    return images, fps


def convert_to_gpu(img):
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    return gpu_img


def save_image(file, img):
    return cv2.imwrite(file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_video(file_pattern, file, fps):
    """
    It takes a file pattern, a file name, and a frame rate, and saves a video file using ffmpeg. Requires a terminal installation of ffmpeg.
    
    :param file_pattern: The pattern of the files to be converted
    :param file: the name of the video file to be saved
    :param fps: frames per second
    """
    import subprocess
    print(file, file_pattern)
    subprocess.run(['ffmpeg', '-framerate', str(int(fps)), '-pattern_type', 'glob', '-i', file_pattern, '-c:v', 'libx265', '-crf', '0', '-c:a', 'aac', '-b:a', '128k', '-tag:v', 'hvc1', '-pix_fmt', 'yuv420p', file])
