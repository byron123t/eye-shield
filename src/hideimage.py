# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import numpy as np
import os
import argparse
from src.utils import BM, open_image, save_image, create_grid_mask, blur, pixelate
from tqdm import tqdm
from src.algorithm import eyeshield_rmse


ROOT = os.path.abspath('data')
TIMES = {'cie': [], 'algo': [], 'blur': [], 'open': [], 'grid': [], 'save': [], 'pixelate': [], 'pixelateblur': []}
PEAK_MEMORY = 0
PEAK_CPU = 0
img_id = 0
grid_id = 0
target_id = 0
threadsperblock = 32
GPU_MEMORY = 0


def write_performance(dataset, mode, colormode, gridsize=1, sigma=1, num_blocks_height=16, log_memory=False, gpu=False):
    """
    It writes the performance data to a csv file
    
    :param dataset: Tte name of the dataset
    :param mode: the method for computing the target ('blur', 'pixelate', or 'pixelateblur')
    :param colormode: the method for computing the complement colors ('avg', 'sqrt', or 'ciecam')
    :param grid_halfsize: the size of each grid checker
    :param sigma: the standard deviation of the Gaussian kernel used to blur the image, defaults to 1 (optional)
    :param num_blocks_height: the number of blocks in the height of the image. The number of blocks in the width is calculated based on the aspect ratio of the image, defaults to 16 (optional)
    :param log_memory: whether to log memory usage, defaults to False (optional)
    :param gpu: whether to use the GPU or not, defaults to False (optional)
    """
    log_memory_string = ''
    gpu_string = ''
    if log_memory:
        log_memory_string = '-memory'
    if gpu:
        gpu_string = '-gpu'
    with open(os.path.join(ROOT, 'csvs/performance_data/performance-{}-{}-{}-{}-{}-{}{}{}.csv'.format(dataset, mode, colormode, gridsize, sigma, num_blocks_height, log_memory_string, gpu_string)), 'w') as outfile:
        if log_memory:
            outfile.write(PEAK_MEMORY)
            outfile.write('\n')
            outfile.write('{}%'.format(PEAK_CPU))
            outfile.write('\n')
            outfile.write('{} cpus\n'.format(psutil.cpu_count()))
            if gpu:
                outfile.write('{}MB'.format(GPU_MEMORY))
        else:
            for key, val in TIMES.items():
                outfile.write('{},{}\n'.format(key, np.median(val)))


def run_batch(imgs, grids, targets, gpu):
    """
    It takes a batch of images, grids, and targets, and returns a batch of protected images
    
    :param imgs: a list of images, each image is a numpy array of shape (height, width, 3)
    :param grids: a list of grids, each grid is a numpy array of shape (height, width, 1)
    :param targets: a list of blurred or pixelated images, each image is a numpy array of shape (height, width, 3)
    :param gpu: whether to use the GPU or not
    :return: the protected image list being returned
    """
    newimgs = []
    for i in tqdm(range(len(imgs))):
        BM.mark('algo')
        if gpu:
            gpu_imgs = cp.array(imgs[i]).astype(cp.int32)
            gpu_grids = cp.array(grids[i])
            gpu_targets = cp.array(targets[i]).astype(cp.int32)
            newimgs.append(cp.asnumpy(cupy_eyeshield_rmse(gpu_imgs, gpu_grids, gpu_targets)))
        else:
            newimgs.append(eyeshield_rmse(imgs[i], grids[i], targets[i]))
        TIMES['algo'].append(BM.mark('algo'))
    return newimgs


def run_and_save(imgs, grids, targets, path, files, gpu, performance):
    """
    It takes a batch of images, runs them through the model, and saves the output
    
    :param imgs: a list of images, each image is a numpy array of shape (height, width, 3)
    :param grids: a list of grids, each grid is a numpy array of shape (height, width, 1)
    :param targets: a list of blurred or pixelated images, each image is a numpy array of shape (height, width, 3)
    :param path: the path to the folder for the output images to be saved in
    :param files: list of output filenames
    :param gpu: whether to use the GPU or not
    :param performance: if True, then the performance metrics are logged
    """
    newimgs = run_batch(imgs, grids, targets, gpu)
    if not performance:
        for filename, newimg in zip(files, newimgs):
            save_image(os.path.join(path, filename), newimg)


def create_batches(path, filenames, mode, colormode, gridsize=1, sigma=1, num_blocks_height=16, gpu=False, performance=False):
    """
    It takes a list of images, and returns a list of outpaths, images, grids, and targets that have been blurred, pixelated, or pixelated and blurred
    
    :param path: the path to the folder containing the images
    :param filenames: list of filenames
    :param mode: the method for computing the target ('blur', 'pixelate', or 'pixelateblur')
    :param colormode: the method for computing the complement colors ('avg', 'sqrt', or 'ciecam')
    :param gridsize: the size of the grid to be used for pixelation, defaults to 1 (optional)
    :param sigma: the standard deviation of the gaussian blur, defaults to 1 (optional)
    :param num_blocks_height: The number of blocks to split the image into vertically, defaults to 16 (optional)
    :param gpu: whether to use the GPU or not, defaults to False (optional)
    :param performance: if True, the code will not save the images to disk. This is useful for benchmarking, defaults to False (optional)
    :return: the outpaths, images, grids, and targets
    """
    if mode == 'blur':
        targetpath = os.path.join(ROOT, 'blurred', 'blur-{}-{}-{}'.format(colormode, gridsize, sigma))
        outpath = os.path.join(ROOT, 'hidden', 'blur-{}-{}-{}'.format(colormode, gridsize, sigma))
    elif mode == 'pixelate':
        targetpath = os.path.join(ROOT, 'pixelated', 'pixelate-{}-{}-{}'.format(colormode, gridsize, num_blocks_height))
        outpath = os.path.join(ROOT, 'hidden', 'pixelate-{}-{}-{}'.format(colormode, gridsize, num_blocks_height))
    elif mode == 'pixelateblur':
        targetpath = os.path.join(ROOT, 'pixelatedblurred', 'pixelateblurred-{}-{}-{}-{}'.format(colormode, gridsize, num_blocks_height, sigma))
        outpath = os.path.join(ROOT, 'hidden', 'pixelateblur-{}-{}-{}-{}'.format(colormode, gridsize, num_blocks_height, sigma))

    if not os.path.exists(outpath):
        os.mkdir(outpath)
    outpath = os.path.join(outpath, path.replace('data/', '').replace('data\\', ''))
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    if not os.path.exists(targetpath):
        os.mkdir(targetpath)
    targetpath = os.path.join(targetpath, path.replace('data/', '').replace('data\\', ''))
    if not os.path.exists(targetpath):
        os.mkdir(targetpath)

    np_imgs = []
    new_files = []
    grids = []
    targets = []

    for file in tqdm(filenames):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            BM.mark('open')
            img = open_image(os.path.join(path, file), gpu)
            TIMES['open'].append(BM.mark('open'))

            if gpu:
                grid_img = img.download()
            else:
                grid_img = img
            BM.mark('grid')
            grid = create_grid_mask(grid_img, gridsize, gpu)
            TIMES['grid'].append(BM.mark('grid'))
            BM.mark('pixelateblur')
        
            if mode == 'blur':
                target = blur(img, sigma, gpu)
            elif mode == 'pixelate':
                target = pixelate(img, num_blocks_height, gpu)
            elif mode == 'pixelateblur':
                target = pixelate(img, num_blocks_height, gpu)
                target = blur(target, sigma, gpu)
            TIMES['pixelateblur'].append(BM.mark('pixelateblur'))
        
            if not performance:
                BM.mark('save')
                save_image(os.path.join(targetpath, file), target.astype(np.uint8))
                TIMES['save'].append(BM.mark('save'))

            grids.append(grid)
            targets.append(target)
            np_imgs.append(grid_img)
            new_files.append(file)

    return outpath, targetpath, new_files, np_imgs, grids, targets


def get_files_from_dataset(dataset):
    """
    It takes a dataset name as input and returns the path to the dataset and a list of all the files in the dataset
    
    :param dataset: The name of the dataset
    :return: The path to the data and the list of files in the data directory
    """
    path = os.path.join('data', dataset)
    return path, os.listdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='div2kvalid', help='The folder of images to protect')
    parser.add_argument('--mode', type=str, default='blur', choices=['blur', 'pixelate', 'pixelateblur'], help='The method to use for computing the target image')
    parser.add_argument('--color', type=str, default='sqrt', choices=['sqrt'], help='The method for computing the complement colors. Only sqrt is currently supported due to being optimal in terms of runtime and visual appeal.')
    parser.add_argument('--grid', type=int, default=1, choices=[1, 2, 3, 4, 6, 8], help='The size in pixels of each grid block')
    parser.add_argument('--sigma', type=int, default=1, choices=[1, 2, 4, 8, 12, 16, 20, 24, 32], help='Sigma for Gaussian blur or block size for pixelation')
    parser.add_argument('--blocks', type=int, default=16, choices=[8, 16, 24, 32, 64, 128], help='Number of pixleation blocks (height)')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Whether to print out runtimes of each portion of code')
    parser.add_argument('--performance', action=argparse.BooleanOptionalAction, help='Whether to log performance metrics like runtimes or memory')
    parser.add_argument('--memory', action=argparse.BooleanOptionalAction, help='Whether to log memory and CPU utilization')
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, help='Whether to run with GPU or CPU')
    args = parser.parse_args()
    if args.verbose:
        BM.set_verbose()
        print(args.dataset, args.mode, args.grid, args.sigma, args.blocks)
    if args.memory:
        import psutil
        import tracemalloc
        process = psutil.Process(os.getpid())
        num_cpus = psutil.cpu_count(logical=False)
        num_cpus = 4
        tracemalloc.start()
        PEAK_CPU = max(process.cpu_percent() / psutil.cpu_count(), PEAK_CPU)
    path, files = get_files_from_dataset(args.dataset)
    outpath, targetpath, new_files, np_imgs, grids, targets = create_batches(path,
                                                                             files,
                                                                             mode=args.mode,
                                                                             colormode=args.color,
                                                                             gridsize=args.grid,
                                                                             sigma=args.sigma,
                                                                             num_blocks_height=args.blocks,
                                                                             gpu=args.gpu,
                                                                             performance=args.performance)
    if args.gpu:
        import cupy as cp
        from algorithm_cuda import cupy_eyeshield_rmse
        mempool = cp.get_default_memory_pool()
        GPU_MEMORY = max(mempool.total_bytes() / 10**6, GPU_MEMORY)
    if args.memory:
        PEAK_CPU = max(process.cpu_percent() / psutil.cpu_count(), PEAK_CPU)
    run_and_save(np_imgs, grids, targets, outpath, new_files, args.gpu, args.performance)
    if args.memory:
        PEAK_CPU = max(process.cpu_percent() / psutil.cpu_count(), PEAK_CPU)
    if args.gpu:
        GPU_MEMORY = max(mempool.total_bytes() / 10**6, GPU_MEMORY)
    if args.memory:
        current, peak = tracemalloc.get_traced_memory()
        PEAK_MEMORY = '{}MB'.format(peak / 10**6)
        tracemalloc.stop()
    write_performance(args.dataset, args.mode, args.color, args.grid, args.sigma, args.blocks, args.memory, args.gpu)
