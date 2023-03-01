# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from src.utils import BM, blur, pixelate, create_grid_mask, save_image, open_video, save_video
from src.algorithm import eyeshield_rmse


TIMES = {'cie': [], 'algo': [], 'blur': [], 'open': [], 'grid': [], 'save': [], 'pixelate': []}


def write_performance(dataset, mode, grid_halfsize=1, sigma=1, num_blocks_height=16):
    """
    It writes the performance data to a csv file
    
    :param dataset: Tte name of the dataset
    :param mode: the method for computing the target ('blur', 'pixelate', or 'pixelateblur')
    :param grid_halfsize: the size of each grid checker
    :param sigma: the standard deviation of the Gaussian kernel used to blur the image, defaults to 1 (optional)
    :param num_blocks_height: the number of blocks in the height of the image. The number of blocks in the width is calculated based on the aspect ratio of the image, defaults to 16 (optional)
    """
    with open('data/csvs/performance-{}-{}-{}-{}-{}.csv'.format(dataset, mode, grid_halfsize, sigma, num_blocks_height), 'w') as outfile:
        for key, val in TIMES.items():
            outfile.write('{},{}\n'.format(key, np.mean(val)))


def run_batch(imgs, grids, targets):
    """
    It takes a batch of images, grids, and targets, and returns a batch of protected images
    
    :param imgs: a list of images, each image is a numpy array of shape (height, width, 3)
    :param grids: a list of grids, each grid is a numpy array of shape (height, width, 1)
    :param targets: a list of blurred or pixelated images, each image is a numpy array of shape (height, width, 3)
    :return: the protected image list being returned
    """
    newimgs = []
    for img, grid, targ in tqdm(zip(imgs, grids, targets)):
        newimgs.append(eyeshield_rmse(img, grid, targ))
    return newimgs


def run_and_save(imgs, grids, targets, path, files, fps):
    """
    It takes a batch of images, runs them through the model, and saves the output
    
    :param imgs: a list of images, each image is a numpy array of shape (height, width, 3)
    :param grids: a list of grids, each grid is a numpy array of shape (height, width, 1)
    :param targets: a list of blurred or pixelated images, each image is a numpy array of shape (height, width, 3)
    :param path: the path to the folder for the output images to be saved in
    :param files: list of output filenames
    :param fps: the fps rate of the video
    """
    newimgs = run_batch(imgs, grids, targets)
    for i, newimg in enumerate(newimgs):
        save_image(os.path.join(path, files.replace('.mov', ''), 'img{}.png'.format(str(i).zfill(4))), cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR))
    save_video(os.path.join(path, files.replace('.mov', ''), '*.png'), os.path.join(path, files.replace('.mov', '.mp4')), fps)


def create_batches(path, filenames, mode, grid_halfsize=1, sigma=1, num_blocks_height=16, verbose=False):
    """
    It takes a list of images, and returns a list of outpaths, images, grids, and targets that have been blurred, pixelated, or pixelated and blurred
    
    :param path: the path to the folder containing the images
    :param filenames: list of filenames
    :param mode: the method for computing the target ('blur', 'pixelate', or 'pixelateblur')
    :param colormode: the method for computing the complement colors ('avg', 'sqrt', or 'ciecam')
    :param gridsize: the size of the grid to be used for pixelation, defaults to 1 (optional)
    :param sigma: the standard deviation of the gaussian blur, defaults to 1 (optional)
    :param num_blocks_height: The number of blocks to split the image into vertically, defaults to 16 (optional)
    :param verbose: whether to print out fps
    :return: the outpaths, images, grids, and targets
    """
    if mode == 'blur':
        targetpath = os.path.join('data', 'blurred', 'blur-{}-{}'.format(grid_halfsize, sigma))
        outpath = os.path.join('data', 'hidden', 'blur-{}-{}'.format(grid_halfsize, sigma))
    elif mode == 'pixelate':
        targetpath = os.path.join('data', 'pixelated', 'pixelate-{}-{}'.format(grid_halfsize, num_blocks_height))
        outpath = os.path.join('data', 'hidden', 'pixelate-{}-{}'.format(grid_halfsize, num_blocks_height))

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

    for file in tqdm(filenames):
        if file.endswith(('.avi', '.mp4', '.mov')):
            imgs, fps = open_video(os.path.join(path, file))
            if verbose:
                print(fps)
            temp_grids = []
            for img in tqdm(imgs):
                temp_grids.append(create_grid_mask(img, grid_halfsize).astype(np.uint8))
            temp_targets = []
            if not os.path.exists(os.path.join(targetpath, file.replace('.mov', ''))):
                os.mkdir(os.path.join(targetpath, file.replace('.mov', '')))
            if not os.path.exists(os.path.join(outpath, file.replace('.mov', ''))):
                os.mkdir(os.path.join(outpath, file.replace('.mov', '')))
            if mode == 'blur':
                for i, img in tqdm(enumerate(imgs)):
                    target = blur(img, sigma)
                    temp_targets.append(target)
                    save_image(os.path.join(targetpath, file.replace('.mov', ''), 'img{}.png'.format(str(i).zfill(4))), cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
                save_video(os.path.join(targetpath, file.replace('.mov', ''), '*.png'), os.path.join(targetpath, file.replace('.mov', '.mp4')), fps)

            elif mode == 'pixelate':
                for i, img in tqdm(enumerate(imgs)):
                    target = pixelate(img, num_blocks_height)
                    temp_targets.append(target)
                    save_image(os.path.join(targetpath, file.replace('.mov', ''), 'img{}.png'.format(str(i).zfill(4))), cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
                save_video(os.path.join(targetpath, file.replace('.mov', ''), '*.png'), os.path.join(targetpath, file.replace('.mov', '.mp4')), fps)

            run_and_save(imgs, temp_grids, temp_targets, outpath, file, fps)


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
    parser.add_argument('--grid', type=int, default=1, choices=[1, 2, 3, 4, 6, 8], help='The size in pixels of each grid block')
    parser.add_argument('--sigma', type=int, default=1, choices=[1, 2, 4, 8, 12, 16, 20, 24, 32], help='Sigma for Gaussian blur or block size for pixelation')
    parser.add_argument('--blocks', type=int, default=16, choices=[8, 16, 24, 32, 64, 128], help='Number of pixleation blocks (height)')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Whether to print out runtimes of each portion of code')
    args = parser.parse_args()
    if args.verbose:
        BM.set_verbose()
        print(args.dataset, args.mode, args.grid, args.sigma, args.blocks)
    path, files = get_files_from_dataset(args.dataset)
    create_batches(path,
                   files,
                   mode=args.mode,
                   grid_halfsize=args.grid,
                   sigma=args.sigma,
                   num_blocks_height=args.blocks,
                   verbose=args.verbose)
