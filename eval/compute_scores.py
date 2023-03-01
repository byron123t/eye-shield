# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import os
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from numba import njit
from torchvision import transforms
from src.utils import *
from algorithm import average_pixelated_average, average_pixelated, average_average


def store_images(img, target_img, hidden_img, dim, resample):
    img_small = cv2.resize(img, dim, interpolation=resample)
    target_img_small = cv2.resize(target_img, dim, interpolation=resample)
    hidden_img_small = cv2.resize(hidden_img, dim, interpolation=resample)
    keymap = {'O': img, 'H': hidden_img, 'T': target_img,
              'DO': img_small, 'DH': hidden_img_small, 'DT': target_img_small}
    return keymap


def iterate_combinations(keymap, df_data):
    for key, value in df_data.items():
        if key.startswith('Entropy '):
            combo = key.replace('Entropy ', '')
            func = get_entropy
            name = 'entropy'
        elif key.startswith('Shannon Entropy '):
            combo = key.replace('Shannon Entropy ', '')
            func = get_shannon_entropy
            name = 'shannon_entropy'
        elif key.startswith('SSIM '):
            combo = key.replace('SSIM ', '')
            func = get_ssim
            name = 'ssim'
        elif key.startswith('MSE '):
            combo = key.replace('MSE ', '')
            func = get_mse
            name = 'mse'
        elif key.startswith('L2 '):
            combo = key.replace('L2 ', '')
            func = get_l2
            name = 'l2'
        elif key.startswith('Filename'):
            continue
        if '-' in combo:
            split = combo.split('-')
            BM.mark(name)
            df_data[key].append(func(keymap[split[0]], keymap[split[1]]))
            BM.mark(name)
        else:
            BM.mark(name)
            df_data[key].append(func(keymap[combo]))
            BM.mark(name)
    return df_data


def create_batches(path, filenames, dataset, mode, colormode, scale='area', size=0.2, grid_halfsize=1, sigma=1, contrast=127, num_blocks_height=16):
    if mode == 'blur':
        targetpath = os.path.join('data', 'blurred', 'blur-{}-{}-{}'.format(grid_halfsize, sigma, contrast))
        outpath = os.path.join('data', 'hidden', 'blur-{}-{}-{}'.format(grid_halfsize, sigma, contrast))
        csvfile = 'blur-{}-{}-{}-{}-{}-{}.csv'.format(grid_halfsize, sigma, dataset, scale, size, contrast)
    elif mode == 'pixelate':
        targetpath = os.path.join('data', 'pixelated', 'pixelate-{}-{}-{}'.format(grid_halfsize, num_blocks_height, contrast))
        outpath = os.path.join('data', 'hidden', 'pixelate-{}-{}-{}'.format(grid_halfsize, num_blocks_height, contrast))
        csvfile = 'pixelate-{}-{}-{}-{}-{}-{}.csv'.format(grid_halfsize, num_blocks_height, dataset, scale, size, contrast)
    elif mode == 'pixelateblur':
        targetpath = os.path.join('data', 'pixelatedblurred', 'pixelateblurred-{}-{}-{}'.format(grid_halfsize, num_blocks_height, sigma))
        outpath = os.path.join('data', 'hidden', 'pixelateblur-{}-{}-{}'.format(grid_halfsize, num_blocks_height, sigma))
        csvfile = 'pixelateblur-{}-{}-{}-{}-{}-{}.csv'.format(grid_halfsize, num_blocks_height, sigma, dataset, scale, size)

    targetpath = os.path.join(targetpath, path.replace('data/', ''))
    outpath = os.path.join(outpath, path.replace('data/', ''))

    if scale == 'bilinear':
        resample = cv2.INTER_LINEAR
    elif scale == 'bicubic':
        resample = cv2.INTER_CUBIC
    elif scale == 'lanczos':
        resample = cv2.INTER_LANCZOS4
    elif scale == 'area':
        resample = cv2.INTER_AREA

    # There's too many combinations, send help...
    'O: original, H: hidden, T: target, D-: downscale, F-: fft, I-: high pass, L-: low pass'
    # df_data = {'Entropy O': [], 'Shannon Entropy O': [], 'SSIM O-H': [], 'MSE O-H': [], 'L2 O-H': [], 'LPIPS O-H': [],
    #            'Entropy H': [], 'Shannon Entropy H': [], 'SSIM O-T': [], 'MSE O-T': [], 'L2 O-T': [], 'LPIPS O-T': [],
    #            'Entropy T': [], 'Shannon Entropy T': [], 'SSIM H-T': [], 'MSE H-T': [], 'L2 H-T': [], 'LPIPS H-T': [],
    #            'Entropy DO': [], 'Shannon Entropy DO': [], 'SSIM DO-DH': [], 'MSE DO-DH': [], 'L2 DO-DH': [], 'LPIPS DO-DH': [],
    #            'Entropy DH': [], 'Shannon Entropy DH': [], 'SSIM DO-DT': [], 'MSE DO-DT': [], 'L2 DO-DT': [], 'LPIPS DO-DT': [],
    #            'Entropy DT': [], 'Shannon Entropy DT': [], 'SSIM DH-DT': [], 'MSE DH-DT': [], 'L2 DH-DT': [], 'LPIPS DH-DT': [],
    #            'Entropy FO': [], 'Shannon Entropy FO': [], 'SSIM FO-FH': [], 'MSE FO-FH': [], 'L2 FO-FH': [], 'LPIPS FO-FH': [],
    #            'Entropy FH': [], 'Shannon Entropy FH': [], 'SSIM FO-FT': [], 'MSE FO-FT': [], 'L2 FO-FT': [], 'LPIPS FO-FT': [],
    #            'Entropy FT': [], 'Shannon Entropy FT': [], 'SSIM FH-FT': [], 'MSE FH-FT': [], 'L2 FH-FT': [], 'LPIPS FH-FT': []
    #            'Entropy IO': [], 'Shannon Entropy IO': [], 'SSIM IO-IH': [], 'MSE IO-IH': [], 'L2 IO-IH': [], 'LPIPS IO-IH': [],
    #            'Entropy IH': [], 'Shannon Entropy IH': [], 'SSIM IO-IT': [], 'MSE IO-IT': [], 'L2 IO-IT': [], 'LPIPS IO-IT': [],
    #            'Entropy IT': [], 'Shannon Entropy IT': [], 'SSIM IH-IT': [], 'MSE IH-IT': [], 'L2 IH-IT': [], 'LPIPS IH-IT': [],
    #            'Entropy LO': [], 'Shannon Entropy LO': [], 'SSIM LO-LH': [], 'MSE LO-LH': [], 'L2 LO-LH': [], 'LPIPS LO-LH': [],
    #            'Entropy LH': [], 'Shannon Entropy LH': [], 'SSIM LO-LT': [], 'MSE LO-LT': [], 'L2 LO-LT': [], 'LPIPS LO-LT': [],
    #            'Entropy LT': [], 'Shannon Entropy LT': [], 'SSIM LH-LT': [], 'MSE LH-LT': [], 'L2 LH-LT': [], 'LPIPS LH-LT': [],
    #            'Entropy DFO': [], 'Shannon Entropy DFO': [], 'SSIM DFO-DFH': [], 'MSE DFO-DFH': [], 'L2 DFO-DFH': [], 'LPIPS DFO-DFH': [],
    #            'Entropy DFH': [], 'Shannon Entropy DFH': [], 'SSIM DFO-DFT': [], 'MSE DFO-DFT': [], 'L2 DFO-DFT': [], 'LPIPS DFO-DFT': [],
    #            'Entropy DFT': [], 'Shannon Entropy DFT': [], 'SSIM DFH-DFT': [], 'MSE DFH-DFT': [], 'L2 DFH-DFT': [], 'LPIPS DFH-DFT': []
    #            'Entropy DIO': [], 'Shannon Entropy DIO': [], 'SSIM DIO-DIH': [], 'MSE DIO-DIH': [], 'L2 DIO-DIH': [], 'LPIPS DIO-DIH': [],
    #            'Entropy DIH': [], 'Shannon Entropy DIH': [], 'SSIM DIO-DIT': [], 'MSE DIO-DIT': [], 'L2 DIO-DIT': [], 'LPIPS DIO-DIT': [],
    #            'Entropy DIT': [], 'Shannon Entropy DIT': [], 'SSIM DIH-DIT': [], 'MSE DIH-DIT': [], 'L2 DIH-DIT': [], 'LPIPS DIH-DIT': [],
    #            'Entropy DLO': [], 'Shannon Entropy DLO': [], 'SSIM DLO-DLH': [], 'MSE DLO-DLH': [], 'L2 DLO-DLH': [], 'LPIPS DLO-DLH': [],
    #            'Entropy DLH': [], 'Shannon Entropy DLH': [], 'SSIM DLO-DLT': [], 'MSE DLO-DLT': [], 'L2 DLO-DLT': [], 'LPIPS DLO-DLT': [],
    #            'Entropy DLT': [], 'Shannon Entropy DLT': [], 'SSIM DLH-DLT': [], 'MSE DLH-DLT': [], 'L2 DLH-DLT': [], 'LPIPS DLH-DLT': []}
    df_data = {'Shannon Entropy O': [], 'SSIM O-H': [],
               'Shannon Entropy H': [],
               'Shannon Entropy T': [],
               'Shannon Entropy DO': [],
               'Shannon Entropy DH': [],
               'Shannon Entropy DT': [], 'SSIM DH-DT': [], 'Filename': []}
               
    keymaps = []
    torchkeys = []
    for file in tqdm(filenames):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img = open_image(os.path.join(path, file))
            target_img = open_image(os.path.join(targetpath, file))
            hidden_img = open_image(os.path.join(outpath, file))
            dim = (int(img.shape[0] * size), int(img.shape[1] * size))
            keymap = store_images(img, target_img, hidden_img, dim, resample)
            keymaps.append(keymap)
    for keymap, file in tqdm(zip(keymaps, filenames), total=len(filenames)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            df_data = iterate_combinations(keymap, df_data)
            df_data['Filename'].append(file)

    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join('data', 'csvs', csvfile))


def get_files_from_dataset(dataset):
    path = os.path.join('data', dataset)
    return path, os.listdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='div2kvalid', help='')
    parser.add_argument('--mode', type=str, default='blur', choices=['blur', 'pixelate', 'pixelateblur'], help='')
    parser.add_argument('--color', type=str, default='sqrt', choices=['avg', 'sqrt', 'ciecam'], help='')
    parser.add_argument('--scale', type=str, default='area', choices=['bilinear', 'bicubic', 'lanczos', 'area'], help='')
    parser.add_argument('--size', type=float, default=0.2, choices=[0.2, 0.25, 0.33, 0.5], help='')
    parser.add_argument('--grid', type=int, default=1, choices=[1, 2, 3, 4, 6, 8], help='')
    parser.add_argument('--sigma', type=int, default=1, choices=[1, 2, 4, 8, 12, 16, 20, 24, 32], help='Sigma for Gaussian blur or block size for pixelation.')
    parser.add_argument('--contrast', type=int, default=127, choices=[127, 100, 75], help='Decrease in contrast of image.')
    parser.add_argument('--blocks', type=int, default=16, choices=[8, 16, 24, 32, 64, 128], help='Number of pixleation blocks. (height)')
    args = parser.parse_args()
    path, files = get_files_from_dataset(args.dataset)
    create_batches(path=path,
                   filenames=files,
                   dataset=args.dataset,
                   mode=args.mode,
                   colormode=args.color,
                   scale=args.scale,
                   size=args.size,
                   grid_halfsize=args.grid,
                   sigma=args.sigma,
                   contrast=args.contrast,
                   num_blocks_height=args.blocks)
