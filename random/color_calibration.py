import os
import math
import pandas as pd
import numpy as np
from PIL import Image
# from ciecam02 import rgb2xyz, xyz2rgb, setconfig, jch2rgb, rgb2jch
from skimage.color import rgb2gray, rgb2yuv, yuv2rgb, rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
from tqdm import tqdm
from itertools import groupby
import matplotlib.pyplot as plt
import seaborn as sns

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def create_grid_mask(img, grid_halfsize):
    """
    Pillow image required
    """
    width = img.shape[1]
    height = img.shape[0]
    grid = np.zeros(img.shape[:3])
    outer_count = 0
    for i in range(0, img.shape[0], grid_halfsize):
        if outer_count % 2 == 1:
            count = 1
        else:
            count = 0
        for j in range(0, img.shape[1], grid_halfsize):
            if (count) % 2 == 1:
                grid[i:i+grid_halfsize,j:j+grid_halfsize] = [1, 1, 1]
            count += 1
        outer_count += 1
    return grid


# colors = {'green': (24, 162, 121),
#           'yellow': (233, 232, 4),
#           'orange': (228, 186, 28),
#           'red': (190, 28, 76),
#           'purple': (172, 36, 132),
#           'pink': (244, 204, 220),
#           'blue': (92, 140, 204),
#           'black': (0, 0, 0),
#           'white': (255, 255, 255)}


# # for n1, c1 in colors.items():
# #     for n2, c2 in colors.items():
# #         for i in range(-10, 11):
# #             if n1 != n2:
# #                 c1 = np.array(c1)
# #                 c2 = np.array(c2)
# #                 avg = np.clip(((c1 + c2) / 2), 0, 255).astype(np.uint8)
# #                 c1lab = np.zeros((1, 1, 3)) + c1
# #                 c2lab = np.zeros((1, 1, 3)) + c2
# #                 c1lab = rgb2yuv(c1lab / 255)
# #                 c2lab = rgb2yuv(c2lab / 255)
# #                 avg = np.clip(yuv2rgb((c1lab + c2lab) / 2).squeeze() * 255, 0, 255).astype(np.uint8)
# #                 img = np.clip(np.zeros((256, 256, 3)) + avg + i, 0, 255).astype(np.uint8)
# #                 small_img = (np.zeros((64, 64, 3)) * 255).astype(np.uint8)
# #                 img = Image.fromarray(img)
# #                 small_img = Image.fromarray(small_img)
# #                 grid_halfsize = 1
# #                 grid = (create_grid_mask(small_img, grid_halfsize) * 255).astype(np.uint8)
# #                 newgrid = np.where(grid == 0, grid, c1.astype(np.uint8))
# #                 grid = np.where(grid == 255, newgrid, c2.astype(np.uint8))
# #                 grid = Image.fromarray(grid)
# #                 img.paste(grid, (0,0))
# #                 if not os.path.exists('drawable/calibration/color_calibration_{}-{}'.format(n1, n2)):
# #                     os.mkdir('drawable/calibration/color_calibration_{}-{}'.format(n1, n2))
# #                 img.save('drawable/calibration/color_calibration_{}-{}/{}.png'.format(n1, n2, i))


# for n1, c1 in colors.items():
#     for n2, c2 in colors.items():
#         if n1 != n2:
#             c1 = np.array(c1)
#             c2 = np.array(c2)
#             avg = np.clip(((c1 + c2) / 2), 0, 255).astype(np.uint8)
#             c1lab = np.zeros((1, 1, 3)) + c1
#             c2lab = np.zeros((1, 1, 3)) + c2
#             c1lab = rgb2yuv(c1lab / 255)
#             c2lab = rgb2yuv(c2lab / 255)
#             avg = np.clip(yuv2rgb((c1lab + c2lab) / 2).squeeze() * 255, 0, 255).astype(np.uint8)
#             img = np.clip(np.zeros((256, 256, 3)) + avg, 0, 255).astype(np.uint8)
#             small_img = (np.zeros((64, 64, 3)) * 255).astype(np.uint8)
#             img = Image.fromarray(img)
#             small_img = Image.fromarray(small_img)
#             grid_halfsize = 2
#             grid = (create_grid_mask(small_img, grid_halfsize) * 255).astype(np.uint8)
#             newgrid = np.where(grid == 0, grid, c1.astype(np.uint8))
#             grid = np.where(grid == 255, newgrid, c2.astype(np.uint8))
#             grid = Image.fromarray(grid)
#             grid.save('drawable/colors/color_calibration_{}-{}.png'.format(n1, n2))
#             with open('drawable/colors/color_calibration_{}-{}.txt'.format(n1, n2), 'w') as outfile:
#                 outfile.write('{},{},{}'.format(avg[0], avg[1], avg[2]))


# setconfig('c', 'average', 'high', 'high')

colors = {'darkskin': (115, 82, 68),
          'lightskin': (194, 150, 130),
          'bluesky': (98, 122, 157),
          'foliage': (87, 108, 67),
          'blueflower': (133, 128, 177),
          'bluishgreen': (103, 189, 170),
          'orange': (214, 126, 44),
          'purplishblue': (80, 91, 166),
          'moderatered': (193, 90, 99),
          'purple': (94, 60, 108),
          'yellowgreen': (157, 188, 64),
          'orangeyellow': (224, 163, 46),
          'blue': (56, 61, 150),
          'green': (70, 148, 73),
          'red': (175, 54, 60),
          'yellow': (231, 199, 31),
          'magenta': (187, 86, 149),
          'cyan': (8, 133, 161),
          'white': (243, 243, 242),
          'neutral8': (200, 200, 200),
          'neutral65': (160, 160, 160),
          'neutral5': (122, 122, 121),
          'neutral35': (85, 85, 85),
          'black': (52, 52, 52)}

random_colors = {'ylw': (213, 179, 62),
                 'grn': (93, 172, 37),
                 'brn': (201, 164, 102),
                 'blu': (16, 122, 245),
                 'org': (254, 99, 33),
                 'pur': (195, 193, 254),
                 'gry': (139, 140, 122),
                 'blk': (40, 40, 40),
                 'wht': (231, 235, 229)}

# for n1, avg in colors.items():
#     for n2, c1 in random_colors.items():
#         c1 = np.array(c1)
#         avg = np.array(avg)
#         avg_tup = ((avg**2) * 2) - (c1**2)
#         dontskip = True
#         for indx in avg_tup:
#             if indx < 0 or indx > 65025:
#                 dontskip = False
#         if dontskip:
#             c2 = np.sqrt(((avg**2) * 2) - (c1**2))
#             # print(c2)
#             jch_c1 = rgb2xyz(np.expand_dims(c1, 0))
#             jch_avg = rgb2xyz(np.expand_dims(avg, 0))
#             # try:
#             jch_c2 = xyz2rgb((jch_avg * 2) - jch_c1)
#             # print(jch_c2)
#             c2 = np.clip(np.sqrt(((avg**2) * 2) - (c1**2)), 0, 255).astype(np.uint8)
#             c2 = jch_c2[0]
#             # c1lab = np.zeros((1, 1, 3)) + c1
#             # c2lab = np.zeros((1, 1, 3)) + c2
#             # c1lab = rgb2yuv(c1lab / 255)
#             # c2lab = rgb2yuv(c2lab / 255)
#             # avg = np.clip(yuv2rgb((c1lab + c2lab) / 2).squeeze() * 255, 0, 255).astype(np.uint8)
#             img = np.clip(np.zeros((64, 64, 3)) + avg, 0, 255).astype(np.uint8)
#             small_img = (np.zeros((64, 64, 3)) * 255).astype(np.uint8)
#             grid_halfsize = 2
#             grid = (create_grid_mask(small_img, grid_halfsize) * 255).astype(np.uint8)
#             newgrid = np.where(grid == 0, grid, c1.astype(np.uint8))
#             grid = np.where(grid == 255, newgrid, c2.astype(np.uint8))
#             # grid = np.concatenate((img, grid), axis=0)
#             grid = Image.fromarray(grid)
#             grid.save('drawable/new_colors3/color_calibration_{}-{}.png'.format(n1, n2))
#             with open('drawable/new_colors3/color_calibration_{}-{}.txt'.format(n1, n2), 'w') as outfile:
#                 outfile.write('{},{},{}'.format(avg[0], avg[1], avg[2]))
#             with open('drawable/new_colors3/color_calibration_orig_{}-{}.txt'.format(n1, n2), 'w') as outfile:
#                 outfile.write('{},{},{}|{},{},{}'.format(c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]))
#             # except Exception as e:
#             #     pass

# df_data = {'avg': [], 'c1': [], 'c2': []}
df_data = {'R': [], 'G': [], 'B': []}
lengths = {'r': [], 'g': [], 'b': []}

for r in tqdm(range(0, 256, 4)):
    df_data = {'R': [], 'G': [], 'B': []}
    for g in tqdm(range(0, 256, 4)):
        for b in range(0, 256, 4):
            lengths['r'] = []
            color1 = np.array([r, g, b])
            for temp_r in range(r, 256):
                lengths['g'] = []
                if len(lengths['r']) >= 3 and all_equal(lengths['r'][-3:]):
                    break
                for temp_g in range(g, 256):
                    lengths['b'] = []
                    if len(lengths['g']) >= 3 and all_equal(lengths['g'][-3:]):
                        break
                    for temp_b in range(b, 256):
                        if len(lengths['b']) >= 3 and all_equal(lengths['b'][-3:]):
                            break
                        color2 = np.array([temp_r, temp_g, temp_b])
                        color3 = ((color1 ** 2) * 2) - (color2 ** 2)
                        dontskip = True
                        for channel in color3:
                            if channel < 0 or channel > 65025:
                                dontskip = False
                        if dontskip:
                            df_data['R'].append(color1[0])
                            df_data['G'].append(color1[1])
                            df_data['B'].append(color1[2])
                        lengths['b'].append(len(df_data['R']))
                    lengths['g'].append(len(df_data['R']))
                lengths['r'].append(len(df_data['R']))
            for temp_r in reversed(range(0, r)):
                lengths['g'] = []
                if len(lengths['r']) >= 3 and all_equal(lengths['r'][-3:]):
                    break
                for temp_g in reversed(range(0, g)):
                    lengths['b'] = []
                    if len(lengths['g']) >= 3 and all_equal(lengths['g'][-3:]):
                        break
                    for temp_b in reversed(range(0, b)):
                        if len(lengths['b']) >= 3 and all_equal(lengths['b'][-3:]):
                            break
                        color2 = np.array([temp_r, temp_g, temp_b])
                        color3 = ((color1 ** 2) * 2) - (color2 ** 2)
                        dontskip = True
                        for channel in color3:
                            if channel < 0 or channel > 65025:
                                dontskip = False
                        if dontskip:
                            df_data['R'].append(color1[0])
                            df_data['G'].append(color1[1])
                            df_data['B'].append(color1[2])
                        lengths['b'].append(len(df_data['R']))
                    lengths['g'].append(len(df_data['R']))
                lengths['r'].append(len(df_data['R']))
    df = pd.DataFrame(df_data)
    sns.set(style='ticks')
    plt.figure(figsize=(16, 12))
    g = sns.displot(data=df, x='G', y='B', kind='hist', binwidth=(4, 4), legend='auto', cbar=True).set(title=r)
    # lgnd = plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join('data/plots', 'colors-{}.pdf'.format(r)))


# keys = []
# avgs = []
# valid_colors = []
# complement_colors = []
# lengths = {'r': [], 'g': [], 'b': []}

# for r in tqdm(range(0, 256)):
#     for g in tqdm(range(0, 256)):
#         counter = []
#         for b in range(0, 256):
#             lengths['r'] = []
#             color1 = np.array([r, g, b])
#             for temp_r in range(r, 256):
#                 lengths['g'] = []
#                 if len(lengths['r']) >= 3 and all_equal(lengths['r'][-3:]):
#                     break
#                 for temp_g in range(g, 256):
#                     lengths['b'] = []
#                     if len(lengths['g']) >= 3 and all_equal(lengths['g'][-3:]):
#                         break
#                     for temp_b in range(b, 256):
#                         if len(lengths['b']) >= 3 and all_equal(lengths['b'][-3:]):
#                             break
#                         color2 = np.array([temp_r, temp_g, temp_b])
#                         color3 = np.array(((color1 ** 2) * 2) - (color2 ** 2))
#                         if np.all((color3 >= 0)&(color3 < 65025)):
#                             if set(color1) not in keys:
#                                 keys.append(set(color1))
#                                 avgs.append(color1)
#                             idx = keys.index(set(color1))
#                             if len(valid_colors) >= idx:
#                                 valid_colors.append([color2])
#                                 complement_colors.append([color3])
#                             else:
#                                 valid_colors[idx].append(color2)
#                                 complement_colors[idx].append(color3)
#                             counter.append(0)
#                         lengths['b'].append(len(counter))
#                     lengths['g'].append(len(counter))
#                 lengths['r'].append(len(counter))
#             for temp_r in reversed(range(0, r)):
#                 lengths['g'] = []
#                 if len(lengths['r']) >= 3 and all_equal(lengths['r'][-3:]):
#                     break
#                 for temp_g in reversed(range(0, g)):
#                     lengths['b'] = []
#                     if len(lengths['g']) >= 3 and all_equal(lengths['g'][-3:]):
#                         break
#                     for temp_b in reversed(range(0, b)):
#                         if len(lengths['b']) >= 3 and all_equal(lengths['b'][-3:]):
#                             break
#                         color2 = np.array([temp_r, temp_g, temp_b])
#                         color3 = np.array(((color1 ** 2) * 2) - (color2 ** 2))
#                         if np.all((color3 >= 0)&(color3 < 65025)):
#                             if set(color1) not in keys:
#                                 keys.append(set(color1))
#                                 avgs.append(color1)
#                             idx = keys.index(set(color1))
#                             if len(valid_colors) >= idx:
#                                 valid_colors.append([color2])
#                                 complement_colors.append([color3])
#                             else:
#                                 valid_colors[idx].append(color2)
#                                 complement_colors[idx].append(color3)
#                             counter.append(0)
#                         lengths['b'].append(len(counter))
#                     lengths['g'].append(len(counter))
#                 lengths['r'].append(len(counter))
#         np.savez('data/colors/color_calibration-{}-{}'.format(r, g), avg=np.array(avgs), c1=np.array(valid_colors), c2=np.array(complement_colors))


# n2 = str(c1)
# c1 = np.array(c1)
# avg = np.array(avg)
# avg_tup = ((avg**2) * 2) - (c1**2)
# dontskip = True
# for indx in avg_tup:
#     if indx < 0 or indx > 65025:
        
