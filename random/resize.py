import cv2
import os
import numpy as np
from src.utils import *
from tqdm import tqdm


# for folder in os.listdir('data/mturk-shoulder-surf/videos_original'):
#     if not folder.endswith('.mp4') and folder != '.DS_Store':
#         for i in os.listdir('data/mturk-shoulder-surf/videos_original/{}'.format(folder)):
#             if i.endswith('.jpg'):
#                 image = open_image(os.path.join('data/mturk-shoulder-surf/videos_original/{}'.format(folder), i))
#                 size = image.shape[:2]
#                 height = 120
#                 width = 214
#                 print(i, width, height)
#                 # newimg = downscale(image, (width, height), cv2.INTER_LINEAR)
#                 newimg = downscale(image, (width, height), cv2.INTER_AREA)
#                 print(newimg.shape)
#                 save_image(os.path.join('data/mturk-shoulder-surf/videos_original_small/{}'.format(folder), i), newimg)

all_folders1 = ['blur-1-8-100', 'blur-1-16-100', 'blur-1-24-100', 'blur-1-32-100', 
               'blur-2-8-100', 'blur-2-16-100', 'blur-2-24-100', 'blur-2-32-100', 
               'blur-3-8-100', 'blur-3-16-100', 'blur-3-24-100', 'blur-3-32-100', 
               'blur-4-8-100', 'blur-4-16-100', 'blur-4-24-100', 'blur-4-32-100',
               'blur-1-8-75', 'blur-1-16-75', 'blur-1-24-75', 'blur-1-32-75', 
               'blur-2-8-75', 'blur-2-16-75', 'blur-2-24-75', 'blur-2-32-75', 
               'blur-3-8-75', 'blur-3-16-75', 'blur-3-24-75', 'blur-3-32-75', 
               'blur-4-8-75', 'blur-4-16-75', 'blur-4-24-75', 'blur-4-32-75']
all_folders = ['pixelate-1-8-100', 'pixelate-1-16-100', 'pixelate-1-24-100', 'pixelate-1-32-100', 
               'pixelate-2-8-100', 'pixelate-2-16-100', 'pixelate-2-24-100', 'pixelate-2-32-100', 
               'pixelate-3-8-100', 'pixelate-3-16-100', 'pixelate-3-24-100', 'pixelate-3-32-100', 
               'pixelate-4-8-100', 'pixelate-4-16-100', 'pixelate-4-24-100', 'pixelate-4-32-100',
               'pixelate-1-8-75', 'pixelate-1-16-75', 'pixelate-1-24-75', 'pixelate-1-32-75', 
               'pixelate-2-8-75', 'pixelate-2-16-75', 'pixelate-2-24-75', 'pixelate-2-32-75', 
               'pixelate-3-8-75', 'pixelate-3-16-75', 'pixelate-3-24-75', 'pixelate-3-32-75', 
               'pixelate-4-8-75', 'pixelate-4-16-75', 'pixelate-4-24-75', 'pixelate-4-32-75']

folder_name = 'ricovalid'
mode_name = 'hidden'

for folder in tqdm(all_folders):
    for i in os.listdir(os.path.join('data', mode_name, folder, folder_name)):
        image = open_image(os.path.join('data', mode_name, folder, folder_name, i))
        size = image.shape[:2]
        height = int(size[0] / 4)
        width = int(size[1] / 4)
        newimg = downscale(image, (width, height), cv2.INTER_AREA)
        if not os.path.exists(os.path.join('data/{}_downscaled'.format(mode_name), folder)):
            os.mkdir(os.path.join('data/{}_downscaled'.format(mode_name), folder))
        if not os.path.exists(os.path.join('data/{}_downscaled'.format(mode_name), folder, folder_name)):
            os.mkdir(os.path.join('data/{}_downscaled'.format(mode_name), folder, folder_name))
        save_image(os.path.join('data/{}_downscaled'.format(mode_name), folder, folder_name, i), newimg)
# for i in os.listdir('data/mturk_shoulder_surf/text_small'):
#     if i.endswith('.jpg'):
#         image = open_image(os.path.join('data/mturk_shoulder_surf/text', i))
#         size = image.shape[:2]
#         if size[0] == 960:
#             height = int(size[0] / 2.5)
#             width = int(size[1] / 2.5)            
#         else:
#             height = int(size[0] / 5)
#             width = int(size[1] / 5)
#         newimg = downscale(image, (width, height), cv2.INTER_LINEAR)
#         save_image(os.path.join('data/mturk_shoulder_surf/text_small', i), newimg)

# sizes = [29, 40, 58, 60, 80, 87, 120, 180, 1024]
# image = cv2.imread(os.path.join('data/iosapp/shoulder.png'), cv2.IMREAD_UNCHANGED)
# print(image[150:160, 150:160])
# image[:,:,0] = np.where(image[:,:,3] == 0, 255, 0)
# image[:,:,1] = np.where(image[:,:,3] == 0, 255, 0)
# image[:,:,2] = np.where(image[:,:,3] == 0, 255, 0)
# print(image)
# for size in sizes:
#     if size == 1024:
#         newimg = downscale(image, (size, size), cv2.INTER_LINEAR)
#     else:
#         newimg = downscale(image, (size, size), cv2.INTER_AREA)
#     print(newimg)
#     save_image(os.path.join('data/iosapp/shoulder{}.png'.format(size)), newimg)
