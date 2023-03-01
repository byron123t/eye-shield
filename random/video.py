# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import os
import cv2
from src.utils import open_video, save_image, convert_to_gpu
from eyeshield import protect_image


imgs, fps = open_video(os.path.join('data', 'recording_demo', 'screen_recording.mp4'), verbose=True, gpu=True)
counter = 0
for i in range(0, len(imgs)):
    img = convert_to_gpu(imgs[i])
    newimg = protect_image(img, strength='strong', gpu=True)
    save_image(os.path.join('data', 'screen_recording', 'img{}.png'.format(str(i).zfill(4))), cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR))
