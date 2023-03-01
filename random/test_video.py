import numpy as np
import cv2
import os
import subprocess
import ffmpeg
from subprocess import call, check_output
from PIL import Image


# vidcap = cv2.VideoCapture('output.mp4')
# hasFrames = True
# sec = 0
# frameRate = 0.016666667
# images = []
# while hasFrames:
#     print('                          ', end='\r')
#     print('Seconds Elapsed: {:2f}'.format(sec), end='\r')
#     sec += frameRate
#     vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#     hasFrames, image = vidcap.read()
#     if hasFrames:
#         # images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         images.append(image)
# vidcap.release()
# vidcap = None

# size = images[0].shape
# out = cv2.VideoWriter('newoutput.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (size[1], size[0]), True)
# for img in images:
#     print(img.shape, img.dtype)
#     print()
#     # data = np.random.randint(0, 256, size, dtype='uint8')
#     out.write(img)
# out.release()

def vidwrite(fn, images, framerate=60, vcodec='libx265'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

# vidwrite('output.mp4', np.random.randint(0, 256, (120, 200, 200, 3), dtype='uint8'))
# subprocess.run(['ffmpeg', '-framerate', '30', '-pattern_type', 'glob', '-i', 'data/480p-chairlift/*.jpg', '-c:v', 'libx265', '-crf', '0', '-c:a', 'aac', '-b:a', '128k', '-tag:v', 'hvc1', '-pix_fmt', 'yuv420p', 'data/chairlift.mp4'])

for folder in os.listdir('data/mturk-shoulder-surf/videos_original_small'):
    subprocess.run(['ffmpeg', '-framerate', '30', '-pattern_type', 'glob', '-i', 'data/mturk-shoulder-surf/videos_original_small/{}/*.png'.format(folder), '-c:v', 'libx264', '-crf', '1', '-pix_fmt', 'yuv420p', 'data/mturk-shoulder-surf/videos_original_small/{}.mp4'.format(folder.replace('480-', ''))])

# for folder in os.listdir('data/mturk-shoulder-surf/videos_original_small'):
#     if not folder.endswith('.mp4'):
#         for img in os.listdir('data/mturk-shoulder-surf/videos_original_small/{}'.format(folder)):
#             if img.endswith('.jpg'):
#                 Image.open(os.path.join('data/mturk-shoulder-surf/videos_original_small', folder, img)).save(os.path.join('data/mturk-shoulder-surf/videos_original_small/', folder, img.replace('.jpg', '.png')))
