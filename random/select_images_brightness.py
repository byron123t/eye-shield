import os
import random
from shutil import copy2

all_images = ['data/dark-close',
              'data/dark-side',
              'data/dark-far',
              'data/light-close',
              'data/light-side',
              'data/light-far',
              'data/orig-close',
              'data/orig-side',
              'data/orig-far']

with open('data/csvs/cloud_images_brightness.csv', 'w') as outfile:
    for dataset in all_images:
        files = os.listdir(dataset)
        outfile.write('{}: {}\n'.format(dataset, files))
