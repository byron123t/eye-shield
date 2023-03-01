import os
import random
from shutil import copy2


def get_computed_images(image_dict):
    """
    Given a dictionary of image paths, return a new dictionary with the same keys, but with the values
    being a list of paths to the computed images
    
    :param image_dict: a dictionary of the form {directory: [list of images in directory]}
    :return: A dictionary of the form {'directory': [list of images]}
    """
    new_image_dict = {}
    for directory, imgs in image_dict.items():
        new_image_dict[directory] = imgs
    parameters_paths = [os.path.join('data/blurred_downscaled', 'blur-1-8'),
                        os.path.join('data/blurred_downscaled', 'blur-1-16'),
                        os.path.join('data/blurred_downscaled', 'blur-1-24'),
                        os.path.join('data/blurred_downscaled', 'blur-1-32'),
                        os.path.join('data/blurred_downscaled', 'blur-2-8'),
                        os.path.join('data/blurred_downscaled', 'blur-2-16'),
                        os.path.join('data/blurred_downscaled', 'blur-2-24'),
                        os.path.join('data/blurred_downscaled', 'blur-2-32'),
                        os.path.join('data/blurred_downscaled', 'blur-3-8'),
                        os.path.join('data/blurred_downscaled', 'blur-3-16'),
                        os.path.join('data/blurred_downscaled', 'blur-3-24'),
                        os.path.join('data/blurred_downscaled', 'blur-3-32'),
                        os.path.join('data/blurred_downscaled', 'blur-4-8'),
                        os.path.join('data/blurred_downscaled', 'blur-4-16'),
                        os.path.join('data/blurred_downscaled', 'blur-4-24'),
                        os.path.join('data/blurred_downscaled', 'blur-4-32'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-1-8'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-1-16'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-1-24'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-1-32'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-2-8'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-2-16'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-2-24'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-2-32'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-3-8'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-3-16'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-3-24'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-3-32'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-4-8'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-4-16'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-4-24'),
                        os.path.join('data/pixelated_downscaled', 'pixelate-4-32'),
                        os.path.join('data/hidden_downscaled', 'blur-1-8'),
                        os.path.join('data/hidden_downscaled', 'blur-1-16'),
                        os.path.join('data/hidden_downscaled', 'blur-1-24'),
                        os.path.join('data/hidden_downscaled', 'blur-1-32'),
                        os.path.join('data/hidden_downscaled', 'blur-2-8'),
                        os.path.join('data/hidden_downscaled', 'blur-2-16'),
                        os.path.join('data/hidden_downscaled', 'blur-2-24'),
                        os.path.join('data/hidden_downscaled', 'blur-2-32'),
                        os.path.join('data/hidden_downscaled', 'blur-3-8'),
                        os.path.join('data/hidden_downscaled', 'blur-3-16'),
                        os.path.join('data/hidden_downscaled', 'blur-3-24'),
                        os.path.join('data/hidden_downscaled', 'blur-3-32'),
                        os.path.join('data/hidden_downscaled', 'blur-4-8'),
                        os.path.join('data/hidden_downscaled', 'blur-4-16'),
                        os.path.join('data/hidden_downscaled', 'blur-4-24'),
                        os.path.join('data/hidden_downscaled', 'blur-4-32'),
                        os.path.join('data/hidden_downscaled', 'pixelate-1-8'),
                        os.path.join('data/hidden_downscaled', 'pixelate-1-16'),
                        os.path.join('data/hidden_downscaled', 'pixelate-1-24'),
                        os.path.join('data/hidden_downscaled', 'pixelate-1-32'),
                        os.path.join('data/hidden_downscaled', 'pixelate-2-8'),
                        os.path.join('data/hidden_downscaled', 'pixelate-2-16'),
                        os.path.join('data/hidden_downscaled', 'pixelate-2-24'),
                        os.path.join('data/hidden_downscaled', 'pixelate-2-32'),
                        os.path.join('data/hidden_downscaled', 'pixelate-3-8'),
                        os.path.join('data/hidden_downscaled', 'pixelate-3-16'),
                        os.path.join('data/hidden_downscaled', 'pixelate-3-24'),
                        os.path.join('data/hidden_downscaled', 'pixelate-3-32'),
                        os.path.join('data/hidden_downscaled', 'pixelate-4-8'),
                        os.path.join('data/hidden_downscaled', 'pixelate-4-16'),
                        os.path.join('data/hidden_downscaled', 'pixelate-4-24'),
                        os.path.join('data/hidden_downscaled', 'pixelate-4-32')]
    for path in parameters_paths:
        for directory, imgs in image_dict.items():
            dataset = directory.split('/')[1]
            if os.path.join(path, dataset) not in new_image_dict:
                new_image_dict[os.path.join(path, dataset)] = imgs
    return new_image_dict


def randomly_sample(all_images, dataset, count):
    """
    This function randomly samples count images from the dataset and returns a list of the sampled
    images
    
    :param all_images: a dictionary with keys as the dataset names and values as the list of images in
    that dataset
    :param dataset: the name of the dataset, such as "div2kvalid", "div2ktrain", or "ricovalid"
    :param count: The number of images to be randomly sampled from the dataset
    :return: A dictionary with the keys being the dataset names and the values being the list of images
    in that dataset.
    """
    all_images[dataset] = os.listdir(dataset)
    random.shuffle(all_images[dataset])
    all_images[dataset] = all_images[dataset][:count]
    return all_images


all_images = {'data/div2kvalid': [],
              'data/div2ktrain': [],
              'data/480p-bmx-rider': [],
              'data/1080p-bmx-rider': [],
              'data/480p-butterfly': [],
              'data/1080p-butterfly': [],
              'data/480p-chairlift': [],
              'data/1080p-chairlift': [],
              'data/480p-dog-competition': [],
              'data/1080p-dog-competition': [],
              'data/480p-dolphins-show': [],
              'data/1080p-dolphins-show': [],
              'data/ricovalid': []}

all_images = randomly_sample(all_images, 'data/div2kvalid', 25)
all_images = randomly_sample(all_images, 'data/div2ktrain', 205)
all_images = randomly_sample(all_images, 'data/480p-bmx-rider', 2)
all_images = randomly_sample(all_images, 'data/1080p-bmx-rider', 2)
all_images = randomly_sample(all_images, 'data/480p-butterfly', 2)
all_images = randomly_sample(all_images, 'data/1080p-butterfly', 2)
all_images = randomly_sample(all_images, 'data/480p-chairlift', 2)
all_images = randomly_sample(all_images, 'data/1080p-chairlift', 2)
all_images = randomly_sample(all_images, 'data/480p-dog-competition', 2)
all_images = randomly_sample(all_images, 'data/1080p-dog-competition', 2)
all_images = randomly_sample(all_images, 'data/480p-dolphins-show', 2)
all_images = randomly_sample(all_images, 'data/1080p-dolphins-show', 2)
all_images = randomly_sample(all_images, 'data/ricovalid', 250)
all_images = get_computed_images(all_images)

with open('data/csvs/cloud_images.csv', 'w') as outfile:
    for key, val in all_images.items():
        outfile.write('{}: {}\n'.format(key, val))
