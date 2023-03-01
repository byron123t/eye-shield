# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

import os
import csv
from google.cloud import vision
from tqdm import tqdm


client = vision.ImageAnnotatorClient()


def load_image(path):
    """
    Loads an image from a file path

    :param path: The path to the image file you're sending to Vision API
    :return: Returns a Vision image
    """
    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    return image


def detect_text(path, image, text_dict):
    """
    The function takes in a path to an image and a dictionary of text.
    It returns a dictionary of text with the path as the key and a list of tuples containing the text
    and the vertices of the bounding box as the value

    :param path: The path to the image to be analyzed
    :param image: The image to be processed
    :param text_dict: a dictionary that stores the text found in each image
    :return: A dictionary with the path to the image as the key and the text as the value.
    """
    response = client.text_detection(image=image)
    texts = response.text_annotations
    text_dict[path] = []

    for text in texts:
        vertices = ([(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])
        text_dict[path].append((text.description,vertices))
    return text_dict


def detect_labels_uri(path, image, label_dict):
    """
    This function takes in a path to an image and a dictionary of labels.
    It returns a dictionary of labels with the path as the key and a list of tuples as the value.
    Each tuple contains the label description, label score, and label topicality

    :param path: The path to the image file you're sending to the Vision API
    :param image: The image to be processed
    :param label_dict: a dictionary that will contain the image path and all of the labels that were
    detected in that image
    :return: A dictionary with the image path as the key and the labels as the values.
    """
    response = client.label_detection(image=image)
    labels = response.label_annotations
    label_dict[path] = []

    for label in labels:
        label_dict[path].append((label.description, label.score, label.topicality))
    return label_dict


def main():
    """
    Reads in a csv file of image paths and their corresponding file names. 
    For each image, it runs the Google Cloud Vision API's label detection and text detection functions. 
    It then writes the results to two separate csv files.
    """
    # infile = 'cloud_images.csv'
    # outlabel = 'cloud_results_labels.csv'
    # outtext = 'cloud_results_text.csv'
    infiles = 'cloud_images_contrast.csv'
    outlabel = 'cloud_results_labels_contrast.csv'
    outtext = 'cloud_results_text_contrast_temp.csv'
    # infiles = 'cloud_images_brightness.csv'
    # outlabel = 'cloud_results_labels_brightness.csv'
    # outtext = 'cloud_results_text_brightness.csv'
    with open('data/csvs/{}'.format(infiles), 'r') as infile:
        label_dict = {}
        text_dict = {}
        for line in tqdm(infile):
            split = line.strip().split(': ')
            path = split[0]
            files = split[1].replace('[', '').replace(']', '').replace('\'', '').split(', ')
            for file in tqdm(files):
                img = load_image(os.path.join(path, file))
                if '/ricovalid' in path:
                    text_dict = detect_text(os.path.join(path, file), img, text_dict)
                else:
                    label_dict = detect_labels_uri(os.path.join(path, file), img, label_dict)
    # with open('data/csvs/{}'.format(outlabel), 'w') as outfile:
    #     writer = csv.DictWriter(outfile, fieldnames=['path', 'label', 'score', 'topicality'])
    #     for path, labels in label_dict.items():
    #         for label in labels:
    #             writer.writerow({'path': path, 'label': label[0], 'score': label[1], 'topicality': label[2]})
    with open('data/csvs/{}'.format(outtext), 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=['path', 'text', 'vertices'])
        for path, texts in text_dict.items():
            for text in texts:
                writer.writerow({'path': path, 'text': text[0], 'vertices': text[1]})


if __name__ == '__main__':
    main()
