'''
Do basic preprocessing of images including resizing, extracting labels,
converting to arrays, and saving arrays to disk. This is a helper function to
avoid having to do it every time the model is trained or tested.
'''
from keras.preprocessing.image import img_to_array
import logging
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import random
from resizeimage import resizeimage
import sys
import tensorflow as tf


TRAIN_DATA_PATH = '../../../kaggle-data/cats-dogs/train/'
TEST_DATA_PATH = '../../../kaggle-data/cats-dogs/test/'
TARGET_SIZE = (224,224)

def extract_labels(train_files, test_files):
    '''
    Image filenames begin with the 'cat' or 'dog'. Use this to extract labels.
    Model is expected to be a binary classifier so cats = 0 and dogs = 1
    '''

    logging.info('Extracting labels ...')
    print('\nExtracting labels ...')

    train_labels = []
    test_labels = []

    for idx, pet in enumerate(train_files):
        if pet.startswith('cat'):
            train_labels.append(0)
        else:
            train_labels.append(1)

    for idx, pet in enumerate(test_files):
        if pet.startswith('cat'):
            test_labels.append(0)
        else:
            test_labels.append(1)

    return train_labels, test_labels

def get_file_name_lists():

    logging.info('Getting image file name lists ...')
    print('\nGetting image file name lists ...')

    train_file_list = [f for f in listdir(TRAIN_DATA_PATH) if isfile(join(TRAIN_DATA_PATH, f))]
    random.shuffle(train_file_list)
    test_file_list = [f for f in listdir(TEST_DATA_PATH) if isfile(join(TEST_DATA_PATH, f))]
    random.shuffle(test_file_list)

    return train_file_list, test_file_list

def process_images(train_files, test_files):

    logging.info('Resizing images and converting to arrays ...')
    print('\nResizing images and converting to arrays ...')

    depth = 3

    train_arr = np.zeros((len(train_files), TARGET_SIZE[0], TARGET_SIZE[1], depth), dtype=np.uint8)
    for idx, img in enumerate(train_files):
        with open(TRAIN_DATA_PATH + img, 'r+b') as f:
            with Image.open(f) as image:
                resized_image = resizeimage.resize_contain(image, TARGET_SIZE)
                resized_image = resized_image.convert("RGB")
                X = img_to_array(resized_image).astype(np.uint8)
                train_arr[idx] = X
        if (idx + 1) % 1000 == 0:
            print(idx+1, "out of", len(train_files), "training images have been processed")

    test_arr = np.zeros((len(test_files), TARGET_SIZE[0], TARGET_SIZE[1], depth), dtype=np.uint8)
    for idx, img in enumerate(test_files):
        with open(TEST_DATA_PATH + img, 'r+b') as f:
            with Image.open(f) as image:
                resized_image = resizeimage.resize_contain(image, TARGET_SIZE)
                resized_image = resized_image.convert("RGB")
                X = img_to_array(resized_image).astype(np.uint8)
                test_arr[idx] = X
        if (idx + 1) % 1000 == 0:
            print(idx+1, "out of", len(test_files), "test images have been processed")

    return train_arr, test_arr

def save_data(train_arr, train_labels, test_arr, test_labels):

    logging.info('Saving data to disk ...')
    print('\nSaving data to disk ...')

    print("Size of training array = ", sys.getsizeof(train_arr))
    np.save('../../../kaggle-data/cats-dogs/processed_training_images.npy', train_arr)

    print("Size of test array = ", sys.getsizeof(test_arr))
    np.save('../../../kaggle-data/cats-dogs/processed_test_images.npy', test_arr)

    np.save('../../../kaggle-data/cats-dogs/training_labels.npy', np.array(train_labels))
    np.save('../../../kaggle-data/cats-dogs/test_labels.npy', np.array(test_labels))

    pass


def main():
    random.seed(39)
    np.random.seed(39)
    tf.set_random_seed(39)

    train_filenames, test_filenames = get_file_name_lists()
    train_labels, test_labels = extract_labels(train_filenames, test_filenames)
    train_arr, test_arr = process_images(train_filenames, test_filenames)
    save_data(train_arr, train_labels, test_arr, test_labels)

    print('\nProcessing complete.')

    pass







if __name__ == '__main__':
    main()
