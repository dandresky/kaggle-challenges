'''
Creates a train test split of files by randomly holding out 2.5k dog images and
2.5k cat images. Only need to call this once
Original training dataset downloaded from Kaggle:
    ../../kaggle-data/cats-dogs/original-train-data/
New data paths:
    ../../kaggle-data/cats-dogs/train/
    ../../kaggle-data/cats-dogs/test/
'''
import logging
from os import listdir
from os import rename
from os.path import isfile, join
import random
from random import shuffle

ORIGINAL_DATA_PATH = '../../kaggle-data/cats-dogs/original-train-data/'
TRAIN_DATA_PATH = '../../kaggle-data/cats-dogs/train/'
TEST_DATA_PATH = '../../kaggle-data/cats-dogs/test/'


def get_file_name_lists():
    '''
    Return lists of image file names for both cats and dogs
    '''
    logging.info('Getting list of file names and splitting between cats and dogs ...')
    print('\nGetting list of file names and splitting ...')

    # get list of all image files and sort by file name
    img_file_list = [f for f in listdir(ORIGINAL_DATA_PATH) if isfile(join(ORIGINAL_DATA_PATH, f))]
    cat_file_list = []
    dog_file_list = []
    for idx, pet in enumerate(img_file_list):
        if pet.startswith('cat'):
            cat_file_list.append(pet)
        else:
            dog_file_list.append(pet)

    return cat_file_list, dog_file_list


def move_files(cat_train_files, cat_test_files, dog_train_files, dog_test_files):

    logging.info('Move files into new folders ... ')
    print('\nMove files into new folders ...')

    for filename in cat_train_files:
        rename(ORIGINAL_DATA_PATH + filename, TRAIN_DATA_PATH + filename)
    for filename in dog_train_files:
        rename(ORIGINAL_DATA_PATH + filename, TRAIN_DATA_PATH + filename)
    for filename in cat_test_files:
        rename(ORIGINAL_DATA_PATH + filename, TEST_DATA_PATH + filename)
    for filename in dog_test_files:
        rename(ORIGINAL_DATA_PATH + filename, TEST_DATA_PATH + filename)
    pass


def split_data(cat_file_list, dog_file_list):
    # randomize the list index
    shuffle(cat_file_list)
    shuffle(dog_file_list)
    cat_train_files = cat_file_list[:int(len(cat_file_list)*0.8)]
    cat_test_files = cat_file_list[int(len(cat_file_list)*0.8):]
    dog_train_files = dog_file_list[:int(len(dog_file_list)*0.8)]
    dog_test_files = dog_file_list[int(len(dog_file_list)*0.8):]
    return cat_train_files, cat_test_files, dog_train_files, dog_test_files


def main():
    random.seed(39)
    cat_file_list, dog_file_list = get_file_name_lists()
    cat_train_files, cat_test_files, dog_train_files, dog_test_files = \
        split_data(cat_file_list, dog_file_list)
    move_files(cat_train_files, cat_test_files, dog_train_files, dog_test_files)
    pass










if __name__ == '__main__':
    main()
