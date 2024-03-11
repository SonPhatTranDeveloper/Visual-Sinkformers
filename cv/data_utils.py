"""
This file prepares the cats and dogs classification dataset
"""
import os
import glob
from sklearn.model_selection import train_test_split


def create_file_lists(train_dir: str, test_dir: str, validation_size: float, random_seed: int):
    """
    Create the train, validation and test file lists containing the image's file path
    :param train_dir: the directory of the train dataset
    :param test_dir: the director of the test dataset
    :param validation_size: the size of the validation set
    :param random_seed: random seed for train-validation split
    :return: train files list, validation files list, test files list
    """
    # Get all the file names from train and test sets
    train = glob.glob(os.path.join(train_dir, '*.jpg'))
    test = glob.glob(os.path.join(test_dir, '*.jpg'))

    # Get the labels of train dataset
    labels = [path.split('/')[-1].split('.')[0] for path in train]

    # Split the train set to obtain the validation set
    train, validation = train_test_split(
        train,
        test_size=validation_size,
        stratify=labels,
        random_state=random_seed
    )

    return train, validation, test


if __name__ == "__main__":
    # Define the paths
    TRAIN_DIR = "../data/cats-and-dogs/train"
    TEST_DIR = "../data/cats-and-dogs/test"

    # Get the file lists
    train_list, validation_list, test_list = create_file_lists(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        validation_size=0.2,
        random_seed=42
    )

    print(len(train_list))