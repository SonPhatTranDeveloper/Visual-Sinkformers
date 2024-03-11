"""
This file defines the dataset for cats and dogs images
"""
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

from data_utils import create_file_lists

# Define the transformation for the images
def create_image_transformation():
    """
    Create transformation pipeline for the images
    :return: transformation pipeline
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )


# Define the Cats and Dogs dataset
class CatsAndDogsDataset(Dataset):
    def __init__(self, file_list, transformation):
        """
        Initialize the Cats and Dogs dataset
        :param file_list: the file list of the images
        :param transformation: transformation to apply
        """
        # Cache the values
        self.file_list = file_list
        self.transform = transformation

    def __len__(self):
        """
        Get the size of the dataset
        :return: the size of the dataset
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Get an image as tensor from the dataset
        :param index: index of the image
        :return: the image
        """
        # Open and transform the image
        image_path = self.file_list[index]
        image = Image.open(image_path)
        transformed_image = self.transform(image)

        # Get the label of the image
        image_label = image_path.split("/")[-1].split(".")[0]
        image_path = 1 if image_label == "dog" else 0

        # Return the image and label
        return transformed_image, image_path


if __name__ == "__main__":
    # Get the file lists for the train, validation, and test images
    train_list, validation_list, test_list = create_file_lists(
        train_dir="../data/cats-and-dogs/train",
        test_dir="../data/cats-and-dogs/test",
        validation_size=0.2,
        random_seed=42
    )

    # Create transformation for train, validation, and test dataset
    TRAIN_TRANSFORMATION = create_image_transformation()
    VALIDATION_TRANSFORMATION = create_image_transformation()
    TEST_TRANSFORMATION = create_image_transformation()

    # Create dataset
    train_dataset = CatsAndDogsDataset(train_list, TRAIN_TRANSFORMATION)
    validation_dataset = CatsAndDogsDataset(validation_list, VALIDATION_TRANSFORMATION)
    test_dataset = CatsAndDogsDataset(test_list, TEST_TRANSFORMATION)

    # Get a single image and label
    image, label = train_dataset[0]
    print(label)

