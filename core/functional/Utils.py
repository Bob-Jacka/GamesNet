"""
File for some useful functions and values.
Include settings for neuro model.
"""
import csv
import os
import stat
from enum import Enum

from PIL import Image
from matplotlib import pyplot as mpl
from torch.utils.data import DataLoader
from torchvision import transforms

from core.functional.Settings import input_img_size, success_img_indicator, failure_img_indicator
from core.functional.custom_types.GameResultsDataset import GameResultsDataset

transform_func_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.ColorJitter(brightness=0.3, contrast=0.4),
    transforms.GaussianBlur(5),
    transforms.RandomCrop(2),
    transforms.RandomErasing(),
    transforms.Resize(input_img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
"""
Функция для преобразования данных для обучения модели.
"""

transform_func_classify = transforms.Compose([
    transforms.Grayscale,
    transforms.Resize(input_img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
"""
Функция для преобразования данных в те которые модель понимает.
"""

transform_func_lite = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(input_img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
"""
Lite version of the transform function.
"""


class Test_results(Enum):
    """
    Enum class for storing results for test.
    """
    SUCCESS = 'Success',
    FAILED = 'Fail',
    SKIPPED = 'Skip'


def proceed_image(path: str):
    """
    Function to proceed image with matplot lib methods.
    :return: torch tensor.
    """
    print('Proceed image invoked.')
    image = Image.open(path)
    image.resize(input_img_size)
    pass  # TODO дописать метод


def get_dataloader(labels_dir_path: str, img_dir_path: str, batch_size=64, shuffle_sets=True) -> tuple[
                                                                                                     DataLoader, DataLoader | None] | None:
    """
    Function for creating dataloader. First DataLoader - train, Second DataLoader - validate.
    :param img_dir_path: path to directory where images are stored.
    :param labels_dir_path: path to directory where labels are stored.
    :param shuffle_sets: True or False.
    :param batch_size: size of the dataloader, by default equals 64.
    :return: DataLoader object with data.
    """
    try:
        train_set = GameResultsDataset(labels_dir_path,
                                       img_dir_path, transform=transform_func_lite)

        test_set = GameResultsDataset(labels_dir_path,
                                      img_dir_path, transform=transform_func_classify)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_sets)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle_sets)
        if train_loader is not None and test_loader is not None:
            return train_loader, test_loader
        elif train_loader is not None and test_loader is None:
            print('Be careful, only first dataloader is not None.')
            return train_loader, None
        else:
            print('Both, train dataloader and test dataloader are None.')
    except RuntimeError as e:
        print(e.__cause__)
        print("Error in dataloader create.")


def show_img(image_size: tuple[int, int] = (700, 700)):
    """
    Function for showing image by matplot library.
    :return: nothing.
    """
    figure = mpl.figure(figsize=image_size)
    for i in range(1):
        # sample_idx = torch.randint(len(training_data), size=(1,)).item()
        # img, label = training_data[sample_idx] # TODO исправить значения
        figure.add_subplot(i)
        mpl.title('Image')
        mpl.axis('off')
        # plt.imshow(img.squeeze(), cmap="gray")
    mpl.show()


test_labels: dict[str, int] = {
    'Success': 0,
    'Failed': 1
}
"""
Map of test labels.
"""


def update_labels(labels_dir_path: str, images_dir_name: str = 'images', labels_file_name: str = 'labels.csv'):
    """
    Static function for updating labels in csv file.
    Opens file if it exits or creates new file if it not and then rewritten (written) data to file on every call.
    Before every call unlocks file for writing, after every call locks file for writing.

    RULES:
        1. Images names must ends with {success_img_indicator} or {failure_img_indicator}.
        2. labels_file_name must include '.csv' extension.
        3. labels_dir_path must fill out in main.py file.

    :param labels_dir_path path where labels are stored.
    :param images_dir_name path where images are stored.
    :param labels_file_name: name of the labels file.
    :return: None.
    """
    os.chmod(labels_dir_path + labels_file_name, stat.S_IWUSR)
    with open(labels_dir_path + labels_file_name, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator='\n')
        label: int
        row: list = list()  # row of the csv file.
        images_path_dir: str = labels_dir_path + images_dir_name
        csvwriter.writerow(['Games', 'Value'])  # writes down headers to csv file.
        for file_name in os.listdir(images_path_dir):
            if file_name.endswith(success_img_indicator):
                label = test_labels['Success']
                row.append(file_name)
                row.append(label)
            elif file_name.endswith(failure_img_indicator):
                label = test_labels['Failed']
                row.append(file_name)
                row.append(label)
            else:
                raise RuntimeError(
                    f'Unknown image identifier. Expected image ending with {success_img_indicator} or {failure_img_indicator}')
            csvwriter.writerow(row)
            row = list()

    csvfile.close()
    os.chmod(labels_dir_path + labels_file_name, stat.SF_IMMUTABLE)
