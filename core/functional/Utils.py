"""
File for some useful functions and values.
Include settings for neuro model.
"""
import csv
import os
import stat
from enum import Enum
from os import PathLike
from os.path import exists

from PIL import Image
from matplotlib import pyplot as mpl
from matplotlib.image import imread
from termcolor import colored
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from core.functional.Settings import (
    input_img_size,
    success_img_indicator,
    failure_img_indicator,
    __static_pic_ext__,
    user_input_cursor
)
from core.functional.custom_types.GameResultsDataset import GameResultsDataset

transform_func_train = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.4),
    transforms.GaussianBlur(5),
    transforms.Grayscale(),
    transforms.RandomCrop(2),
    transforms.RandomErasing(),
    transforms.Resize(input_img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
"""
Heavy version of transform function.
Performs a lot of transformation, like color jitter, random crop and random erasing.
"""

transform_func_classify = transforms.Compose([
    transforms.Grayscale,
    transforms.Resize(input_img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
"""
Version of transform function for classify task.
Performs transformation for images before their classification.
"""

transform_func_lite = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(input_img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
"""
Lite version of the transform function.
Performs lite number of transformation. Same as classify transformations, but for semantics value their different functions.
"""


class Test_results(Enum):
    """
    Enum class for storing results for test.
    """
    SUCCESS = 'Success',
    FAILED = 'Fail',
    SKIPPED = 'Skip'

    def get_value(self):
        """
        Method for receiving value of enum object.
        :return: string value of enum object.
        """
        return self.value


def proceed_image(path: str | PathLike) -> Tensor:
    """
    Function to proceed image with matplotlib methods.
    :return: torch tensor.
    """
    try:
        print('Proceed image invoked.')
        image = Image.open(path)
        image.resize(input_img_size)
        proceeded_image = transform_func_classify(image)
        return proceeded_image
    except Exception as e:
        print(e.__cause__)
        print_error(f'Error occurred in proceed_image function - {e.with_traceback(None)}.')


def get_dataloader(labels_dir_path: str | PathLike, img_dir_path: str | PathLike, batch_size=64, shuffle_sets=True) -> \
        tuple[DataLoader, DataLoader | None] | None:
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
            print_info('Be careful, only first dataloader is not None.')
            return train_loader, None
        else:
            print_error('Both, train dataloader and test dataloader are None.')
    except RuntimeError as e:
        print(e.__cause__)
        print_error(f'Error in dataloader create - {e.with_traceback(None)}.')


def show_img(path: str | PathLike, image_size: tuple[int, int] | list[int, int] = (300, 300)):
    """
    Function for showing image by matplot library.
    :return: nothing.
    """
    try:
        with open(path):
            image = imread(path)
            mpl.imshow(image, cmap='gray')
            mpl.title('Image')
            mpl.axis('off')
            mpl.show()
            mpl.close()
    except Exception as e:
        print(e.__cause__)
        print_error(f'Error occurred in show_img function - {e.with_traceback(None)}.')


test_labels: dict[str, int] = {
    'Success': 0,
    'Failed': 1
}
"""
Map of test labels.
"""


def update_labels(labels_dir_path: str | PathLike, images_dir_name: str = 'images',
                  labels_file_name: str = 'labels.csv'):
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
    concat_path = labels_dir_path + labels_file_name
    try:
        os.chmod(concat_path, stat.S_IWUSR)
        with open(concat_path, 'w+') as csvfile:
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
                    raise RuntimeError(colored(f'Unknown image identifier. Expected image ending with {success_img_indicator} or {failure_img_indicator}.', 'red'))
                csvwriter.writerow(row)
                row = list()
        csvfile.close()
        os.chmod(concat_path, stat.SF_IMMUTABLE)
    except Exception as e:
        print_error(f'Error in update labels because - {e.with_traceback(None)}.')
        os.chmod(concat_path, stat.SF_IMMUTABLE)


def get_max_size(path_to_images_dir: str, is_beautiful_output: bool = True) -> tuple[int, int] | None:
    """
    Static function for receiving maximum image size in Image directory.

    :param path_to_images_dir string path to images directory.
    :param is_beautiful_output: beautiful values output (optional).
    :return: tuple with (width x height).
    """
    _max_width: int = 0
    _max_height: int = 0
    try:
        for image in os.listdir(path_to_images_dir):
            if image.endswith(__static_pic_ext__):
                im = Image.open(path_to_images_dir + image)
                width, height = im.size
                if width > _max_width:
                    _max_width = width
                if height > _max_height:
                    _max_height = height
        if is_beautiful_output:
            print_info(f'Maximum width of the images - "{_max_width}", Maximum height of the images - "{_max_height}".')
        else:
            return _max_width, _max_height
    except Exception as e:
        print_error(f'Error in get max size of image because - {e.with_traceback(None)}.')


def select_terminal(items_directory_path: str, is_full_path_ret: bool = False) -> str | PathLike:
    """
    Selects item to be proceeded in neuro network.
    :param is_full_path_ret: indicates if you need full path to directory or not.
    :param items_directory_path: directory where you need terminal.
    :return: path to use.
    """
    try:
        dir_container = os.listdir(items_directory_path)
        if exists(items_directory_path) and len(dir_container) != 0:
            image_counter = 0
            items_list: list[str] = list()
            for image in dir_container:
                print(f'\t №{image_counter}. {image}')
                items_list.append(image)
                image_counter += 1
            while True:
                print('Select Item, enter number of item.')
                print(user_input_cursor, end='')
                user_input = int(input())
                if user_input.is_integer() and user_input in range(len(items_list)):
                    if is_full_path_ret:
                        return f'{items_directory_path}{items_list[user_input]}'
                    else:
                        return items_list[user_input]
                else:
                    print_error('Wrong argument, try again.')
                    continue
    except Exception as e:
        print(e.__cause__)
        print_error(f'Error occurred in Terminal function - {e.with_traceback(None)}.')


def input_from_user(values_range: int = None) -> int:
    """
    Error safety static function for input integer number from user.
    :param values_range: accepts if inputted value in this range.
    :return: integer number from user.
    """
    try:
        print(user_input_cursor, end='')
        user_input = int(input())
        if values_range is not None and user_input in values_range:
            print('Value in given range.')
            return user_input
        return user_input
    except Exception as e:
        print(e.__cause__)
        print_error(f'Error occurred in input_from_user function - {e.with_traceback(None)}.')


def user_input_with_exit(values_range: int = None) -> int | str:
    """
    Error safety static function for input string or integer number from user, that supports exit from loop.
    Input value can be string or integer.
    :param values_range: accepts if inputted value in this range.
    :return: integer number from user or "exit" value if user wants to exit loop.
    """
    try:
        print(user_input_cursor, end='')
        user_input: int | str = input()
        if user_input.isdigit():
            user_input = int(user_input)
            if values_range is not None and user_input in values_range:
                print('Value in given range.')
                return user_input
            return user_input
        else:
            if user_input == 'exit':
                return user_input
    except Exception as e:
        print(e.__cause__)
        print_error(f'Error occurred in input_from_user function - {e.with_traceback(None)}.')


def get_model_parameters(model_state_dict: dict):
    """
    Static function for retrieving data from model state_dict.
    :param model_state_dict: full architecture of model, include weights and biases.
    :return: nothing.
    """
    keys = model_state_dict.keys()
    keys_length = len(keys)
    keys_list: list = list()
    counter: int = 0
    func = lambda: print('Model layer:')
    if keys_length != 0:
        while True:
            print('Select one value, of:')
            print('Enter "exit" to out.')
            func()
            for key in keys:
                keys_list.append(key)
                key: str = key.replace('__model__.', '')
                print(f'\t №{counter}. {key}')
                if key.endswith('.bias') and len(keys_list) != keys_length:
                    func()
                counter += 1
            user_input = user_input_with_exit(values_range=len(keys_list))
            if user_input == 'exit':
                break
            selected_key = keys_list[user_input]
            print(selected_key)
            print(model_state_dict[selected_key])
            break
    else:
        print_error('Given state dict is None.')


def print_error(msg: str):
    """
    Static function for printing error with red color.
    :param msg: message to print.
    """
    print(colored(msg, 'red'))


def print_success(msg: str):
    """
    Static function for printing success with green color.
    :param msg: message to print.
    """
    print(colored(msg, 'green'))


def print_info(msg: str):
    """
    Static function for printing info messages with blue color.
    :param msg: message to print.
    """
    print(colored(msg, 'blue'))
