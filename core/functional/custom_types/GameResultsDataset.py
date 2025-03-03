from typing import Any

import pandas as pd
from torch.utils.data import (
    Dataset,
    ConcatDataset
)
from torch.utils.data.dataset import _T_co
from torchvision.transforms import Compose

from core.functional.Utils import *


class GameResultsDataset(Dataset):
    img_labels: Any
    """
    Image labels, that read from csv file.
    """
    img_dir: str
    """
    Directory, where stored images.
    """
    target_transform: Any

    def __init__(self, labels_file_path: str, img_dir_path: str, transform, target_transform=None):
        """
        Exception safety Custom data set object.
        Inherits Dataset class of the torch.util.
        :param labels_file_path: CSV file where contains annotation labels.
        :param img_dir_path: directory where contains images to train on.
        :param transform: transformations which will be applied to image.
        :param target_transform: ?.
        """
        try:
            self.img_labels = pd.read_csv(labels_file_path, lineterminator='\n',
                                          skiprows=1)  # skip first line, because first line is header.
            self.img_dir = img_dir_path
            self.transform = transform
            self.target_transform = target_transform
        except Exception as e:
            print(e.with_traceback(None))
            print_error("An error occurred in init custom game results dataset.")

    def __add__(self, other: "Dataset[_T_co]") -> ConcatDataset[_T_co] | None:
        """
        Method for adding dataset in inner structure.
        :param other: Object to add.
        :return: nothing.
        """
        if other is not None:
            return super().__add__(other)
        else:
            print_error('Error in adding element to dataset.')

    def __getitem__(self, idx: int) -> _T_co:
        """
        Method for getting item by index parameter.
        :param idx: index of which item get.
        :return: generic type of return value.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        if img_path is not None:
            image = Image.open(img_path)
            label = self.img_labels.iloc[idx, 1]
            image = self.transform(image).unsqueeze(0)
            if self.target_transform:
                label = self.target_transform(label)
                print('Label target transform invoked.')
            return image, label
        else:
            print_error('Error occurred in inner get item method of custom dataset.')

    def __len__(self) -> int:
        """
        Zero safety length function of the dataset.
        :return: length (int parameter).
        """
        length = len(self.img_labels)
        if length != 0:
            return length
        else:
            print('Zero value length.')
            return 0

    def get_image_labels(self) -> Any:
        """
        None safety method for get image labels.
        :return: image labels of dataset.
        """
        if self.img_labels is not None:
            return self.img_labels
        else:
            print_error('Image labels are None.')

    def get_img_dir(self) -> str | None:
        """
        None safety method for get image directory.
        :return: path to image directory.
        """
        if self.img_dir is not None:
            return self.img_dir
        else:
            print_error('Image directory is None.')

    def get_transform(self) -> Compose | None:
        """
        None safety method for get transformation.
        :return: transform, that will be used.
        """
        if self.transform is not None:
            return self.transform
        else:
            print_error('Transform is None.')

    def get_target_transform(self) -> Any | None:
        """
        None safety method for get target transformation.
        :return: target_transform.
        """
        if self.target_transform is not None:
            return self.target_transform
        else:
            print_error('Target transform is None.')
